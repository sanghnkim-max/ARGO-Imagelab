//! The part of the core every surface shares: registry + shared state, the swappable event
//! sink, `DocChanged` detection, running scripts and their cancel flags.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, Weak};
use std::time::Duration;

use imagelab_api::{ApiResult, Ctx, Event, EventSink, NoEvents, Registry, Shared, Source};
use imagelab_script::{Budget, ScriptError, ScriptOutcome, ScriptRuntime};
use serde_json::Value;

/// How often the watcher looks for edits made through PhotoCraft's HTTP MCP endpoint.
const WATCH_INTERVAL: Duration = Duration::from_millis(250);

/// An [`EventSink`] whose target can be replaced at any time (the FFI installs the Kotlin
/// listener after the core exists). A sink that panics loses that event, nothing more.
pub(crate) struct ForwardingSink {
    inner: RwLock<Arc<dyn EventSink>>,
}

impl Default for ForwardingSink {
    fn default() -> Self {
        Self { inner: RwLock::new(Arc::new(NoEvents)) }
    }
}

impl ForwardingSink {
    pub(crate) fn set(&self, sink: Arc<dyn EventSink>) {
        *self.inner.write().unwrap_or_else(PoisonError::into_inner) = sink;
    }
}

impl EventSink for ForwardingSink {
    fn emit(&self, event: Event) {
        let sink = self.inner.read().unwrap_or_else(PoisonError::into_inner).clone();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sink.emit(event)));
    }
}

/// Decrements the in-flight counter when dropped.
pub(crate) struct Busy<'a>(&'a AtomicUsize);

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The active document and its revision as last reported, shared by the core and the
/// registry's after-edit hook, so a change is announced once whoever notices it first.
#[derive(Default)]
pub(crate) struct DocWatch {
    seen: Mutex<Option<(usize, u64)>>,
}

impl DocWatch {
    /// Emit `DocChanged` attributed to `source` if the active document or its revision moved.
    pub(crate) fn report(&self, shared: &Shared, current: Option<(usize, u64)>, source: Source) {
        let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
        if *seen == current {
            return;
        }
        *seen = current;
        drop(seen);
        let (doc, revision) = current.unwrap_or((0, 0));
        shared.events.emit(Event::DocChanged { doc, revision, source });
    }

    /// Read the session (it is free between calls) and report.
    pub(crate) fn notice(&self, shared: &Shared, source: Source) {
        let list = shared.session.lock().unwrap_or_else(PoisonError::into_inner).session_list();
        self.report(shared, active_revision(&list), source);
    }
}

pub(crate) struct Core {
    pub registry: Registry,
    pub shared: Shared,
    pub sink: Arc<ForwardingSink>,
    watch: Arc<DocWatch>,
    /// Calls in flight inside the core; the watcher stays quiet while any run, so an edit is
    /// attributed to the surface that made it.
    in_flight: AtomicUsize,
    /// Cancel flags of the scripts running now.
    running: Mutex<Vec<Arc<AtomicBool>>>,
    /// Held by an editing (not read-only) program for its whole run: every step works on the
    /// active document, so two editing programs at once (the in-app agent and an MCP client,
    /// say) would land steps on each other's photo. Read-only programs never wait.
    editing: Mutex<()>,
}

impl Core {
    pub(crate) fn new(mut registry: Registry, shared: Shared, sink: Arc<ForwardingSink>) -> Self {
        let watch = Arc::new(DocWatch::default());
        // Live preview: every editing call (each step of a script included) announces its
        // change as soon as it returns.
        let w = watch.clone();
        registry.set_after_mutation(Arc::new(move |ctx: &Ctx| w.notice(&ctx.shared, ctx.source)));
        let core = Self { registry, shared, sink, watch, in_flight: AtomicUsize::new(0), running: Mutex::new(Vec::new()), editing: Mutex::new(()) };
        // Start from the current state, so the first event is a real change.
        let current = core.current();
        *core.watch.seen.lock().unwrap_or_else(PoisonError::into_inner) = current;
        core
    }

    pub(crate) fn busy(&self) -> Busy<'_> {
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        Busy(&self.in_flight)
    }

    pub(crate) fn call(&self, id: &str, params: Value, source: Source) -> ApiResult<Value> {
        let _busy = self.busy();
        let ctx = Ctx::new(self.shared.clone(), source);
        let result = self.registry.call(&ctx, id, params);
        self.notice(source);
        result
    }

    pub(crate) fn run_script(&self, code: &str, read_only: bool, timeout: Duration, source: Source, cancel: Arc<AtomicBool>) -> ScriptOutcome {
        let _busy = self.busy();
        let mut ctx = Ctx::new(self.shared.clone(), source);
        ctx.cancel = cancel.clone();
        // An agent's program never overwrites or deletes on its own: the person is asked.
        if matches!(source, Source::Agent | Source::Mcp) {
            ctx = ctx.asking_before_destroying();
        }
        if read_only {
            ctx = ctx.read_only();
        }
        let _editing = if read_only {
            None
        } else {
            match self.wait_to_edit(timeout, &cancel) {
                Some(guard) => Some(guard),
                None => {
                    return ScriptOutcome::failed(ScriptError::new(
                        "busy",
                        "another editing program is running on this photo and did not finish in time; try again when it is done",
                    ));
                }
            }
        };
        self.running.lock().unwrap_or_else(PoisonError::into_inner).push(cancel.clone());
        let outcome = ScriptRuntime::run(&self.registry, &ctx, code, Budget::with_wall(timeout));
        self.running.lock().unwrap_or_else(PoisonError::into_inner).retain(|c| !Arc::ptr_eq(c, &cancel));
        self.notice(source);
        outcome
    }

    /// The editing lock, waiting at most the run's own budget (or until it is cancelled).
    fn wait_to_edit(&self, timeout: Duration, cancel: &AtomicBool) -> Option<std::sync::MutexGuard<'_, ()>> {
        let budget = if timeout.is_zero() { Budget::default().wall } else { timeout };
        let give_up = std::time::Instant::now() + budget;
        loop {
            match self.editing.try_lock() {
                Ok(guard) => return Some(guard),
                Err(std::sync::TryLockError::Poisoned(p)) => return Some(p.into_inner()),
                Err(std::sync::TryLockError::WouldBlock) => {}
            }
            if cancel.load(Ordering::Relaxed) || std::time::Instant::now() >= give_up {
                return None;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    pub(crate) fn cancel_scripts(&self) {
        for flag in self.running.lock().unwrap_or_else(PoisonError::into_inner).iter() {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// The active document and its revision, from PhotoCraft's `session_list`.
    fn current(&self) -> Option<(usize, u64)> {
        let list = self.shared.session.lock().unwrap_or_else(PoisonError::into_inner).session_list();
        active_revision(&list)
    }

    /// Emit `DocChanged` attributed to `source` if the active document or its revision moved.
    pub(crate) fn notice(&self, source: Source) {
        let current = self.current();
        self.report(current, source);
    }

    /// [`Core::notice`] that gives up when the session is busy (the watcher).
    fn try_notice(&self, source: Source) {
        let current = match self.shared.session.try_lock() {
            Ok(guard) => active_revision(&guard.session_list()),
            Err(_) => return,
        };
        self.report(current, source);
    }

    fn report(&self, current: Option<(usize, u64)>, source: Source) {
        self.watch.report(&self.shared, current, source);
    }
}

fn active_revision(list: &Value) -> Option<(usize, u64)> {
    let active = usize::try_from(list.get("active")?.as_u64()?).ok()?;
    let doc = list.get("documents")?.as_array()?.get(active)?;
    Some((active, doc.get("revision")?.as_u64()?))
}

/// Notices edits that arrive through PhotoCraft's HTTP MCP endpoint (they bypass the core).
/// Runs only while the HTTP server does.
pub(crate) struct Watcher {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Watcher {
    pub(crate) fn start(core: Weak<Core>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let thread = std::thread::Builder::new()
            .name("imagelab-watch".into())
            .spawn(move || {
                while !flag.load(Ordering::Relaxed) {
                    std::thread::sleep(WATCH_INTERVAL);
                    let Some(core) = core.upgrade() else { break };
                    if core.in_flight.load(Ordering::Acquire) == 0 {
                        core.try_notice(Source::Mcp);
                    }
                }
            })
            .ok();
        Self { stop, thread }
    }

    pub(crate) fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_the_active_revision() {
        let list = json!({"active": 1, "documents": [{"revision": 3}, {"revision": 9}]});
        assert_eq!(active_revision(&list), Some((1, 9)));
        assert_eq!(active_revision(&json!({"active": null, "documents": []})), None);
    }
}
