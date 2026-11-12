//! # imagelab-script
//!
//! The agent's code runtime (docs/INTERFACES.md §3): one JavaScript program per tool call, run
//! in a fresh QuickJS realm (rquickjs) over the imagelab-api registry. Adapted, much smaller,
//! from ARGO's `argo-code-runtime` (realm per program, budgets, abort, error values) and its
//! `device` script tool (one tool, disclosed progressively through `help(area)`).
//!
//! ## Inside a program
//!
//! * The body is an async function: top-level `await`, and its `return` value is the result.
//! * `lab.<area>.<fn>(params)` is an async function per registry function; it resolves with
//!   the function's JSON result or rejects with a `LabCallError {code, message, fn}`.
//!   Unknown functions and areas reject with `code: "not_found"` and a pointer to `lab.help`.
//! * `lab.help()` lists areas and function names; `lab.help("cv")` (or `"cv.canny"`) returns
//!   full descriptions and parameter schemas. Synchronous; `await` works too.
//! * `lab.keep("img:3")` keeps a handle the run created alive after the run (see below).
//! * `console.log/info/debug/warn/error` are captured, bounded ([`Budget::max_log_bytes`]).
//! * Nothing else: no `fetch`, no timers, no file system, no modules.
//!
//! ## Budgets and stopping
//!
//! Wall time (an interrupt handler stops synchronous JS at its next check; a registry call in
//! progress finishes first), QuickJS heap limit, a maximum number of `lab` calls, and the
//! caller's [`Ctx::cancel`](imagelab_api::Ctx) flag (checked by the interrupt handler and by
//! every registry call). A read-only [`Ctx`] makes every non-ReadOnly function reject with
//! `LabCallError {code: "read_only"}`: the registry enforces it per call.
//!
//! ## Images and handles
//!
//! In the returned value, every object `{image: "img:N" | "mask:N", …}` is encoded to PNG
//! (masks as grayscale), appended to [`ScriptOutcome::images`], and gets `attached: k` (its
//! index there) added. The same handle twice is attached once.
//!
//! Handles a run creates are freed when it ends, unless the result mentions them (any string
//! equal to the handle, anywhere in the returned value) or the program called `lab.keep`. A
//! handle the result only shows as `{image: h}` is freed once encoded (the PNG is the result).
//! Inside a program, `cv.free` frees only handles its own run made.
//! "Created by the run" = a handle that appears in some `lab.*` call result of this run and
//! did not exist when the run started. Concurrent runs therefore never free each other's
//! handles, and handles that existed before the run are never freed by it.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod png;
mod realm;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use imagelab_api::{ApiError, Ctx, HandleId, Registry};
use serde::Serialize;
use serde_json::{Value, json};

pub use png::encode_handle_png;

/// The realm thread's stack: QuickJS and the JSON conversions recurse with program depth.
pub(crate) const REALM_THREAD_STACK: usize = 16 << 20;
/// Largest program source accepted.
pub const MAX_CODE_BYTES: usize = 256 << 10;
/// Longest wall budget anyone may ask for (batch jobs).
pub const MAX_WALL: Duration = Duration::from_secs(30 * 60);
/// After the wall budget, how long the host waits for a realm that does not stop on its own
/// (a registry call that ignores cancellation) before it reports a timeout without it.
const STOP_GRACE: Duration = Duration::from_secs(10);
/// How often the host loop wakes to relay the caller's cancel flag to the run.
const TICK: Duration = Duration::from_millis(50);

/// Limits for one run.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Wall time for the whole program, calls included. Capped at [`MAX_WALL`].
    pub wall: Duration,
    /// QuickJS heap limit (handles' pixels live outside it, in the arena).
    pub memory_bytes: usize,
    /// Most `lab.*` calls (help included).
    pub max_calls: u32,
    /// Console output kept.
    pub max_log_bytes: usize,
    /// Largest returned value, as JSON text.
    pub max_result_bytes: usize,
    /// Most images attached from the result.
    pub max_images: usize,
    /// Attached images are scaled down to this longest side (0: full size).
    pub max_image_side: u32,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            wall: Duration::from_secs(120),
            memory_bytes: 256 << 20,
            max_calls: 10_000,
            max_log_bytes: 64 << 10,
            max_result_bytes: 8 << 20,
            max_images: 16,
            max_image_side: 1568,
        }
    }
}

impl Budget {
    /// The default budget with another wall time (capped at [`MAX_WALL`]; 0 means default).
    pub fn with_wall(wall: Duration) -> Self {
        let mut b = Self::default();
        if !wall.is_zero() {
            b.wall = wall.min(MAX_WALL);
        }
        b
    }
}

/// Why a run failed. `code` is stable: `syntax`, `exception` (an uncaught JS error), a
/// `LabCallError`'s own code when one was not caught (`read_only`, `not_found`, `bad_params`…),
/// `timeout`, `cancelled`, `out_of_memory`, `too_many_calls`, `too_large`, `bad_result`
/// (the return value is not JSON), `never_settled`, `internal`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScriptError {
    pub code: String,
    pub message: String,
}

impl ScriptError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self { code: code.to_string(), message: message.into() }
    }
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

/// One image attached from the result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptImage {
    /// The handle it came from (`img:N` / `mask:N`).
    pub handle: String,
    /// Always `image/png` today.
    pub mime: String,
    pub bytes: Vec<u8>,
}

/// What a run produced. `value` is `null` when `error` is set.
#[derive(Debug, Clone)]
pub struct ScriptOutcome {
    pub value: Value,
    pub logs: Vec<String>,
    pub images: Vec<ScriptImage>,
    pub error: Option<ScriptError>,
    /// `lab.*` calls made.
    pub calls: u32,
    pub elapsed: Duration,
}

impl ScriptOutcome {
    pub fn failed(error: ScriptError) -> Self {
        Self { value: Value::Null, logs: Vec::new(), images: Vec::new(), error: Some(error), calls: 0, elapsed: Duration::ZERO }
    }

    /// The result as one JSON object for a text tool result: `{value, logs?, error?, images?}`
    /// (images listed by handle and index; their bytes travel separately).
    pub fn to_json(&self) -> Value {
        let mut out = serde_json::Map::new();
        out.insert("value".into(), self.value.clone());
        if !self.logs.is_empty() {
            out.insert("logs".into(), json!(self.logs));
        }
        if let Some(e) = &self.error {
            out.insert("error".into(), json!(e));
        }
        if !self.images.is_empty() {
            out.insert("images".into(), json!(self.images.iter().enumerate().map(|(k, i)| json!({"attached": k, "image": i.handle})).collect::<Vec<_>>()));
        }
        Value::Object(out)
    }
}

/// The script runtime. Stateless: every [`ScriptRuntime::run`] builds and drops its own realm.
pub struct ScriptRuntime;

impl ScriptRuntime {
    /// Run `code` against `registry` with `ctx` (its source, read-only flag and cancel flag
    /// apply to every call). Never panics; every failure is [`ScriptOutcome::error`].
    pub fn run(registry: &Registry, ctx: &Ctx, code: &str, budget: Budget) -> ScriptOutcome {
        let started = Instant::now();
        let mut outcome = Self::run_inner(registry, ctx, code, budget);
        outcome.elapsed = started.elapsed();
        outcome
    }

    fn run_inner(registry: &Registry, ctx: &Ctx, code: &str, mut budget: Budget) -> ScriptOutcome {
        if code.len() > MAX_CODE_BYTES {
            return ScriptOutcome::failed(ScriptError::new("too_large", format!("the program is {} bytes; the limit is {MAX_CODE_BYTES}", code.len())));
        }
        budget.wall = if budget.wall.is_zero() { Budget::default().wall } else { budget.wall.min(MAX_WALL) };
        let handles = ctx.shared.handles.clone();
        let before: HashSet<HandleId> = handles.ids().into_iter().collect();
        // This run's own cancel flag: set when the caller cancels (its flag may be shared with
        // other runs, so it is only read) or when the wall budget runs out, so a long registry
        // call in progress (`analyze.quality`, a batch) stops at its next check too.
        let caller_cancel = ctx.cancel.clone();
        let run_cancel = Arc::new(AtomicBool::new(false));
        let mut run_ctx = ctx.clone();
        run_ctx.cancel = run_cancel.clone();
        let ctx = &run_ctx;

        let mut spec: Vec<(String, Vec<String>)> = Vec::new();
        for area in registry.areas() {
            if matches!(area.name.as_str(), "help" | "keep") {
                continue;
            }
            let names: Vec<String> = registry.functions().filter(|f| f.area == area.name).map(|f| f.name.clone()).collect();
            spec.push((area.name.clone(), names));
        }

        let (tx, rx) = mpsc::channel::<realm::ToHost>();
        let stop = Arc::new(AtomicBool::new(false));
        let setup = realm::RealmSetup { code: code.to_string(), spec, budget: budget.clone(), cancel: ctx.cancel.clone(), stop: stop.clone(), host: tx };
        let thread = std::thread::Builder::new().name("imagelab-script".into()).stack_size(REALM_THREAD_STACK).spawn(move || realm::run_realm(setup));
        let thread = match thread {
            Ok(t) => t,
            Err(e) => return ScriptOutcome::failed(ScriptError::new("internal", format!("cannot start the script thread: {e}"))),
        };

        // Serve the realm's calls until it drops its end of the channel (it finished).
        let deadline = Instant::now() + budget.wall;
        let mut created: HashSet<HandleId> = HashSet::new();
        let mut kept: HashSet<HandleId> = HashSet::new();
        let mut abandoned = false;
        loop {
            let now = Instant::now();
            let wait = if now < deadline + STOP_GRACE {
                deadline + STOP_GRACE - now
            } else {
                // The realm ignored the stop: give up on it (it holds nothing of ours but the
                // channel, and stops at its next interrupt check).
                stop.store(true, Ordering::Relaxed);
                abandoned = true;
                break;
            };
            // Wake at least every TICK to relay the caller's cancel to this run.
            match rx.recv_timeout(wait.min(TICK)) {
                Ok(realm::ToHost::Call { id, args, reply }) => {
                    // Past the budget (or cancelled): refuse further calls at once instead of
                    // dispatching them; the realm stops at its next interrupt check.
                    let answer = if Instant::now() >= deadline {
                        stop.store(true, Ordering::Relaxed);
                        run_cancel.store(true, Ordering::Relaxed);
                        reply_err("timeout", &format!("the program ran past its {} ms budget", budget.wall.as_millis()))
                    } else if caller_cancel.load(Ordering::Relaxed) {
                        run_cancel.store(true, Ordering::Relaxed);
                        reply_err("cancelled", "the program was cancelled")
                    } else {
                        Self::dispatch(registry, ctx, &id, &args, &before, &mut created)
                    };
                    let _ = reply.send(answer);
                }
                Ok(realm::ToHost::Keep(handle)) => {
                    if let Some(id) = HandleId::parse(&handle) {
                        kept.insert(id);
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if caller_cancel.load(Ordering::Relaxed) {
                        run_cancel.store(true, Ordering::Relaxed);
                    }
                    if Instant::now() >= deadline {
                        stop.store(true, Ordering::Relaxed);
                        run_cancel.store(true, Ordering::Relaxed);
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        let done = if abandoned {
            None
        } else {
            thread.join().ok()
        };
        let Some(done) = done else {
            free_unreferenced(ctx, &created, &kept, &Value::Null);
            return ScriptOutcome::failed(if abandoned {
                ScriptError::new("timeout", format!("the program ran past its {} ms budget and did not stop", budget.wall.as_millis()))
            } else {
                ScriptError::new("internal", "the script realm died")
            });
        };

        let (mut value, error) = match done.result {
            Ok(v) => (v, None),
            Err(e) => (Value::Null, Some(e)),
        };
        free_unreferenced(ctx, &created, &kept, &value);
        let images = if error.is_none() { attach_images(ctx, &mut value, &budget) } else { Vec::new() };
        // An image the result only shows (`{image: h}`) is encoded now and needs no handle
        // afterwards: free it unless kept or mentioned elsewhere in the value. Otherwise every
        // "look" call would leave its preview in the arena until it filled.
        free_shown_images(ctx, &created, &kept, &value, &images);
        ScriptOutcome { value, logs: done.logs, images, error, calls: done.calls, elapsed: Duration::ZERO }
    }

    /// One `lab.*` call from the realm: `{"ok": result}` or `{"err": {code, message}}`.
    fn dispatch(registry: &Registry, ctx: &Ctx, id: &str, args: &str, before: &HashSet<HandleId>, created: &mut HashSet<HandleId>) -> String {
        let params: Value = match serde_json::from_str(args) {
            Ok(v) => v,
            Err(e) => return reply(Err(ApiError::bad_params(format!("arguments are not JSON: {e}")))),
        };
        // A program may only free handles its own run made: the UI's selection mask, another
        // run's images and the live preview are not its to drop (and a read-only run must not
        // change anything the person can see).
        let params = if id == "cv.free" { own_handles_only(params, created) } else { params };
        let result = if id == "$help" {
            match &params {
                Value::Null => registry.help(None),
                Value::String(area) => registry.help(Some(area)),
                _ => Err(ApiError::bad_params("lab.help takes an area name")),
            }
        } else {
            registry.call(ctx, id, params)
        };
        if let Ok(v) = &result {
            let mut found = HashSet::new();
            collect_handles(v, &mut found);
            created.extend(found.into_iter().filter(|h| !before.contains(h)));
        }
        reply(result)
    }
}

fn reply_err(code: &str, message: &str) -> String {
    json!({"err": {"code": code, "message": message}}).to_string()
}

fn reply(result: Result<Value, ApiError>) -> String {
    let v = match result {
        Ok(v) => json!({"ok": v}),
        Err(e) => json!({"err": {"code": e.code, "message": e.message}}),
    };
    v.to_string()
}

/// Every string in `v` that parses as a handle.
/// `cv.free` params with `handles` narrowed to those in `created`; the rest are reported back
/// by the function as `missing` (it never sees them).
fn own_handles_only(mut params: Value, created: &HashSet<HandleId>) -> Value {
    if let Some(Value::Array(list)) = params.get_mut("handles") {
        list.retain(|h| h.as_str().and_then(HandleId::parse).is_some_and(|id| created.contains(&id)));
    }
    params
}

fn collect_handles(v: &Value, out: &mut HashSet<HandleId>) {
    match v {
        Value::String(s) => {
            if s.len() <= 32
                && let Some(h) = HandleId::parse(s)
            {
                out.insert(h);
            }
        }
        Value::Array(items) => items.iter().for_each(|i| collect_handles(i, out)),
        Value::Object(map) => map.values().for_each(|i| collect_handles(i, out)),
        _ => {}
    }
}

fn free_unreferenced(ctx: &Ctx, created: &HashSet<HandleId>, kept: &HashSet<HandleId>, value: &Value) {
    let mut referenced = HashSet::new();
    collect_handles(value, &mut referenced);
    for h in created {
        if !kept.contains(h) && !referenced.contains(h) {
            ctx.shared.handles.free(*h);
        }
    }
}

fn free_shown_images(ctx: &Ctx, created: &HashSet<HandleId>, kept: &HashSet<HandleId>, value: &Value, images: &[ScriptImage]) {
    if images.is_empty() {
        return;
    }
    let mut rest = value.clone();
    strip_attached_images(&mut rest, 0);
    let mut mentioned = HashSet::new();
    collect_handles(&rest, &mut mentioned);
    for image in images {
        if let Some(h) = HandleId::parse(&image.handle)
            && created.contains(&h)
            && !kept.contains(&h)
            && !mentioned.contains(&h)
        {
            ctx.shared.handles.free(h);
        }
    }
}

/// Remove the `image` member of every object that got an `attached` index.
fn strip_attached_images(value: &mut Value, depth: usize) {
    if depth > 64 {
        return;
    }
    match value {
        Value::Array(items) => items.iter_mut().for_each(|i| strip_attached_images(i, depth + 1)),
        Value::Object(map) => {
            if map.get("attached").is_some_and(Value::is_u64) {
                map.remove("image");
            }
            map.values_mut().for_each(|i| strip_attached_images(i, depth + 1));
        }
        _ => {}
    }
}

/// Replace every `{image: handle}` object in `value` with itself plus `attached: k`, encoding
/// the handle to PNG.
fn attach_images(ctx: &Ctx, value: &mut Value, budget: &Budget) -> Vec<ScriptImage> {
    let mut images = Vec::new();
    let mut seen: HashMap<HandleId, usize> = HashMap::new();
    walk_images(ctx, value, budget, &mut images, &mut seen, 0);
    images
}

fn walk_images(ctx: &Ctx, value: &mut Value, budget: &Budget, images: &mut Vec<ScriptImage>, seen: &mut HashMap<HandleId, usize>, depth: usize) {
    if depth > 64 {
        return;
    }
    match value {
        Value::Array(items) => items.iter_mut().for_each(|i| walk_images(ctx, i, budget, images, seen, depth + 1)),
        Value::Object(map) => {
            let handle = map.get("image").and_then(Value::as_str).and_then(HandleId::parse);
            if let Some(h) = handle {
                if let Some(&k) = seen.get(&h) {
                    map.insert("attached".into(), json!(k));
                } else if images.len() >= budget.max_images {
                    map.insert("attached".into(), Value::Null);
                    map.insert("attachError".into(), json!(format!("more than {} images in one result", budget.max_images)));
                } else {
                    match encode_handle_png(&ctx.shared.handles, h, budget.max_image_side) {
                        Ok(bytes) => {
                            let k = images.len();
                            images.push(ScriptImage { handle: h.to_string(), mime: "image/png".into(), bytes });
                            seen.insert(h, k);
                            map.insert("attached".into(), json!(k));
                        }
                        Err(e) => {
                            map.insert("attached".into(), Value::Null);
                            map.insert("attachError".into(), json!(e.to_string()));
                        }
                    }
                }
                return;
            }
            map.values_mut().for_each(|i| walk_images(ctx, i, budget, images, seen, depth + 1));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;
