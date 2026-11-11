//! # imagelab-api
//!
//! The one registry every ImageLab surface dispatches through (docs/INTERFACES.md §2):
//! the agent's script tool, MCP, the CLI and the app UI all call
//! [`Registry::call`]`(ctx, "area.name", params)`.
//!
//! A [`Function`] declares its [`Effects`] in code. A read-only call context refuses anything
//! that is not [`Effects::ReadOnly`] — "words are not a boundary" (ARGO's plan-mode rule).
//!
//! Areas live in their own crates and register with a [`Module`]:
//! `library`/`analyze` (imagelab-library), `doc`/`edit`/`batch` (imagelab-edit),
//! `select`/`segment` (imagelab-segment), `cv` (imagelab-cv).
//!
//! Images and masks never cross as bytes in JSON: they live in the [`Handles`] arena and are
//! referenced as `"img:12"` / `"mask:3"` strings.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod handles;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

pub use handles::{HandleId, Handles, ImageBuf, MaskBuf};
pub use photocraft_automation::Headless;

/// What a function may do. Declared by the function's author, checked by [`Registry::call`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Effects {
    /// Reads only: library queries, analysis, previews, help. Allowed everywhere.
    ReadOnly,
    /// Changes state that history or a sidecar can undo (document edits, ratings, tags).
    Revertible,
    /// Writes or deletes something that cannot be undone from inside ImageLab (export over a
    /// file, delete a photo, download a model).
    Destructive,
}

/// Who started a call; carried into events so the UI can tell its own edits from an agent's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Source {
    Ui,
    Agent,
    Mcp,
    Cli,
}

/// Errors are values with a stable `code` the agent can branch on.
#[derive(Debug, Clone, thiserror::Error, Serialize)]
#[error("{code}: {message}")]
pub struct ApiError {
    pub code: String,
    pub message: String,
}

impl ApiError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self { code: code.to_string(), message: message.into() }
    }
    pub fn bad_params(message: impl Into<String>) -> Self {
        Self::new("bad_params", message)
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new("not_found", message)
    }
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new("unavailable", message)
    }
    pub fn too_large(message: impl Into<String>) -> Self {
        Self::new("too_large", message)
    }
    pub fn cancelled() -> Self {
        Self::new("cancelled", "the call was cancelled")
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new("internal", message)
    }
}

impl From<photocraft_automation::AutomationError> for ApiError {
    fn from(e: photocraft_automation::AutomationError) -> Self {
        ApiError::new("engine", e.to_string())
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

/// Events pushed to the UI and to MCP clients (docs/INTERFACES.md §5).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Event {
    DocChanged { doc: usize, revision: u64, source: Source },
    LibraryChanged { photos: Vec<String>, source: Source },
    JobProgress { job: String, done: u64, total: u64, message: String },
    JobFinished { job: String, ok: bool, summary: String },
    ModelStatus { model: String, state: String, progress: f32 },
}

/// Where events go. The FFI forwards to a Kotlin callback; tests collect them.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: Event);
}

/// Drops every event.
pub struct NoEvents;
impl EventSink for NoEvents {
    fn emit(&self, _event: Event) {}
}

/// Collects events (tests, the CLI's `--events`).
#[derive(Default)]
pub struct CollectEvents(pub Mutex<Vec<Event>>);
impl EventSink for CollectEvents {
    fn emit(&self, event: Event) {
        if let Ok(mut v) = self.0.lock() {
            v.push(event);
        }
    }
}

/// Shared state of the app: one per process, cheap to clone into a [`Ctx`].
#[derive(Clone)]
pub struct Shared {
    /// The workspace root (canonical). Areas keep their data under it: `library/`, `edits/`,
    /// `recipes/`, `exports/`, `.imagelab/` (caches, models, catalog).
    pub root: PathBuf,
    /// The PhotoCraft session the UI shows.
    pub session: Arc<Mutex<Headless>>,
    pub handles: Arc<Handles>,
    pub events: Arc<dyn EventSink>,
}

/// One call's context: the shared state plus who is calling and under which rules.
#[derive(Clone)]
pub struct Ctx {
    pub shared: Shared,
    pub source: Source,
    /// Read-only realm (`lab_read`, plan mode): non-ReadOnly functions are refused.
    pub read_only: bool,
    /// Set to cancel; long functions check it between steps ([`Ctx::check_cancel`]).
    pub cancel: Arc<AtomicBool>,
}

impl Ctx {
    pub fn new(shared: Shared, source: Source) -> Self {
        Self { shared, source, read_only: false, cancel: Arc::new(AtomicBool::new(false)) }
    }

    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    pub fn check_cancel(&self) -> ApiResult<()> {
        if self.cancel.load(Ordering::Relaxed) { Err(ApiError::cancelled()) } else { Ok(()) }
    }

    pub fn emit(&self, event: Event) {
        self.shared.events.emit(event);
    }

    /// Run `f` on the PhotoCraft session (after applying finished background jobs).
    pub fn with_session<R>(&self, f: impl FnOnce(&mut Headless) -> ApiResult<R>) -> ApiResult<R> {
        let mut guard = self.shared.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.sync_jobs();
        f(&mut guard)
    }
}

type Handler = Arc<dyn Fn(&Ctx, Value) -> ApiResult<Value> + Send + Sync>;

/// One callable function.
#[derive(Clone)]
pub struct Function {
    pub area: String,
    pub name: String,
    /// One line, shown in `help(area)` listings.
    pub summary: String,
    /// Full description for `help(area)`; may describe params in prose.
    pub description: String,
    /// JSON Schema of the params object.
    pub params: Value,
    pub effects: Effects,
    handler: Handler,
}

impl Function {
    /// The qualified name, `area.name`.
    pub fn id(&self) -> String {
        format!("{}.{}", self.area, self.name)
    }
}

/// An area: a namespace with a one-line summary (the script tool's description lists these).
#[derive(Debug, Clone, Serialize)]
pub struct Area {
    pub name: String,
    pub summary: String,
}

/// Builder for one function, so registrations read as a table.
pub struct FunctionBuilder<'r> {
    registry: &'r mut Registry,
    f: Function,
}

impl FunctionBuilder<'_> {
    pub fn description(mut self, d: &str) -> Self {
        self.f.description = d.to_string();
        self
    }
    pub fn params(mut self, schema: Value) -> Self {
        self.f.params = schema;
        self
    }
    pub fn effects(mut self, e: Effects) -> Self {
        self.f.effects = e;
        self
    }
    /// Untyped handler.
    pub fn handler(mut self, h: impl Fn(&Ctx, Value) -> ApiResult<Value> + Send + Sync + 'static) -> Self {
        self.f.handler = Arc::new(h);
        self
    }
    /// Typed handler: params deserialize into `P` (errors become `bad_params`), the result
    /// serializes from `R`.
    pub fn typed<P, R>(mut self, h: impl Fn(&Ctx, P) -> ApiResult<R> + Send + Sync + 'static) -> Self
    where
        P: DeserializeOwned,
        R: Serialize,
    {
        self.f.handler = Arc::new(move |ctx, params| {
            let params = if params.is_null() { json!({}) } else { params };
            let p: P = serde_json::from_value(params).map_err(|e| ApiError::bad_params(e.to_string()))?;
            let r = h(ctx, p)?;
            serde_json::to_value(r).map_err(|e| ApiError::internal(e.to_string()))
        });
        self
    }
    /// Finish. A duplicate id replaces the earlier one (last registration wins) and is logged
    /// in [`Registry::warnings`].
    pub fn register(self) {
        let id = self.f.id();
        if self.registry.functions.insert(id.clone(), self.f).is_some() {
            self.registry.warnings.push(format!("{id} registered twice; the later one wins"));
        }
    }
}

/// A crate that contributes areas and functions.
pub trait Module {
    fn register(&self, registry: &mut Registry);
}

/// Called after every successful call of a function that is not ReadOnly.
pub type AfterMutation = Arc<dyn Fn(&Ctx) + Send + Sync>;

/// All areas and functions.
#[derive(Default, Clone)]
pub struct Registry {
    areas: BTreeMap<String, Area>,
    functions: BTreeMap<String, Function>,
    pub warnings: Vec<String>,
    after_mutation: Option<AfterMutation>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run `f` after every successful call of a function that is not ReadOnly — each step of
    /// a script included — so the core can announce `DocChanged` as edits land (live preview)
    /// rather than when a whole program ends.
    pub fn set_after_mutation(&mut self, f: AfterMutation) {
        self.after_mutation = Some(f);
    }

    pub fn with(mut self, module: &dyn Module) -> Self {
        module.register(&mut self);
        self
    }

    /// Declare an area (idempotent; the first summary wins).
    pub fn area(&mut self, name: &str, summary: &str) {
        self.areas.entry(name.to_string()).or_insert_with(|| Area { name: name.to_string(), summary: summary.to_string() });
    }

    /// Start a function in `area` (declare the area first).
    pub fn function(&mut self, area: &str, name: &str, summary: &str) -> FunctionBuilder<'_> {
        let f = Function {
            area: area.to_string(),
            name: name.to_string(),
            summary: summary.to_string(),
            description: String::new(),
            params: json!({"type": "object"}),
            effects: Effects::ReadOnly,
            handler: Arc::new(|_, _| Err(ApiError::internal("function has no handler"))),
        };
        FunctionBuilder { registry: self, f }
    }

    pub fn areas(&self) -> impl Iterator<Item = &Area> {
        self.areas.values()
    }

    pub fn functions(&self) -> impl Iterator<Item = &Function> {
        self.functions.values()
    }

    pub fn get(&self, id: &str) -> Option<&Function> {
        self.functions.get(id)
    }

    /// One line per area, for a tool description.
    pub fn areas_summary(&self) -> String {
        self.areas.values().map(|a| format!("- {}: {}", a.name, a.summary)).collect::<Vec<_>>().join("\n")
    }

    /// `help()`: areas with function names; `help("cv")`: that area's functions in full.
    pub fn help(&self, area: Option<&str>) -> ApiResult<Value> {
        match area {
            None => Ok(Value::Array(
                self.areas
                    .values()
                    .map(|a| {
                        let names: Vec<&str> = self.functions.values().filter(|f| f.area == a.name).map(|f| f.name.as_str()).collect();
                        json!({"area": a.name, "summary": a.summary, "functions": names})
                    })
                    .collect(),
            )),
            Some(area) => {
                let area = area.trim().trim_end_matches('.');
                // Accept "cv" or "cv.canny".
                let (area, only) = area.split_once('.').map_or((area, None), |(a, f)| (a, Some(f)));
                if !self.areas.contains_key(area) {
                    return Err(ApiError::not_found(format!("no area `{area}`; areas: {}", self.areas.keys().cloned().collect::<Vec<_>>().join(", "))));
                }
                let fns: Vec<Value> = self
                    .functions
                    .values()
                    .filter(|f| f.area == area && only.is_none_or(|n| n == f.name))
                    .map(|f| json!({"name": f.id(), "summary": f.summary, "description": f.description, "params": f.params, "effects": f.effects}))
                    .collect();
                Ok(Value::Array(fns))
            }
        }
    }

    /// Call `id` ("area.name") with `params`. Checks effects against the context, catches
    /// panics, and never lets a handler crash the caller.
    pub fn call(&self, ctx: &Ctx, id: &str, params: Value) -> ApiResult<Value> {
        let f = self.functions.get(id).ok_or_else(|| {
            let area = id.split('.').next().unwrap_or(id);
            ApiError::not_found(format!("no function `{id}`; see help(\"{area}\")"))
        })?;
        if ctx.read_only && f.effects != Effects::ReadOnly {
            return Err(ApiError::new("read_only", format!("`{id}` changes things ({:?}) and this call is read-only", f.effects)));
        }
        ctx.check_cancel()?;
        let handler = f.handler.clone();
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler(ctx, params))) {
            Ok(result) => result,
            Err(_) => Err(ApiError::internal(format!("`{id}` panicked"))),
        };
        if result.is_ok() && f.effects != Effects::ReadOnly {
            if let Some(hook) = &self.after_mutation {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| hook(ctx)));
            }
        }
        result
    }
}

/// Parse a handle string or fail with `bad_params`.
pub fn parse_handle(s: &str) -> ApiResult<HandleId> {
    HandleId::parse(s).ok_or_else(|| ApiError::bad_params(format!("`{s}` is not a handle (img:N or mask:N)")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    fn shared() -> Shared {
        Shared {
            root: std::env::temp_dir(),
            session: Arc::new(Mutex::new(Headless::new())),
            handles: Arc::new(Handles::default()),
            events: Arc::new(NoEvents),
        }
    }

    #[derive(Deserialize)]
    struct Add {
        a: i64,
        b: i64,
    }

    fn registry() -> Registry {
        let mut r = Registry::new();
        r.area("math", "arithmetic for tests");
        r.function("math", "add", "add two numbers").typed(|_, p: Add| Ok(p.a + p.b)).register();
        r.function("math", "store", "pretend to write").effects(Effects::Revertible).handler(|_, _| Ok(json!("stored"))).register();
        #[allow(clippy::panic)]
        r.function("math", "boom", "panics").handler(|_, _| panic!("boom")).register();
        r
    }

    #[test]
    fn typed_call_and_errors() {
        let r = registry();
        let ctx = Ctx::new(shared(), Source::Agent);
        assert_eq!(r.call(&ctx, "math.add", json!({"a": 2, "b": 3})).unwrap(), json!(5));
        assert_eq!(r.call(&ctx, "math.add", json!({"a": "x"})).unwrap_err().code, "bad_params");
        assert_eq!(r.call(&ctx, "math.nope", json!({})).unwrap_err().code, "not_found");
        assert_eq!(r.call(&ctx, "math.boom", json!({})).unwrap_err().code, "internal");
    }

    #[test]
    fn after_mutation_runs_for_effects_only() {
        let mut r = registry();
        let n = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = n.clone();
        r.set_after_mutation(Arc::new(move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        }));
        let ctx = Ctx::new(shared(), Source::Agent);
        r.call(&ctx, "math.add", json!({"a": 1, "b": 1})).unwrap();
        assert_eq!(n.load(Ordering::Relaxed), 0);
        r.call(&ctx, "math.store", json!({})).unwrap();
        assert_eq!(n.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn read_only_refuses_effects() {
        let r = registry();
        let ctx = Ctx::new(shared(), Source::Agent).read_only();
        assert_eq!(r.call(&ctx, "math.store", json!({})).unwrap_err().code, "read_only");
        assert!(r.call(&ctx, "math.add", json!({"a": 1, "b": 1})).is_ok());
    }

    #[test]
    fn cancel_is_checked() {
        let r = registry();
        let ctx = Ctx::new(shared(), Source::Agent);
        ctx.cancel.store(true, Ordering::Relaxed);
        assert_eq!(r.call(&ctx, "math.add", json!({"a": 1, "b": 1})).unwrap_err().code, "cancelled");
    }

    #[test]
    fn help_lists_and_details() {
        let r = registry();
        let all = r.help(None).unwrap();
        assert_eq!(all[0]["area"], "math");
        let one = r.help(Some("math.add")).unwrap();
        assert_eq!(one.as_array().map(Vec::len), Some(1));
        assert_eq!(r.help(Some("nope")).unwrap_err().code, "not_found");
        assert!(r.areas_summary().contains("math: arithmetic"));
    }
}
