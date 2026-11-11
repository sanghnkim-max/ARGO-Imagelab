//! # imagelab-core
//!
//! ImageLab is a thin host around PhotoCraft's engine. Everything that edits an image is a
//! PhotoCraft engine command (~700 of them, by id, with JSON params). This crate owns **one**
//! `photocraft_automation::Headless` session and hands it to four clients, so they all see the
//! same open documents:
//!
//! * the app UI (through `imagelab-ffi`): open, run a command, render a preview, save;
//! * the in-app agent: PhotoCraft's own MCP server, called in-process ([`mcp::InProcess`]);
//! * external agents (Claude Code on a laptop over `adb forward`): the same MCP server over
//!   loopback streamable HTTP with a bearer token ([`mcp::HttpServer`]);
//! * the CLI: `photocraft-cli`'s argument parser and commands, run in-process ([`cli`]).
//!
//! On top of the session sits the ImageLab function registry (`imagelab-api`, composed here
//! from the area crates) and the script runtime (`imagelab-script`): [`Lab::call`],
//! [`Lab::help`], [`Lab::run_script`] are what the app UI, the in-app agent (`lab_script` /
//! `lab_read`), MCP and the CLI all dispatch through (docs/INTERFACES.md §2-3).
//!
//! **Events.** After every call, script, in-process CLI run or in-process MCP request, the core
//! compares the active document's revision with the last one it saw and emits
//! `Event::DocChanged {doc, revision, source}` when it moved. Edits arriving over the HTTP MCP
//! server (PhotoCraft's raw endpoint) are noticed by a 250 ms watcher thread that runs only
//! while the HTTP server does and only when no in-core call is in flight. `revision: 0` means
//! no document is open any more. The sink is swappable at runtime ([`Lab::set_event_sink`]).
//!
//! Files live under one workspace folder (the app's private storage on Android). The session is
//! capability-scoped to it: PhotoCraft's `AuthorizedWorkspace` refuses absolute paths outside
//! it, `..` and symlink escapes, for the UI and for agents alike.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod cli;
mod hub;
pub mod mcp;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

pub use imagelab_api;
use imagelab_api::{ApiError, ApiResult, EventSink, Handles, Registry, Shared, Source};
pub use imagelab_script;
use imagelab_script::ScriptOutcome;
use photocraft_automation::{AuthorizedWorkspace, AutomationError, Backend, Headless, PhotocraftMcp};
use serde_json::{Value, json};

use crate::hub::{Core, ForwardingSink};

/// Every area ImageLab ships, composed from the area crates. The CLI's `help` uses it without
/// opening a workspace.
pub fn build_registry() -> Registry {
    Registry::new()
        .with(&imagelab_library::LibraryModule)
        .with(&imagelab_cv::CvModule)
        .with(&imagelab_segment::SegmentModule)
        .with(&imagelab_edit::EditModule)
}

/// Folders ImageLab creates inside the workspace.
pub const IMPORTS_DIR: &str = "imports";
pub const EXPORTS_DIR: &str = "exports";

/// Longest side of a preview the UI may ask for. PhotoCraft's own preview budget for untrusted
/// callers applies on top of this.
pub const MAX_PREVIEW_SIDE: u32 = 4096;

#[derive(Debug, thiserror::Error)]
pub enum LabError {
    #[error("{0}")]
    BadRequest(String),
    #[error("I/O: {0}")]
    Io(String),
    #[error(transparent)]
    Automation(#[from] AutomationError),
    /// A registry call failed; `code` is stable (`bad_params`, `not_found`, `read_only`...).
    #[error(transparent)]
    Api(#[from] ApiError),
    #[error("{0}")]
    Other(String),
}

impl From<std::io::Error> for LabError {
    fn from(e: std::io::Error) -> Self {
        LabError::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, LabError>;

/// Options for [`Lab::new`].
#[derive(Debug, Clone)]
pub struct LabOptions {
    /// Use PhotoCraft's wgpu renderer for full-size previews and exports (lazy: the device is
    /// created on first use, and a device that fails falls back to the CPU compositor).
    pub gpu: bool,
}

impl Default for LabOptions {
    fn default() -> Self {
        Self { gpu: true }
    }
}

/// The ImageLab core: one PhotoCraft session plus the ways in to it.
pub struct Lab {
    root: PathBuf,
    session: Arc<Mutex<Headless>>,
    core: Arc<Core>,
    mcp: PhotocraftMcp,
    runtime: tokio::runtime::Runtime,
    in_process: mcp::InProcess,
    lab_in_process: mcp::InProcess,
    http: Mutex<Option<(mcp::HttpServer, hub::Watcher)>>,
    gpu_disabled: AtomicBool,
}

impl Lab {
    /// Open (creating if needed) the workspace at `root`.
    pub fn new(root: &Path, options: &LabOptions) -> Result<Self> {
        std::fs::create_dir_all(root.join(IMPORTS_DIR))?;
        std::fs::create_dir_all(root.join(EXPORTS_DIR))?;
        let root = root.canonicalize()?;
        let workspace = AuthorizedWorkspace::new(Some(&root), Some(&root))?;
        #[allow(unused_mut)]
        let mut headless = Headless::with_workspace(workspace);
        #[cfg(feature = "gpu")]
        let gpu_failed = options.gpu && !gpu_works();
        #[cfg(feature = "gpu")]
        if options.gpu && !gpu_failed {
            headless.enable_gpu();
        }
        #[cfg(not(feature = "gpu"))]
        let gpu_failed = false;
        #[cfg(not(feature = "gpu"))]
        let _ = options.gpu;
        // Fonts the workspace installed (Google Fonts…) join PhotoCraft's text engine.
        imagelab_edit::fonts::register_all(&root);
        let session = Arc::new(Mutex::new(headless));
        let sink = Arc::new(ForwardingSink::default());
        let shared = Shared { root: root.clone(), session: session.clone(), handles: Arc::new(Handles::default()), events: sink.clone() };
        let core = Arc::new(Core::new(build_registry(), shared, sink));
        let mcp = PhotocraftMcp::with_backend(Backend::Headless(session.clone()));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("imagelab-mcp")
            .enable_all()
            .build()
            .map_err(|e| LabError::Other(format!("tokio runtime: {e}")))?;
        let in_process = runtime.block_on(mcp::InProcess::start(mcp.clone()))?;
        // The in-app agent is the person's own: edits allowed, events attributed to the agent.
        let lab_server = mcp::LabMcp::new(core.clone(), true, Source::Agent);
        let lab_in_process = runtime.block_on(mcp::InProcess::start_lab(lab_server))?;
        Ok(Self { root, session, core, mcp, runtime, in_process, lab_in_process, http: Mutex::new(None), gpu_disabled: AtomicBool::new(gpu_failed) })
    }

    /// The workspace folder (canonical).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Run `f` on the session, after applying background jobs that finished.
    fn with<R>(&self, f: impl FnOnce(&mut Headless) -> std::result::Result<R, AutomationError>) -> Result<R> {
        let mut guard = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        guard.sync_jobs();
        Ok(f(&mut guard)?)
    }

    /// [`Lab::with`] for a UI action that may change the document: emits `DocChanged` (source
    /// `Ui`) afterwards when the active revision moved.
    fn with_ui<R>(&self, f: impl FnOnce(&mut Headless) -> std::result::Result<R, AutomationError>) -> Result<R> {
        let _busy = self.core.busy();
        let result = self.with(f);
        self.core.notice(Source::Ui);
        result
    }

    // ---------------------------------------------------------------- registry and scripts

    /// The function registry (all areas).
    pub fn registry(&self) -> &Registry {
        &self.core.registry
    }

    /// The shared state handed to every call (workspace, session, handles, events).
    pub fn shared(&self) -> &Shared {
        &self.core.shared
    }

    /// `Registry::call` as `source` (`Ui` for buttons, `Agent`, `Mcp`, `Cli`).
    pub fn call(&self, id: &str, params: Value, source: Source) -> ApiResult<Value> {
        self.core.call(id, params, source)
    }

    /// `help()` (areas and function names) or `help("cv")` / `help("cv.canny")` (full entries).
    pub fn help(&self, area: Option<&str>) -> ApiResult<Value> {
        self.core.registry.help(area)
    }

    /// One line per area, for tool descriptions.
    pub fn areas_summary(&self) -> String {
        self.core.registry.areas_summary()
    }

    /// Run a `lab_script` program as the in-app agent (`read_only`: a `lab_read` realm).
    /// `timeout` 0 means the default (120 s); capped at 30 min.
    pub fn run_script(&self, code: &str, read_only: bool, timeout: Duration) -> ScriptOutcome {
        self.run_script_as(code, read_only, timeout, Source::Agent)
    }

    /// [`Lab::run_script`] attributed to `source`.
    pub fn run_script_as(&self, code: &str, read_only: bool, timeout: Duration, source: Source) -> ScriptOutcome {
        self.core.run_script(code, read_only, timeout, source, Arc::new(AtomicBool::new(false)))
    }

    /// Cancel every script running now (from any surface).
    pub fn cancel_scripts(&self) {
        self.core.cancel_scripts();
    }

    /// Where events go from now on (the FFI's Kotlin listener, a test collector...).
    pub fn set_event_sink(&self, sink: Arc<dyn EventSink>) {
        self.core.sink.set(sink);
    }

    /// PNG of an `img:N` / `mask:N` handle (masks as grayscale), longest side at most
    /// `max_side` (0: full size, otherwise capped at [`MAX_PREVIEW_SIDE`]).
    pub fn render_handle(&self, handle: &str, max_side: u32) -> Result<Vec<u8>> {
        if max_side > MAX_PREVIEW_SIDE {
            return Err(LabError::BadRequest(format!("max_side {max_side} exceeds {MAX_PREVIEW_SIDE}")));
        }
        let id = imagelab_api::parse_handle(handle)?;
        Ok(imagelab_script::encode_handle_png(&self.core.shared.handles, id, max_side)?)
    }

    // ---------------------------------------------------------------- documents

    /// Open a file beneath the workspace and make it the active document.
    pub fn open(&self, path: &str) -> Result<Value> {
        let path = relative(path)?;
        self.with_ui(|h| h.open(&path))
    }

    /// Copy `bytes` into `imports/` under a safe, unique version of `name`, then open it.
    /// This is how the UI brings in a photo from the system picker or the camera.
    pub fn import(&self, name: &str, bytes: &[u8]) -> Result<Value> {
        let file = unique_name(&self.root.join(IMPORTS_DIR), &sanitize_file_name(name))?;
        std::fs::write(self.root.join(IMPORTS_DIR).join(&file), bytes)?;
        let mut opened = self.open(&format!("{IMPORTS_DIR}/{file}"))?;
        if let Value::Object(map) = &mut opened {
            map.insert("path".into(), json!(format!("{IMPORTS_DIR}/{file}")));
        }
        Ok(opened)
    }

    /// A new blank document (PhotoCraft's `file.new`; params as that command takes them).
    pub fn new_document(&self, params: Value) -> Result<Value> {
        self.run("file.new", params)
    }

    /// Save the active document. `.pcraft` is lossless and layered; `.png`/`.jpg`/`.webp`/
    /// `.psd`/`.tiff`… export by extension. `path` is beneath the workspace.
    pub fn save(&self, path: &str, format: Option<&str>) -> Result<Value> {
        let path = relative(path)?;
        let options = photocraft_io::ExportOptions::default();
        self.with_ui(|h| h.save(None, Some(&path), format, &options))
    }

    /// Close the active document without saving.
    pub fn close(&self) -> Result<Value> {
        self.with_ui(|h| h.close(None))
    }

    /// Make the document at `index` active.
    pub fn select(&self, index: usize) -> Result<Value> {
        self.with_ui(|h| h.select(index))
    }

    /// Open documents and the active index.
    pub fn session_info(&self) -> Value {
        let guard = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        guard.session_list()
    }

    /// The active document as JSON: layer tree, history, selection.
    pub fn inspect(&self) -> Result<Value> {
        self.with(|h| h.inspect(None))
    }

    // ---------------------------------------------------------------- commands

    /// Every engine command: `[{id, label, menu, params, ...}]`.
    pub fn commands(&self) -> Value {
        let guard = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        guard.command_list()
    }

    /// Run an engine command by id (waits for job-capable commands such as filters).
    pub fn run(&self, id: &str, params: Value) -> Result<Value> {
        self.with_ui(|h| h.command_run(id, params))
    }

    /// Step back / forward in history.
    pub fn undo(&self) -> Result<Value> {
        self.run("edit.undo", json!({}))
    }

    pub fn redo(&self) -> Result<Value> {
        self.run("edit.redo", json!({}))
    }

    // ---------------------------------------------------------------- rendering

    /// The active document flattened to a PNG whose longest side is at most `max_side`.
    /// Full size goes through the GPU renderer when it is on; scaled previews through
    /// PhotoCraft's reduced CPU compositor.
    pub fn render_png(&self, max_side: u32) -> Result<Vec<u8>> {
        if max_side > MAX_PREVIEW_SIDE {
            return Err(LabError::BadRequest(format!("max_side {max_side} exceeds {MAX_PREVIEW_SIDE}")));
        }
        let mut guard = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        guard.sync_jobs();
        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| guard.render_png(None, max_side)));
        match first {
            Ok(Ok(png)) => return Ok(png),
            // A document/engine error: the CPU compositor would fail the same way.
            Ok(Err(e)) if guard.renderer_mut().is_none() => return Err(e.into()),
            Err(_) if guard.renderer_mut().is_none() => return Err(LabError::Other("rendering panicked".into())),
            _ => {}
        }
        // The GPU renderer failed or panicked (seen on emulators: wgpu's GL fallback panics while
        // creating a shader program). Drop it for the rest of the session and render on the CPU
        // compositor, which is PhotoCraft's reference renderer anyway.
        guard.set_renderer(None);
        self.gpu_disabled.store(true, Ordering::Relaxed);
        Ok(guard.render_png(None, max_side)?)
    }

    /// The active document flattened to premultiplied RGBA8, longest side at most `max_side`:
    /// the live-preview path. PhotoCraft's reduced compositor works from a downsampled proxy
    /// (milliseconds, no full-size composite), and skipping PNG encode/decode keeps a frame
    /// cheap enough to follow a slider. Premultiplied, so Android copies it straight into an
    /// ARGB_8888 bitmap.
    pub fn render_rgba(&self, max_side: u32) -> Result<(u32, u32, Vec<u8>)> {
        if max_side == 0 || max_side > MAX_PREVIEW_SIDE {
            return Err(LabError::BadRequest(format!("max_side must be 1..={MAX_PREVIEW_SIDE}")));
        }
        let mut guard = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        guard.sync_jobs();
        let doc = guard.session.active().map(|d| &d.doc).ok_or_else(|| LabError::BadRequest("no document is open".into()))?;
        let img = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| photocraft_compose::thumbnail(doc, max_side)))
            .map_err(|_| LabError::Other("rendering panicked".into()))?;
        drop(guard);
        let mut px = img.pixels;
        for p in px.chunks_exact_mut(4) {
            let a = u16::from(p[3]);
            if a < 255 {
                for c in &mut p[..3] {
                    *c = ((u16::from(*c) * a + 127) / 255) as u8;
                }
            }
        }
        Ok((img.width, img.height, px))
    }

    /// True once the GPU renderer failed and the session fell back to the CPU compositor.
    pub fn gpu_disabled(&self) -> bool {
        self.gpu_disabled.load(Ordering::Relaxed)
    }

    /// Warnings from the renderer (a GPU fallback, an export that lost something), each once.
    pub fn drain_diagnostics(&self) -> Vec<String> {
        let mut guard = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        guard.drain_diagnostics()
    }

    // ---------------------------------------------------------------- MCP

    /// Send one MCP JSON-RPC request to PhotoCraft's server in-process and wait for its
    /// response. `method` is e.g. `tools/list` or `tools/call`. Edits are attributed to the
    /// agent in `DocChanged`.
    pub fn mcp_request(&self, method: &str, params: Value) -> Result<Value> {
        let _busy = self.core.busy();
        let result = self.runtime.block_on(self.in_process.request(method, params));
        self.core.notice(Source::Agent);
        result
    }

    /// Send one MCP request to ImageLab's own server (`lab_script`, `lab_read`, `lab_help`)
    /// in-process, edits allowed (the in-app agent).
    pub fn lab_mcp_request(&self, method: &str, params: Value) -> Result<Value> {
        self.runtime.block_on(self.lab_in_process.request(method, params))
    }

    /// Serve MCP over streamable HTTP on `127.0.0.1:port` (0 picks a free port), requiring
    /// `Authorization: Bearer <token>`: ImageLab's tools at `/mcp`, PhotoCraft's raw tools at
    /// `/photocraft/mcp`. With `allow_edits` false, `lab_script` and the PhotoCraft endpoint are
    /// refused (only `lab_read` / `lab_help`). Replaces a server already running.
    pub fn start_http(&self, port: u16, token: &str, allow_edits: bool) -> Result<u16> {
        self.stop_http();
        let lab = mcp::LabMcp::new(self.core.clone(), allow_edits, Source::Mcp);
        let server = self.runtime.block_on(mcp::HttpServer::start(self.mcp.clone(), lab, port, token.to_owned(), allow_edits))?;
        let bound = server.port();
        let watcher = hub::Watcher::start(Arc::downgrade(&self.core));
        *self.http.lock().unwrap_or_else(PoisonError::into_inner) = Some((server, watcher));
        Ok(bound)
    }

    /// Stop the HTTP MCP server if it runs.
    pub fn stop_http(&self) {
        let taken = self.http.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some((server, watcher)) = taken {
            server.stop();
            watcher.stop();
        }
    }

    /// The HTTP MCP server's port, if it runs.
    pub fn http_port(&self) -> Option<u16> {
        self.http.lock().unwrap_or_else(PoisonError::into_inner).as_ref().map(|(s, _)| s.port())
    }

    // ---------------------------------------------------------------- CLI

    /// Run `photocraft-cli` with `args`, in-process, with the workspace as its folder. The CLI
    /// is its own headless session (as on a desktop shell), not the app's open documents.
    pub fn cli(&self, args: &[String]) -> cli::CliOutput {
        let _busy = self.core.busy();
        let out = cli::run(&self.root, args);
        self.core.notice(Source::Cli);
        out
    }
}

impl Drop for Lab {
    fn drop(&mut self) {
        self.stop_http();
    }
}

/// Render one tiny full-size document through PhotoCraft's GPU renderer in a throwaway session.
/// Every path that renders (the UI, MCP previews and exports) shares the main session's
/// renderer, so a GPU that fails must be found before any of them runs. Seen on the Android
/// emulator: without real Vulkan, wgpu falls back to GL and panics creating a shader program.
#[cfg(feature = "gpu")]
fn gpu_works() -> bool {
    let probe = std::panic::catch_unwind(|| {
        let mut h = Headless::new().with_gpu();
        h.command_run("file.new", json!({"width": 16, "height": 16}))?;
        h.render_png(None, 0)
    });
    matches!(probe, Ok(Ok(png)) if png.starts_with(b"\x89PNG"))
}

/// A workspace-relative path: forward slashes, no absolute paths, no `..`. PhotoCraft's
/// workspace checks again (and catches symlinks); this gives the UI a clear early error.
fn relative(path: &str) -> Result<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(LabError::BadRequest("empty path".into()));
    }
    let p = Path::new(trimmed);
    if p.is_absolute() || p.components().any(|c| !matches!(c, std::path::Component::Normal(_))) {
        return Err(LabError::BadRequest(format!("path must stay inside the workspace: {trimmed}")));
    }
    Ok(p.to_path_buf())
}

/// Keep a picked file's name readable but safe on every file system: no separators, no
/// control or reserved characters, no leading dots, bounded length.
pub fn sanitize_file_name(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let cleaned: String = base
        .chars()
        .map(|c| if c.is_control() || matches!(c, '<' | '>' | ':' | '"' | '|' | '?' | '*') { '_' } else { c })
        .collect();
    let cleaned = cleaned.trim().trim_start_matches('.').trim_end_matches(['.', ' ']).to_string();
    let cleaned = if cleaned.is_empty() { "image".to_string() } else { cleaned };
    // Bound the name, keeping the extension.
    if cleaned.chars().count() <= 120 {
        return cleaned;
    }
    let (stem, ext) = match cleaned.rsplit_once('.') {
        Some((s, e)) if e.len() <= 10 => (s.to_string(), format!(".{e}")),
        _ => (cleaned.clone(), String::new()),
    };
    stem.chars().take(120 - ext.len()).collect::<String>() + &ext
}

/// `name`, or `name (2)`, `name (3)`… — the first that does not exist in `dir`.
fn unique_name(dir: &Path, name: &str) -> Result<String> {
    if !dir.join(name).exists() {
        return Ok(name.to_string());
    }
    let (stem, ext) = match name.rsplit_once('.') {
        Some((s, e)) if !s.is_empty() => (s, format!(".{e}")),
        _ => (name, String::new()),
    };
    for n in 2..10_000 {
        let candidate = format!("{stem} ({n}){ext}");
        if !dir.join(&candidate).exists() {
            return Ok(candidate);
        }
    }
    Err(LabError::Io(format!("too many files named {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_keeps_names_safe() {
        assert_eq!(sanitize_file_name("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_file_name("C:\\pics\\a:b?.jpg"), "a_b_.jpg");
        assert_eq!(sanitize_file_name("..."), "image");
        assert_eq!(sanitize_file_name(".hidden.png"), "hidden.png");
        let long = format!("{}.jpeg", "x".repeat(400));
        let s = sanitize_file_name(&long);
        assert!(s.ends_with(".jpeg") && s.chars().count() <= 120);
    }

    #[test]
    fn relative_refuses_escapes() {
        assert!(relative("a/b.png").is_ok());
        for bad in ["", "/etc/passwd", "../x.png", "a/../../x", "./x"] {
            assert!(relative(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn new_document_render_save_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: false }).unwrap();
        lab.new_document(json!({"width": 64, "height": 48})).unwrap();
        let png = lab.render_png(32).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        lab.save("exports/out.png", None).unwrap();
        assert!(dir.path().join("exports/out.png").is_file());
        // Escapes are refused before PhotoCraft sees them.
        assert!(lab.save("../out.png", None).is_err());
    }

    #[test]
    fn gpu_on_always_renders_and_saves() {
        // With a working GPU (Metal here) the probe enables it; without one the session stays
        // on the CPU. Either way previews and exports work.
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: true }).unwrap();
        lab.new_document(json!({"width": 40, "height": 30})).unwrap();
        assert!(lab.render_png(0).unwrap().starts_with(b"\x89PNG"));
        lab.save("exports/gpu.png", None).unwrap();
        assert!(dir.path().join("exports/gpu.png").is_file());
        eprintln!("gpu disabled: {}", lab.gpu_disabled());
    }

    #[test]
    fn render_rgba_is_premultiplied_and_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: false }).unwrap();
        assert!(lab.render_rgba(256).is_err(), "no document yet");
        lab.new_document(json!({"width": 400, "height": 200})).unwrap();
        let (w, h, px) = lab.render_rgba(100).unwrap();
        assert_eq!((w, h), (100, 50));
        assert_eq!(px.len(), (w * h * 4) as usize);
        assert_eq!(&px[..4], &[255, 255, 255, 255]);
        assert!(lab.render_rgba(0).is_err());
    }

    #[test]
    fn import_opens_a_copy_under_imports() {
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: false }).unwrap();
        lab.new_document(json!({"width": 8, "height": 8})).unwrap();
        lab.save("exports/seed.png", None).unwrap();
        let bytes = std::fs::read(dir.path().join("exports/seed.png")).unwrap();
        let first = lab.import("../photo.png", &bytes).unwrap();
        assert_eq!(first["path"], "imports/photo.png");
        let second = lab.import("photo.png", &bytes).unwrap();
        assert_eq!(second["path"], "imports/photo (2).png");
    }

    #[test]
    fn mcp_in_process_lists_photocraft_tools_and_shares_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: false }).unwrap();
        let tools = lab.mcp_request("tools/list", json!({})).unwrap();
        let names: Vec<&str> = tools["tools"].as_array().unwrap().iter().filter_map(|t| t["name"].as_str()).collect();
        for expected in ["doc_new", "doc_inspect", "command_run", "command_list", "doc_render_preview"] {
            assert!(names.contains(&expected), "{expected} missing from {names:?}");
        }
        // A document made over MCP is the UI's document too.
        lab.mcp_request("tools/call", json!({"name": "doc_new", "arguments": {"width": 20, "height": 10}})).unwrap();
        assert_eq!(lab.session_info()["documents"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn cli_runs_in_process() {
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: false }).unwrap();
        let out = lab.cli(&["--version".to_string()]);
        assert_eq!(out.code, 0, "{out:?}");
        assert!(out.stdout.contains("photocraft"), "{out:?}");
    }

    fn doc_changes(events: &imagelab_api::CollectEvents) -> Vec<(u64, Source)> {
        events
            .0
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                imagelab_api::Event::DocChanged { revision, source, .. } => Some((*revision, *source)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn registry_help_and_scripts() {
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: false }).unwrap();
        let areas: Vec<String> = lab.help(None).unwrap().as_array().unwrap().iter().map(|a| a["area"].as_str().unwrap().to_string()).collect();
        for expected in ["library", "analyze", "cv", "select", "segment", "doc", "edit", "batch"] {
            assert!(areas.iter().any(|a| a == expected), "{expected} missing from {areas:?}");
        }
        assert!(lab.areas_summary().contains("- cv:"));
        assert_eq!(lab.call("nope.nothing", json!({}), Source::Ui).unwrap_err().code, "not_found");
        let o = lab.run_script("const h = lab.help(); return h.length + 1;", true, Duration::ZERO);
        assert_eq!(o.error, None);
        assert_eq!(o.value, json!(areas.len() + 1));
    }

    #[test]
    fn render_handle_and_script_images() {
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: false }).unwrap();
        let h = lab.shared().handles.put_mask(imagelab_api::MaskBuf::new(8, 8, vec![255; 64]).unwrap()).unwrap();
        let png = lab.render_handle(&h.to_string(), 4).unwrap();
        assert!(png.starts_with(b"\x89PNG"));
        assert!(lab.render_handle("img:999", 0).is_err());
        assert!(lab.render_handle("bogus", 0).is_err());
        // Over in-process MCP, an image in the result comes back as image content.
        let result = lab
            .lab_mcp_request("tools/call", json!({"name": "lab_read", "arguments": {"code": format!("return {{mask: {{image: '{h}'}}}};"), "description": "show"}}))
            .unwrap();
        let content = result["content"].as_array().unwrap();
        assert_eq!(content.len(), 2, "{result}");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["mimeType"], "image/png");
        assert!(content[0]["text"].as_str().unwrap().contains("\"attached\": 0"));
    }

    #[test]
    fn lab_mcp_tools_in_process() {
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: false }).unwrap();
        let tools = lab.lab_mcp_request("tools/list", json!({})).unwrap();
        let names: Vec<&str> = tools["tools"].as_array().unwrap().iter().filter_map(|t| t["name"].as_str()).collect();
        assert_eq!(names, ["lab_read", "lab_script", "lab_help"]);
        let help = lab.lab_mcp_request("tools/call", json!({"name": "lab_help", "arguments": {"area": "cv"}})).unwrap();
        assert_ne!(help["isError"], json!(true), "{help}");
        let failed = lab.lab_mcp_request("tools/call", json!({"name": "lab_script", "arguments": {"code": "throw new Error('x')", "description": "fail"}})).unwrap();
        assert_eq!(failed["isError"], json!(true));
        // The PhotoCraft in-process client still works next to it.
        let pc = lab.mcp_request("tools/list", json!({})).unwrap();
        assert!(pc["tools"].as_array().unwrap().len() > 5);
    }

    #[test]
    fn doc_changed_events_carry_the_source() {
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: false }).unwrap();
        let events = Arc::new(imagelab_api::CollectEvents::default());
        lab.set_event_sink(events.clone());
        lab.new_document(json!({"width": 16, "height": 16})).unwrap();
        lab.mcp_request("tools/call", json!({"name": "command_run", "arguments": {"id": "image.adjustments.invert", "params": {}}})).unwrap();
        lab.inspect().unwrap();
        let changes = doc_changes(&events);
        assert_eq!(changes.len(), 2, "{changes:?}");
        assert_eq!(changes[0].1, Source::Ui);
        assert_eq!(changes[1].1, Source::Agent);
        assert!(changes[1].0 > changes[0].0);
        // Swapping the sink takes effect at once.
        let second = Arc::new(imagelab_api::CollectEvents::default());
        lab.set_event_sink(second.clone());
        lab.undo().unwrap();
        assert_eq!(doc_changes(&second).len(), 1);
        assert_eq!(doc_changes(&events).len(), 2);
    }

    #[test]
    fn watcher_reports_edits_that_bypass_the_core() {
        let dir = tempfile::tempdir().unwrap();
        let lab = Lab::new(dir.path(), &LabOptions { gpu: false }).unwrap();
        lab.new_document(json!({"width": 16, "height": 16})).unwrap();
        let events = Arc::new(imagelab_api::CollectEvents::default());
        lab.set_event_sink(events.clone());
        lab.start_http(0, "0123456789abcdef", true).unwrap();
        // As PhotoCraft's HTTP endpoint would: straight on the session.
        lab.shared().session.lock().unwrap().command_run("image.adjustments.invert", json!({})).unwrap();
        let start = std::time::Instant::now();
        while doc_changes(&events).is_empty() && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(50));
        }
        lab.stop_http();
        assert_eq!(doc_changes(&events).first().map(|c| c.1), Some(Source::Mcp));
    }
}
