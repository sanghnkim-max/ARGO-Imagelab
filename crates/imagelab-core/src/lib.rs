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
//! Files live under one workspace folder (the app's private storage on Android). The session is
//! capability-scoped to it: PhotoCraft's `AuthorizedWorkspace` refuses absolute paths outside
//! it, `..` and symlink escapes, for the UI and for agents alike.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod cli;
pub mod mcp;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use photocraft_automation::{AuthorizedWorkspace, AutomationError, Backend, Headless, PhotocraftMcp};
use serde_json::{Value, json};

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
    mcp: PhotocraftMcp,
    runtime: tokio::runtime::Runtime,
    in_process: mcp::InProcess,
    http: Mutex<Option<mcp::HttpServer>>,
    gpu_disabled: std::sync::atomic::AtomicBool,
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
        let session = Arc::new(Mutex::new(headless));
        let mcp = PhotocraftMcp::with_backend(Backend::Headless(session.clone()));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("imagelab-mcp")
            .enable_all()
            .build()
            .map_err(|e| LabError::Other(format!("tokio runtime: {e}")))?;
        let in_process = runtime.block_on(mcp::InProcess::start(mcp.clone()))?;
        Ok(Self { root, session, mcp, runtime, in_process, http: Mutex::new(None), gpu_disabled: std::sync::atomic::AtomicBool::new(gpu_failed) })
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

    // ---------------------------------------------------------------- documents

    /// Open a file beneath the workspace and make it the active document.
    pub fn open(&self, path: &str) -> Result<Value> {
        let path = relative(path)?;
        self.with(|h| h.open(&path))
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
        self.with(|h| h.save(None, Some(&path), format, &options))
    }

    /// Close the active document without saving.
    pub fn close(&self) -> Result<Value> {
        self.with(|h| h.close(None))
    }

    /// Make the document at `index` active.
    pub fn select(&self, index: usize) -> Result<Value> {
        self.with(|h| h.select(index))
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
        self.with(|h| h.command_run(id, params))
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
        self.gpu_disabled.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(guard.render_png(None, max_side)?)
    }

    /// True once the GPU renderer failed and the session fell back to the CPU compositor.
    pub fn gpu_disabled(&self) -> bool {
        self.gpu_disabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Warnings from the renderer (a GPU fallback, an export that lost something), each once.
    pub fn drain_diagnostics(&self) -> Vec<String> {
        let mut guard = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        guard.drain_diagnostics()
    }

    // ---------------------------------------------------------------- MCP

    /// Send one MCP JSON-RPC request to PhotoCraft's server in-process and wait for its
    /// response (the in-app agent's way in). `method` is e.g. `tools/list` or `tools/call`.
    pub fn mcp_request(&self, method: &str, params: Value) -> Result<Value> {
        self.runtime.block_on(self.in_process.request(method, params))
    }

    /// Serve the MCP server over streamable HTTP on `127.0.0.1:port` (0 picks a free port),
    /// requiring `Authorization: Bearer <token>`. Replaces a server already running.
    pub fn start_http(&self, port: u16, token: &str) -> Result<u16> {
        self.stop_http();
        let server = self.runtime.block_on(mcp::HttpServer::start(self.mcp.clone(), port, token.to_owned()))?;
        let bound = server.port();
        *self.http.lock().unwrap_or_else(PoisonError::into_inner) = Some(server);
        Ok(bound)
    }

    /// Stop the HTTP MCP server if it runs.
    pub fn stop_http(&self) {
        if let Some(server) = self.http.lock().unwrap_or_else(PoisonError::into_inner).take() {
            server.stop();
        }
    }

    /// The HTTP MCP server's port, if it runs.
    pub fn http_port(&self) -> Option<u16> {
        self.http.lock().unwrap_or_else(PoisonError::into_inner).as_ref().map(mcp::HttpServer::port)
    }

    // ---------------------------------------------------------------- CLI

    /// Run `photocraft-cli` with `args`, in-process, with the workspace as its folder. The CLI
    /// is its own headless session (as on a desktop shell), not the app's open documents.
    pub fn cli(&self, args: &[String]) -> cli::CliOutput {
        cli::run(&self.root, args)
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
}
