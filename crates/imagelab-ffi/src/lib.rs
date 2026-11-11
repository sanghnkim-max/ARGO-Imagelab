//! # imagelab-ffi
//!
//! The Android app's view of [`imagelab_core::Lab`], through uniffi (Kotlin + JNA). JSON crosses
//! as strings: the engine's params and results are open-ended JSON already, and the agent hands
//! them straight to the model, so typing them twice would add nothing.
//!
//! Every method blocks; the app calls them off the main thread. A panic inside becomes a
//! Kotlin exception (uniffi catches it), never a crash of the app.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use imagelab_core::imagelab_api::{ApiError, Event, EventSink, Source};
use imagelab_core::{Lab, LabError, LabOptions};
use serde_json::Value;

uniffi::setup_scaffolding!();

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum ImageLabError {
    /// The request was refused (a bad path, bad JSON, an unknown command, bad params).
    #[error("{reason}")]
    BadRequest { reason: String },
    /// The engine or the file system failed.
    #[error("{reason}")]
    Failed { reason: String },
}

impl From<LabError> for ImageLabError {
    fn from(e: LabError) -> Self {
        match e {
            LabError::BadRequest(reason) => ImageLabError::BadRequest { reason },
            LabError::Api(api) => api.into(),
            other => ImageLabError::Failed { reason: other.to_string() },
        }
    }
}

/// Registry errors keep their stable code at the front of `reason` (`read_only: ...`).
impl From<ApiError> for ImageLabError {
    fn from(e: ApiError) -> Self {
        let reason = e.to_string();
        match e.code.as_str() {
            "bad_params" | "not_found" | "read_only" | "too_large" => ImageLabError::BadRequest { reason },
            _ => ImageLabError::Failed { reason },
        }
    }
}

fn parse_source(source: &str) -> Result<Source> {
    match source.trim().to_ascii_lowercase().as_str() {
        "" | "ui" => Ok(Source::Ui),
        "agent" => Ok(Source::Agent),
        "mcp" => Ok(Source::Mcp),
        "cli" => Ok(Source::Cli),
        other => Err(ImageLabError::BadRequest { reason: format!("unknown source `{other}` (ui, agent, mcp, cli)") }),
    }
}

/// Receives core events (docs/INTERFACES.md §5) as JSON, e.g.
/// `{"type":"docChanged","doc":0,"revision":7,"source":"agent"}`. Called off the main thread.
#[uniffi::export(callback_interface)]
pub trait EventListener: Send + Sync {
    fn on_event(&self, event_json: String);
}

struct ListenerSink(Box<dyn EventListener>);

impl EventSink for ListenerSink {
    fn emit(&self, event: Event) {
        if let Ok(json) = serde_json::to_string(&event) {
            self.0.on_event(json);
        }
    }
}

/// One image a script returned (`{image: "img:N"}` in its value).
#[derive(Debug, Clone, uniffi::Record)]
pub struct ScriptImage {
    pub mime: String,
    pub bytes: Vec<u8>,
}

/// What a `lab_script` / `lab_read` program produced. `value_json` is `null` when `error` is
/// set; `error` is `"<code>: <message>"` (codes: syntax, exception, timeout, cancelled,
/// read_only, ...). `images[k]` is the one whose object in the value has `attached: k`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ScriptResult {
    pub value_json: String,
    pub logs: Vec<String>,
    pub images: Vec<ScriptImage>,
    pub error: Option<String>,
}

type Result<T> = std::result::Result<T, ImageLabError>;

fn parse(json: &str) -> Result<Value> {
    if json.trim().is_empty() {
        return Ok(Value::Object(Default::default()));
    }
    serde_json::from_str(json).map_err(|e| ImageLabError::BadRequest { reason: format!("invalid JSON: {e}") })
}

fn text(value: &Value) -> String {
    value.to_string()
}

/// What `photocraft-cli` printed and returned.
#[derive(Debug, Clone, uniffi::Record)]
pub struct CliResult {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// The core: one PhotoCraft session over the app's workspace folder.
#[derive(uniffi::Object)]
pub struct ImageLab {
    lab: Lab,
}

#[uniffi::export]
impl ImageLab {
    /// Open the workspace at `root` (created if missing). `gpu` turns on PhotoCraft's wgpu
    /// renderer (Vulkan) for full-size previews and exports.
    #[uniffi::constructor]
    pub fn new(root: String, gpu: bool) -> Result<Arc<Self>> {
        let lab = Lab::new(Path::new(&root), &LabOptions { gpu })?;
        Ok(Arc::new(Self { lab }))
    }

    pub fn workspace_root(&self) -> String {
        self.lab.root().display().to_string()
    }

    // ------------------------------------------------------------ registry and scripts

    /// `Registry::call("area.name", params)`; `source` is `ui`, `agent`, `mcp` or `cli`.
    pub fn call(&self, id: String, params_json: String, source: String) -> Result<String> {
        let source = parse_source(&source)?;
        Ok(text(&self.lab.call(&id, parse(&params_json)?, source)?))
    }

    /// Areas and function names (no area), or one area's / function's full entries.
    pub fn help(&self, area: Option<String>) -> Result<String> {
        Ok(text(&self.lab.help(area.as_deref().filter(|a| !a.trim().is_empty()))?))
    }

    /// One line per area (for the agent's tool description).
    pub fn areas_summary(&self) -> String {
        self.lab.areas_summary()
    }

    /// Run a program as the in-app agent: `lab_script` (`read_only` false) or `lab_read`.
    /// `timeout_ms` 0 means the default (120 s); capped at 30 min. Never throws.
    pub fn run_script(&self, code: String, read_only: bool, timeout_ms: u64) -> ScriptResult {
        let outcome = self.lab.run_script(&code, read_only, Duration::from_millis(timeout_ms));
        ScriptResult {
            value_json: text(&outcome.value),
            logs: outcome.logs,
            images: outcome.images.into_iter().map(|i| ScriptImage { mime: i.mime, bytes: i.bytes }).collect(),
            error: outcome.error.map(|e| e.to_string()),
        }
    }

    /// Cancel every running script (they finish with error `cancelled: ...`).
    pub fn cancel_scripts(&self) {
        self.lab.cancel_scripts();
    }

    /// Where core events go from now on (replaces the previous listener).
    pub fn set_event_listener(&self, listener: Box<dyn EventListener>) {
        self.lab.set_event_sink(Arc::new(ListenerSink(listener)));
    }

    /// PNG of an `img:N` / `mask:N` handle (masks grayscale), longest side at most `max_side`
    /// (0: full size). For selection overlays.
    pub fn render_handle(&self, handle: String, max_side: u32) -> Result<Vec<u8>> {
        Ok(self.lab.render_handle(&handle, max_side)?)
    }

    // ------------------------------------------------------------ documents

    /// Open a workspace-relative file. Returns PhotoCraft's JSON (index, size, warnings).
    pub fn open(&self, path: String) -> Result<String> {
        Ok(text(&self.lab.open(&path)?))
    }

    /// Copy picked bytes into `imports/` and open them.
    pub fn import_bytes(&self, name: String, bytes: Vec<u8>) -> Result<String> {
        Ok(text(&self.lab.import(&name, &bytes)?))
    }

    /// `file.new` with JSON params, e.g. `{"width":1080,"height":1350}`.
    pub fn new_document(&self, params_json: String) -> Result<String> {
        Ok(text(&self.lab.new_document(parse(&params_json)?)?))
    }

    /// Save or export the active document (by extension, or `format`).
    pub fn save(&self, path: String, format: Option<String>) -> Result<String> {
        Ok(text(&self.lab.save(&path, format.as_deref())?))
    }

    /// Close the active document without saving. (Not `close`: Kotlin objects already have one.)
    pub fn close_document(&self) -> Result<String> {
        Ok(text(&self.lab.close()?))
    }

    pub fn select(&self, index: u32) -> Result<String> {
        Ok(text(&self.lab.select(index as usize)?))
    }

    pub fn session_info(&self) -> String {
        text(&self.lab.session_info())
    }

    pub fn inspect(&self) -> Result<String> {
        Ok(text(&self.lab.inspect()?))
    }

    // ------------------------------------------------------------ commands

    /// Every engine command as a JSON array.
    pub fn commands(&self) -> String {
        text(&self.lab.commands())
    }

    /// Run an engine command by id with JSON params.
    pub fn run(&self, id: String, params_json: String) -> Result<String> {
        Ok(text(&self.lab.run(&id, parse(&params_json)?)?))
    }

    pub fn undo(&self) -> Result<String> {
        Ok(text(&self.lab.undo()?))
    }

    pub fn redo(&self) -> Result<String> {
        Ok(text(&self.lab.redo()?))
    }

    // ------------------------------------------------------------ rendering

    /// The active document as PNG bytes, longest side at most `max_side` (0: full size).
    pub fn render_png(&self, max_side: u32) -> Result<Vec<u8>> {
        Ok(self.lab.render_png(max_side)?)
    }

    /// True once the GPU renderer failed and rendering fell back to the CPU.
    pub fn gpu_disabled(&self) -> bool {
        self.lab.gpu_disabled()
    }

    pub fn drain_diagnostics(&self) -> Vec<String> {
        self.lab.drain_diagnostics()
    }

    // ------------------------------------------------------------ MCP

    /// One MCP request to PhotoCraft's server, in-process: returns the JSON `result`.
    pub fn mcp_request(&self, method: String, params_json: String) -> Result<String> {
        Ok(text(&self.lab.mcp_request(&method, parse(&params_json)?)?))
    }

    /// One MCP request to ImageLab's own server (`lab_script`, `lab_read`, `lab_help`),
    /// in-process with edits allowed: returns the JSON `result`.
    pub fn lab_mcp_request(&self, method: String, params_json: String) -> Result<String> {
        Ok(text(&self.lab.lab_mcp_request(&method, parse(&params_json)?)?))
    }

    /// Serve MCP over HTTP on 127.0.0.1:`port` with a bearer token: ImageLab's tools at `/mcp`,
    /// PhotoCraft's at `/photocraft/mcp`. Without `allow_edits` only `lab_read`/`lab_help`
    /// are served. Returns the port.
    pub fn start_mcp_http(&self, port: u16, token: String, allow_edits: bool) -> Result<u16> {
        Ok(self.lab.start_http(port, &token, allow_edits)?)
    }

    pub fn stop_mcp_http(&self) {
        self.lab.stop_http();
    }

    pub fn mcp_http_port(&self) -> Option<u16> {
        self.lab.http_port()
    }

    // ------------------------------------------------------------ CLI

    /// Run a `photocraft-cli` command line in the workspace, e.g. `inspect imports/a.jpg`.
    pub fn cli(&self, command_line: String) -> CliResult {
        match imagelab_core::cli::split_command_line(&command_line) {
            Ok(args) => self.cli_args(args),
            Err(message) => CliResult { code: 2, stdout: String::new(), stderr: message },
        }
    }

    /// Run `photocraft-cli` with already-split arguments.
    pub fn cli_args(&self, args: Vec<String>) -> CliResult {
        let out = self.lab.cli(&args);
        CliResult { code: out.code, stdout: out.stdout, stderr: out.stderr }
    }
}
