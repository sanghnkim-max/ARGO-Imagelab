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
            other => ImageLabError::Failed { reason: other.to_string() },
        }
    }
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

    /// Serve MCP over HTTP at 127.0.0.1:`port`/mcp with a bearer token. Returns the port.
    pub fn start_mcp_http(&self, port: u16, token: String) -> Result<u16> {
        Ok(self.lab.start_http(port, &token)?)
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
