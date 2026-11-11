//! `photocraft-cli`, in-process.
//!
//! PhotoCraft's CLI is a library (`photocraft_cli::run(args, out, err)`), so the app runs the
//! real thing (convert, inspect, run, batch, actions…) rather than a re-implementation. The
//! `imagelab` binary (crates/imagelab-cli) is the same call behind `main`, for a desktop shell or
//! `adb shell`.
//!
//! The CLI resolves relative paths against the process's current directory, so a run takes a
//! process-wide lock and sets the current directory to the workspace for its duration. The
//! app's own session is unaffected: its file access goes through directory capabilities opened
//! at the workspace's absolute path.

use std::path::Path;
use std::sync::{Mutex, PoisonError};

/// One CLI run at a time (the current directory is process-wide).
static CLI_LOCK: Mutex<()> = Mutex::new(());

/// Largest stdout/stderr kept from one run (an `inspect` of a big PSD can be long).
const MAX_OUTPUT_BYTES: usize = 4 << 20;

#[derive(Debug, Clone)]
pub struct CliOutput {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Run `photocraft-cli <args>` with `root` as the current directory.
pub fn run(root: &Path, args: &[String]) -> CliOutput {
    let _guard = CLI_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let previous = std::env::current_dir().ok();
    if let Err(e) = std::env::set_current_dir(root) {
        return CliOutput { code: 1, stdout: String::new(), stderr: format!("cannot enter the workspace: {e}") };
    }
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| photocraft_cli::run(args.iter().cloned(), &mut out, &mut err))) {
        Ok(code) => code,
        Err(_) => {
            err.extend_from_slice(b"photocraft-cli panicked\n");
            101
        }
    };
    if let Some(previous) = previous {
        let _ = std::env::set_current_dir(previous);
    }
    CliOutput { code, stdout: bounded(out), stderr: bounded(err) }
}

fn bounded(mut bytes: Vec<u8>) -> String {
    if bytes.len() > MAX_OUTPUT_BYTES {
        bytes.truncate(MAX_OUTPUT_BYTES);
        bytes.extend_from_slice(b"\n[output truncated]\n");
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Split a command line the way a shell would for the simple cases a console needs: spaces,
/// single and double quotes, backslash escapes. A leading `photocraft-cli` or `imagelab` is
/// dropped, so people can paste commands from PhotoCraft's docs.
pub fn split_command_line(line: &str) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some('"'), '\\') | (None, '\\') => {
                if let Some(next) = chars.next() {
                    current.push(next);
                    in_word = true;
                }
            }
            (Some(_), c) => current.push(c),
            (None, '"' | '\'') => {
                quote = Some(c);
                in_word = true;
            }
            (None, c) if c.is_whitespace() => {
                if in_word {
                    args.push(std::mem::take(&mut current));
                    in_word = false;
                }
            }
            (None, c) => {
                current.push(c);
                in_word = true;
            }
        }
    }
    if quote.is_some() {
        return Err("unterminated quote".into());
    }
    if in_word {
        args.push(current);
    }
    if matches!(args.first().map(String::as_str), Some("photocraft-cli" | "imagelab")) {
        args.remove(0);
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_a_shell() {
        assert_eq!(split_command_line("inspect a.psd").unwrap(), ["inspect", "a.psd"]);
        assert_eq!(split_command_line(r#"photocraft-cli run "my file.png" '{"a": 1}'"#).unwrap(), ["run", "my file.png", r#"{"a": 1}"#]);
        assert_eq!(split_command_line(r"convert a\ b.png out.jpg").unwrap(), ["convert", "a b.png", "out.jpg"]);
        assert_eq!(split_command_line(r#"x """#).unwrap(), ["x", ""]);
        assert!(split_command_line("\"open").is_err());
    }
}
