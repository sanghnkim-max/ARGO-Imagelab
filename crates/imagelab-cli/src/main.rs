//! `imagelab`: PhotoCraft's CLI as it ships inside ImageLab, plus ImageLab's own registry.
//!
//! * `imagelab call area.name '{json}' --root DIR`: one registry call; prints the JSON result.
//! * `imagelab help [area | area.fn] [--root DIR]`: the registry's areas, or one area in full.
//! * `imagelab script FILE.js --root DIR [--read-only] [--out-dir D] [--timeout-ms N]`: run a
//!   `lab_script` program (`-` reads stdin); prints `{value, logs, error, images}` and writes
//!   attached images to `D/image-<k>.png`.
//! * `imagelab serve-http --root DIR [--port N] [--token T] [--allow-edits]`: ImageLab's core
//!   with its streamable HTTP MCP server, the same server the app runs.
//! * `imagelab <photocraft-cli args>`: exactly `photocraft-cli` (convert, inspect, run, batch,
//!   `mcp` over stdio, …). On a phone: `adb shell` into the app's bundled copy (see the README).
//!
//! Exit codes: 0 ok, 1 the call/program failed, 2 usage.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::io::{Read as _, Write as _};
use std::path::PathBuf;
use std::time::Duration;

use imagelab_core::imagelab_api::Source;
use imagelab_core::{Lab, LabOptions};
use serde_json::Value;

/// Largest program file read.
const MAX_SCRIPT_FILE: u64 = 1 << 20;

fn main() {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let rest = || args[1..].iter().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>();
    let code = match args.first().and_then(|a| a.to_str()) {
        Some("serve-http") => serve(&rest()),
        Some("call") => call(&rest()),
        Some("help") => help(&rest()),
        Some("script") => script(&rest()),
        _ => photocraft_cli::run(args, &mut std::io::stdout(), &mut std::io::stderr()),
    };
    std::process::exit(code);
}

/// Flags shared by the ImageLab subcommands, plus positional arguments in order.
#[derive(Default)]
struct Flags {
    root: Option<PathBuf>,
    port: Option<u16>,
    token: Option<String>,
    allow_edits: bool,
    read_only: bool,
    out_dir: Option<PathBuf>,
    timeout_ms: u64,
    gpu: bool,
    positional: Vec<String>,
}

fn parse_flags(args: &[String]) -> Result<Flags, String> {
    let mut f = Flags::default();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--root" => f.root = Some(PathBuf::from(value("--root")?)),
            "--port" => f.port = Some(value("--port")?.parse().map_err(|_| "--port needs a number".to_string())?),
            "--token" => f.token = Some(value("--token")?),
            "--out-dir" => f.out_dir = Some(PathBuf::from(value("--out-dir")?)),
            "--timeout-ms" => f.timeout_ms = value("--timeout-ms")?.parse().map_err(|_| "--timeout-ms needs a number".to_string())?,
            "--allow-edits" => f.allow_edits = true,
            "--read-only" => f.read_only = true,
            "--gpu" => f.gpu = true,
            "-h" | "--help" => return Err(String::new()),
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => f.positional.push(other.to_string()),
        }
    }
    Ok(f)
}

fn open_lab(f: &Flags, gpu: bool) -> Result<Lab, i32> {
    let Some(root) = &f.root else { return Err(usage("--root DIR is required")) };
    Lab::new(root, &LabOptions { gpu }).map_err(|e| {
        eprintln!("imagelab: {e}");
        1
    })
}

fn print_json(v: &Value) {
    let text = serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string());
    let mut out = std::io::stdout();
    let _ = writeln!(out, "{text}");
    let _ = out.flush();
}

fn call(args: &[String]) -> i32 {
    let f = match parse_flags(args) {
        Ok(f) => f,
        Err(e) => return usage(&e),
    };
    let (id, params) = match f.positional.as_slice() {
        [id] => (id.clone(), Value::Object(Default::default())),
        [id, json] => match serde_json::from_str(json) {
            Ok(v) => (id.clone(), v),
            Err(e) => return usage(&format!("params are not JSON: {e}")),
        },
        _ => return usage("call takes area.name and an optional JSON object"),
    };
    let lab = match open_lab(&f, f.gpu) {
        Ok(l) => l,
        Err(code) => return code,
    };
    // `--read-only` as a `lab_read` realm would apply it: only ReadOnly functions run.
    if f.read_only && lab.registry().get(&id).is_some_and(|fun| fun.effects != imagelab_core::imagelab_api::Effects::ReadOnly) {
        eprintln!("imagelab: read_only: {id} changes things; drop --read-only to run it");
        return 1;
    }
    match lab.call(&id, params, Source::Cli) {
        Ok(v) => {
            print_json(&v);
            0
        }
        Err(e) => {
            eprintln!("imagelab: {e}");
            1
        }
    }
}

fn help(args: &[String]) -> i32 {
    let f = match parse_flags(args) {
        Ok(f) => f,
        Err(e) => return usage(&e),
    };
    // The registry needs no workspace; --root is accepted for symmetry.
    let registry = imagelab_core::build_registry();
    match registry.help(f.positional.first().map(String::as_str)) {
        Ok(v) => {
            print_json(&v);
            0
        }
        Err(e) => {
            eprintln!("imagelab: {e}");
            1
        }
    }
}

fn script(args: &[String]) -> i32 {
    let f = match parse_flags(args) {
        Ok(f) => f,
        Err(e) => return usage(&e),
    };
    let [file] = f.positional.as_slice() else { return usage("script takes one FILE.js (or - for stdin)") };
    let code = match read_program(file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("imagelab: {e}");
            return 1;
        }
    };
    let lab = match open_lab(&f, f.gpu) {
        Ok(l) => l,
        Err(code) => return code,
    };
    let outcome = lab.run_script_as(&code, f.read_only, Duration::from_millis(f.timeout_ms), Source::Cli);
    let mut report = outcome.to_json();
    if let Some(dir) = &f.out_dir {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("imagelab: cannot create {}: {e}", dir.display());
            return 1;
        }
        let mut written = Vec::new();
        for (k, image) in outcome.images.iter().enumerate() {
            let path = dir.join(format!("image-{k}.png"));
            match std::fs::write(&path, &image.bytes) {
                Ok(()) => written.push(Value::String(path.display().to_string())),
                Err(e) => eprintln!("imagelab: cannot write {}: {e}", path.display()),
            }
        }
        if let Value::Object(map) = &mut report {
            map.insert("written".into(), Value::Array(written));
        }
    } else if !outcome.images.is_empty() {
        eprintln!("imagelab: {} image(s) attached; pass --out-dir DIR to write them", outcome.images.len());
    }
    print_json(&report);
    if outcome.error.is_some() { 1 } else { 0 }
}

fn read_program(file: &str) -> Result<String, String> {
    let mut code = String::new();
    if file == "-" {
        std::io::stdin().take(MAX_SCRIPT_FILE + 1).read_to_string(&mut code).map_err(|e| format!("stdin: {e}"))?;
    } else {
        let f = std::fs::File::open(file).map_err(|e| format!("{file}: {e}"))?;
        f.take(MAX_SCRIPT_FILE + 1).read_to_string(&mut code).map_err(|e| format!("{file}: {e}"))?;
    }
    if code.len() as u64 > MAX_SCRIPT_FILE {
        return Err(format!("{file} is larger than {MAX_SCRIPT_FILE} bytes"));
    }
    Ok(code)
}

fn serve(args: &[String]) -> i32 {
    let f = match parse_flags(args) {
        Ok(f) => f,
        Err(e) => return usage(&e),
    };
    if !f.positional.is_empty() {
        return usage(&format!("unexpected argument {}", f.positional[0]));
    }
    let Some(token) = f.token.clone().or_else(|| std::env::var("IMAGELAB_MCP_TOKEN").ok()) else {
        return usage("--token (or IMAGELAB_MCP_TOKEN) is required");
    };
    let lab = match open_lab(&f, true) {
        Ok(l) => l,
        Err(code) => return code,
    };
    match lab.start_http(f.port.unwrap_or(7700), &token, f.allow_edits) {
        Ok(bound) => {
            let mode = if f.allow_edits { "edits allowed" } else { "read-only" };
            println!("ImageLab MCP: http://127.0.0.1:{bound}/mcp ({mode}; workspace {})", lab.root().display());
            if f.allow_edits {
                println!("PhotoCraft MCP: http://127.0.0.1:{bound}/photocraft/mcp");
            }
            let _ = std::io::stdout().flush();
        }
        Err(e) => {
            eprintln!("imagelab: {e}");
            return 1;
        }
    }
    // Serve until killed.
    loop {
        std::thread::park();
    }
}

fn usage(problem: &str) -> i32 {
    if !problem.is_empty() {
        eprintln!("imagelab: {problem}");
    }
    eprintln!("usage: imagelab call area.name ['{{json}}'] --root DIR");
    eprintln!("       imagelab help [area | area.fn]");
    eprintln!("       imagelab script FILE.js|- --root DIR [--read-only] [--out-dir DIR] [--timeout-ms N]");
    eprintln!("       imagelab serve-http --root DIR [--port 7700] [--token TOKEN | env IMAGELAB_MCP_TOKEN] [--allow-edits]");
    eprintln!("       imagelab <photocraft-cli arguments>   (see `imagelab --help`)");
    eprintln!("       (call/script take --gpu to use PhotoCraft's GPU renderer)");
    if problem.is_empty() { 0 } else { 2 }
}
