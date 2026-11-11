//! `imagelab`: PhotoCraft's CLI as it ships inside ImageLab.
//!
//! * `imagelab <photocraft-cli args>`: exactly `photocraft-cli` (convert, inspect, run, batch,
//!   `mcp` over stdio, …). On a phone: `adb shell` into the app's bundled copy (see the README).
//! * `imagelab serve-http --root DIR [--port N] [--token T]`: ImageLab's own core with its streamable
//!   HTTP MCP server, the same server the app runs, for trying MCP clients on a laptop.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::io::Write as _;
use std::path::PathBuf;

fn main() {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    if args.first().and_then(|a| a.to_str()) == Some("serve-http") {
        std::process::exit(serve(&args[1..]));
    }
    let code = photocraft_cli::run(args, &mut std::io::stdout(), &mut std::io::stderr());
    std::process::exit(code);
}

fn serve(args: &[std::ffi::OsString]) -> i32 {
    let mut root: Option<PathBuf> = None;
    let mut port: u16 = 7700;
    let mut token: Option<String> = std::env::var("IMAGELAB_MCP_TOKEN").ok();
    let mut it = args.iter().map(|a| a.to_string_lossy().into_owned());
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--root" => root = it.next().map(PathBuf::from),
            "--port" => match it.next().and_then(|p| p.parse().ok()) {
                Some(p) => port = p,
                None => return usage("--port needs a number"),
            },
            "--token" => token = it.next(),
            "-h" | "--help" => return usage(""),
            other => return usage(&format!("unknown argument {other}")),
        }
    }
    let Some(root) = root else { return usage("--root is required") };
    let Some(token) = token else { return usage("--token (or IMAGELAB_MCP_TOKEN) is required") };
    let lab = match imagelab_core::Lab::new(&root, &imagelab_core::LabOptions::default()) {
        Ok(lab) => lab,
        Err(e) => {
            eprintln!("imagelab: {e}");
            return 1;
        }
    };
    match lab.start_http(port, &token) {
        Ok(bound) => {
            println!("ImageLab MCP: http://127.0.0.1:{bound}/mcp (workspace {})", lab.root().display());
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
        eprintln!("imagelab serve-http: {problem}");
    }
    eprintln!("usage: imagelab serve-http --root DIR [--port 7700] [--token TOKEN | env IMAGELAB_MCP_TOKEN]");
    eprintln!("       imagelab <photocraft-cli arguments>   (see `imagelab --help`)");
    if problem.is_empty() { 0 } else { 2 }
}
