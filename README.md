# ImageLab

A standalone Android photo editor on [PhotoCraft](https://github.com/storytold/photocraft)'s engine,
built to try PhotoCraft's **MCP server and CLI** on a phone: an in-app Claude agent, an MCP server
that agents on your laptop can drive over `adb`, and a `photocraft-cli` console, all on the
document you see. Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

| Path | What |
|---|---|
| `crates/imagelab-core` | One PhotoCraft session shared by UI, in-app agent, MCP (in-process + HTTP) and CLI |
| `crates/imagelab-ffi` | uniffi bindings for Kotlin (`libimagelab_ffi.so`) |
| `crates/imagelab-cli` | `imagelab`: `photocraft-cli`, plus `call` / `help` / `script` over the registry and `serve-http` (the app's HTTP MCP server, off-device) |
| `android/` | The app: Kotlin + Jetpack Compose; agent on the Anthropic Java SDK |
| `crates/imagelab-script` | The agent's JavaScript runtime (rquickjs, realm per program, budgets, read-only realms) |
| `skills/imagelab` | Agent Skill for `lab_read` / `lab_script` / `lab_help` over the ImageLab MCP server |
| `tools/uniffi-bindgen` | Host tool: Kotlin binding generator |

PhotoCraft is a pinned git dependency (root `Cargo.toml`), never vendored.

## Build

Prerequisites: Rust (stable ≥ 1.95) with `aarch64-linux-android`, `cargo install cargo-ndk`,
Android SDK (platform 37) + NDK 27, JDK 17.

```bash
cargo test -p imagelab-core
```

```bash
cd android && ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/27.2.12479018 ./gradlew :app:installDebug
```

Gradle runs cargo-ndk (core + CLI) and uniffi-bindgen before compiling Kotlin.

## Try the MCP server from your laptop

In the app: **MCP** tab → switch on (it shows the token; "allow edits" is off by default). Then:

```bash
adb forward tcp:7700 tcp:7700
```

```bash
claude mcp add --transport http imagelab http://localhost:7700/mcp --header "Authorization: Bearer <token>"
```

`/mcp` is ImageLab's own server: `lab_read` (a read-only JavaScript program over the `lab.*`
registry), `lab_help`, and — only with **allow edits** — `lab_script` (the same program, allowed
to change things). Images a program returns (`{image: "img:N"}`) come back as image content.
PhotoCraft's raw tools (`doc_inspect`, `command_run`, …) are at `/photocraft/mcp`, also only
with allow edits. Every request needs the bearer token; `Host` must be loopback, and
cross-origin / cross-site browser requests are refused. See `skills/imagelab/SKILL.md`.

Or export `IMAGELAB_MCP_TOKEN=<token>` and use this repo's `.mcp.json`. Without a phone:

```bash
cargo run -p imagelab-cli -- serve-http --root ./workspace --token "$(openssl rand -hex 24)" --allow-edits
```

## CLI

`imagelab` is `photocraft-cli` plus ImageLab's registry:

```bash
cargo run -p imagelab-cli -- help                      # areas and function names
cargo run -p imagelab-cli -- help cv                   # one area: descriptions, schemas, effects
cargo run -p imagelab-cli -- call library.query '{"limit": 5}' --root ./workspace
cargo run -p imagelab-cli -- script cull.js --root ./workspace --read-only --out-dir out/
```

`script` runs a `lab_script` program (`-` reads stdin; `--timeout-ms`, `--read-only`), prints
`{value, logs, error, images}` and writes returned images to `--out-dir`. Exit code 1 when the
call or program failed. Anything else is passed to `photocraft-cli` (`imagelab inspect a.psd`).

## CLI on the phone

In-app: the **CLI** tab runs `photocraft-cli` in the workspace (`commands --filter blur`,
`info imports/<file>`). Over adb, the APK's bundled binary (its path is shown in the MCP tab):

```bash
APK_DIR=$(adb shell pm path dev.argo.imagelab | sed 's/package://; s/base.apk//' | tr -d '\r')
```

```bash
adb shell "run-as dev.argo.imagelab sh -c 'cd files/ImageLab && ${APK_DIR}lib/arm64/libimagelab_cli.so info exports/from-laptop.png'"
```

The bundled binary also takes `call`, `help` and `script` with `--root .` (its own core over
the workspace, not the open document).

## In-app agent

**Agent** tab → Settings → your Anthropic API key (stored in app-private storage). Default model
`claude-opus-5-5`. The agent's tools are PhotoCraft's MCP tools plus `photocraft_cli`.

Licence: Apache-2.0. PhotoCraft is MIT OR Apache-2.0.
