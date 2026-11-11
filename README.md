# ImageLab

A standalone Android photo editor on [PhotoCraft](https://github.com/storytold/photocraft)'s engine,
built to try PhotoCraft's **MCP server and CLI** on a phone: an in-app Claude agent, an MCP server
that agents on your laptop can drive over `adb`, and a `photocraft-cli` console, all on the
document you see. Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

| Path | What |
|---|---|
| `crates/imagelab-core` | One PhotoCraft session shared by UI, in-app agent, MCP (in-process + HTTP) and CLI |
| `crates/imagelab-ffi` | uniffi bindings for Kotlin (`libimagelab_ffi.so`) |
| `crates/imagelab-cli` | `imagelab`: `photocraft-cli`, plus `serve-http` (the app's HTTP MCP server, off-device) |
| `android/` | The app: Kotlin + Jetpack Compose; agent on the Anthropic Java SDK |
| `skills/imagelab` | Agent Skill for editing over the ImageLab MCP server |
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

In the app: **MCP** tab → switch on (it shows the token). Then:

```bash
adb forward tcp:7700 tcp:7700
```

```bash
claude mcp add --transport http imagelab http://localhost:7700/mcp --header "Authorization: Bearer <token>"
```

Or export `IMAGELAB_MCP_TOKEN=<token>` and use this repo's `.mcp.json`. Without a phone:

```bash
cargo run -p imagelab-cli -- serve-http --root ./workspace --token "$(openssl rand -hex 24)"
```

## CLI on the phone

In-app: the **CLI** tab runs `photocraft-cli` in the workspace (`commands --filter blur`,
`info imports/<file>`). Over adb, the APK's bundled binary (its path is shown in the MCP tab):

```bash
APK_DIR=$(adb shell pm path dev.argo.imagelab | sed 's/package://; s/base.apk//' | tr -d '\r')
```

```bash
adb shell "run-as dev.argo.imagelab sh -c 'cd files/ImageLab && ${APK_DIR}lib/arm64/libimagelab_cli.so info exports/from-laptop.png'"
```

## In-app agent

**Agent** tab → Settings → your Anthropic API key (stored in app-private storage). Default model
`claude-opus-5-5`. The agent's tools are PhotoCraft's MCP tools plus `photocraft_cli`.

Licence: Apache-2.0. PhotoCraft is MIT OR Apache-2.0.
