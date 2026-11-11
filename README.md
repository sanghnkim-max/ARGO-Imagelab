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

In the app: **Settings → Developer → MCP server** → switch on (it shows the token; "Allow edits"
turns on the edit tools). Then:

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

In-app: **Settings → Developer → CLI console** runs `photocraft-cli` in the workspace (`commands --filter blur`,
`info imports/<file>`). Over adb, the APK's bundled binary (its path is shown under Developer):

```bash
APK_DIR=$(adb shell pm path dev.argo.imagelab | sed 's/package://; s/base.apk//' | tr -d '\r')
```

```bash
adb shell "run-as dev.argo.imagelab sh -c 'cd files/ImageLab && ${APK_DIR}lib/arm64/libimagelab_cli.so info exports/from-laptop.png'"
```

## In-app agent

**Settings → Agent** → your Anthropic API key (sealed with an Android Keystore AES-GCM key).
Default model `claude-opus-5-5`. The agent writes small JS programs against the function registry:
tools `lab_script` (may edit) and `lab_read` (read-only realm), plus `photocraft_cli`
(docs/INTERFACES.md §3). Each program shows as a collapsible row with its code, previews and the
photos it touched.

## App UI

Docs: [docs/INTERFACES.md §10](docs/INTERFACES.md). Code: `android/app/src/main/java/dev/argo/imagelab/`.

* **Library** (home): photo grid with ratings/flags, long-press multi-select, rate/flag,
  "Ask agent about these", batch export with a progress card, import from the system picker.
* **Editor**: top bar (Library, undo, redo, export, agent, settings), pinch/pan canvas, layer strip,
  floating add (photo layer, text, fills, adjustment), tools Layers · Select · Adjust · Retouch · Crop.
  Select: Subject/Sky chips, tap = point, drag = box, or type a concept; the mask shows as an overlay.
* **Adaptive**: compact → bottom tool bar + agent sheet; medium/expanded → navigation rail + right
  pane (agent or tool options), expanded adds a layers column and a list/detail library;
  foldables → tabletop (canvas above the hinge) and book (canvas | pane) postures.
* The UI reaches the core only through `core/LabApi.kt`; `core/PendingFfi.kt` holds stand-ins for
  §11 FFI calls not yet exported (delete it once the bindings have them).

Screenshots: `android/docs/screenshots/`.

Licence: Apache-2.0. PhotoCraft is MIT OR Apache-2.0.
