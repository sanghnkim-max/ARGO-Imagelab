# ImageLab — architecture

**Goal:** a standalone Android photo editor whose real purpose is to try PhotoCraft's **MCP
server and CLI** on a phone: the same agent surface PhotoCraft has on the desktop, driving the
document a person sees, from an in-app agent or from an agent on a laptop. Editing UI is kept
small on purpose; every edit is a PhotoCraft engine command, so the agent can do far more than
the buttons show. AI image generation comes next, then a video editor on the same skeleton.

```
┌─────────────────────────── Android app (Kotlin, Jetpack Compose) ───────────────────────────┐
│  Editor: preview (pinch/pan) · Adjust · Filters · Agent · CLI console · MCP switch           │
│  Agent: Claude tool loop (Anthropic Java SDK) ──┐                                            │
└────────────────────────────────────────────────┼─────────────────────────────────────────────┘
        uniffi (JNA) · libimagelab_ffi.so         │ tools/list, tools/call (in-process)
┌─────────────────────────────── imagelab-core (Rust) ────────────────────────────────────────┐
│   one photocraft_automation::Headless session  (Arc<Mutex<…>>, capability-scoped workspace)   │
│        ▲ UI calls        ▲ PhotocraftMcp (in-process pipe)   ▲ PhotocraftMcp (HTTP, :7700)   │
│        │ open/run/render │                                   │  Bearer token, loopback only   │
│   photocraft_cli::run (in-process console)          libimagelab_cli.so (adb shell)           │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
        PhotoCraft crates as git deps: engine (~1000 commands) · automation (MCP) · cli · io · gpu (wgpu → Vulkan)
```

## 1. What is reused, and how (the ARGO-Workspace rule)

Like ARGO-Workspace with the craft engines, ImageLab consumes PhotoCraft **as-is, as Cargo git
dependencies pinned by rev** (root `Cargo.toml`, `Cargo.lock` seeded from PhotoCraft's own lock).
No vendoring, no fork, no patch of PhotoCraft code. ImageLab touches it only through narrow,
stable seams:

| PhotoCraft seam | Used for |
|---|---|
| `photocraft_automation::Headless` | The session: open (all PhotoCraft formats incl. PSD, HEIC, RAW, .pcraft), `command_run`, `render_png`, `save`, `inspect`, jobs |
| `AuthorizedWorkspace` | Capability-scoped file access: the app's `files/ImageLab/` folder; no absolute paths, `..` or symlink escapes, for UI and agents alike |
| `PhotocraftMcp::with_backend(Backend::Headless(arc))` + `serve_io` | The MCP server, sharing the UI's session; served in-process and over streamable HTTP |
| `photocraft_cli::run(args, out, err)` | The CLI, in-process (console + agent tool) and as the `imagelab` binary |
| `Headless::enable_gpu` (photocraft-gpu, wgpu) | GPU compositing of full-size renders and exports, on Vulkan |

What ImageLab adds is only the host: the FFI, the HTTP transport + bearer auth, the in-process
MCP client, the Android UI, and the agent loop. The compact-projection idea from ARGO (agents
read a text form, not the binary model) is already PhotoCraft's: `doc_inspect` / `info` are the
JSON projection, and `.pcraft` is the native file.

**What we could not reuse:** PhotoCraft's egui UI (desktop Photoshop layout; unusable on a
phone). LightCraft is reference only: its RAW develop pipeline isn't needed on mobile yet;
PhotoCraft already opens camera raws through `photocraft-raw` if a person picks one.

## 2. Why native Rust, not WASM or a WebView

ARGO-Workspace runs its core as wasm in the browser and natively under Tauri. For pixels that
is the wrong trade: wasm is single-threaded here (PhotoCraft's filters use rayon across cores),
size-gated, and has no Vulkan; a WebView canvas means copying every preview across a JS bridge.
On Android the core runs **natively** (`aarch64-linux-android`, cargo-ndk): rayon on all cores,
wgpu on Vulkan, and the preview crosses JNI once as an encoded image. The whole PhotoCraft
dependency tree (engine, automation with GPU, cli) builds for Android unchanged — verified when
this repo was created.

## 3. Rendering

v1: `Lab.render_png(2048)` → PNG bytes → `BitmapFactory` → Compose `Image` with
pinch-zoom/pan. Full-size renders (exports) go through PhotoCraft's GPU renderer when on;
reduced previews use its reduced CPU compositor (PhotoCraft's own split). At startup the core
renders a 16×16 document through the GPU renderer in a throwaway session; only if that works is
the GPU turned on for the real session (on the emulator, without real Vulkan, wgpu falls back to
GL and panics). A GPU failure later still drops to the CPU for the rest of the session.

Next (the "graph acceleration" step), in order of payoff:
1. **Raw RGBA instead of PNG** across the FFI (skip encode/decode). Needs read access to the
   session's `Document` (`Headless` doesn't expose it today) — an upstream accessor, or keep a
   `Session` of our own.
2. **Tile/viewport rendering**: render only the visible region at the zoom level, so zooming in
   shows real pixels instead of an upscaled 2048 px preview (PhotoCraft's raster is tiled and
   mip-mapped already).
3. **Zero-copy GPU surface**: a `SurfaceView` whose `ANativeWindow` becomes a wgpu surface;
   photocraft-gpu composites straight into it. The interface to keep: the UI asks for "a view
   of the active document at (zoom, pan, size)"; how it is produced stays behind the FFI.

## 4. MCP and CLI — the point of the app

* **In-app agent** (`agent/Agent.kt`): a manual Claude tool loop on the official Anthropic Java
  SDK, `claude-opus-5-5`, adaptive thinking (always on), effort `high`, prompt caching (system +
  automatic), server-side refusal fallbacks. Its tools are exactly PhotoCraft's MCP tools
  (`tools/list` over the in-process connection, minus bridge-only `ui_*`/`control_call`), plus
  `photocraft_cli`. `doc_render_preview` returns an image, which goes back to Claude as an image
  tool result, so the agent sees its work.
* **External agents:** the MCP switch serves the same server on `http://127.0.0.1:7700/mcp`
  (rmcp streamable HTTP). Loopback-only, rmcp's Host allow-list against DNS rebinding, and a
  random bearer token — on Android any app can reach loopback, so the token is the access
  control. From a laptop: `adb forward tcp:7700 tcp:7700` and `claude mcp add --transport http …`.
  Edits made this way bump the document revision; the editor polls it and re-renders.
* **CLI:** the CLI console runs `photocraft-cli` in-process with the workspace as current
  directory (a process-wide lock guards the chdir). The APK also ships `libimagelab_cli.so` —
  the `imagelab` executable named as a library so the installer extracts it with exec rights —
  for `adb shell`. The CLI is its own headless session (as on a desktop), not the open document.
* **Off-device:** `imagelab serve-http --root DIR --token T` runs the identical core + HTTP
  server on a laptop, for trying MCP clients without a phone.

## 5. Extending: AI image generation, then video

The seams are chosen so neither needs a redesign:

* **AI generation as tools, not UI code.** A generator (remote API or on-device model) becomes
  either (a) another MCP server the agent lists next to PhotoCraft's (`Agent.loadTools` merges
  sources; remote servers can also come in through the Messages API MCP connector), or (b) a
  core command that writes a new layer through PhotoCraft (`file.place`-style import of the
  generated PNG into `imports/`, then `command_run`). Inpainting = selection/mask from PhotoCraft
  + generator + place-as-layer. The UI gets a "Generate" panel that calls the same tool.
* **Video app:** same repo shape: `crates/<app>-core` around FilmCraft's (or the video craft
  engine's) headless session, the same FFI pattern, the same HTTP MCP transport and agent loop
  (lift `imagelab-core::mcp` and `agent/` into shared crates/modules when the second app lands,
  not before).

## 6. Rules carried over

* Never crash: no `unwrap`/`expect`/`panic` in production Rust (workspace lints); uniffi turns a
  panic into a Kotlin exception; the CLI run is under `catch_unwind`.
* Hostile input: paths are relative and re-checked by PhotoCraft's workspace; picked file names
  sanitized; JSON from the UI parsed with errors, not panics; HTTP body and line sizes bounded.
* Secrets: the API key lives in app-private SharedPreferences (`allowBackup=false`); the MCP
  token is random per install and rotatable.

## 7. Upstream status

* Pinned to PhotoCraft `20fcb61`. `main` at `6d99175` (2026-11-11) does not compile:
  `crates/codecs/src/codecs/tiff/decode.rs` defines `TAG_PREDICTOR` and `TAG_JPEG_TABLES` twice
  after #9781 was merged. Move the pin forward once that's fixed upstream.
* Wanted upstream (would simplify §3): read access to the active `Document` from `Headless`.
