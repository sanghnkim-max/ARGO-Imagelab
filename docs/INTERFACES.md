# ImageLab — interfaces

Interfaces first: what people ask an agent to do with their photos, the one API every surface
dispatches through, and how agents reach it. Implementation follows this document; when they
disagree, fix one of them in the same change.

## 1. How people will use ImageLab with agents

| Job | Example ask | What the agent needs |
|---|---|---|
| **Cull** | "Pick the best 10 of yesterday's 300 shots." | Library query by date/album; per-photo quality scores (sharpness, exposure, noise, faces/eyes); near-duplicate clusters; a contact sheet it can *look at*; ratings/flags to record the choice |
| **Batch enhance** | "Make these consistent and warm, export 2048px JPEGs for Instagram." | Batch map over a selection with progress + cancel; auto tone/color; adjustment presets; export presets |
| **Recipe reuse** | "Do to the others what you did to this one." | An edit as a replayable recipe (PhotoCraft actions / JS); apply to N photos |
| **Targeted edit** | "Blur the background." "Make the sky dramatic." "Remove the person on the left." | Segmentation by point/box/text (SAM), cheap PhotoCraft selections first; masks as first-class objects feeding adjustment layers, fill and inpaint |
| **Fix** | "Straighten the horizon, crop to the subject, denoise the night shots." | OpenCV-style analysis (lines, contours, histograms), transforms, denoise |
| **Organize** | "Album the beach photos, tag people, rename by date." | Content/face tags, albums, metadata read/write |
| **Generate** (next) | "Extend the canvas", "replace the background with a sunset." | AI generation/inpaint as another area; results land as layers |

Two consequences: (1) most jobs touch **many** photos and **many** small operations, so the
agent works in **programs**, not one tool call per step; (2) the agent must be able to **see**
(previews, contact sheets) and **measure** (scores, histograms) before it acts.

## 2. One API, every surface (the ARGO rule)

All capability is a registry of functions in Rust (`crates/imagelab-api`):

```
Function { area, name, summary, description, params (JSON Schema), effects, handler }
Effects  = ReadOnly | Revertible | Destructive      // declared in code, never inferred
```

Areas (each owned by one crate, registered at startup):

| Area | Crate | Contents |
|---|---|---|
| `library` | imagelab-library | list/query/import photos, albums, ratings, flags, tags, metadata |
| `analyze` | imagelab-library | quality scores, duplicate clusters, histogram, faces, contact sheet |
| `doc` | imagelab-edit | open/close/inspect/preview/save the PhotoCraft document; history |
| `edit` | imagelab-edit | PhotoCraft engine commands (allowlisted, generated from the registry), masked adjustments, recipes |
| `batch` | imagelab-edit | apply a recipe/program to many photos; export presets |
| `select` / `segment` | imagelab-segment | PhotoCraft selections + SAM 3 masks (point, box, text concept) |
| `cv` | imagelab-cv | OpenCV-style image ops on handles |
| `ai` | (later) | generation/inpaint providers |

Surfaces, all dispatching `Registry::call(ctx, "area.name", params)`:

* **In-app agent** — tools `lab_script` (Revertible) and `lab_read` (ReadOnly), §3.
* **MCP** (loopback HTTP + bearer): the same two tools plus `lab_help`; and, opt-in, PhotoCraft's
  own MCP tools for people who want the raw engine. Read-only unless "allow edits" is on.
* **CLI**: `imagelab call area.name '{json}'`, `imagelab script file.js`, `imagelab help [area]`;
  plus the existing `photocraft-cli` passthrough.
* **App UI**: Kotlin calls the same functions through the FFI (`call(name, json)`), so a button
  and an agent step are the same code path.

## 3. Agent surface: one script tool, disclosed progressively

Modelled on ARGO's `device` tool (`argo-tool-builtin/src/device_script.rs`) and its code
runtime (`argo-code-runtime`, rquickjs, one realm per program):

1. The **tool description** lists one line per area — enough to know what exists.
2. `lab.help('cv')` inside a program returns that area's functions with schemas.
3. `await lab.cv.canny({img, low: 50, high: 150})` is the call, dispatched through the registry
   (effects, budgets and approvals apply per call).

```js
// Cull: top 10 sharp, well-exposed, one per near-duplicate cluster
const photos = await lab.library.query({ takenAfter: "2026-11-10", limit: 500 });
const scores = await lab.analyze.quality({ photos: photos.map(p => p.id) });
const clusters = await lab.analyze.duplicates({ photos: photos.map(p => p.id), threshold: 10 });
const best = clusters.map(c => c.members.sort((a, b) => scores[b].overall - scores[a].overall)[0]);
best.sort((a, b) => scores[b].overall - scores[a].overall);
await lab.library.setRating({ photos: best.slice(0, 10), rating: 5 });
return { picked: best.slice(0, 10), sheet: await lab.analyze.contactSheet({ photos: best.slice(0, 24) }) };
```

```js
// Targeted edit: blur the background of the open photo
const subject = await lab.segment.subject({});                     // SAM 3, cached embedding
const bg = await lab.cv.invertMask({ mask: subject.mask });
await lab.edit.run({ id: "filter.blur.gaussianBlur", params: { radius: 12 }, mask: bg });
return await lab.doc.preview({ maxSide: 768 });
```

Rules of the runtime (`crates/imagelab-script`):
* rquickjs, a fresh realm per program; no `fetch`, no timers beyond the budget, no FS — only `lab`.
* Budgets: wall time (default 120 s, batch up to 30 min with progress), calls, memory; abort on
  cancel. Errors are values (`LabCallError` with `code`, `message`), never crashes.
* The program's `return` value is the tool result; images in it (`{image: handle}` from
  `preview`/`contactSheet`) are returned to the model as image blocks.
* `console.log` is captured into the result (bounded).

`lab_read` runs the same program in a **read-only** realm: any non-ReadOnly function throws
`LabCallError{code:"read_only"}`. That is how plan mode / "look but don't touch" works: by
effects declared in code, not by words.

## 4. Data model

* **Photo** — a file in the workspace `library/` (imported from the gallery/camera by copy), id =
  stable string; metadata + ratings in the catalog (LightCraft's `lightcraft-catalog` where it
  fits). Originals are never modified; edits live in a `.pcraft` sidecar (`edits/<id>.pcraft`).
* **Handles** — images and masks inside a run or session live in a handle arena
  (`img:12`, `mask:3`), never as bytes in JS. `ImageBuf` = 8-bit RGBA (16-bit later);
  `MaskBuf` = 8-bit coverage. Handles from a run are freed when it ends unless `keep`.
* **Document** — the PhotoCraft session's documents (layers, history). `edit.*` with `mask`
  turns the mask into the layer mask of an adjustment layer (non-destructive by default).
* **Recipe** — a list of `{fn, params}` steps (or a JS program) stored in `recipes/`; replayable.

## 5. Events: push, not poll

The core emits events through one sink (uniffi callback interface to Kotlin; SSE notifications
over MCP):

```
DocChanged { doc, revision, source: Ui|Agent|Mcp|Cli }    LibraryChanged { photos }
JobProgress { job, done, total, message }                  JobFinished { job, ok, summary }
ModelStatus { model, state: missing|downloading|ready, progress }
```

`source` lets the UI tell its own edits from the agent's (ARGO's `writeId`). The UI re-renders on
`DocChanged`; the revision poll goes away.

## 6. Segmentation, used wisely

Three engines, cheapest that does the job first:

* **Manual / PhotoCraft** (no model): rectangle, ellipse and freehand lasso (`select.rect`,
  `select.lasso`), plus Select Subject / Sky / Object / Color Range. Instant, always available.
* **MobileSAM** (the phone's model): points and boxes. TinyViT-5M encoder + SAM's prompt encoder and
  mask decoder on candle (pure Rust), 40.6 MB weights, Apache-2.0, public. The app downloads it once
  (platform TLS) and the core installs it after checking size and SHA-256 (`segment.installModel
  {model:"mobilesam"}`). Measured with the real weights: IoU 0.99 on point and box prompts.
* **SAM 3** (desktop, optional): text concepts and the best quality; 3.4 GB, SAM License (gated,
  per-user acceptance); refused on low-memory devices.

Used wisely:
* The encoder (the slow part) runs **once per image** on a ≤1008 px proxy, ahead of time:
  `segment.prepare` when the Select tool opens. Each tap or box then only runs the decoder.
* Embeddings are cached by a coarse image fingerprint (16×16 block means, small tolerance), so a
  selection change or render jitter reuses the embedding and a real edit encodes again.
* Point prompts accumulate: *Add* / *Subtract* append include / exclude points to the same prompt.
  Shapes combine with the current mask (`cv.combineMasks` union / subtract).
* "Subject" = PhotoCraft Select Subject for *where*, then MobileSAM with that box for a clean edge.
* Text concepts other than sky/subject need SAM 3; without it the agent looks at a preview,
  locates the thing and calls `segment.box` (MobileSAM segments it precisely).

Speed: the encoder and the mask decoder are ImageLab's own (`imagelab-segment/src/tiny_vit.rs`,
`sam_decoder.rs`): the same weights and output as candle's (encoder to 1e-6, mask logits to 1e-5
relative). Batch norms are folded into the convs. Depthwise convs, windowed and cross attention
and the upscaler's pixel shuffle are parallel kernels. Bias, GELU, ReLU and residual adds run fused
after each gemm. The prompt encoder stays candle's, with its constant positional encoding computed once.

| | candle | ImageLab |
|---|---|---|
| Encode, M2 Pro, 12 threads | 2.4 s | 0.23 s |
| Encode, M2 Pro, 1 thread | 3.7 s | 1.15 s |
| Tap (decode), M2 Pro, 12 threads | 80 ms | 21 ms |
| Tap, M2 Pro, 1 thread | 139 ms | 64 ms |
| Tap, one-core emulator | 248 ms | 95 ms |

On the one-core emulator an encode takes ~1.9 s with the host idle. Under heavy host load it was
75 s with candle's encoder and 11 s with ours. What remains is mostly gemm. Open: a GPU (wgpu)
or NNAPI backend; caching the first cross-attention's image-side projections per embedding (~15%
of a tap).

## 7. `cv`: OpenCV-style, for agents

Pure Rust (no OpenCV C++), names and parameters close to OpenCV so models already know them:
`read, write, info, resize, crop, rotate, flip, warpAffine, warpPerspective, cvtColor,
gaussianBlur, medianBlur, bilateralFilter, canny, sobel, threshold, adaptiveThreshold,
morphology (erode/dilate/open/close), findContours (→ boxes, areas), houghLines,
connectedComponents, histogram, meanStd, laplacianVariance, matchTemplate, inpaint, denoise,
maskFromColor, invertMask, combineMasks, featherMask, applyMask, toDocumentLayer`.
Images in, handles out; every op bounded by size.

## 8. Security and trust

* Tool results that contain photo-derived text (EXIF, OCR, file names) are untrusted data.
* MCP HTTP: loopback, bearer token, Host + Origin checks, read-only tools unless the person
  turns on "allow edits".
* API key: Android Keystore (move from SharedPreferences).
* Same never-crash rules as the core; every function bounds its inputs (pixels, counts, time).

## 9. Departures from ARGOv2, on purpose

* One shared Rust session for UI + agent + MCP (ARGO: file on disk is the truth, separate
  cores). Revisions + events + the session lock replace file versions.
* Core in the main process (ARGO: `:core` service). Fine for a standalone editor; revisit if
  long batch jobs must survive backgrounding (then a foreground service hosts the core).
* Agent loop in Kotlin (ARGO: Rust loop). The loop is thin because all tools are Rust; move it
  into Rust when the CLI needs `imagelab agent "…"` headless.

## 10. App UI

### Layout (phone, portrait)

Reference: a mobile layered editor — full-bleed canvas, chrome at the edges, our own icons and names.

```
┌──────────────────────────────────────────┐
│ [⌂]        [↶][↷]  [⇪ share] [✦ agent] [⚙]│  top bar: home (library), undo/redo, share/export, agent, settings
│                                          │
│              canvas (pinch/pan)     [▣]◀ │  layer strip: collapsible thumbnails on the right edge
│                                          │
│                                     (+)  │  floating add: photo layer, text, fill, adjustment, generate
├──────────────────────────────────────────┤
│ Layers │ Select │ Adjust │ Retouch │ Crop │  tool bar → contextual sheet above it
└──────────────────────────────────────────┘
```

* **Library** (home): photo grid with ratings/flags, multi-select, "Ask agent about these" — the
  entry point for cull and batch jobs; batch progress shows as a card.
* **Select**: tap = SAM point, drag = box, type = concept ("sky"); Subject/Sky quick chips
  (PhotoCraft, no model). The mask shows as an overlay; actions: adjust inside, blur, remove, cut to layer.
* **Adjust**: adjustment layers (light, color, effects); applies to the selection if one exists.
* **Retouch**: heal/remove brush, content-aware fill, denoise.
* **Agent**: a sheet (phone) or side pane (large screens); shows the program it runs (collapsed),
  progress, previews it looked at, and chips for photos it touched.
* CLI console and MCP switch live under Settings → Developer.

### Foldables and tablets

Window size classes (androidx.window / Material 3 adaptive), never device checks:

| Class | Layout |
|---|---|
| Compact (phone, folded) | as above: bottom tool bar, sheets over the canvas, agent as a sheet |
| Medium (unfolded portrait, small tablet) | navigation rail on the left (tools), layer strip expanded, agent/tool panel as a right pane |
| Expanded (tablet, unfolded landscape) | rail + canvas + persistent right pane (agent or tool options) + layers column; library as a two-pane list/detail |

Foldable postures (WindowLayoutInfo / FoldingFeature):
* **Tabletop** (half-open, horizontal hinge): canvas on the top half, tools + agent on the bottom half.
* **Book** (vertical hinge): canvas on one side, agent/tool pane on the other; nothing straddles the hinge.

State survives configuration changes (fold/unfold, rotation): the session lives in the core, the
UI keeps only view state in ViewModels. Stylus input (tablets): pressure for retouch brushes;
hover shows the brush cursor.

## 11. FFI contract (Kotlin ↔ core)

Added to `imagelab-ffi` next to the existing calls:

```
call(id: String, paramsJson: String, source: String) -> String     // Registry::call; throws ImageLabException
help(area: String?) -> String                                       // Registry::help
areasSummary() -> String
runScript(code: String, readOnly: Boolean, timeoutMs: ULong) -> ScriptResult
    ScriptResult { valueJson: String, logs: List<String>, images: List<ScriptImage>, error: String? }
    ScriptImage  { mime: String, bytes: ByteArray }
cancelScripts()
setEventListener(listener: EventListener)        // callback interface
    EventListener.onEvent(eventJson: String)      // Event serialized (§5), called off the main thread
renderHandle(handle: String, maxSide: UInt) -> ByteArray   // PNG of img:N / mask:N for overlays
```
