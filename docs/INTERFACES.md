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
const bg = await lab.cv.invertMask({ mask: subject.mask });        // {mask: "mask:N", coverage, …}
await lab.select.fromMask({ mask: bg.mask, feather: 2 });          // the background is the selection
await lab.edit.run({ id: "filter.blur.gaussianBlur", params: { radius: 12 } });  // applies inside it
await lab.select.none({});
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
  `MaskBuf` = 8-bit coverage. Handles from a run are freed when it ends unless the result names
  them or the program keeps them (`lab.keep`); an image the result only shows is freed once
  encoded.
* **Document** — the PhotoCraft session's documents (layers, history). To edit part of a photo,
  load a mask as the selection (`select.fromMask`); `edit.*` calls then apply inside it, and a
  new adjustment layer takes it as its layer mask. `select.none` ends it.
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
* Detail pass (`refine: true` on `segment.point` / `segment.box`, the Select tool's *Fine
  detail* switch): the whole-image proxy shows a 12 MP photo at ~1/4 scale, and SAM's mask grid
  is 256² over that. So after the first mask, MobileSAM runs again on a crop of the full-resolution
  pixels around it: the mask's box plus the prompt, with a 15% margin. Decoded photos are resampled
  and the open document is rendered (PhotoCraft's `render_reduced_rect`). The crop is prompted with
  the same points and the first mask's box, and the result is pasted back. The pass skips when
  the crop would add less than 1.4× detail. It keeps the first mask when the two disagree
  (IoU < 0.5). `detail` in the result says which happened. Cost: one more encode (~0.2 s on a
  phone GPU). Measured: a rippled edge on a 3200 px image goes from IoU 0.930 to 0.979 (3.2× detail).
  On a real 3000×4000 photo, a pylon's soft blob becomes straight, tapered sides with the legs
  apart (4× detail, 0.47 s on the phone).
* Then, on the open document, PhotoCraft's Refine Edge runs on the result. The mask is loaded as
  the selection, refined, read back, and the document rolled back, so its selection and history
  are untouched. The smart radius is 3 mask cells of what SAM last saw: ~12 px after a detail pass
  on a crop, up to 32 px on a whole large photo. Calibrated by `calibrate_refine_edge` in
  `tests/mobilesam.rs`:
  * IoU against exact shapes rises up to ~3 cells, then stays flat.
  * Thin 6 px spokes stay intact even at 12 cells.
  * Smooth and contrast made no difference, so they are 0.

  Measured on the document: detail pass 0.978 → with Refine Edge 0.998. Photos and image handles
  have no document to sample, so `detail.edge` says it did not run.
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

GPU (`tiny_vit/gpu.rs`, feature `mobile-sam-gpu`, on by default): the encoder and the mask decoder
as wgpu compute kernels, from the CPU modules' weights. An encode is one submission and one
read-back. So is a tap: all four masks come back with the IoU head and the best is picked on the
CPU. Both match their CPU versions (encoder to 4e-6, mask logits to 2e-6 relative). With no usable
GPU (the emulator has no Vulkan adapter), a failed startup self-test, or any GPU error, MobileSAM
falls back to the CPU. `IMAGELAB_MOBILESAM_GPU=0` forces the CPU, and `segment.models` →
`mobilesam.encoder` says which runs.

| | CPU | GPU |
|---|---|---|
| Encode, M2 Pro | 0.23 s | 0.13 s |
| Encode, Galaxy SM-S948N (Snapdragon SM8850, Adreno 840), on battery | 1.2–2.2 s | 0.18–0.24 s |
| Tap, that phone | 117–205 ms | 34–36 ms |

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
* **Select**: tap = SAM point, drag = box, type = concept ("sky"); Subject/Sky/In focus/All quick
  chips (PhotoCraft, no model); Magic wand and Color modes (a tap: `select.magicWand` /
  `select.colorRange`, with a tolerance). The mask shows as an overlay; actions: adjust inside,
  blur, remove, cut to layer, and refine (invert, feather, expand, contract, smooth, border,
  similar colors). A PhotoCraft selection is copied into a mask handle and then dropped from the
  document, so the handle is the only selection (masked edits load it back).
* **Enhance** (first tool): one-slider, photo-aware looks from `edit.look`, live as the slider
  moves, each one history step. "For this photo" suggests templates for the scene
  (`edit.lookSuggest`: portrait if faces, landscape if a sky). Groups: Templates; Essentials
  (Enhance + sky, Sky, Structure, Details, Denoise, B&W, Vignette); Landscape (Golden hour,
  Foliage, Atmosphere, Sky replacement, Relight); Creative (Dramatic, Glow, Matte, Toning, Film
  grain); Portrait (Smooth skin, Face light, Eyes, Red eye, Background blur, Face shape).
* **Adjust**: quick sliders and auto fixes; Light & color (`filter.cameraRaw`: light, color,
  effects, tone curve, HSL hue/saturation/luminance, color grading, detail, vignette & grain;
  Shadows/Highlights, HDR toning); adjustment layers (Levels, Hue/Saturation, Vibrance, Color
  balance, Selective color, Channel mixer, Gradient map, Photo filter, Look, B&W, Posterize,
  Threshold, Invert); color tools: Replace color (tap the photo to sample; `points`), Match color
  (to another pixel layer, `sourceLayer`). **Curves**: a Curves adjustment layer edited on a draggable curve (RGB,
  red, green, blue; up to 19 points each; presets), drawn with the same natural cubic spline the
  engine applies (checked level by level), over the photo's histogram; a point is added only by
  touching the line, the end points move too, and dragging a middle point out of the box removes
  it. Adjustment layers preview on the proxy while dragging. Applies to the selection if one exists. Catalog keys may address part of a
  structured param (`hslSat[2]`, `gradeShadows.hue`; a `json:` choice value is sent as JSON).
* **Filters**: Blur (incl. lens, spin, field, smart, box), Sharpen, Noise, Lens (Lens
  correction), Painting / Sketch / Texture (PhotoCraft's Filter Gallery, picture frames),
  Stylize, Distort, Pixelate, Render. **Effects**: layer styles.
* **Layers**: order, duplicate, merge, group, clipping masks, masks, remove background
  (`layer.removeBackground`), defringe, opacity / fill / blend.
* **Crop & size**: aspect crops, rotate (90°, 180°), flip, straighten, Geometry (lens
  correction: perspective, distortion), image and canvas size.
* **Retouch**: brushes painted on the canvas. Repair: Heal (`paint.spotHealing`), Remove
  (`paint.remove`), Clone / Heal from (`paint.cloneStamp` / `paint.healingBrush`: tap sets the
  source, the first stroke sends `source`, later strokes the returned aligned `offset`), Patch
  and Move (`paint.patch` content-aware / `paint.contentAwareMove`: drag the selection's content;
  the selection is dropped after). Tone: Lighten/Darken (`paint.dodge`/`paint.burn`),
  Saturate/Desaturate (`paint.sponge`), Blur, Sharpen, Smudge. Liquify Push/Bloat/Pinch/Restore
  (`filter.liquify` strokes); content-aware fill, denoise, a shortcut to Face shape.
* **Paint**: Brush, Mixer (`paint.mixerBrush`), Gradient (drag from → to, `paint.gradient`,
  five styles, to a colour or transparent), Bucket (`paint.bucket`); symmetry
  (`paint.setSymmetry`); "On layer mask" paints the active layer's mask (`target: "mask"`, white
  shows / black hides, a mask added first if none) — with Gradient, a graduated mask.
* **Transform**: flip/rotate the layer, Move & scale (`edit.transform`, live), Warp
  (`edit.transform.warp` shapes), Content-aware scale, Perspective (drag a quad's four corners:
  Crop to it = `image.perspectiveCrop`, Straighten = `edit.perspectiveWarp` to the quad's
  rectangle), Puppet warp (tap pins, drag them: `edit.puppetWarp`), align to the canvas
  (`layer.align.*`). A locked Background is made a layer first (`layer.new.layerFromBackground`).
* **History**: every step (past, current, greyed redo) from `doc.history` `states`; a tap returns
  there (`doc.historyGo` = `history.selectState`). Snapshots: `doc.snapshot`,
  `doc.restoreSnapshot` (a new step), `doc.deleteSnapshot`.
* **Before/after**: hold the compare button in the top bar to see `doc.original` (the oldest
  history state, rendered to a handle) in place of the edit.
* **Re-editing adjustments**: `doc.inspect` layers carry `adjustment {kind, params}`; Layers shows
  "Edit <kind>" for an adjustment layer, the same controls starting at its settings, applied
  with `layer.setAdjustment` (one "Modify Adjustment" step).
* **Export**: format (JPEG, PNG, WebP, TIFF), quality, longest side (`doc.export`), copied to
  Pictures/ImageLab.
* **Per-document state**: opening, creating or closing a document drops the looks' measurements,
  the preview proxy (session, frame, temp file) and the before/after render (`doc.*`); the app
  also clears its selection, SAM prompt, clone source, handles and paint layer, and calls
  `segment.forgetImages` (SAM embeddings), whenever the document's file or size changes.

#### Looks (`edit.look`) and face liquify (`edit.faceLiquify`)

Built only from PhotoCraft commands. A look becomes a plan: one `filter.cameraRaw` pass for the
whole photo, masked passes for areas PhotoCraft finds itself (Select Sky, `image.detectFaces`
ellipses, skin tones via `select.colorRange`, the subject's background), and for sky replacement
a gradient fill layer masked by the sky, blended *Color* so clouds keep their texture and the
soft sky edge shows no halo. The plan plays as a temporary action, so the look is one history
step ("Action: Enhance"); the action is deleted afterwards. Enhance measures the photo
(luminance percentiles, dark/bright share, saturation, from a ≤ 256 px render cached per
revision) and only corrects what it finds. Sky and face looks return `applied:false` with
`skipped` when there is no sky or face, and change nothing.

**Fast preview** (`edit.preview`): while a slider of a scale-free item moves (looks, face shape,
Camera Raw, relight, lens correction; no selection), the app runs the step on a ≤ 768 px flattened proxy of the
photo in a session of its own and shows that frame; the full-resolution edit runs once, when the
slider is released. The proxy is snapshotted when the tool opens (`begin`), so the tool's own
full-size run landing meanwhile is never previewed on top of itself. Each frame carries a
version and is shown only if newer than the one on screen. On the Mac, proxy previews of
Enhance take ~0.7 s (shadows/highlights are the costly sliders, ~180 ms each), Dramatic 0.35 s,
plain Camera Raw 30 ms, versus 1.6–3 s at 3000×4000.

Face shape sliders (eyes, nose, mouth, smile, face width, chin, forehead; −100..100) become
`filter.liquify` strokes sized from each face's width: bloat/pucker held at the eye, nose and
mouth landmarks, forward-warp drags for the smile corners, five rows along each cheek and jaw,
chin and hairline. One call is one Liquify step. Checked on a grid
(`render_face_liquify_grid`), and every look on a photo (`render_looks`, both `#[ignore]`d in
imagelab-edit; on a 3000×4000 photo a whole-photo look takes 0.15–1.6 s on the Mac).
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

## 12. ARGO companion

ImageLab is a companion of the ARGO app (ARGOv2, `apps/argo-android`): ARGO's own on-device agent
edits photos through ImageLab, from ARGO's chat, with ARGO's model, memory and approvals.

**The wire.** `dev.argo.imagelab.agent.LabService`, an exported bound service speaking
`ILabService` (`android/app/src/main/aidl`; ARGO keeps an identical copy, add methods only at the
end). Any app can bind an exported service, so the service checks the **caller's signing
certificate on every call** (`LabService.ALLOWED`: `com.argo.android` with ARGO's debug key; a
release ARGO adds its release key). ARGO pins ImageLab's certificate the same way before binding
(`LabLink.TRUSTED`), so neither side talks to a look-alike package.

**MCP over the wire.** ImageLab is an MCP server to ARGO (`agent/CompanionMcp.kt`): ARGO's core is
the MCP client (`companion_mcp` in `argo-android-ffi`), and each JSON-RPC message travels as
`call("mcp", message)`, its response on the pipe (nothing for a notification). ARGO mounts it as
the server `imagelab`, so the tools reach the model as `imagelab__<tool>`, deferred behind one
roster line (`select:imagelab`) like every MCP server's.

| Method | What |
|---|---|
| `initialize` | protocol `2025-06-18` (or the older one asked), `capabilities.tools`, `serverInfo`, `instructions` |
| `tools/list` | the five tools below, with `annotations` (all `openWorldHint: false`, `destructiveHint: false`) |
| `tools/call` | a `CallToolResult`: text and `image` blocks (JPEG unless transparent), `structuredContent`, `isError` |
| `ping` | `{}` |
| `resources/list`, `resources/templates/list`, `resources/read` | the resources below |
| `prompts/list`, `prompts/get` | the prompts below |
| `notifications/cancelled` | stop the running programs (the turn was stopped) |

| Tool | What |
|---|---|
| `open {photo}` | Open a gallery photo by MediaStore id (ImageLab reads it with its own photo permission) |
| `read {code, description, timeoutMs?}` | `lab_read`: a read-only program; ARGO declares it read-only |
| `edit {code, description, timeoutMs?}` | `lab_script`: an editing program, one undoable step each |
| `help {area?}` | `lab_help` |
| `save {format?, quality?, maxSide?}` | Export to the gallery (Pictures/ImageLab): `structuredContent {photo, uri, photos:[{id,name}]}`, which ARGO's card shows |

| Resource | What |
|---|---|
| `imagelab://document` | `doc.inspect` of the open document (JSON) |
| `imagelab://document/history` | `doc.history` (JSON) |
| `imagelab://document/preview` | the document now, a JPEG at most 1024 px a side (`blob`) |
| `imagelab://reference`, `imagelab://reference/{area}` | `lab_help`: every area, or one area or function (Markdown); the second is a resource template |

| Prompt | Arguments | What |
|---|---|---|
| `enhance` | `photo?`, `style?` | a natural overall enhancement, checked in a preview, then saved |
| `sky` | `photo?`, `look?` | only the sky, through `select.sky` with `edgeFill`, then saved |
| `caption` | `text`, `photo?`, `position?` | a readable caption, then saved |

In ARGO the resources are read through its MCP resource tools, and each prompt is a command
(`/imagelab-enhance`, …) where the surface has a command registry.

`call("status")` answers `{"ok": {contract, version, document}}`. `importPhoto(name, fd)` and
`cancel()` remain on the AIDL for a caller that holds a photo ImageLab cannot read itself.

Results travel through a **pipe** (`ParcelFileDescriptor`), never in the Binder reply: images ride
inline as base64 and outgrow a transaction. Images are encoded at most 1568 px a side (ARGO's
attachment store refuses more than 2000).

**In ARGO** (`com.argo.android.tools.lab`): `LabLink` binds the service, `LabMcp` carries the
messages, and the core mounts the server at boot and when ImageLab is installed while ARGO runs.
None of the tools asks first: every edit is an undoable step in ImageLab, the gallery original is
never changed, and saving makes a new photo. The server is mounted person-only (refused to other
devices and automations) with private output (the person's photos).

**ARGO's core client allowlist.** ImageLab and VideoLab are listed in ARGO's
`res/values/core_clients.xml` (debug: their shared debug key `647e1822…`) as client packages, as ARGO
Browser and VIA are, so they may bind ARGO's `:core` and open a client tab on ARGO's agent.
