//! Area `doc`: the UI's PhotoCraft session — open, inspect, preview, save, export, history.

use std::path::Path;

use imagelab_api::{ApiError, ApiResult, Ctx, Effects, Event, Headless, ImageBuf, Registry};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::ops::{self, ExportSpec, MAX_PREVIEW_SIDE, MAX_SIDE};
use crate::paths;

/// Run `f` on the UI session, then emit `DocChanged` for the active document (outside the lock,
/// so a listener may call back in).
pub(crate) fn mutate<R>(ctx: &Ctx, f: impl FnOnce(&mut Headless) -> ApiResult<R>) -> ApiResult<R> {
    let mut changed = None;
    let result = ctx.with_session(|h| {
        let before = ops::active_revision(h);
        let r = f(h);
        let after = ops::active_revision(h);
        if after != before || r.is_ok() {
            changed = after;
        }
        r
    });
    if let Some((doc, revision)) = changed {
        ctx.emit(Event::DocChanged { doc, revision, source: ctx.source });
    }
    result
}

/// Make sure the parent folder of a workspace-relative `path` exists.
pub(crate) fn ensure_parent(root: &Path, path: &str) -> ApiResult<()> {
    if let Some((dir, _)) = path.rsplit_once('/') {
        std::fs::create_dir_all(root.join(dir)).map_err(|e| ApiError::internal(format!("create {dir}/: {e}")))?;
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenParams {
    #[serde(default)]
    photo: Option<String>,
    #[serde(default)]
    path: Option<String>,
    /// Open the original even when a sidecar exists.
    #[serde(default)]
    original: bool,
}

/// Which file to open for `photo`: its sidecar when one exists (unless `original`).
pub(crate) fn resolve_open(root: &Path, photo: &str, original: bool) -> ApiResult<(String, bool)> {
    let photo = paths::relative(photo)?;
    let sidecar = paths::sidecar_for(&photo);
    if !original && sidecar != photo && root.join(&sidecar).is_file() {
        return Ok((sidecar, true));
    }
    if !root.join(&photo).is_file() {
        return Err(ApiError::not_found(format!("no file `{photo}` in the workspace")));
    }
    Ok((photo, false))
}

fn open(ctx: &Ctx, p: OpenParams) -> ApiResult<Value> {
    forget_document_caches(ctx);
    let photo = p.photo.or(p.path).ok_or_else(|| ApiError::bad_params("pass `photo` (a library id) or `path`"))?;
    let (file, from_sidecar) = resolve_open(&ctx.shared.root, &photo, p.original)?;
    let photo = paths::relative(&photo)?;
    mutate(ctx, |h| {
        let opened = h.open(Path::new(&file))?;
        let (index, revision) = ops::active_revision(h).unwrap_or((0, 0));
        Ok(json!({
            "photo": photo,
            "opened": file,
            "fromSidecar": from_sidecar,
            "sidecar": paths::sidecar_for(&photo),
            "index": index,
            "revision": revision,
            "info": opened,
        }))
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NewParams {
    width: u32,
    height: u32,
    #[serde(default)]
    background: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

fn new_doc(ctx: &Ctx, p: NewParams) -> ApiResult<Value> {
    forget_document_caches(ctx);
    if p.width == 0 || p.height == 0 || p.width > MAX_SIDE || p.height > MAX_SIDE {
        return Err(ApiError::bad_params(format!("width and height must be 1..{MAX_SIDE}")));
    }
    let mut params = json!({"width": p.width, "height": p.height});
    if let Some(b) = p.background {
        params["background"] = json!(b);
    }
    if let Some(n) = p.name {
        params["name"] = json!(n.chars().take(128).collect::<String>());
    }
    mutate(ctx, |h| {
        h.command_run("file.new", params)?;
        Ok(h.session_list())
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexParams {
    index: usize,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct InspectParams {
    /// Most layers listed (default 100, at most 500).
    #[serde(default)]
    max_layers: Option<usize>,
    /// Most history entries listed, newest last (default 30, at most 200).
    #[serde(default)]
    max_history: Option<usize>,
}

/// Keep the parts of a layer an agent needs, bounded in count and depth.
fn trim_layers(layers: &[Value], budget: &mut usize, depth: usize) -> Vec<Value> {
    let mut out = Vec::new();
    for l in layers {
        if *budget == 0 {
            out.push(json!({"truncated": true}));
            break;
        }
        *budget -= 1;
        let mut t = serde_json::Map::new();
        // `adjustment` = {kind, params} as the adjustment commands take them (re-editing).
        for k in ["id", "name", "kind", "visible", "opacity", "blend", "hasMask", "bounds", "clipped", "selected", "adjustment"] {
            if let Some(v) = l.get(k) {
                t.insert(k.into(), v.clone());
            }
        }
        if let Some(children) = l.get("children").and_then(Value::as_array) {
            if depth < 6 {
                t.insert("children".into(), Value::Array(trim_layers(children, budget, depth + 1)));
            } else {
                t.insert("children".into(), json!(format!("{} nested", children.len())));
            }
        }
        out.push(Value::Object(t));
    }
    out
}

pub(crate) fn trimmed_inspect(h: &Headless, max_layers: usize, max_history: usize) -> ApiResult<Value> {
    let full = h.inspect(None)?;
    let mut out = serde_json::Map::new();
    for k in ["name", "path", "width", "height", "mode", "depth", "resolution", "activeLayer", "selectedLayers", "hasSelection", "selectionBounds", "canUndo", "canRedo", "revision", "quickMask"] {
        if let Some(v) = full.get(k) {
            out.insert(k.into(), v.clone());
        }
    }
    let mut budget = max_layers;
    let layers = full.get("layers").and_then(Value::as_array).map(|l| trim_layers(l, &mut budget, 0)).unwrap_or_default();
    out.insert("layers".into(), Value::Array(layers));
    let history: Vec<Value> = full.get("history").and_then(Value::as_array).cloned().unwrap_or_default();
    let skip = history.len().saturating_sub(max_history);
    out.insert("historyCount".into(), json!(history.len()));
    out.insert("history".into(), Value::Array(history.into_iter().skip(skip).collect()));
    out.insert("index".into(), json!(h.session.active_index()));
    Ok(Value::Object(out))
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PreviewParams {
    #[serde(default)]
    max_side: Option<u32>,
}

/// Render the active document to PNG, dropping a failing GPU renderer for the CPU compositor
/// (as imagelab-core does).
pub(crate) fn render_png(h: &mut Headless, max_side: u32) -> ApiResult<Vec<u8>> {
    let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.render_png(None, max_side)));
    match first {
        Ok(Ok(png)) => return Ok(png),
        Ok(Err(e)) if h.renderer_mut().is_none() => return Err(e.into()),
        Err(_) if h.renderer_mut().is_none() => return Err(ApiError::internal("rendering panicked")),
        _ => {}
    }
    h.set_renderer(None);
    Ok(h.render_png(None, max_side)?)
}

fn preview(ctx: &Ctx, p: PreviewParams) -> ApiResult<Value> {
    let max_side = p.max_side.unwrap_or(1024);
    if max_side == 0 || max_side > MAX_PREVIEW_SIDE {
        return Err(ApiError::bad_params(format!("maxSide must be 1..{MAX_PREVIEW_SIDE}")));
    }
    let png = ctx.with_session(|h| render_png(h, max_side))?;
    let image = photocraft_codecs::decode(&png).map_err(|e| ApiError::internal(format!("decode preview: {e}")))?;
    let (width, height) = image.dimensions();
    let buf = ImageBuf::new(width, height, image.to_rgba8())?;
    let handle = ctx.shared.handles.put_image(buf)?;
    Ok(json!({"image": handle.to_string(), "width": width, "height": height}))
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SaveParams {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    format: Option<String>,
}

/// Where a plain `doc.save` writes: the document's sidecar (`edits/….pcraft`).
pub(crate) fn default_save_target(doc_path: Option<&str>) -> ApiResult<String> {
    let p = doc_path.ok_or_else(|| ApiError::bad_params("this document has no file (made with doc.new); pass `path`, e.g. edits/name.pcraft"))?;
    // A document opened from inside the workspace has a relative path; anything else is unusual.
    let rel = paths::relative(p).map_err(|_| ApiError::bad_params(format!("the document's path `{p}` is not workspace-relative; pass `path`")))?;
    Ok(paths::sidecar_for(&rel))
}

fn save(ctx: &Ctx, p: SaveParams) -> ApiResult<Value> {
    let root = ctx.shared.root.clone();
    mutate(ctx, |h| {
        let doc_path = h.session.active().and_then(|d| d.path.clone());
        let target = match &p.path {
            Some(path) => paths::writable(path)?,
            None => default_save_target(doc_path.as_deref())?,
        };
        let layered = p.format.as_deref().map(|f| f.trim_start_matches('.').eq_ignore_ascii_case("pcraft")).unwrap_or_else(|| target.ends_with(".pcraft"));
        // Anything but the document's own file (or its own edits sidecar) is a new file: never
        // over an existing one, layered or flat.
        let own = doc_path.as_deref() == Some(target.as_str())
            || (p.path.is_none() && default_save_target(doc_path.as_deref()).ok().as_deref() == Some(target.as_str()));
        if !own && root.join(&target).exists() {
            return Err(ApiError::new("exists", format!("`{target}` exists; pick another name (doc.exportOverwrite replaces a file, and needs the person's OK)")));
        }
        ensure_parent(&root, &target)?;
        let spec = ExportSpec { format: p.format.clone(), ..ExportSpec::default() };
        let saved = ops::save_as(h, &target, &spec)?;
        Ok(json!({"path": target, "layered": layered, "result": saved}))
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ExportParams {
    path: String,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    max_side: Option<u32>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    aspect: Option<String>,
    #[serde(default)]
    quality: Option<u8>,
}

/// Export the active document of the UI session. Resizing or cropping happens on a duplicate,
/// so the open document is unchanged.
fn export(ctx: &Ctx, p: ExportParams, overwrite: bool) -> ApiResult<Value> {
    let root = ctx.shared.root.clone();
    let target = paths::writable(&p.path)?;
    if !overwrite && root.join(&target).exists() {
        return Err(ApiError::new("exists", format!("`{target}` exists; pick another path, or doc.exportOverwrite to replace it (Destructive)")));
    }
    let spec = ExportSpec { format: p.format, max_side: p.max_side, width: p.width, aspect: p.aspect, quality: p.quality };
    ensure_parent(&root, &target)?;
    ctx.with_session(|h| {
        let (w, hgt) = h.session.active().map(|d| (d.doc.size.width, d.doc.size.height)).ok_or_else(|| ApiError::not_found("no open document"))?;
        let reshape = spec.aspect.is_some() || ops::target_size(w, hgt, &spec)?.is_some();
        if !reshape {
            let r = ops::save_as(h, &target, &spec)?;
            return Ok(json!({"path": target, "width": w, "height": hgt, "result": r}));
        }
        let original = h.session.active_index().unwrap_or(0);
        h.command_run("image.duplicate", json!({"name": "imagelab-export"}))?;
        let dup = h.session.active_index();
        let result = (|| {
            if dup == Some(original) {
                return Err(ApiError::internal("image.duplicate did not make a new document"));
            }
            ops::shape_for_export(h, &spec)?;
            let size = h.session.active().map(|d| (d.doc.size.width, d.doc.size.height)).unwrap_or((0, 0));
            let r = ops::save_as(h, &target, &spec)?;
            Ok(json!({"path": target, "width": size.0, "height": size.1, "result": r}))
        })();
        if dup.is_some() && dup != Some(original) {
            let _ = h.close(dup);
        }
        let _ = h.select(original);
        result
    })
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct CloseParams {
    #[serde(default)]
    index: Option<usize>,
    /// Close even with unsaved changes.
    #[serde(default)]
    discard: bool,
}

fn close(ctx: &Ctx, p: CloseParams) -> ApiResult<Value> {
    forget_document_caches(ctx);
    mutate(ctx, |h| {
        let i = p.index.or_else(|| h.session.active_index()).ok_or_else(|| ApiError::not_found("no open document"))?;
        let dirty = h.session.documents().get(i).map(|d| d.is_dirty()).ok_or_else(|| ApiError::not_found(format!("no document at index {i}")))?;
        if dirty && !p.discard {
            return Err(ApiError::new("unsaved", "the document has unsaved changes: doc.save first, or pass discard: true"));
        }
        Ok(h.close(Some(i))?)
    })
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct UndoParams {
    #[serde(default)]
    steps: Option<usize>,
}

fn history(h: &Headless) -> ApiResult<Value> {
    let d = h.session.active().ok_or_else(|| ApiError::not_found("no open document"))?;
    let entries = d.history.entries();
    let skip = entries.len().saturating_sub(100);
    // `states`: every History panel row (past, current, then redo), each with the `state`
    // index `doc.historyGo` takes; `current` is the current row's index.
    let current = d.history.past_len();
    let mut states: Vec<Value> = entries.iter().enumerate().map(|(i, l)| json!({"state": i, "label": l, "redo": false})).collect();
    states.extend(d.history.redo_labels().enumerate().map(|(i, l)| json!({"state": current + 1 + i, "label": l, "redo": true})));
    let skip_states = states.len().saturating_sub(200);
    Ok(json!({
        "count": entries.len(),
        "entries": entries.into_iter().skip(skip).collect::<Vec<_>>(),
        "states": states.into_iter().skip(skip_states).collect::<Vec<_>>(),
        "current": current,
        "snapshots": d.snapshots.iter().map(|s| json!({"id": s.id, "name": s.name})).collect::<Vec<_>>(),
        "canUndo": d.history.can_undo(),
        "canRedo": d.history.can_redo(),
        "revision": d.revision,
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GoParams {
    state: usize,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SnapshotParams {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    snapshot: Option<Value>,
}

/// A History command the `edit.run` allowlist keeps out (panel commands), run for one purpose.
fn history_cmd(ctx: &Ctx, id: &str, params: Value) -> ApiResult<Value> {
    mutate(ctx, |h| {
        let r = h.command_run(id, params)?;
        let mut v = history(h)?;
        v["result"] = r;
        Ok(v)
    })
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct OriginalParams {
    #[serde(default)]
    max_side: Option<u32>,
}

/// The document as it was at its oldest history state (the photo as opened, unless the
/// history has dropped its oldest states), rendered into an image handle: before/after.
/// Per-document caches dropped when a document is opened, created or closed: the looks'
/// measurements, the preview proxy (session, frame, temp file) and the before/after render.
/// (The app also calls `segment.forgetImages` for SAM's embeddings.)
fn forget_document_caches(ctx: &Ctx) {
    crate::looks::forget_caches();
    crate::preview::forget(ctx);
    if let Some(old) = LAST_ORIGINAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
        ctx.shared.handles.free(old);
    }
}

/// The last `doc.original` handle, freed by the next call (one at a time, never a pile-up).
static LAST_ORIGINAL: std::sync::Mutex<Option<imagelab_api::HandleId>> = std::sync::Mutex::new(None);

fn original(ctx: &Ctx, p: OriginalParams) -> ApiResult<Value> {
    let side = p.max_side.unwrap_or(1024).clamp(64, MAX_PREVIEW_SIDE);
    let img = ctx.with_session(|h| {
        let d = h.session.active().ok_or_else(|| ApiError::not_found("no open document"))?;
        // The automatic snapshot taken at open (id 0) is the photo as opened even after the
        // history drops old states or is cleared; else the oldest state still held.
        let snap = d.snapshots.iter().find(|s| s.id == 0).map(|s| s.doc.clone());
        let doc = snap.or_else(|| d.history.state(0));
        let doc = doc.as_deref().unwrap_or(&d.doc);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| photocraft_compose::thumbnail(doc, side))).map_err(|_| ApiError::internal("rendering the original panicked"))
    })?;
    let (width, height) = (img.width, img.height);
    let handle = ctx.shared.handles.put_image(ImageBuf::new(width, height, img.pixels)?)?;
    if let Some(old) = LAST_ORIGINAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner).replace(handle) {
        ctx.shared.handles.free(old);
    }
    Ok(json!({"image": handle.to_string(), "width": width, "height": height}))
}

fn step_history(ctx: &Ctx, p: UndoParams, id: &str) -> ApiResult<Value> {
    let n = p.steps.unwrap_or(1).clamp(1, 100);
    mutate(ctx, |h| {
        let mut done = 0;
        for _ in 0..n {
            let possible = h.session.active().is_some_and(|d| if id == "edit.undo" { d.history.can_undo() } else { d.history.can_redo() });
            if !possible {
                break;
            }
            h.command_run(id, json!({}))?;
            done += 1;
        }
        let mut v = history(h)?;
        v["stepped"] = json!(done);
        Ok(v)
    })
}

const EXPORT_PARAMS: &str = r#"{"type":"object","required":["path"],"properties":{"path":{"type":"string","description":"workspace-relative target, e.g. exports/IMG_0012.jpg (not under library/)"},"format":{"type":"string","enum":["jpg","png","webp","tif","psd","pcraft"]},"maxSide":{"type":"integer","minimum":1,"maximum":16384,"description":"longest side in px; never upscales"},"width":{"type":"integer","minimum":1,"maximum":16384},"aspect":{"type":"string","description":"crop to W:H around the centre first, e.g. 4:5"},"quality":{"type":"integer","minimum":1,"maximum":100}}}"#;

fn schema(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| json!({"type": "object"}))
}

pub fn register(r: &mut Registry) {
    r.function("doc", "open", "Open a photo (its edit sidecar if one exists) as the active document")
        .description(
            "Opens `photo` (a library id such as library/2026-11/IMG_0012.jpg) or any workspace-relative `path`. If the photo has an edit sidecar (edits/<id with / as __>.pcraft) that is opened instead, so earlier edits continue; `original: true` opens the original. Originals are never modified: doc.save writes the sidecar. Returns {photo, opened, fromSidecar, sidecar, index, revision, info}.",
        )
        .params(json!({"type":"object","properties":{"photo":{"type":"string"},"path":{"type":"string"},"original":{"type":"boolean","default":false}}}))
        .effects(Effects::Revertible)
        .typed(open)
        .register();
    r.function("doc", "new", "New blank document")
        .params(json!({"type":"object","required":["width","height"],"properties":{"width":{"type":"integer","minimum":1,"maximum":MAX_SIDE},"height":{"type":"integer","minimum":1,"maximum":MAX_SIDE},"background":{"type":"string","description":"white|black|transparent|#rrggbb"},"name":{"type":"string"}}}))
        .effects(Effects::Revertible)
        .typed(new_doc)
        .register();
    r.function("doc", "list", "Open documents and the active index")
        .description("Returns {active, documents:[{index, name, path, width, height, dirty, revision}]}.")
        .typed(|ctx: &Ctx, _: Value| ctx.with_session(|h| Ok(h.session_list())))
        .register();
    r.function("doc", "select", "Make the document at `index` active")
        .params(json!({"type":"object","required":["index"],"properties":{"index":{"type":"integer","minimum":0}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: IndexParams| mutate(ctx, |h| Ok(h.select(p.index)?)))
        .register();
    r.function("doc", "inspect", "The active document: size, layers, selection, recent history (trimmed)")
        .description("Layers top to bottom with id, name, kind, visible, opacity, blend, hasMask, bounds and children; `hasSelection` and `selectionBounds` [x,y,w,h]; the last `max_history` history entries. Bounded by `max_layers` (default 100).")
        .params(json!({"type":"object","properties":{"max_layers":{"type":"integer","minimum":1,"maximum":500},"max_history":{"type":"integer","minimum":0,"maximum":200}}}))
        .typed(|ctx: &Ctx, p: InspectParams| ctx.with_session(|h| trimmed_inspect(h, p.max_layers.unwrap_or(100).clamp(1, 500), p.max_history.unwrap_or(30).min(200))))
        .register();
    r.function("doc", "preview", "Render the active document to an image handle {image: \"img:N\", width, height}")
        .description("Flattened render whose longest side is at most `maxSide` (default 1024, at most 2048). Return {image} from a program and the model sees it. Free the handle when done if you keep calling.")
        .params(json!({"type":"object","properties":{"maxSide":{"type":"integer","minimum":1,"maximum":MAX_PREVIEW_SIDE,"default":1024}}}))
        .typed(preview)
        .register();
    r.function("doc", "save", "Save the active document (default: its .pcraft edit sidecar)")
        .description("Without `path`, writes the photo's sidecar edits/<id>.pcraft (layered, lossless; the original stays untouched). A `path` ending in .pcraft saves there; another extension (.jpg, .png, .webp, .tif, .psd) exports a copy and refuses to overwrite an existing file. Paths under library/ are refused.")
        .params(json!({"type":"object","properties":{"path":{"type":"string"},"format":{"type":"string"}}}))
        .effects(Effects::Revertible)
        .typed(save)
        .register();
    r.function("doc", "export", "Export the active document flattened, optionally resized/cropped (never overwrites)")
        .description("Writes `path` (e.g. exports/IMG_0012.jpg). `maxSide`/`width` downscale and `aspect` crops — on a duplicate, so the open document is unchanged. Refuses an existing file (code `exists`); doc.exportOverwrite replaces one.")
        .params(schema(EXPORT_PARAMS))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: ExportParams| export(ctx, p, false))
        .register();
    r.function("doc", "exportOverwrite", "doc.export that may replace an existing file")
        .params(schema(EXPORT_PARAMS))
        .effects(Effects::Destructive)
        .typed(|ctx: &Ctx, p: ExportParams| export(ctx, p, true))
        .register();
    r.function("doc", "close", "Close the active document (or `index`); refuses unsaved changes unless discard: true")
        .params(json!({"type":"object","properties":{"index":{"type":"integer","minimum":0},"discard":{"type":"boolean","default":false}}}))
        .effects(Effects::Revertible)
        .typed(close)
        .register();
    r.function("doc", "undo", "Undo `steps` (default 1) history steps")
        .params(json!({"type":"object","properties":{"steps":{"type":"integer","minimum":1,"maximum":100}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: UndoParams| step_history(ctx, p, "edit.undo"))
        .register();
    r.function("doc", "redo", "Redo `steps` (default 1) history steps")
        .params(json!({"type":"object","properties":{"steps":{"type":"integer","minimum":1,"maximum":100}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: UndoParams| step_history(ctx, p, "edit.redo"))
        .register();
    r.function("doc", "history", "History of the active document {count, entries (last 100), states (past, current, redo; each {state, label, redo}), current, snapshots [{id, name}], canUndo, canRedo, revision}")
        .typed(|ctx: &Ctx, _: Value| ctx.with_session(|h| history(h)))
        .register();
    r.function("doc", "historyGo", "Return the document to History row `state` (an index from doc.history `states`); later rows stay redoable")
        .params(json!({"type":"object","required":["state"],"properties":{"state":{"type":"integer","minimum":0}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: GoParams| history_cmd(ctx, "history.selectState", json!({"state": p.state})))
        .register();
    r.function("doc", "snapshot", "Keep a named snapshot of the document for this session (not a history step)")
        .params(json!({"type":"object","properties":{"name":{"type":"string"}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: SnapshotParams| {
            let mut params = json!({});
            if let Some(n) = p.name.filter(|n| !n.trim().is_empty()) {
                params["name"] = json!(n.chars().take(64).collect::<String>());
            }
            history_cmd(ctx, "history.newSnapshot", params)
        })
        .register();
    r.function("doc", "restoreSnapshot", "Return the document to snapshot `snapshot` (an id from doc.history), as a new history step")
        .params(json!({"type":"object","required":["snapshot"],"properties":{"snapshot":{}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: SnapshotParams| history_cmd(ctx, "history.selectSnapshot", json!({"snapshot": p.snapshot.unwrap_or(Value::Null)})))
        .register();
    r.function("doc", "deleteSnapshot", "Delete snapshot `snapshot` (the document is unchanged; it cannot be brought back)")
        .params(json!({"type":"object","required":["snapshot"],"properties":{"snapshot":{}}}))
        .effects(Effects::Destructive)
        .typed(|ctx: &Ctx, p: SnapshotParams| history_cmd(ctx, "history.deleteSnapshot", json!({"snapshot": p.snapshot.unwrap_or(Value::Null)})))
        .register();
    r.function("doc", "original", "Render the document as first opened (its oldest history state) to an image handle {image, width, height}: before/after")
        .params(json!({"type":"object","properties":{"maxSide":{"type":"integer","minimum":64,"maximum":MAX_PREVIEW_SIDE,"default":1024}}}))
        .typed(original)
        .register();
}
