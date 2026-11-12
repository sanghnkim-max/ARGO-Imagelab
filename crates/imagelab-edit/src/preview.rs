//! Fast previews on a proxy: while a slider moves, run the edit on a small copy of the photo
//! instead of the full-resolution document, so frames keep up.
//!
//! `edit.preview {begin: true, …}` snapshots the active document, flattened, into a ≤ `maxSide`
//! proxy (default 1024 px) opened in a session of its own. Later calls reuse that snapshot, so
//! a full-resolution run of the same tool landing in the document meanwhile is never previewed
//! on top of itself. Each call undoes the previous preview step on the proxy, runs the step
//! (`edit.look`, `edit.faceLiquify` or an allowlisted `edit.run` command), renders the proxy
//! into an `img:N` handle (the previous preview handle is freed) and returns it. The document
//! itself is never touched: apply the edit with the real function when the slider is released.
//!
//! The proxy is flattened, so a command meant for one layer previews on the composite, and
//! pixel-unit parameters (a blur radius) are not rescaled: use previews for scale-free edits
//! (looks, face liquify, `filter.cameraRaw`, relight).

use std::sync::Mutex;
use std::time::Instant;

use imagelab_api::{ApiError, ApiResult, Ctx, HandleId, Headless, ImageBuf};
use photocraft_automation::AuthorizedWorkspace;
use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::ops;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct PreviewParams {
    /// Snapshot the active document as the proxy (start of a tool).
    #[serde(default)]
    pub begin: bool,
    #[serde(default)]
    pub max_side: Option<u32>,
    /// `{fn: "edit.look" | "edit.faceLiquify" | "edit.run", params}`; absent with `begin`
    /// renders the proxy as it is.
    #[serde(default)]
    pub step: Option<PreviewStep>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewStep {
    #[serde(rename = "fn")]
    pub func: String,
    #[serde(default)]
    pub params: Value,
}

struct Proxy {
    session: Headless,
    /// History length of the proxy document as opened; previews are undone back to it.
    base: usize,
    /// The handle of the last preview, freed when the next one lands.
    last: Option<HandleId>,
}

static PROXY: Mutex<Option<Proxy>> = Mutex::new(None);

const DEFAULT_SIDE: u32 = 1024;

fn history_len(h: &Headless) -> usize {
    h.session.active().map_or(0, |d| d.history.entries().len())
}

/// The active document flattened to ≤ `side`, opened in a fresh session.
fn snapshot(ctx: &Ctx, side: u32) -> ApiResult<Proxy> {
    let img = ctx.with_session(|h| {
        let d = h.session.active().ok_or_else(|| ApiError::not_found("no open document; doc.open a photo first"))?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| photocraft_compose::thumbnail(&d.doc, side))).map_err(|_| ApiError::internal("rendering the proxy panicked"))
    })?;
    let image = Image::from_u8(img.width, img.height, ChannelLayout::Rgba, img.pixels).map_err(|e| ApiError::internal(format!("proxy image: {e}")))?;
    let png = photocraft_codecs::encode(&image, Format::Png, &EncodeOptions::default()).map_err(|e| ApiError::internal(format!("proxy encode: {e}")))?;
    let root = &ctx.shared.root;
    let rel = ".imagelab/tmp/preview-proxy.png";
    let path = root.join(rel);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| ApiError::internal(format!("proxy dir: {e}")))?;
    }
    std::fs::write(&path, png).map_err(|e| ApiError::internal(format!("proxy write: {e}")))?;
    let ws = AuthorizedWorkspace::new(Some(root), Some(root))?;
    let mut session = Headless::with_workspace(ws);
    session.open(&path)?;
    let base = history_len(&session);
    Ok(Proxy { session, base, last: None })
}

fn run_step(h: &mut Headless, step: &PreviewStep) -> ApiResult<Value> {
    match step.func.as_str() {
        "edit.look" => {
            let p: crate::looks::LookParams = serde_json::from_value(step.params.clone()).map_err(|e| ApiError::bad_params(format!("edit.look params: {e}")))?;
            crate::looks::apply(h, &p)
        }
        "edit.faceLiquify" => {
            let p: crate::face::FaceParams = serde_json::from_value(step.params.clone()).map_err(|e| ApiError::bad_params(format!("edit.faceLiquify params: {e}")))?;
            crate::face::apply(h, &p)
        }
        "edit.run" => {
            let id = step.params.get("id").and_then(Value::as_str).ok_or_else(|| ApiError::bad_params("edit.run needs `id`"))?;
            ops::run_allowed(h, id, step.params.get("params").cloned().unwrap_or(Value::Null))
        }
        other => Err(ApiError::bad_params(format!("edit.preview runs edit.look, edit.faceLiquify or edit.run, not `{other}`"))),
    }
}

/// `edit.preview`.
pub fn preview(ctx: &Ctx, p: PreviewParams) -> ApiResult<Value> {
    let t = Instant::now();
    let side = p.max_side.unwrap_or(DEFAULT_SIDE).clamp(256, 2048);
    let mut guard = PROXY.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if p.begin || guard.is_none() {
        if let Some(old) = guard.take().and_then(|o| o.last) {
            ctx.shared.handles.free(old);
        }
        *guard = Some(snapshot(ctx, side)?);
    }
    let proxy = guard.as_mut().ok_or_else(|| ApiError::internal("no proxy"))?;
    let h = &mut proxy.session;
    // Back to the snapshot: the previous preview step is undone.
    let extra = history_len(h).saturating_sub(proxy.base);
    if extra > 0 {
        ops::undo_n(h, extra)?;
    }
    let result = match &p.step {
        Some(step) => Some(run_step(h, step)?),
        None => None,
    };
    let d = h.session.active().ok_or_else(|| ApiError::internal("the proxy document is gone"))?;
    let img = photocraft_compose::thumbnail(&d.doc, side);
    let (w, hh) = (img.width, img.height);
    let id = ctx.shared.handles.put_image(ImageBuf::new(w, hh, img.pixels)?)?;
    if let Some(old) = proxy.last.replace(id) {
        ctx.shared.handles.free(old);
    }
    Ok(json!({"image": id.to_string(), "width": w, "height": hh, "ms": t.elapsed().as_millis() as u64, "result": result}))
}
