//! The `segment` area: SAM 3 masks from points, a box, a text concept or the subject, with
//! PhotoCraft's selections as the fallback when the model is missing or fails (the result says
//! which: `"engine": "sam3" | "photocraft"`, and `samError` why SAM was not used).
//!
//! Every `segment.*` mask call is read-only: SAM only reads pixels, and the PhotoCraft fallback
//! runs its selection command and rolls the step back ([`crate::select::probe`]), so the
//! document's selection and history are untouched. Use `select.fromMask` to apply a result.

use std::sync::Arc;

use imagelab_api::{ApiError, ApiResult, Ctx, Effects, MaskBuf, Registry};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::input::{self, Input, Source};
use crate::mask::{self, Mode};
use crate::select;
use crate::{Mobile, Sam, State};

/// Most points one call takes (SAM 3's interactive decoder limit).
pub const MAX_POINTS: usize = 64;
/// Most instances `segment.text` returns.
const MAX_INSTANCES: usize = 32;

/// One click, normalized to the image (0..1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Click {
    pub x: f32,
    pub y: f32,
    pub positive: bool,
}

/// One prediction at the model's mask resolution (`side²` logits over the stretched image).
pub struct Pred {
    pub logits: Vec<f32>,
    pub side: usize,
    pub score: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PointParams {
    #[serde(flatten)]
    src: Source,
    points: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoxParams {
    #[serde(flatten)]
    src: Source,
    #[serde(rename = "box")]
    rect: [f64; 4],
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TextParams {
    #[serde(flatten)]
    src: Source,
    prompt: String,
    threshold: Option<f32>,
    max_instances: Option<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubjectParams {
    #[serde(flatten)]
    src: Source,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EnsureParams {
    #[serde(default)]
    confirm: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InstallParams {
    path: String,
    /// `sam3` (a folder, default) or `mobilesam` (the weights file).
    #[serde(default)]
    model: Option<String>,
}

/// A full-size mask into the arena; the JSON every single-mask result shares.
fn put(ctx: &Ctx, w: u32, h: u32, m: Vec<u8>, score: Option<f32>) -> ApiResult<Value> {
    let bbox = mask::bbox(&m, w as usize, h as usize);
    let coverage = mask::coverage(&m);
    let id = ctx.shared.handles.put_mask(MaskBuf::new(w, h, m)?)?;
    Ok(json!({"mask": id.to_string(), "width": w, "height": h, "bbox": bbox, "coverage": coverage, "score": score}))
}

fn sam_mask(input: &Input, p: &Pred) -> Vec<u8> {
    mask::logits_to_mask(&p.logits, p.side, input.width as usize, input.height as usize)
}

fn check_point(x: f64, y: f64, w: u32, h: u32) -> ApiResult<()> {
    if !(x.is_finite() && y.is_finite() && x >= 0.0 && y >= 0.0 && x <= w as f64 && y <= h as f64) {
        return Err(ApiError::bad_params(format!("point ({x}, {y}) is outside the {w}x{h} image")));
    }
    Ok(())
}

/// Fallback over PhotoCraft (the open document only): run each selection command as a probe
/// and combine the masks.
fn photocraft(ctx: &Ctx, steps: &[(&str, Value, Mode)]) -> ApiResult<(u32, u32, Vec<u8>)> {
    ctx.with_session(|h| {
        let mut acc: Option<Vec<u8>> = None;
        let (mut w, mut ht) = (0, 0);
        for (id, params, mode) in steps {
            let (pw, ph, sel) = select::probe(h, id, params.clone())?;
            (w, ht) = (pw, ph);
            let sel = sel.unwrap_or_else(|| vec![0; pw as usize * ph as usize]);
            acc = Some(match acc {
                None if *mode == Mode::Subtract => vec![0; sel.len()],
                None => sel,
                Some(a) => mask::combine(Some(&a), &sel, *mode),
            });
        }
        let acc = acc.ok_or_else(|| ApiError::bad_params("nothing to select"))?;
        Ok((w, ht, acc))
    })
}

fn fallback_result(ctx: &Ctx, steps: &[(&str, Value, Mode)], sam_error: Option<String>) -> ApiResult<Value> {
    let (w, h, m) = photocraft(ctx, steps)?;
    let mut v = put(ctx, w, h, m, None)?;
    v["engine"] = json!("photocraft");
    v["samError"] = json!(sam_error);
    Ok(v)
}

/// Try MobileSAM; on success the mask result (`engine: "mobilesam"`), on failure why.
fn mobile_try(ctx: &Ctx, mobile: &Mobile, input: &Input, clicks: &[Click], rect: Option<[f32; 4]>, clip: Option<[f64; 4]>) -> Result<Value, String> {
    let (pred, cached) = mobile.segment(&ctx.shared.root, input, clicks, rect)?;
    let mut m = sam_mask(input, &pred);
    if let Some([x, y, w, h]) = clip {
        let pad = (f64::from(input.width.max(input.height)) * 0.01) as f32;
        mask::clip_to_box(&mut m, input.width as usize, input.height as usize, [x as f32, y as f32, w as f32, h as f32], pad);
    }
    let mut v = put(ctx, input.width, input.height, m, Some(pred.score)).map_err(|e| e.message)?;
    v["engine"] = json!("mobilesam");
    v["cached"] = json!(cached);
    Ok(v)
}

/// Both model errors in one line, for `samError`.
fn both(sam: String, mobile: String) -> String {
    format!("SAM 3: {sam}; MobileSAM: {mobile}")
}

/// SAM is unusable for an image/photo input: say so (PhotoCraft works on the document only).
fn no_fallback(kind: &str, err: String) -> ApiError {
    ApiError::unavailable(format!("SAM 3 could not segment this {kind} ({err}); the PhotoCraft fallback works on the open document only (omit image/photo)"))
}

fn point(ctx: &Ctx, st: &State, p: PointParams) -> ApiResult<Value> {
    let sam = &st.sam;
    if p.points.is_empty() || p.points.len() > MAX_POINTS {
        return Err(ApiError::bad_params(format!("points: 1..={MAX_POINTS} [x, y, label] triples")));
    }
    let input = input::resolve(ctx, &p.src)?;
    for [x, y, l] in &p.points {
        check_point(*x, *y, input.width, input.height)?;
        if *l != 0.0 && *l != 1.0 {
            return Err(ApiError::bad_params("label is 1 (include) or 0 (exclude)"));
        }
    }
    if !p.points.iter().any(|q| q[2] == 1.0) {
        return Err(ApiError::bad_params("at least one point must have label 1"));
    }
    ctx.check_cancel()?;
    let clicks: Vec<Click> =
        p.points.iter().map(|[x, y, l]| Click { x: (*x / input.width as f64) as f32, y: (*y / input.height as f64) as f32, positive: *l == 1.0 }).collect();
    let err = match sam.clicks(&ctx.shared.root, &input, &clicks) {
        Ok((pred, cached)) => {
            let mut v = put(ctx, input.width, input.height, sam_mask(&input, &pred), Some(pred.score))?;
            v["engine"] = json!("sam3");
            v["cached"] = json!(cached);
            return Ok(v);
        }
        Err(e) => e,
    };
    let err = match mobile_try(ctx, &st.mobile, &input, &clicks, None, None) {
        Ok(mut v) => {
            v["samError"] = json!(err);
            return Ok(v);
        }
        Err(m) => both(err, m),
    };
    if !p.src.is_document() {
        return Err(no_fallback(input.kind, err));
    }
    // Object Finder under each point: includes add, excludes subtract.
    let mut steps: Vec<(&str, Value, Mode)> = Vec::new();
    for [x, y, l] in p.points.iter().filter(|q| q[2] == 1.0).chain(p.points.iter().filter(|q| q[2] == 0.0)) {
        let mode = if *l == 1.0 { Mode::Add } else { Mode::Subtract };
        steps.push(("select.object", json!({"point": [x, y], "mode": "replace"}), mode));
    }
    fallback_result(ctx, &steps, Some(err))
}

fn boxed(ctx: &Ctx, st: &State, p: BoxParams) -> ApiResult<Value> {
    let sam = &st.sam;
    let [x, y, w, h] = p.rect;
    if !(w.is_finite() && h.is_finite() && w >= 1.0 && h >= 1.0) {
        return Err(ApiError::bad_params("box is [x, y, w, h] with w, h ≥ 1"));
    }
    let input = input::resolve(ctx, &p.src)?;
    check_point(x, y, input.width, input.height)?;
    let (iw, ih) = (input.width as f64, input.height as f64);
    let (w, h) = (w.min(iw - x), h.min(ih - y));
    ctx.check_cancel()?;
    // The interactive decoder takes clicks: include the centre, exclude just outside the
    // corners (where inside the image); the mask is then clipped to the box.
    let mut clicks = vec![Click { x: ((x + w / 2.0) / iw) as f32, y: ((y + h / 2.0) / ih) as f32, positive: true }];
    let (mx, my) = (w * 0.05 + 2.0, h * 0.05 + 2.0);
    for (cx, cy) in [(x - mx, y - my), (x + w + mx, y - my), (x - mx, y + h + my), (x + w + mx, y + h + my)] {
        if cx >= 0.0 && cy >= 0.0 && cx < iw && cy < ih {
            clicks.push(Click { x: (cx / iw) as f32, y: (cy / ih) as f32, positive: false });
        }
    }
    let err = match sam.clicks(&ctx.shared.root, &input, &clicks) {
        Ok((pred, cached)) => {
            let mut m = sam_mask(&input, &pred);
            let pad = (iw.max(ih) * 0.01) as f32;
            mask::clip_to_box(&mut m, input.width as usize, input.height as usize, [x as f32, y as f32, w as f32, h as f32], pad);
            let mut v = put(ctx, input.width, input.height, m, Some(pred.score))?;
            v["engine"] = json!("sam3");
            v["cached"] = json!(cached);
            return Ok(v);
        }
        Err(e) => e,
    };
    // MobileSAM takes the box as a real box prompt.
    let nbox = [(x / iw) as f32, (y / ih) as f32, ((x + w) / iw) as f32, ((y + h) / ih) as f32];
    let err = match mobile_try(ctx, &st.mobile, &input, &[], Some(nbox), Some([x, y, w, h])) {
        Ok(mut v) => {
            v["samError"] = json!(err);
            return Ok(v);
        }
        Err(m) => both(err, m),
    };
    if !p.src.is_document() {
        return Err(no_fallback(input.kind, err));
    }
    fallback_result(ctx, &[("select.object", json!({"rect": [x, y, w, h], "mode": "replace"}), Mode::Replace)], Some(err))
}

/// Concepts PhotoCraft can stand in for without a model.
fn text_fallback(prompt: &str) -> Option<(&'static str, Value)> {
    let p = prompt.trim().to_lowercase();
    match p.as_str() {
        "sky" | "the sky" | "skies" => Some(("select.sky", json!({"mode": "replace"}))),
        "subject" | "the subject" | "person" | "people" | "a person" | "the person" | "main subject" => Some(("select.subject", json!({"mode": "replace"}))),
        _ => None,
    }
}

fn text(ctx: &Ctx, sam: &Sam, p: TextParams) -> ApiResult<Value> {
    let prompt = p.prompt.trim();
    if prompt.is_empty() || prompt.chars().count() > 200 {
        return Err(ApiError::bad_params("prompt: a short noun phrase (1..200 characters), e.g. \"sky\""));
    }
    let threshold = p.threshold.unwrap_or(0.5);
    if !(threshold.is_finite() && (0.0..1.0).contains(&threshold)) {
        return Err(ApiError::bad_params("threshold is 0..1"));
    }
    let max = p.max_instances.unwrap_or(8).clamp(1, MAX_INSTANCES);
    let input = input::resolve(ctx, &p.src)?;
    ctx.check_cancel()?;
    let err = match sam.text(&ctx.shared.root, &input, prompt, threshold, max) {
        Ok((preds, cached)) => {
            let mut instances = Vec::with_capacity(preds.len());
            for (pred, bbox) in preds {
                ctx.check_cancel()?;
                let mut v = put(ctx, input.width, input.height, sam_mask(&input, &pred), Some(pred.score))?;
                if let Some([x0, y0, x1, y1]) = bbox {
                    let (w, h) = (input.width as f32, input.height as f32);
                    v["bbox"] = json!([(x0 * w).round(), (y0 * h).round(), ((x1 - x0) * w).round(), ((y1 - y0) * h).round()]);
                }
                instances.push(v);
            }
            return Ok(json!({"instances": instances, "engine": "sam3", "cached": cached, "prompt": prompt}));
        }
        Err(e) => e,
    };
    match (p.src.is_document(), text_fallback(prompt)) {
        (true, Some((id, params))) => {
            let v = fallback_result(ctx, &[(id, params, Mode::Replace)], None)?;
            let empty = v["coverage"].as_f64().is_some_and(|c| c <= 0.0);
            let instances = if empty { json!([]) } else { json!([{"mask": v["mask"], "score": null, "bbox": v["bbox"], "coverage": v["coverage"]}]) };
            Ok(json!({"instances": instances, "engine": "photocraft", "samError": err, "prompt": prompt}))
        }
        (true, None) => Err(ApiError::unavailable(format!(
            "text concepts other than sky/subject need SAM 3 ({err}). Instead: look at a preview (doc.preview), locate the thing yourself and call segment.box {{box:[x,y,w,h]}} or segment.point (MobileSAM segments it precisely), or try select.colorRange"
        ))),
        (false, _) => Err(no_fallback(input.kind, err)),
    }
}

fn subject(ctx: &Ctx, st: &State, p: SubjectParams) -> ApiResult<Value> {
    let sam = &st.sam;
    let input = input::resolve(ctx, &p.src)?;
    ctx.check_cancel()?;
    let err = match sam.subject(&ctx.shared.root, &input) {
        Ok((Some(pred), cached)) => {
            let mut v = put(ctx, input.width, input.height, sam_mask(&input, &pred), Some(pred.score))?;
            v["engine"] = json!("sam3");
            v["cached"] = json!(cached);
            return Ok(v);
        }
        Ok((None, _)) => "SAM 3 found no person or animal".to_string(),
        Err(e) => e,
    };
    if !p.src.is_document() {
        return Err(no_fallback(input.kind, err));
    }
    // PhotoCraft's Select Subject finds *where* the subject is; MobileSAM, given that box,
    // draws a cleaner edge. Without MobileSAM, PhotoCraft's own mask is the answer.
    let (w, h, m) = photocraft(ctx, &[("select.subject", json!({"mode": "replace"}), Mode::Replace)])?;
    if Mobile::installed(&ctx.shared.root)
        && let Some([bx, by, bw, bh]) = mask::bbox(&m, w as usize, h as usize)
    {
        let (fw, fh) = (f64::from(input.width), f64::from(input.height));
        let (sx, sy) = (fw / f64::from(w.max(1)), fh / f64::from(h.max(1)));
        let r = [f64::from(bx) * sx, f64::from(by) * sy, f64::from(bw) * sx, f64::from(bh) * sy];
        let nbox = [(r[0] / fw) as f32, (r[1] / fh) as f32, ((r[0] + r[2]) / fw) as f32, ((r[1] + r[3]) / fh) as f32];
        if let Ok(mut v) = mobile_try(ctx, &st.mobile, &input, &[], Some(nbox), Some(r)) {
            v["engine"] = json!("mobilesam");
            v["guide"] = json!("photocraft select.subject");
            v["samError"] = json!(err);
            return Ok(v);
        }
    }
    let mut v = put(ctx, w, h, m, None)?;
    v["engine"] = json!("photocraft");
    v["samError"] = json!(err);
    Ok(v)
}

const SOURCE_PROPS: &str = r#"{"image":{"type":"string","description":"img:N handle"},"photo":{"type":"string","description":"workspace-relative photo path"}}"#;

fn schema(extra: Value, required: &[&str]) -> Value {
    let mut props: serde_json::Map<String, Value> = serde_json::from_str(SOURCE_PROPS).unwrap_or_default();
    if let Value::Object(e) = extra {
        props.extend(e);
    }
    json!({"type": "object", "properties": props, "required": required})
}

/// Register the `segment` area.
pub fn register(r: &mut Registry, state: Arc<State>) {
    let input_note = "Input: `image` (img:N), `photo` (workspace path) or, by default, the open document; coordinates are in that image's pixels and the mask comes back at its full size. SAM 3 encodes each image once on a ≤1008 px proxy (cached by content); prompts only run the decoder. Without the model (or if it fails) the open document falls back to PhotoCraft: `engine` says which, `samError` why. Read-only: the document's selection is not changed (apply with select.fromMask).";

    let s = state.clone();
    r.function("segment", "point", "Mask of the object at points [[x,y,label 1|0],…] (SAM 3; PhotoCraft Object Finder fallback)")
        .description(&format!("{input_note} Returns {{mask, width, height, bbox, coverage, score, engine, cached}}."))
        .params(schema(json!({"points":{"type":"array","items":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"minItems":1,"maxItems":MAX_POINTS}}), &["points"]))
        .effects(Effects::ReadOnly)
        .typed(move |ctx, p: PointParams| point(ctx, &s, p))
        .register();

    let s = state.clone();
    r.function("segment", "box", "Mask of the object in box [x,y,w,h] (SAM 3; PhotoCraft Object Finder fallback)")
        .description(&format!("{input_note} Returns {{mask, width, height, bbox, coverage, score, engine, cached}}."))
        .params(schema(json!({"box":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}}), &["box"]))
        .effects(Effects::ReadOnly)
        .typed(move |ctx, p: BoxParams| boxed(ctx, &s, p))
        .register();

    let s = state.clone();
    r.function("segment", "text", "Masks of every instance of a concept, e.g. {prompt:\"sky\"} (SAM 3)")
        .description(&format!("{input_note} Returns {{instances: [{{mask, score, bbox, coverage}}], engine}}, best first; `threshold` 0..1 (default 0.5), `maxInstances` (default 8). Without the model only sky and subject/person fall back to PhotoCraft."))
        .params(schema(json!({"prompt":{"type":"string"},"threshold":{"type":"number","minimum":0,"maximum":1},"maxInstances":{"type":"integer","minimum":1,"maximum":MAX_INSTANCES}}), &["prompt"]))
        .effects(Effects::ReadOnly)
        .typed(move |ctx, p: TextParams| text(ctx, &s.sam, p))
        .register();

    let s = state.clone();
    r.function("segment", "subject", "Mask of the photo's subject (SAM 3 people/animals; PhotoCraft Select Subject fallback)")
        .description(&format!("{input_note} Returns {{mask, width, height, bbox, coverage, score, engine}}."))
        .params(schema(json!({}), &[]))
        .effects(Effects::ReadOnly)
        .typed(move |ctx, p: SubjectParams| subject(ctx, &s, p))
        .register();

    let s = state.clone();
    r.function("segment", "prepare", "Encode an image ahead of prompts (MobileSAM), so later segment.point/box only decode")
        .description(&format!("{input_note} The encoder is the slow part (seconds on a phone CPU); call this when segmentation is likely (the app does when the Select tool opens) and prompts on the same pixels take a fraction of a second. Returns {{engine, cached, ms}}; `engine: null` when no model is installed."))
        .params(schema(json!({}), &[]))
        .effects(Effects::ReadOnly)
        .typed(move |ctx, p: SubjectParams| {
            let input = input::resolve(ctx, &p.src)?;
            let t = std::time::Instant::now();
            match s.mobile.prepare(&ctx.shared.root, &input) {
                Ok(cached) => Ok(json!({"engine": "mobilesam", "cached": cached, "ms": t.elapsed().as_millis() as u64})),
                Err(e) => Ok(json!({"engine": null, "reason": e})),
            }
        })
        .register();

    let s = state.clone();
    r.function("segment", "models", "Model status: SAM 3 (desktop) and MobileSAM (phone): missing | ready, size, licence, download URL")
        .description("Returns {sam3: missing|downloading|ready|unavailable, sizeBytes, path, loaded, cachedEmbeddings, licence, downloadConfigured, progress?, lastError?, mobilesam: {state, sizeBytes, sha256, url, file, path, licence, loaded, cachedEmbeddings}}. MobileSAM (40.6 MB, Apache-2.0, public) is what phones use for points and boxes: the app downloads `mobilesam.url` and calls installModel {model:\"mobilesam\", path}.")
        .params(json!({"type":"object","properties":{}}))
        .effects(Effects::ReadOnly)
        .typed(move |ctx, _: Empty| {
            let mut v = s.sam.status(&ctx.shared.root);
            v["mobilesam"] = s.mobile.status(&ctx.shared.root);
            Ok(v)
        })
        .register();

    let s = state.clone();
    r.function("segment", "ensureModel", "Download the SAM 3 weights (needs confirm:true after the person approves the size)")
        .description("Without `confirm: true` fails with `confirm_required` and states the size, folder and licence: ask the person first. With it, starts a background download into <workspace>/.imagelab/models/sam3 (resumable, SHA-256 checked) and returns {sam3: downloading}; progress arrives as ModelStatus events and in segment.models. Download locations come from IMAGELAB_SAM3_MIRRORS or .imagelab/models/sam3-mirrors.txt, never from params.")
        .params(json!({"type":"object","properties":{"confirm":{"type":"boolean"}}}))
        .effects(Effects::Destructive)
        .typed(move |ctx, p: EnsureParams| s.sam.ensure(ctx, p.confirm))
        .register();

    let s = state.clone();
    r.function("segment", "installModel", "Install model files the app downloaded, after checking size and SHA-256")
        .description("model `sam3` (default): `path` is a workspace folder with model.safetensors, vocab.json, merges.txt matching the official facebook/sam3 checkpoint. model `mobilesam`: `path` is the downloaded mobile_sam-tiny-vitt.safetensors (see segment.models for URL, size, SHA-256). Files are hard-linked into .imagelab/models/<model>, or copied.")
        .params(json!({"type":"object","required":["path"],"properties":{"path":{"type":"string"},"model":{"type":"string","enum":["sam3","mobilesam"]}}}))
        .effects(Effects::Destructive)
        .typed(move |ctx, p: InstallParams| {
            let path = input::workspace_path(ctx, &p.path)?;
            match p.model.as_deref().unwrap_or("sam3") {
                "sam3" => s.sam.install(ctx, &path),
                "mobilesam" => s.mobile.install(ctx, &path),
                other => Err(ApiError::bad_params(format!("model `{other}`: sam3 or mobilesam"))),
            }
        })
        .register();

    let s = state.clone();
    r.function("segment", "cancelDownload", "Stop a running SAM 3 download (it resumes next time)")
        .params(json!({"type":"object","properties":{}}))
        .effects(Effects::Revertible)
        .typed(move |_, _: Empty| Ok(json!({"cancelled": s.sam.cancel_download()})))
        .register();

    let s = state;
    r.function("segment", "unload", "Free the loaded models (SAM 3, MobileSAM) and cached embeddings now (they reload on demand)")
        .params(json!({"type":"object","properties":{}}))
        .effects(Effects::ReadOnly)
        .typed(move |_, _: Empty| Ok(json!({"unloaded": s.sam.unload() | s.mobile.unload()})))
        .register();
}
