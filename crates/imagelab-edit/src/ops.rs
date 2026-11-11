//! Operations on one PhotoCraft session (`&mut Headless`), shared by the UI-session areas
//! (`doc`, `edit`) and by `batch`, which runs them on throwaway sessions.

use std::path::Path;

use imagelab_api::{ApiError, ApiResult, Headless};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::allow;
use crate::paths;

/// Most steps in one `edit.steps` call or recipe.
pub const MAX_STEPS: usize = 200;
/// Largest side of a new document or export.
pub const MAX_SIDE: u32 = 16_384;
/// Largest preview the Headless workspace session renders (PhotoCraft's automation budget).
pub const MAX_PREVIEW_SIDE: u32 = 2048;

/// Run an allowlisted engine command (and PhotoCraft's automation gate) on the active document.
pub fn run_allowed(h: &mut Headless, id: &str, params: Value) -> ApiResult<Value> {
    let id = id.trim();
    if !allow::is_allowed(id) {
        return Err(ApiError::new("not_allowed", allow::refusal(id)));
    }
    let params = if params.is_null() { json!({}) } else { params };
    if !params.is_object() {
        return Err(ApiError::bad_params(format!("params of `{id}` must be an object")));
    }
    photocraft_automation::workspace::authorize_engine_command(id, &params).map_err(|e| ApiError::new("not_allowed", e.to_string()))?;
    match h.command_run(id, params.clone()) {
        Ok(v) => Ok(v),
        // Pixel commands (filters, auto tone/color, direct adjustments, retouch) refuse an
        // adjustment/fill/text layer. People and agents mean "the photo": make the topmost
        // visible pixel layer active and run once more, saying so in the result.
        Err(e) if needs_pixels(&e.to_string()) => match top_pixel_layer(h) {
            Some(layer) => {
                h.command_run("layer.select", json!({"layer": layer}))?;
                let mut v = h.command_run(id, params)?;
                if let Value::Object(m) = &mut v {
                    m.insert("retargetedLayer".into(), json!(layer));
                } else {
                    v = json!({"result": v, "retargetedLayer": layer});
                }
                Ok(v)
            }
            None => Err(e.into()),
        },
        Err(e) => Err(e.into()),
    }
}

/// PhotoCraft's refusal for a pixel command on a layer without pixels.
fn needs_pixels(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("has no pixels") || m.contains("need a pixel layer") || m.contains("needs a pixel layer")
}

/// The topmost visible pixel layer of the active document, from the inspect projection.
fn top_pixel_layer(h: &Headless) -> Option<u64> {
    fn walk(layers: &Value) -> Option<u64> {
        for l in layers.as_array()? {
            if l.get("visible").and_then(Value::as_bool) == Some(false) {
                continue;
            }
            if l.get("kind").and_then(Value::as_str) == Some("Pixel")
                && let Some(id) = l.get("id").and_then(Value::as_u64)
            {
                return Some(id);
            }
            if let Some(found) = l.get("children").and_then(walk) {
                return Some(found);
            }
        }
        None
    }
    walk(h.inspect(None).ok()?.get("layers")?)
}

/// Index and revision of the active document.
pub fn active_revision(h: &Headless) -> Option<(usize, u64)> {
    let i = h.session.active_index()?;
    h.session.documents().get(i).map(|d| (i, d.revision))
}

fn history_mark(h: &Headless) -> (usize, u64) {
    h.session.active().map_or((0, 0), |d| (d.history.entries().len(), d.revision))
}

/// How many history steps a change from `before` to now made (at least one when the revision
/// moved, so a history at its undo limit still counts).
fn steps_since(h: &Headless, before: (usize, u64)) -> usize {
    let after = history_mark(h);
    let grown = after.0.saturating_sub(before.0);
    if grown > 0 {
        grown
    } else if after.1 != before.1 {
        1
    } else {
        0
    }
}

/// Undo `n` steps (as many as history holds).
pub fn undo_n(h: &mut Headless, n: usize) -> ApiResult<usize> {
    let mut done = 0;
    for _ in 0..n {
        if !h.session.active().is_some_and(|d| d.history.can_undo()) {
            break;
        }
        h.command_run("edit.undo", json!({}))?;
        done += 1;
    }
    Ok(done)
}

/// One step of `edit.steps`, a recipe or a batch: an engine command `{id, params}` or an
/// ImageLab edit function `{fn: "edit.adjust" | "edit.auto" | "edit.straighten" |
/// "edit.cropTo" | "edit.run", params}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "fn", skip_serializing_if = "Option::is_none")]
    pub func: Option<String>,
    #[serde(default)]
    pub params: Value,
}

/// The ImageLab edit functions a step may name.
pub const STEP_FUNCTIONS: &[&str] = &["edit.run", "edit.adjust", "edit.auto", "edit.straighten", "edit.cropTo"];

/// Check a step without running it (recipes are validated when saved).
pub fn validate_step(step: &Step) -> ApiResult<()> {
    match (&step.id, &step.func) {
        (Some(id), None) => {
            if matches!(id.as_str(), "edit.undo" | "edit.redo") {
                return Err(ApiError::bad_params("edit.undo / edit.redo cannot be a step (steps roll back as a unit); use doc.undo"));
            }
            if !allow::is_allowed(id) {
                return Err(ApiError::new("not_allowed", allow::refusal(id)));
            }
            Ok(())
        }
        (None, Some(f)) if f == "edit.run" => {
            let id = step.params.get("id").and_then(Value::as_str).ok_or_else(|| ApiError::bad_params("edit.run step needs params.id"))?;
            validate_step(&Step { id: Some(id.to_string()), func: None, params: Value::Null })
        }
        (None, Some(f)) if STEP_FUNCTIONS.contains(&f.as_str()) => Ok(()),
        (None, Some(f)) => Err(ApiError::bad_params(format!("step fn `{f}` is not one of {}", STEP_FUNCTIONS.join(", ")))),
        _ => Err(ApiError::bad_params("a step has exactly one of `id` (engine command) or `fn` (ImageLab edit function)")),
    }
}

fn parse<T: serde::de::DeserializeOwned>(what: &str, params: &Value) -> ApiResult<T> {
    let params = if params.is_null() { json!({}) } else { params.clone() };
    serde_json::from_value(params).map_err(|e| ApiError::bad_params(format!("{what}: {e}")))
}

/// Run one step (no rollback of its own beyond what the function does).
pub fn apply_step(h: &mut Headless, step: &Step) -> ApiResult<Value> {
    validate_step(step)?;
    if let Some(id) = &step.id {
        return run_allowed(h, id, step.params.clone());
    }
    match step.func.as_deref().unwrap_or_default() {
        "edit.run" => {
            let p: RunParams = parse("edit.run", &step.params)?;
            run_allowed(h, &p.id, p.params)
        }
        "edit.adjust" => adjust(h, &parse("edit.adjust", &step.params)?),
        "edit.auto" => auto(h, &parse("edit.auto", &step.params)?),
        "edit.straighten" => straighten(h, &parse("edit.straighten", &step.params)?),
        "edit.cropTo" => crop_to(h, &parse("edit.cropTo", &step.params)?),
        other => Err(ApiError::bad_params(format!("unknown step fn `{other}`"))),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunParams {
    pub id: String,
    #[serde(default)]
    pub params: Value,
}

/// Run `steps` in order. With `atomic`, a failing step undoes the steps already applied (through
/// history) and the error says which step failed; without, it stops at the failure and keeps
/// what was applied.
pub fn apply_steps(h: &mut Headless, steps: &[Step], atomic: bool) -> ApiResult<Vec<Value>> {
    if steps.len() > MAX_STEPS {
        return Err(ApiError::too_large(format!("{} steps; at most {MAX_STEPS}", steps.len())));
    }
    for (i, s) in steps.iter().enumerate() {
        validate_step(s).map_err(|e| ApiError::new(&e.code, format!("step {i}: {}", e.message)))?;
    }
    let mut results = Vec::with_capacity(steps.len());
    let mut applied_history = 0usize;
    for (i, step) in steps.iter().enumerate() {
        let before = history_mark(h);
        match apply_step(h, step) {
            Ok(v) => {
                applied_history += steps_since(h, before);
                results.push(v);
            }
            Err(e) => {
                // A failed command may still have left history (a multi-step function).
                applied_history += steps_since(h, before);
                let mut message = format!("step {i} ({}) failed: {}", step.id.as_deref().or(step.func.as_deref()).unwrap_or("?"), e.message);
                if atomic && applied_history > 0 {
                    match undo_n(h, applied_history) {
                        Ok(n) => message.push_str(&format!("; rolled back {n} history step(s), the document is as before")),
                        Err(u) => message.push_str(&format!("; rollback failed: {}", u.message)),
                    }
                } else if !atomic {
                    message.push_str(&format!("; {i} step(s) before it stay applied"));
                }
                return Err(ApiError::new(&e.code, message));
            }
        }
    }
    Ok(results)
}

// ---------------------------------------------------------------- adjust / auto

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdjustParams {
    pub kind: String,
    #[serde(default)]
    pub params: Value,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub opacity: Option<f64>,
    #[serde(default)]
    pub mode: Option<String>,
}

/// Adjustment kinds (`layer.newAdjustmentLayer.<kind>`), from the allowlist.
pub fn adjustment_kinds() -> Vec<&'static str> {
    allow::ids_with_prefix("layer.newAdjustmentLayer.").into_iter().filter_map(|id| id.strip_prefix("layer.newAdjustmentLayer.")).collect()
}

/// Add a non-destructive adjustment layer. The active selection becomes its layer mask.
pub fn adjust(h: &mut Headless, p: &AdjustParams) -> ApiResult<Value> {
    let kind = p.kind.trim();
    let id = format!("layer.newAdjustmentLayer.{kind}");
    if !allow::is_allowed(&id) {
        return Err(ApiError::bad_params(format!("adjustment kind `{kind}` is not one of {}", adjustment_kinds().join(", "))));
    }
    let mut params: Map<String, Value> = match &p.params {
        Value::Null => Map::new(),
        Value::Object(m) => m.clone(),
        _ => return Err(ApiError::bad_params("adjust params must be an object")),
    };
    if let Some(name) = &p.name {
        params.insert("name".into(), json!(name.chars().take(128).collect::<String>()));
    }
    if let Some(o) = p.opacity {
        params.insert("opacity".into(), json!(o));
    }
    if let Some(m) = &p.mode {
        params.insert("mode".into(), json!(m));
    }
    let result = run_allowed(h, &id, Value::Object(params))?;
    let layer = h.session.active().and_then(|d| d.active_layer).map(|l| l.0);
    Ok(json!({"layer": layer, "kind": kind, "result": result}))
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutoParams {
    #[serde(default)]
    pub tone: Option<bool>,
    #[serde(default)]
    pub contrast: Option<bool>,
    #[serde(default)]
    pub color: Option<bool>,
}

/// Auto Tone / Contrast / Color on the active pixel layer (tone alone when nothing is chosen).
pub fn auto(h: &mut Headless, p: &AutoParams) -> ApiResult<Value> {
    let none = p.tone.is_none() && p.contrast.is_none() && p.color.is_none();
    let mut ran = Vec::new();
    for (on, id) in [(p.tone.unwrap_or(none), "image.autoTone"), (p.contrast.unwrap_or(false), "image.autoContrast"), (p.color.unwrap_or(false), "image.autoColor")] {
        if on {
            run_allowed(h, id, json!({}))?;
            ran.push(id);
        }
    }
    Ok(json!({"ran": ran}))
}

// ---------------------------------------------------------------- geometry

fn doc_size(h: &Headless) -> ApiResult<(u32, u32)> {
    h.session.active().map(|d| (d.doc.size.width, d.doc.size.height)).ok_or_else(|| ApiError::not_found("no open document; doc.open a photo first"))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StraightenParams {
    /// Degrees, clockwise positive, |angle| ≤ 45.
    pub angle: f64,
    #[serde(default = "yes")]
    pub crop: bool,
}

fn yes() -> bool {
    true
}

/// The largest axis-aligned rectangle inside a `w`×`h` rectangle rotated by `angle` radians.
pub fn inscribed(w: f64, h: f64, angle: f64) -> (f64, f64) {
    if w <= 0.0 || h <= 0.0 {
        return (0.0, 0.0);
    }
    let (sin_a, cos_a) = (angle.sin().abs(), angle.cos().abs());
    let width_longer = w >= h;
    let (long, short) = if width_longer { (w, h) } else { (h, w) };
    if short <= 2.0 * sin_a * cos_a * long || (sin_a - cos_a).abs() < 1e-10 {
        let x = 0.5 * short;
        if sin_a < 1e-12 {
            return (w, h);
        }
        if width_longer { (x / sin_a, x / cos_a) } else { (x / cos_a, x / sin_a) }
    } else {
        let cos_2a = cos_a * cos_a - sin_a * sin_a;
        ((w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a)
    }
}

/// Rotate the canvas by `angle` and crop away the empty corners.
pub fn straighten(h: &mut Headless, p: &StraightenParams) -> ApiResult<Value> {
    if !p.angle.is_finite() || p.angle.abs() > 45.0 {
        return Err(ApiError::bad_params("angle must be between -45 and 45 degrees"));
    }
    if p.angle.abs() < 1e-3 {
        return Ok(json!({"rotated": false}));
    }
    let (w, hgt) = doc_size(h)?;
    let before = history_mark(h);
    let direction = if p.angle >= 0.0 { "cw" } else { "ccw" };
    run_allowed(h, "image.rotation.arbitrary", json!({"angle": p.angle.abs(), "direction": direction}))?;
    if !p.crop {
        return Ok(json!({"rotated": true, "cropped": false}));
    }
    let (nw, nh) = doc_size(h)?;
    let (cw, ch) = inscribed(f64::from(w), f64::from(hgt), p.angle.to_radians());
    let cw = (cw.floor() as u32).clamp(1, nw);
    let ch = (ch.floor() as u32).clamp(1, nh);
    let x = (nw - cw) / 2;
    let y = (nh - ch) / 2;
    if let Err(e) = run_allowed(h, "image.crop", json!({"x": x, "y": y, "width": cw, "height": ch})) {
        let n = steps_since(h, before);
        let _ = undo_n(h, n);
        return Err(ApiError::new(&e.code, format!("straighten: crop failed ({}); the rotation was undone", e.message)));
    }
    Ok(json!({"rotated": true, "cropped": true, "width": cw, "height": ch}))
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CropParams {
    #[serde(default)]
    pub x: Option<i64>,
    #[serde(default)]
    pub y: Option<i64>,
    #[serde(default)]
    pub w: Option<i64>,
    #[serde(default)]
    pub h: Option<i64>,
    /// `"4:5"`, `"1:1"`, `"16:9"`, any `"W:H"`.
    #[serde(default)]
    pub aspect: Option<String>,
    /// `center` (default) or `subject` (centre on PhotoCraft's Select Subject bounds).
    #[serde(default)]
    pub anchor: Option<String>,
}

/// Parse `"W:H"` into a positive ratio W/H.
pub fn parse_aspect(s: &str) -> ApiResult<f64> {
    let (a, b) = s.split_once([':', 'x', '/']).ok_or_else(|| ApiError::bad_params(format!("aspect `{s}` must look like \"4:5\"")))?;
    let a: f64 = a.trim().parse().map_err(|_| ApiError::bad_params(format!("aspect `{s}` must look like \"4:5\"")))?;
    let b: f64 = b.trim().parse().map_err(|_| ApiError::bad_params(format!("aspect `{s}` must look like \"4:5\"")))?;
    if !(a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 && a / b < 100.0 && b / a < 100.0) {
        return Err(ApiError::bad_params(format!("aspect `{s}` must be two positive numbers within 1:100")));
    }
    Ok(a / b)
}

/// The largest `ratio` rectangle inside `w`×`h`, centred on `center` and kept inside the canvas.
pub fn aspect_rect(w: u32, h: u32, ratio: f64, center: (f64, f64)) -> (u32, u32, u32, u32) {
    let (wf, hf) = (f64::from(w), f64::from(h));
    let (cw, ch) = if wf / hf > ratio { (hf * ratio, hf) } else { (wf, wf / ratio) };
    let cw = (cw.round() as u32).clamp(1, w);
    let ch = (ch.round() as u32).clamp(1, h);
    let x = (center.0 - f64::from(cw) / 2.0).round().clamp(0.0, f64::from(w - cw)) as u32;
    let y = (center.1 - f64::from(ch) / 2.0).round().clamp(0.0, f64::from(h - ch)) as u32;
    (x, y, cw, ch)
}

/// The centre of PhotoCraft's Select Subject, leaving the document's selection and history as
/// they were. `None` when it finds nothing.
fn subject_center(h: &mut Headless) -> Option<(f64, f64)> {
    let before = history_mark(h);
    let found = run_allowed(h, "select.subject", json!({})).ok().and_then(|_| {
        let info = h.inspect(None).ok()?;
        let b = info.get("selectionBounds")?.as_array()?.iter().filter_map(Value::as_f64).collect::<Vec<_>>();
        (b.len() == 4 && b[2] > 0.0 && b[3] > 0.0).then(|| (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0))
    });
    let n = steps_since(h, before);
    let _ = undo_n(h, n);
    found
}

/// Crop to a rectangle, or to an aspect ratio around the centre or the subject.
pub fn crop_to(h: &mut Headless, p: &CropParams) -> ApiResult<Value> {
    let (w, hgt) = doc_size(h)?;
    let (x, y, cw, ch, anchor) = match (&p.aspect, p.x, p.y, p.w, p.h) {
        (None, Some(x), Some(y), Some(cw), Some(ch)) => {
            if cw <= 0 || ch <= 0 || x < 0 || y < 0 || x + cw > i64::from(w) || y + ch > i64::from(hgt) {
                return Err(ApiError::bad_params(format!("crop {x},{y} {cw}x{ch} must lie inside the {w}x{hgt} canvas")));
            }
            (x as u32, y as u32, cw as u32, ch as u32, "rect")
        }
        (Some(a), None, None, None, None) => {
            let ratio = parse_aspect(a)?;
            let mut center = (f64::from(w) / 2.0, f64::from(hgt) / 2.0);
            let anchor = match p.anchor.as_deref().unwrap_or("center") {
                "center" | "centre" => "center",
                "subject" => match subject_center(h) {
                    Some(c) => {
                        center = c;
                        "subject"
                    }
                    None => "center (no subject found)",
                },
                other => return Err(ApiError::bad_params(format!("anchor `{other}` must be center or subject"))),
            };
            let (x, y, cw, ch) = aspect_rect(w, hgt, ratio, center);
            (x, y, cw, ch, anchor)
        }
        _ => return Err(ApiError::bad_params("pass either {x, y, w, h} or {aspect: \"4:5\", anchor?: center|subject}")),
    };
    run_allowed(h, "image.crop", json!({"x": x, "y": y, "width": cw, "height": ch}))?;
    Ok(json!({"x": x, "y": y, "width": cw, "height": ch, "anchor": anchor}))
}

// ---------------------------------------------------------------- export

/// How an export is sized and encoded. Presets are values of this type.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExportSpec {
    /// `jpg`, `png`, `webp`, `tif`, `psd` or `pcraft`; default from the path's extension.
    #[serde(default)]
    pub format: Option<String>,
    /// Longest side in px (never upscales).
    #[serde(default)]
    pub max_side: Option<u32>,
    /// Width in px (never upscales); height follows.
    #[serde(default)]
    pub width: Option<u32>,
    /// Crop to this aspect (`"4:5"`) around the centre first.
    #[serde(default)]
    pub aspect: Option<String>,
    /// JPEG / WebP quality 1..100.
    #[serde(default)]
    pub quality: Option<u8>,
}

/// Formats an export may write.
pub const EXPORT_FORMATS: &[&str] = &["jpg", "jpeg", "png", "webp", "tif", "tiff", "psd", "pcraft"];

pub fn normalize_format(f: &str) -> ApiResult<String> {
    let f = f.trim().trim_start_matches('.').to_ascii_lowercase();
    if !EXPORT_FORMATS.contains(&f.as_str()) {
        return Err(ApiError::bad_params(format!("format `{f}` is not one of {}", EXPORT_FORMATS.join(", "))));
    }
    Ok(match f.as_str() {
        "jpeg" => "jpg".into(),
        "tiff" => "tif".into(),
        _ => f,
    })
}

/// PhotoCraft export options for `spec`.
pub fn export_options(spec: &ExportSpec) -> ApiResult<photocraft_io::ExportOptions> {
    let mut opts = photocraft_io::ExportOptions::default();
    if let Some(q) = spec.quality {
        if !(1..=100).contains(&q) {
            return Err(ApiError::bad_params("quality must be 1..100"));
        }
        opts.encode.jpeg_quality = q;
        opts.encode.webp_quality = q;
        opts.encode.webp_lossless = false;
    }
    Ok(opts)
}

/// The size the active document resizes to under `spec`, or `None` to keep it.
pub fn target_size(w: u32, h: u32, spec: &ExportSpec) -> ApiResult<Option<(u32, u32)>> {
    let scale = match (spec.max_side, spec.width) {
        (Some(m), _) if m == 0 || m > MAX_SIDE => return Err(ApiError::bad_params(format!("maxSide must be 1..{MAX_SIDE}"))),
        (_, Some(m)) if m == 0 || m > MAX_SIDE => return Err(ApiError::bad_params(format!("width must be 1..{MAX_SIDE}"))),
        (Some(m), _) => f64::from(m) / f64::from(w.max(h)),
        (None, Some(m)) => f64::from(m) / f64::from(w),
        (None, None) => return Ok(None),
    };
    if scale >= 1.0 {
        return Ok(None);
    }
    let nw = ((f64::from(w) * scale).round() as u32).max(1);
    let nh = ((f64::from(h) * scale).round() as u32).max(1);
    Ok(Some((nw, nh)))
}

/// Crop and resize the active document of `h` in place for `spec` (for a throwaway session or
/// a duplicate). Returns whether anything changed.
pub fn shape_for_export(h: &mut Headless, spec: &ExportSpec) -> ApiResult<bool> {
    let mut changed = false;
    if let Some(a) = &spec.aspect {
        crop_to(h, &CropParams { aspect: Some(a.clone()), ..CropParams::default() })?;
        changed = true;
    }
    let (w, hgt) = doc_size(h)?;
    if let Some((nw, nh)) = target_size(w, hgt, spec)? {
        run_allowed(h, "image.imageSize", json!({"width": nw, "height": nh, "resample": "bicubic"}))?;
        changed = true;
    }
    Ok(changed)
}

/// Save the active document of `h` to `path` (workspace-relative, not under `library/`) with
/// `spec`'s format and quality; does not resize.
pub fn save_as(h: &mut Headless, path: &str, spec: &ExportSpec) -> ApiResult<Value> {
    let path = paths::writable(path)?;
    let format = match &spec.format {
        Some(f) => Some(normalize_format(f)?),
        None => paths::extension(&path).map(|e| normalize_format(&e)).transpose()?,
    };
    let opts = export_options(spec)?;
    Ok(h.save(None, Some(Path::new(&path)), format.as_deref(), &opts)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn inscribed_rect_is_smaller_and_sane() {
        let (w, h) = inscribed(400.0, 300.0, 5f64.to_radians());
        assert!(w < 400.0 && h < 300.0 && w > 300.0 && h > 200.0, "{w}x{h}");
        assert_eq!(inscribed(400.0, 300.0, 0.0), (400.0, 300.0));
    }

    #[test]
    fn aspect_and_size() {
        assert_eq!(parse_aspect("4:5").unwrap(), 0.8);
        assert!(parse_aspect("0:5").is_err());
        assert_eq!(aspect_rect(1000, 1000, 0.8, (500.0, 500.0)), (100, 0, 800, 1000));
        assert_eq!(aspect_rect(1000, 500, 1.0, (0.0, 250.0)), (0, 0, 500, 500));
        let spec = ExportSpec { max_side: Some(100), ..Default::default() };
        assert_eq!(target_size(400, 200, &spec).unwrap(), Some((100, 50)));
        assert_eq!(target_size(50, 20, &spec).unwrap(), None);
        let spec = ExportSpec { width: Some(1080), ..Default::default() };
        assert_eq!(target_size(2160, 2700, &spec).unwrap(), Some((1080, 1350)));
    }
}
