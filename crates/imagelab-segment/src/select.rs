//! The `select` area: PhotoCraft's own selections on the open document (no model), and the two
//! bridges between the document's selection and the handle arena's masks.
//!
//! The bridges read and write `Document::selection` (a GRAY8 coverage surface) directly:
//! `toMask` copies it out; `fromMask` writes it through `Session::edit`, the same entry every
//! PhotoCraft command uses, so the change is one history step ("Load Selection"), undoable,
//! journalled, and re-rendered like `select.all`.

use imagelab_api::{ApiError, ApiResult, Ctx, Event, Headless, MaskBuf};
use photocraft_engine::doc::{PixelFormat, Rect, Surface};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::mask::{self, Mode};

/// Largest document whose selection is copied in one piece (side limit of the handle arena).
const MAX_SIDE: u32 = imagelab_api::handles::MAX_SIDE;

/// Document size and current selection (`None` = nothing selected) as GRAY8 coverage.
pub fn read_selection(h: &Headless) -> ApiResult<(u32, u32, Option<Vec<u8>>)> {
    let d = h.session.active().ok_or_else(|| ApiError::not_found("no document is open"))?;
    let (w, ht) = (d.doc.size.width, d.doc.size.height);
    if w == 0 || ht == 0 || w > MAX_SIDE || ht > MAX_SIDE {
        return Err(ApiError::too_large(format!("document is {w}x{ht}; masks are limited to {MAX_SIDE} px a side")));
    }
    let Some(sel) = d.doc.selection.as_ref() else { return Ok((w, ht, None)) };
    let r = Rect::from_xywh(0, 0, w, ht);
    let bytes = if sel.format() == PixelFormat::GRAY8 {
        sel.try_to_interleaved(r)
    } else {
        sel.try_convert(PixelFormat::GRAY8).and_then(|s| s.try_to_interleaved(r))
    }
    .map_err(|e| ApiError::too_large(format!("reading the selection: {e:?}")))?;
    Ok((w, ht, Some(bytes)))
}

/// Replace the document's selection with `cov` (document size), as one history step. An empty
/// mask deselects.
pub fn write_selection(h: &mut Headless, label: &str, w: u32, ht: u32, cov: &[u8]) -> ApiResult<()> {
    let surface = if cov.iter().all(|v| *v == 0) { None } else { Some(Surface::from_interleaved(PixelFormat::GRAY8, Rect::from_xywh(0, 0, w, ht), cov)) };
    h.session
        .edit(label, move |doc, _| {
            doc.selection = surface;
            Ok(())
        })
        .map_err(|e| ApiError::new("engine", e.to_string()))
}

fn revision(h: &Headless) -> Option<(usize, u64)> {
    let i = h.session.active_index()?;
    h.session.active().map(|d| (i, d.revision))
}

/// Run a PhotoCraft command; emit `DocChanged` when it changed the document.
pub fn run_command(ctx: &Ctx, id: &str, params: Value) -> ApiResult<Value> {
    let (out, changed) = ctx.with_session(|h| {
        let before = revision(h);
        let out = h.command_run(id, params)?;
        let after = revision(h);
        Ok((out, if after != before { after } else { None }))
    })?;
    if let Some((doc, revision)) = changed {
        ctx.emit(Event::DocChanged { doc, revision, source: ctx.source });
    }
    Ok(out)
}

/// Run a selection command and take it back again, returning the selection it made (document
/// size). The document's own selection and history are as before (the step is rolled back).
/// This is how `segment.*` uses PhotoCraft as its fallback while staying read-only.
pub fn probe(h: &mut Headless, id: &str, params: Value) -> ApiResult<(u32, u32, Option<Vec<u8>>)> {
    let before = revision(h);
    let ran = h.command_run(id, params);
    let changed = revision(h) != before;
    let result = ran.map_err(ApiError::from).and_then(|_| read_selection(h));
    if changed {
        h.session.rollback();
    }
    result
}

/// [`probe`] starting from `cov` as the selection: load it, run `id`, read the result, and roll
/// both steps back (the document's selection and history end as they were).
pub fn probe_from(h: &mut Headless, w: u32, ht: u32, cov: &[u8], id: &str, params: Value) -> ApiResult<(u32, u32, Option<Vec<u8>>)> {
    let before = revision(h);
    write_selection(h, "Load Selection", w, ht, cov)?;
    let wrote = revision(h) != before;
    let result = probe(h, id, params);
    if wrote {
        h.session.rollback();
    }
    result
}

/// After a selection command: what is selected now.
fn summary(ctx: &Ctx, engine: Value) -> ApiResult<Value> {
    let (w, h, sel) = ctx.with_session(|h| read_selection(h))?;
    Ok(match sel {
        None => json!({"selected": false, "coverage": 0.0, "bounds": null, "result": engine}),
        Some(s) => json!({
            "selected": true,
            "coverage": mask::coverage(&s),
            "bounds": mask::bbox(&s, w as usize, h as usize),
            "result": engine,
        }),
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FromMask {
    mask: String,
    #[serde(default)]
    mode: Mode,
    #[serde(default)]
    feather: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}

/// Register the `select` area.
pub fn register(r: &mut imagelab_api::Registry) {
    use imagelab_api::Effects::{ReadOnly, Revertible};

    // Thin wrappers: params go to PhotoCraft as given (it validates them); the result adds what
    // is selected afterwards.
    let wrappers: &[(&str, &str, &str, Value)] = &[
        (
            "subject",
            "select.subject",
            "Select the main subject (PhotoCraft, no model)",
            json!({"type":"object","properties":{"mode":{"enum":["replace","add","subtract","intersect"]},"sampleAllLayers":{"type":"boolean"}}}),
        ),
        (
            "object",
            "select.object",
            "Select the object in a rect [x,y,w,h] or under a point [x,y] (PhotoCraft Object Finder)",
            json!({"type":"object","properties":{"rect":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"point":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2},"mode":{"enum":["replace","add","subtract","intersect"]},"sampleAllLayers":{"type":"boolean"}}}),
        ),
        (
            "colorRange",
            "select.colorRange",
            "Select by colour: sampled colours, reds…blues, highlights/midtones/shadows, skin tones",
            json!({"type":"object","description":"PhotoCraft select.colorRange params: select (sampledColors|reds|yellows|greens|cyans|blues|magentas|highlights|midtones|shadows|skinTones|outOfGamut), color, colors, points, subtractPoints, fuzziness, mode…"}),
        ),
        (
            "magicWand",
            "select.magicWand",
            "Magic wand at (x, y) with a tolerance",
            json!({"type":"object","required":["x","y"],"properties":{"x":{"type":"number"},"y":{"type":"number"},"tolerance":{"type":"number","minimum":0,"maximum":255},"contiguous":{"type":"boolean"},"antiAlias":{"type":"boolean"},"sampleAllLayers":{"type":"boolean"},"mode":{"enum":["replace","add","subtract","intersect"]}}}),
        ),
        ("all", "select.all", "Select the whole canvas", json!({"type":"object","properties":{}})),
        ("none", "select.deselect", "Deselect", json!({"type":"object","properties":{}})),
        ("inverse", "select.inverse", "Invert the selection", json!({"type":"object","properties":{}})),
    ];
    for (name, cmd, summary_text, schema) in wrappers {
        let cmd = cmd.to_string();
        r.function("select", name, summary_text)
            .description(&format!("Runs PhotoCraft `{cmd}` on the open document (one undoable step). Returns {{selected, coverage 0..1, bounds [x,y,w,h], result}}."))
            .params(schema.clone())
            .effects(Revertible)
            .handler(move |ctx, params| {
                let params = if params.is_null() { json!({}) } else { params };
                if !params.is_object() {
                    return Err(ApiError::bad_params("params must be an object"));
                }
                let out = run_command(ctx, &cmd, params)?;
                summary(ctx, out)
            })
            .register();
    }

    r.function("select", "sky", "Select the sky (PhotoCraft, no model), its edge completed to the skyline")
        .description("Runs PhotoCraft `select.sky` on the open document, then (edgeFill, default true) completes its edge: PhotoCraft's mask stops short of the skyline, and an edit through it leaves a band of the old sky along ridges and treetops (a halo). edgeFill grows the mask into sky-coloured pixels next to it (CIELAB distance to the sky's mean colour) and feathers it 1.5 px, so the selection reaches the skyline without crossing it. One undoable step per command. Returns {selected, coverage 0..1, bounds [x,y,w,h], result, edgeFill}.")
        .params(json!({"type":"object","properties":{"mode":{"enum":["replace","add","subtract","intersect"]},"threshold":{"type":"number","minimum":0,"maximum":100},"softness":{"type":"number","minimum":0,"maximum":100},"sampleAllLayers":{"type":"boolean"},"edgeFill":{"type":"boolean","description":"Complete the edge to the skyline (default true)."}}}))
        .effects(Revertible)
        .handler(|ctx, params| {
            let mut params = if params.is_null() { json!({}) } else { params };
            let obj = params.as_object_mut().ok_or_else(|| ApiError::bad_params("params must be an object"))?;
            let fill = match obj.remove("edgeFill") {
                None | Some(Value::Null) => true,
                Some(Value::Bool(b)) => b,
                Some(_) => return Err(ApiError::bad_params("edgeFill must be true or false")),
            };
            // Completing the edge works on the sky alone: run PhotoCraft's select.sky as a fresh
            // selection, complete it, then combine with what was selected by `mode`.
            let mode: Mode = match obj.remove("mode") {
                None | Some(Value::Null) => Mode::default(),
                Some(m) => serde_json::from_value(m).map_err(|e| ApiError::bad_params(format!("mode: {e}")))?,
            };
            if !fill {
                obj.insert("mode".into(), json!(mode.as_str()));
                let out = run_command(ctx, "select.sky", params)?;
                let mut v = summary(ctx, out)?;
                v["edgeFill"] = json!(false);
                return Ok(v);
            }
            let (changed, filled) = ctx.with_session(|h| {
                let (w, ht, before) = read_selection(h)?;
                let (_, _, sky) = probe(h, "select.sky", params.clone())?;
                let Some(sky) = sky else {
                    return Ok((None, json!({"grown": 0.0, "note": "no sky found"})));
                };
                let (pw, ph, rgba) = imagelab_cv::io::document_rgba(ctx, h)?;
                if pw != w as usize || ph != ht as usize {
                    return Err(ApiError::internal("the render and the document differ in size"));
                }
                let (completed, grown) = complete_sky_edge(&sky, &rgba, w as usize, ht as usize);
                let out = mask::combine(before.as_deref(), &completed, mode);
                write_selection(h, "Select Sky", w, ht, &out)?;
                Ok((revision(h), json!({"grown": grown})))
            })?;
            if let Some((doc, revision)) = changed {
                ctx.emit(Event::DocChanged { doc, revision, source: ctx.source });
            }
            let mut v = summary(ctx, json!({"mode": mode.as_str()}))?;
            v["edgeFill"] = filled;
            Ok(v)
        })
        .register();

    r.function("select", "refineEdge", "Refine the selection's edge (radius, smooth, feather, contrast, shiftEdge)")
        .description("Runs PhotoCraft `select.refineEdge` with output `selection` (other outputs belong to `edit`). Params: radius, smartRadius, smooth 0..100, feather px, contrast 0..100, shiftEdge -100..100, decontaminate, amount, sampleAllLayers, refineStrokes.")
        .params(json!({"type":"object","properties":{"radius":{"type":"number"},"smartRadius":{"type":"boolean"},"smooth":{"type":"number"},"feather":{"type":"number"},"contrast":{"type":"number"},"shiftEdge":{"type":"number"}}}))
        .effects(Revertible)
        .handler(|ctx, params| {
            let mut params = if params.is_null() { json!({}) } else { params };
            let obj = params.as_object_mut().ok_or_else(|| ApiError::bad_params("params must be an object"))?;
            match obj.get("output").and_then(Value::as_str) {
                None | Some("selection") => {}
                Some(o) => return Err(ApiError::bad_params(format!("output `{o}` is not a selection; select.refineEdge only refines the selection"))),
            }
            obj.insert("output".into(), json!("selection"));
            let out = run_command(ctx, "select.refineEdge", params)?;
            summary(ctx, out)
        })
        .register();

    r.function("select", "toMask", "The document's selection as a mask handle (document size)")
        .description("Copies the open document's selection into the handle arena: {mask: \"mask:N\", width, height, bounds [x,y,w,h], coverage}. Fails with `not_found` when nothing is selected.")
        .params(json!({"type":"object","properties":{}}))
        .effects(ReadOnly)
        .typed(|ctx, _: Empty| {
            let (w, h, sel) = ctx.with_session(|h| read_selection(h))?;
            let sel = sel.ok_or_else(|| ApiError::not_found("the document has no selection"))?;
            let bounds = mask::bbox(&sel, w as usize, h as usize);
            let coverage = mask::coverage(&sel);
            let id = ctx.shared.handles.put_mask(MaskBuf::new(w, h, sel)?)?;
            Ok(json!({"mask": id.to_string(), "width": w, "height": h, "bounds": bounds, "coverage": coverage}))
        })
        .register();

    r.function("select", "fromMask", "Make a mask handle the document's selection (replace/add/subtract/intersect, feather)")
        .description("Loads `mask` (mask:N; resampled to the document size if it differs) as the selection, combined with the current one by `mode` (default replace), feathered by `feather` px. One undoable history step (\"Load Selection\"); later edit/adjustment calls apply to it. An empty result deselects. Returns {selected, coverage, bounds}.")
        .params(json!({"type":"object","required":["mask"],"properties":{"mask":{"type":"string"},"mode":{"enum":["replace","add","subtract","intersect"]},"feather":{"type":"number","minimum":0,"maximum":250}}}))
        .effects(Revertible)
        .typed(|ctx, p: FromMask| {
            let m = ctx.shared.handles.mask_str(&p.mask)?;
            if !(p.feather.is_finite() && (0.0..=250.0).contains(&p.feather)) {
                return Err(ApiError::bad_params("feather must be 0..250 px"));
            }
            let changed = ctx.with_session(|h| {
                let (w, ht, current) = read_selection(h)?;
                let new = mask::resample(&m.alpha, m.width as usize, m.height as usize, w as usize, ht as usize);
                let mut out = mask::combine(current.as_deref(), &new, p.mode);
                mask::feather(&mut out, w as usize, ht as usize, p.feather);
                write_selection(h, "Load Selection", w, ht, &out)?;
                Ok(revision(h))
            })?;
            if let Some((doc, revision)) = changed {
                ctx.emit(Event::DocChanged { doc, revision, source: ctx.source });
            }
            summary(ctx, json!({"mode": p.mode.as_str()}))
        })
        .register();
}

/// How far past PhotoCraft's sky mask the edge may grow, as a share of the long side.
const SKY_REACH: f32 = 0.015;
/// CIELAB ΔE76 to the sky's mean colour that still counts as sky, and the ramp after it.
const SKY_TOLERANCE: f32 = 22.0;
const SKY_SOFTNESS: f32 = 10.0;

/// `sky` (GRAY8 coverage) grown into the sky-coloured pixels of `rgba` next to it, feathered
/// 1.5 px. Returns the completed mask and the coverage it added (0..1).
pub fn complete_sky_edge(sky: &[u8], rgba: &[u8], w: usize, h: usize) -> (Vec<u8>, f64) {
    let lab_of = |i: usize| imagelab_cv::color::rgb_to_lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]);
    let (mut sum, mut n) = ((0f64, 0f64, 0f64), 0f64);
    for (i, &c) in sky.iter().enumerate() {
        if c >= 128 {
            let (l, a, b) = lab_of(i);
            sum = (sum.0 + l as f64, sum.1 + a as f64, sum.2 + b as f64);
            n += 1.0;
        }
    }
    if n == 0.0 {
        return (sky.to_vec(), 0.0);
    }
    let mean = ((sum.0 / n) as f32, (sum.1 / n) as f32, (sum.2 / n) as f32);
    let reach = (w.max(h) as f32 * SKY_REACH).max(4.0);
    let near = imagelab_cv::morph::grow_buf(sky, w, h, reach);
    let mut out = sky.to_vec();
    for i in 0..out.len() {
        if out[i] == 255 || near[i] == 0 {
            continue;
        }
        let (l, a, b) = lab_of(i);
        let de = ((l - mean.0).powi(2) + (a - mean.1).powi(2) + (b - mean.2).powi(2)).sqrt();
        let like = ((SKY_TOLERANCE + SKY_SOFTNESS - de) / SKY_SOFTNESS).clamp(0.0, 1.0);
        let band = (like * near[i] as f32).round() as u8;
        out[i] = out[i].max(band);
    }
    mask::feather(&mut out, w, h, 1.5);
    // Feathering must not eat into the sky PhotoCraft found.
    for (o, &s) in out.iter_mut().zip(sky) {
        *o = (*o).max(s);
    }
    let added = (mask::coverage(&out) - mask::coverage(sky)).max(0.0);
    (out, added)
}

#[cfg(test)]
mod sky_edge_tests {
    use super::complete_sky_edge;

    /// A 100×100 picture (reach 4 px): blue sky rows 0..40, a band of slightly lighter sky rows 40..44 that
    /// PhotoCraft's mask missed, then brown ground. The completed mask covers the band, not the
    /// ground.
    #[test]
    fn the_edge_grows_into_sky_coloured_pixels_only() {
        let (w, h) = (100usize, 100usize);
        let mut rgba = vec![0u8; w * h * 4];
        let mut sky = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let px: [u8; 4] = if y < 40 {
                    sky[i] = 255;
                    [90, 140, 230, 255]
                } else if y < 44 {
                    [120, 165, 240, 255]
                } else {
                    [110, 80, 50, 255]
                };
                rgba[i * 4..i * 4 + 4].copy_from_slice(&px);
            }
        }
        let (out, added) = complete_sky_edge(&sky, &rgba, w, h);
        assert!(added > 0.03, "the band was added: {added}");
        assert!(out[42 * w + 50] > 200, "inside the band: {}", out[42 * w + 50]);
        assert!(out[60 * w + 50] == 0, "the ground stays out: {}", out[60 * w + 50]);
        assert!(out.iter().zip(&sky).all(|(o, s)| o >= s), "never less than PhotoCraft's sky");
    }
}
