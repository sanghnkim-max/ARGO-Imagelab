//! matchTemplate (normalized cross-correlation, coarse to fine) and inpaint (PatchMatch).

use std::sync::atomic::Ordering;

use imagelab_api::{ApiError, ApiResult, Ctx};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::geom::{Interp, crop_buf, resize_buf};
use crate::raster::{gray_of, load, load_img, load_mask, put_img, round4, same_size};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TmMethod {
    #[default]
    #[serde(alias = "tm_ccoeff_normed", alias = "ccoeffNormed", alias = "zncc")]
    CcoeffNormed,
    #[serde(alias = "tm_ccorr_normed", alias = "ccorrNormed")]
    CcorrNormed,
    #[serde(alias = "tm_sqdiff_normed", alias = "sqdiffNormed")]
    SqdiffNormed,
}

/// Best match of `t` (tw×th) in `img` (iw×ih) over positions x in `xs`, y in `ys` (inclusive
/// ranges). Returns (x, y, score); for sqdiff lower is better, otherwise higher.
#[allow(clippy::too_many_arguments)]
fn search(img: &[u8], iw: usize, t: &[u8], tw: usize, th: usize, xs: (usize, usize), ys: (usize, usize), m: TmMethod) -> (usize, usize, f64) {
    let n = (tw * th) as f64;
    let st: f64 = t.iter().map(|&v| v as f64).sum();
    let st2: f64 = t.iter().map(|&v| (v as f64).powi(2)).sum();
    let better = |a: f64, b: f64| if m == TmMethod::SqdiffNormed { a < b } else { a > b };
    let worst = if m == TmMethod::SqdiffNormed { f64::INFINITY } else { f64::NEG_INFINITY };
    (ys.0..=ys.1)
        .into_par_iter()
        .map(|y| {
            let mut best = (xs.0, y, worst);
            // Running window sums along the row.
            for x in xs.0..=xs.1 {
                let (mut si, mut si2, mut cross) = (0u64, 0u64, 0u64);
                for ty in 0..th {
                    let row = &img[(y + ty) * iw + x..(y + ty) * iw + x + tw];
                    let trow = &t[ty * tw..(ty + 1) * tw];
                    for (&a, &b) in row.iter().zip(trow) {
                        let a = a as u64;
                        si += a;
                        si2 += a * a;
                        cross += a * b as u64;
                    }
                }
                let (si, si2, cross) = (si as f64, si2 as f64, cross as f64);
                let score = match m {
                    TmMethod::CcoeffNormed => {
                        let den = ((st2 - st * st / n) * (si2 - si * si / n)).max(0.0).sqrt();
                        if den < 1e-6 { 0.0 } else { (cross - st * si / n) / den }
                    }
                    TmMethod::CcorrNormed => {
                        let den = (st2 * si2).sqrt();
                        if den < 1e-6 { 0.0 } else { cross / den }
                    }
                    TmMethod::SqdiffNormed => {
                        let den = (st2 * si2).sqrt();
                        if den < 1e-6 { if si2 + st2 < 1e-6 { 0.0 } else { 1.0 } } else { (si2 - 2.0 * cross + st2) / den }
                    }
                };
                if better(score, best.2) {
                    best = (x, y, score);
                }
            }
            best
        })
        .reduce(|| (0, 0, worst), |a, b| if better(b.2, a.2) || (b.2 == a.2 && (b.1, b.0) < (a.1, a.0)) { b } else { a })
}

/// Largest cost (template pixels × positions) a single search level may take.
const MAX_LEVEL_COST: f64 = 4e8;

pub fn match_template_buf(img: &[u8], iw: usize, ih: usize, t: &[u8], tw: usize, th: usize, m: TmMethod) -> ApiResult<(usize, usize, f64, usize)> {
    if tw > iw || th > ih {
        return Err(ApiError::bad_params("the template must not be larger than the image"));
    }
    let positions = |f: usize| ((iw / f).saturating_sub(tw / f) + 1) as f64 * ((ih / f).saturating_sub(th / f) + 1) as f64;
    let mut f = 1usize;
    while positions(f) * ((tw / f) * (th / f)) as f64 > MAX_LEVEL_COST && tw / (f * 2) >= 8 && th / (f * 2) >= 8 {
        f *= 2;
    }
    if positions(f) * ((tw / f) * (th / f)) as f64 > MAX_LEVEL_COST * 8.0 {
        return Err(ApiError::too_large("template too small for this image size; crop the search area or resize both first"));
    }
    if f == 1 {
        let (x, y, s) = search(img, iw, t, tw, th, (0, iw - tw), (0, ih - th), m);
        return Ok((x, y, s, 1));
    }
    let (ciw, cih, ctw, cth) = (iw / f, ih / f, tw / f, th / f);
    let ci = resize_buf(img, iw, ih, 1, ciw, cih, Interp::Area);
    let ct = resize_buf(t, tw, th, 1, ctw, cth, Interp::Area);
    let (cx, cy, _) = search(&ci, ciw, &ct, ctw, cth, (0, ciw - ctw), (0, cih - cth), m);
    let r = 2 * f;
    let xs = ((cx * f).saturating_sub(r), (cx * f + r).min(iw - tw));
    let ys = ((cy * f).saturating_sub(r), (cy * f + r).min(ih - th));
    let (x, y, s) = search(img, iw, t, tw, th, xs, ys, m);
    Ok((x, y, s, f))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchTemplate {
    #[serde(alias = "image")]
    img: String,
    #[serde(alias = "templ")]
    template: String,
    #[serde(default)]
    method: TmMethod,
}

pub fn match_template(ctx: &Ctx, p: MatchTemplate) -> ApiResult<Value> {
    let a = load(ctx, &p.img)?;
    let b = load(ctx, &p.template)?;
    let (ga, gb) = (gray_of(&a), gray_of(&b));
    let (x, y, score, f) = match_template_buf(&ga, a.w(), a.h(), &gb, b.w(), b.h(), p.method)?;
    Ok(json!({
        "x": x, "y": y, "width": b.w(), "height": b.h(), "score": round4(score),
        "center": [x as f64 + b.w() as f64 / 2.0, y as f64 + b.h() as f64 / 2.0],
        "method": format!("{:?}", p.method), "pyramidFactor": f,
        "note": if p.method == TmMethod::SqdiffNormed { "lower score = better match" } else { "higher score = better match (1 = perfect)" },
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inpaint {
    img: String,
    mask: String,
    #[serde(default, alias = "inpaintRadius")]
    radius: Option<usize>,
    #[serde(default)]
    method: Option<String>,
}

/// Largest working region (hole bounding box plus context) inpaint processes.
pub const INPAINT_MAX_PIXELS: usize = 16_000_000;

pub fn inpaint(ctx: &Ctx, p: Inpaint) -> ApiResult<Value> {
    let img = load_img(ctx, &p.img)?;
    let m = load_mask(ctx, &p.mask)?;
    let (w, h) = (img.width as usize, img.height as usize);
    same_size((w, h), (m.width as usize, m.height as usize), "inpaint")?;
    let radius = p.radius.unwrap_or(3);
    if !(1..=8).contains(&radius) {
        return Err(ApiError::bad_params("radius (patch radius) must be in 1..=8"));
    }
    let patch = match p.method.as_deref().unwrap_or("patchmatch") {
        "patchmatch" | "patch" | "content-aware" => true,
        "membrane" | "telea" | "ns" | "diffusion" => false,
        _ => return Err(ApiError::bad_params("method must be patchmatch or membrane (telea/ns are accepted as membrane)")),
    };
    // Hole bounding box.
    let (mut x0, mut y0, mut x1, mut y1, mut any) = (w, h, 0usize, 0usize, false);
    for (i, &v) in m.alpha.iter().enumerate() {
        if v >= 128 {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            any = true;
        }
    }
    if !any {
        let mut r = put_img(ctx, w, h, img.rgba.clone())?;
        r["note"] = json!("the mask is empty; nothing to fill");
        return Ok(r);
    }
    let ext = (x1 - x0).max(y1 - y0) + 1;
    let margin = (2 * ext).max(48);
    let (rx0, ry0) = (x0.saturating_sub(margin), y0.saturating_sub(margin));
    let (rx1, ry1) = ((x1 + margin + 1).min(w), (y1 + margin + 1).min(h));
    let (rw, rh) = (rx1 - rx0, ry1 - ry0);
    if rw * rh > INPAINT_MAX_PIXELS {
        return Err(ApiError::too_large(format!("the hole and its context span {rw}x{rh} px; inpaint at most {INPAINT_MAX_PIXELS} px (resize first)")));
    }
    let region = crop_buf(&img.rgba, w, 4, rx0, ry0, rw, rh);
    let hmask = crop_buf(&m.alpha, w, 1, rx0, ry0, rw, rh);
    let hole: Vec<bool> = hmask.iter().map(|&v| v >= 128).collect();
    let rgb: Vec<f32> = region.chunks(4).flat_map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]).collect();
    let cancel = ctx.cancel.clone();
    let mut used = if patch { "patchmatch" } else { "membrane" };
    let filled = if patch {
        let params = lightcraft_inpaint::CompleteParams { patch_radius: radius, ..Default::default() };
        match lightcraft_inpaint::complete_with(rw, rh, 3, &rgb, &hole, &params, &|| cancel.load(Ordering::Relaxed)) {
            Ok(Some(v)) => v,
            Ok(None) => {
                used = "membrane";
                lightcraft_inpaint::membrane_fill(rw, rh, 3, &rgb, &hole)
            }
            Err(_) => return Err(ApiError::cancelled()),
        }
    } else {
        lightcraft_inpaint::membrane_fill(rw, rh, 3, &rgb, &hole)
    };
    if filled.len() != rgb.len() {
        return Err(ApiError::internal("inpaint returned a buffer of the wrong size"));
    }
    let mut out = img.rgba.clone();
    for ry in 0..rh {
        for rx in 0..rw {
            let i = ry * rw + rx;
            if !hole[i] {
                continue;
            }
            let o = ((ry0 + ry) * w + rx0 + rx) * 4;
            for c in 0..3 {
                out[o + c] = (filled[i * 3 + c] * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    let mut r = put_img(ctx, w, h, out)?;
    r["method"] = json!(used);
    r["region"] = json!({"x": rx0, "y": ry0, "w": rw, "h": rh});
    Ok(r)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn textured(w: usize, h: usize) -> Vec<u8> {
        (0..w * h).map(|i| {
            let (x, y) = (i % w, i / w);
            ((x * 37 + y * 91 + (x * y) % 23) % 256) as u8
        }).collect()
    }

    #[test]
    fn finds_template_exact_and_pyramid() {
        let (w, h) = (120, 90);
        let img = textured(w, h);
        let (tx, ty, tw, th) = (70, 33, 24, 20);
        let t = crop_buf(&img, w, 1, tx, ty, tw, th);
        for m in [TmMethod::CcoeffNormed, TmMethod::CcorrNormed, TmMethod::SqdiffNormed] {
            let (x, y, s, _) = match_template_buf(&img, w, h, &t, tw, th, m).unwrap();
            assert_eq!((x, y), (tx, ty), "{m:?}");
            if m == TmMethod::SqdiffNormed { assert!(s < 1e-9) } else { assert!(s > 0.999) }
        }
        // Large enough to use the pyramid: a smooth blob pattern.
        let (w, h) = (1200, 900);
        let img: Vec<u8> = (0..w * h).map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            (127.0 + 60.0 * (x / 37.0).sin() * (y / 23.0).cos() + 60.0 * ((x + y) / 71.0).sin()) as u8
        }).collect();
        let (tx, ty, tw, th) = (803, 411, 160, 120);
        let t = crop_buf(&img, w, 1, tx, ty, tw, th);
        let (x, y, s, f) = match_template_buf(&img, w, h, &t, tw, th, TmMethod::CcoeffNormed).unwrap();
        assert!(f > 1, "pyramid used");
        assert_eq!((x, y), (tx, ty));
        assert!(s > 0.999);
        assert!(match_template_buf(&t, tw, th, &img, w, h, TmMethod::CcoeffNormed).is_err());
    }
}
