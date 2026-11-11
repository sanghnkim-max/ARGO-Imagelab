//! Mask ops: maskFromColor, invertMask, combineMasks, maskBounds, applyMask, composite.

use imagelab_api::{ApiError, ApiResult, Ctx};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::color::{rgb_to_hsv, rgb_to_lab};
use crate::raster::{coverage, load_gray, load_img, load_mask, put_img, put_mask, round4, same_size};

#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub enum Color {
    Hex(String),
    Rgb(Vec<f64>),
}

impl Color {
    pub fn rgb(&self) -> ApiResult<[u8; 3]> {
        match self {
            Color::Rgb(v) if v.len() >= 3 && v.iter().take(3).all(|c| (0.0..=255.0).contains(c)) => Ok([v[0].round() as u8, v[1].round() as u8, v[2].round() as u8]),
            Color::Hex(s) => {
                let t = s.trim().trim_start_matches('#');
                if !t.is_ascii() {
                    return Err(ApiError::bad_params(format!("color `{s}`: use \"#rrggbb\" or [r, g, b]")));
                }
                let parse = |h: &str| u8::from_str_radix(h, 16).ok();
                let rgb = match t.len() {
                    6 | 8 => (parse(&t[0..2]), parse(&t[2..4]), parse(&t[4..6])),
                    3 => {
                        let d = |i: usize| parse(&t[i..i + 1]).map(|v| v * 17);
                        (d(0), d(1), d(2))
                    }
                    _ => (None, None, None),
                };
                match rgb {
                    (Some(r), Some(g), Some(b)) if t.is_ascii() => Ok([r, g, b]),
                    _ => Err(ApiError::bad_params(format!("color `{s}`: use \"#rrggbb\" or [r, g, b]"))),
                }
            }
            Color::Rgb(_) => Err(ApiError::bad_params("color: [r, g, b] with values 0..=255")),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskFromColor {
    img: String,
    color: Color,
    #[serde(default)]
    tolerance: Option<f32>,
    #[serde(default)]
    softness: f32,
    #[serde(default)]
    space: Option<String>,
}

/// Coverage of pixels close to `color`: 255 within `tol`, ramping to 0 at `tol + soft`.
pub fn mask_from_color_buf(rgba: &[u8], color: [u8; 3], space: &str, tol: f32, soft: f32) -> ApiResult<Vec<u8>> {
    let ramp = move |d: f32| -> u8 {
        if d <= tol {
            255
        } else if soft > 0.0 && d < tol + soft {
            ((1.0 - (d - tol) / soft) * 255.0).round() as u8
        } else {
            0
        }
    };
    let [r0, g0, b0] = color;
    Ok(match space {
        "rgb" => rgba
            .par_chunks(4)
            .map(|p| {
                let d = ((p[0] as f32 - r0 as f32).powi(2) + (p[1] as f32 - g0 as f32).powi(2) + (p[2] as f32 - b0 as f32).powi(2)).sqrt();
                ramp(d)
            })
            .collect(),
        "lab" => {
            let (l0, a0, bb0) = rgb_to_lab(r0, g0, b0);
            rgba.par_chunks(4)
                .map(|p| {
                    let (l, a, b) = rgb_to_lab(p[0], p[1], p[2]);
                    ramp(((l - l0).powi(2) + (a - a0).powi(2) + (b - bb0).powi(2)).sqrt())
                })
                .collect()
        }
        "hsv" | "hue" => {
            let (h0, s0, _) = rgb_to_hsv(r0, g0, b0);
            if s0 < 0.1 {
                return Err(ApiError::bad_params("the color is (nearly) gray, so it has no hue; use space lab or rgb"));
            }
            rgba.par_chunks(4)
                .map(|p| {
                    let (h, s, v) = rgb_to_hsv(p[0], p[1], p[2]);
                    if s < 0.1 || v < 0.1 {
                        return 0;
                    }
                    let d = (h - h0).abs();
                    ramp(d.min(360.0 - d))
                })
                .collect()
        }
        _ => return Err(ApiError::bad_params("space must be lab, rgb or hsv")),
    })
}

pub fn mask_from_color(ctx: &Ctx, p: MaskFromColor) -> ApiResult<Value> {
    let img = load_img(ctx, &p.img)?;
    let space = p.space.as_deref().unwrap_or("lab");
    let tol = p.tolerance.unwrap_or(match space {
        "rgb" => 40.0,
        "hsv" | "hue" => 15.0,
        _ => 15.0,
    });
    if !(0.0..=442.0).contains(&tol) || !(0.0..=442.0).contains(&p.softness) {
        return Err(ApiError::bad_params("tolerance and softness must be in 0..=442"));
    }
    let out = mask_from_color_buf(&img.rgba, p.color.rgb()?, space, tol, p.softness)?;
    put_mask(ctx, img.width as usize, img.height as usize, out)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OneMask {
    #[serde(alias = "img")]
    mask: String,
}

pub fn invert_mask(ctx: &Ctx, p: OneMask) -> ApiResult<Value> {
    let (w, h, g) = load_gray(ctx, &p.mask)?;
    put_mask(ctx, w, h, g.par_iter().map(|&v| 255 - v).collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CombineOp {
    #[serde(alias = "or", alias = "add", alias = "max")]
    Union,
    #[serde(alias = "and", alias = "intersection", alias = "min")]
    Intersect,
    #[serde(alias = "minus", alias = "difference", alias = "sub")]
    Subtract,
    Xor,
    #[serde(alias = "mul")]
    Multiply,
}

pub fn combine_buf(a: &[u8], b: &[u8], op: CombineOp) -> Vec<u8> {
    let mul = |x: u8, y: u8| ((x as u32 * y as u32 + 127) / 255) as u8;
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| match op {
            CombineOp::Union => x.max(y),
            CombineOp::Intersect => x.min(y),
            CombineOp::Subtract => x.min(255 - y),
            CombineOp::Xor => x.min(255 - y).max(y.min(255 - x)),
            CombineOp::Multiply => mul(x, y),
        })
        .collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CombineMasks {
    a: String,
    b: String,
    op: CombineOp,
}

pub fn combine_masks(ctx: &Ctx, p: CombineMasks) -> ApiResult<Value> {
    let (w, h, a) = load_gray(ctx, &p.a)?;
    let (w2, h2, b) = load_gray(ctx, &p.b)?;
    same_size((w, h), (w2, h2), "combineMasks")?;
    put_mask(ctx, w, h, combine_buf(&a, &b, p.op))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskBounds {
    #[serde(alias = "img")]
    mask: String,
    #[serde(default = "zero")]
    threshold: u8,
}

fn zero() -> u8 {
    0
}

pub fn mask_bounds(ctx: &Ctx, p: MaskBounds) -> ApiResult<Value> {
    let (w, h, g) = load_gray(ctx, &p.mask)?;
    let (mut x0, mut y0, mut x1, mut y1, mut n) = (usize::MAX, usize::MAX, 0usize, 0usize, 0u64);
    for (i, &v) in g.iter().enumerate() {
        if v > p.threshold {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            n += 1;
        }
    }
    if n == 0 {
        return Ok(json!({"empty": true, "area": 0, "coverage": 0.0, "width": w, "height": h}));
    }
    Ok(json!({
        "empty": false, "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
        "area": n, "coverage": round4(coverage(&g)), "width": w, "height": h,
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplyMask {
    img: String,
    mask: String,
    #[serde(default)]
    invert: bool,
}

pub fn apply_mask(ctx: &Ctx, p: ApplyMask) -> ApiResult<Value> {
    let img = load_img(ctx, &p.img)?;
    let m = load_mask(ctx, &p.mask)?;
    same_size((img.width as usize, img.height as usize), (m.width as usize, m.height as usize), "applyMask")?;
    let mut out = img.rgba.clone();
    out.par_chunks_mut(4).zip(m.alpha.par_iter()).for_each(|(px, &a)| {
        let a = if p.invert { 255 - a } else { a };
        px[3] = ((px[3] as u32 * a as u32 + 127) / 255) as u8;
    });
    put_img(ctx, img.width as usize, img.height as usize, out)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Composite {
    bg: String,
    fg: String,
    #[serde(default)]
    mask: Option<String>,
    #[serde(default)]
    x: i64,
    #[serde(default)]
    y: i64,
    #[serde(default = "one")]
    opacity: f32,
}

fn one() -> f32 {
    1.0
}

/// Porter-Duff "over" of `fg` at (`ox`, `oy`) onto `bg` (straight alpha), weighted by `mask`.
#[allow(clippy::too_many_arguments)]
pub fn composite_buf(bg: &[u8], bw: usize, _bh: usize, fg: &[u8], fw: usize, fh: usize, mask: Option<&[u8]>, ox: i64, oy: i64, opacity: f32) -> Vec<u8> {
    let mut out = bg.to_vec();
    out.par_chunks_mut(bw * 4).enumerate().for_each(|(y, row)| {
        let fy = y as i64 - oy;
        if fy < 0 || fy >= fh as i64 {
            return;
        }
        let fy = fy as usize;
        for x in 0..bw {
            let fx = x as i64 - ox;
            if fx < 0 || fx >= fw as i64 {
                continue;
            }
            let fi = fy * fw + fx as usize;
            let f = &fg[fi * 4..fi * 4 + 4];
            let m = mask.map_or(1.0, |m| m[fi] as f32 / 255.0);
            let fa = f[3] as f32 / 255.0 * m * opacity;
            if fa <= 0.0 {
                continue;
            }
            let b = &mut row[x * 4..x * 4 + 4];
            let ba = b[3] as f32 / 255.0;
            let oa = fa + ba * (1.0 - fa);
            for c in 0..3 {
                let v = (f[c] as f32 * fa + b[c] as f32 * ba * (1.0 - fa)) / oa.max(1e-6);
                b[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            b[3] = (oa * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    });
    out
}

pub fn composite(ctx: &Ctx, p: Composite) -> ApiResult<Value> {
    if !(0.0..=1.0).contains(&p.opacity) {
        return Err(ApiError::bad_params("opacity must be in 0..=1"));
    }
    let bg = load_img(ctx, &p.bg)?;
    let fg = load_img(ctx, &p.fg)?;
    let mask = match &p.mask {
        Some(m) => {
            let m = load_mask(ctx, m)?;
            same_size((fg.width as usize, fg.height as usize), (m.width as usize, m.height as usize), "composite mask (must match fg)")?;
            Some(m)
        }
        None => None,
    };
    let (bw, bh) = (bg.width as usize, bg.height as usize);
    let out = composite_buf(&bg.rgba, bw, bh, &fg.rgba, fg.width as usize, fg.height as usize, mask.as_ref().map(|m| m.alpha.as_slice()), p.x, p.y, p.opacity);
    put_img(ctx, bw, bh, out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn combine_truth_tables() {
        let a = [0u8, 0, 255, 255];
        let b = [0u8, 255, 0, 255];
        assert_eq!(combine_buf(&a, &b, CombineOp::Union), vec![0, 255, 255, 255]);
        assert_eq!(combine_buf(&a, &b, CombineOp::Intersect), vec![0, 0, 0, 255]);
        assert_eq!(combine_buf(&a, &b, CombineOp::Subtract), vec![0, 0, 255, 0]);
        assert_eq!(combine_buf(&a, &b, CombineOp::Xor), vec![0, 255, 255, 0]);
        assert_eq!(combine_buf(&a, &b, CombineOp::Multiply), vec![0, 0, 0, 255]);
    }

    #[test]
    fn color_masks() {
        let px = [255u8, 0, 0, 255, 250, 10, 5, 255, 0, 0, 255, 255, 128, 128, 128, 255];
        for space in ["rgb", "lab", "hsv"] {
            let m = mask_from_color_buf(&px, [255, 0, 0], space, if space == "rgb" { 30.0 } else { 10.0 }, 0.0).unwrap();
            assert_eq!(m, vec![255, 255, 0, 0], "{space}");
        }
        assert_eq!(Color::Hex("#f00".into()).rgb().unwrap(), [255, 0, 0]);
        assert_eq!(Color::Hex("00ff80".into()).rgb().unwrap(), [0, 255, 128]);
        assert!(Color::Hex("#zz0000".into()).rgb().is_err());
        assert!(Color::Rgb(vec![1.0, 2.0]).rgb().is_err());
    }

    #[test]
    fn composite_over() {
        let bg = vec![0u8, 0, 255, 255, 0, 0, 255, 255];
        let fg = vec![255u8, 0, 0, 255];
        let out = composite_buf(&bg, 2, 1, &fg, 1, 1, None, 1, 0, 1.0);
        assert_eq!(out, vec![0, 0, 255, 255, 255, 0, 0, 255]);
        let half = composite_buf(&bg, 2, 1, &fg, 1, 1, Some(&[128]), 0, 0, 1.0);
        assert!(half[0] > 120 && half[0] < 135 && half[2] > 120 && half[2] < 135 && half[3] == 255);
        // Off-canvas placement leaves the background alone.
        assert_eq!(composite_buf(&bg, 2, 1, &fg, 1, 1, None, 5, 5, 1.0), bg);
    }
}
