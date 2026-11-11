//! Geometry: resize, crop, rotate, flip, warpAffine, warpPerspective. Work on images and masks.

use imagelab_api::handles::MAX_SIDE;
use imagelab_api::{ApiError, ApiResult, Ctx};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;

use crate::raster::{load, put};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Interp {
    #[serde(alias = "inter_nearest", alias = "nn")]
    Nearest,
    #[serde(alias = "bilinear", alias = "inter_linear")]
    Linear,
    #[serde(alias = "inter_area", alias = "box")]
    Area,
    #[serde(alias = "bicubic", alias = "inter_cubic", alias = "lanczos", alias = "inter_lanczos4")]
    Cubic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Border {
    /// Fill with `borderValue` (default transparent black).
    #[default]
    #[serde(alias = "border_constant")]
    Constant,
    #[serde(alias = "border_replicate", alias = "clamp", alias = "edge")]
    Replicate,
    #[serde(alias = "border_reflect", alias = "border_reflect_101", alias = "mirror")]
    Reflect,
}

fn cubic_w(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let w0 = ((A * (t + 1.0) - 5.0 * A) * (t + 1.0) + 8.0 * A) * (t + 1.0) - 4.0 * A;
    let w1 = ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0;
    let w2 = ((A + 2.0) * (1.0 - t) - (A + 3.0)) * (1.0 - t) * (1.0 - t) + 1.0;
    [w0, w1, w2, 1.0 - w0 - w1 - w2]
}

/// Per output index: the source taps `(index, weight)`.
fn contribs(src_n: usize, dst_n: usize, interp: Interp) -> Vec<Vec<(usize, f32)>> {
    let scale = src_n as f64 / dst_n as f64;
    let clamp = |i: i64| i.clamp(0, src_n as i64 - 1) as usize;
    (0..dst_n)
        .map(|i| match interp {
            Interp::Nearest => vec![(clamp((i as f64 * scale).floor() as i64), 1.0)],
            Interp::Area if scale > 1.0 => {
                let a = i as f64 * scale;
                let b = a + scale;
                let mut v = Vec::with_capacity(scale.ceil() as usize + 1);
                let mut j = a.floor() as i64;
                while (j as f64) < b {
                    let lo = (j as f64).max(a);
                    let hi = ((j + 1) as f64).min(b);
                    if hi > lo {
                        v.push((clamp(j), ((hi - lo) / scale) as f32));
                    }
                    j += 1;
                }
                v
            }
            Interp::Linear | Interp::Area => {
                let f = (i as f64 + 0.5) * scale - 0.5;
                let x0 = f.floor();
                let t = (f - x0) as f32;
                let x0 = x0 as i64;
                vec![(clamp(x0), 1.0 - t), (clamp(x0 + 1), t)]
            }
            Interp::Cubic => {
                let f = (i as f64 + 0.5) * scale - 0.5;
                let x0 = f.floor();
                let w = cubic_w((f - x0) as f32);
                let x0 = x0 as i64;
                (0..4).map(|k| (clamp(x0 - 1 + k as i64), w[k])).collect()
            }
        })
        .collect()
}

fn has_translucency(src: &[u8], ch: usize) -> bool {
    ch == 4 && src.par_chunks(4).any(|p| p[3] != 255)
}

/// Resize an interleaved buffer. RGBA with transparency is filtered premultiplied.
pub fn resize_buf(src: &[u8], w: usize, h: usize, ch: usize, nw: usize, nh: usize, interp: Interp) -> Vec<u8> {
    let cx = contribs(w, nw, interp);
    let cy = contribs(h, nh, interp);
    let premul = has_translucency(src, ch);
    let stride = w * ch;
    let mut out = vec![0u8; nw * nh * ch];
    out.par_chunks_mut(nw * ch).enumerate().for_each(|(y, orow)| {
        let mut row = vec![0f32; stride];
        for &(sy, wy) in &cy[y] {
            let s = &src[sy * stride..(sy + 1) * stride];
            if premul {
                for (d, p) in row.chunks_mut(4).zip(s.chunks(4)) {
                    let a = p[3] as f32 / 255.0;
                    d[0] += wy * p[0] as f32 * a;
                    d[1] += wy * p[1] as f32 * a;
                    d[2] += wy * p[2] as f32 * a;
                    d[3] += wy * p[3] as f32;
                }
            } else {
                for (d, &v) in row.iter_mut().zip(s) {
                    *d += wy * v as f32;
                }
            }
        }
        for (x, taps) in cx.iter().enumerate() {
            let mut acc = [0f32; 4];
            for &(sx, wx) in taps {
                for c in 0..ch {
                    acc[c] += wx * row[sx * ch + c];
                }
            }
            if premul && acc[3] > 0.0 {
                let k = 255.0 / acc[3];
                for v in acc.iter_mut().take(3) {
                    *v *= k;
                }
            }
            for c in 0..ch {
                orow[x * ch + c] = (acc[c] + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

fn check_out_size(w: f64, h: f64) -> ApiResult<(usize, usize)> {
    if !(w.is_finite() && h.is_finite()) || w < 1.0 || h < 1.0 || w > MAX_SIDE as f64 || h > MAX_SIDE as f64 {
        return Err(ApiError::too_large(format!("output size {w}x{h} is outside 1..={MAX_SIDE}")));
    }
    Ok((w as usize, h as usize))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resize {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default, alias = "fx")]
    scale: Option<f64>,
    #[serde(default, alias = "maxSide")]
    max_side: Option<u32>,
    #[serde(default)]
    interpolation: Option<Interp>,
}

pub fn resize(ctx: &Ctx, p: Resize) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let (w, h) = (src.w() as f64, src.h() as f64);
    let (nw, nh) = match (p.width, p.height, p.scale, p.max_side) {
        (Some(nw), Some(nh), None, None) => (nw as f64, nh as f64),
        (Some(nw), None, None, None) => (nw as f64, (h * nw as f64 / w).round()),
        (None, Some(nh), None, None) => ((w * nh as f64 / h).round(), nh as f64),
        (None, None, Some(s), None) if s > 0.0 => ((w * s).round(), (h * s).round()),
        (None, None, None, Some(m)) => {
            let s = (m as f64 / w.max(h)).min(1.0);
            ((w * s).round(), (h * s).round())
        }
        _ => return Err(ApiError::bad_params("give width and/or height, or scale (> 0), or maxSide")),
    };
    let (nw, nh) = check_out_size(nw.max(1.0), nh.max(1.0))?;
    let interp = p.interpolation.unwrap_or(if nw * nh < src.w() * src.h() { Interp::Area } else { Interp::Linear });
    let out = resize_buf(src.data(), src.w(), src.h(), src.ch(), nw, nh, interp);
    put(ctx, nw, nh, src.ch(), out)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Crop {
    #[serde(alias = "mask")]
    img: String,
    x: i64,
    y: i64,
    #[serde(alias = "width")]
    w: i64,
    #[serde(alias = "height")]
    h: i64,
}

pub fn crop_buf(src: &[u8], w: usize, ch: usize, x: usize, y: usize, cw: usize, chh: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(cw * chh * ch);
    for row in y..y + chh {
        out.extend_from_slice(&src[(row * w + x) * ch..(row * w + x + cw) * ch]);
    }
    out
}

pub fn crop(ctx: &Ctx, p: Crop) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let (w, h) = (src.w() as i64, src.h() as i64);
    let x0 = p.x.clamp(0, w);
    let y0 = p.y.clamp(0, h);
    let x1 = p.x.saturating_add(p.w).clamp(0, w);
    let y1 = p.y.saturating_add(p.h).clamp(0, h);
    if x1 <= x0 || y1 <= y0 {
        return Err(ApiError::bad_params(format!("crop rect {},{} {}x{} does not overlap the {w}x{h} image", p.x, p.y, p.w, p.h)));
    }
    let (cw, chh) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let out = crop_buf(src.data(), src.w(), src.ch(), x0 as usize, y0 as usize, cw, chh);
    let mut r = put(ctx, cw, chh, src.ch(), out)?;
    r["rect"] = serde_json::json!({"x": x0, "y": y0, "w": cw, "h": chh});
    Ok(r)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Flip {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default)]
    mode: Option<String>,
    /// OpenCV flipCode: 0 = vertical (around the x axis), 1 = horizontal, -1 = both.
    #[serde(default, alias = "flipCode")]
    flip_code: Option<i32>,
}

pub fn flip_buf(src: &[u8], w: usize, h: usize, ch: usize, horizontal: bool, vertical: bool) -> Vec<u8> {
    let stride = w * ch;
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(stride).enumerate().for_each(|(y, orow)| {
        let sy = if vertical { h - 1 - y } else { y };
        let s = &src[sy * stride..(sy + 1) * stride];
        if horizontal {
            for x in 0..w {
                orow[x * ch..(x + 1) * ch].copy_from_slice(&s[(w - 1 - x) * ch..(w - x) * ch]);
            }
        } else {
            orow.copy_from_slice(s);
        }
    });
    out
}

pub fn flip(ctx: &Ctx, p: Flip) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let (hz, vt) = match (p.mode.as_deref(), p.flip_code) {
        (Some("horizontal" | "h" | "x" | "leftright" | "mirror"), _) | (None, Some(1)) => (true, false),
        (Some("vertical" | "v" | "y" | "updown"), _) | (None, Some(0)) => (false, true),
        (Some("both" | "hv"), _) | (None, Some(-1)) => (true, true),
        (None, None) => (true, false),
        _ => return Err(ApiError::bad_params("mode must be horizontal, vertical or both (or flipCode 1, 0, -1)")),
    };
    let out = flip_buf(src.data(), src.w(), src.h(), src.ch(), hz, vt);
    put(ctx, src.w(), src.h(), src.ch(), out)
}

/// Sample `src` at the inverse-mapped position of every output pixel.
#[allow(clippy::too_many_arguments)]
pub fn warp_buf(
    src: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    ow: usize,
    oh: usize,
    inv: &(dyn Fn(f64, f64) -> Option<(f64, f64)> + Sync),
    interp: Interp,
    border: Border,
    bval: [u8; 4],
) -> Vec<u8> {
    let premul = ch == 4;
    let fetch = |xi: i64, yi: i64| -> Option<&[u8]> {
        let (x, y) = match border {
            Border::Constant => {
                if xi < 0 || yi < 0 || xi >= w as i64 || yi >= h as i64 {
                    return None;
                }
                (xi as usize, yi as usize)
            }
            Border::Replicate => (xi.clamp(0, w as i64 - 1) as usize, yi.clamp(0, h as i64 - 1) as usize),
            Border::Reflect => (crate::raster::reflect(xi as isize, w), crate::raster::reflect(yi as isize, h)),
        };
        let i = (y * w + x) * ch;
        Some(&src[i..i + ch])
    };
    let bv: Vec<u8> = bval[..ch].to_vec();
    let mut out = vec![0u8; ow * oh * ch];
    out.par_chunks_mut(ow * ch).enumerate().for_each(|(y, orow)| {
        for x in 0..ow {
            let o = &mut orow[x * ch..(x + 1) * ch];
            let Some((sx, sy)) = inv(x as f64, y as f64) else {
                o.copy_from_slice(&bv);
                continue;
            };
            if !(sx.is_finite() && sy.is_finite()) || sx.abs() > 1e7 || sy.abs() > 1e7 {
                o.copy_from_slice(&bv);
                continue;
            }
            if interp == Interp::Nearest {
                match fetch(sx.round() as i64, sy.round() as i64) {
                    Some(p) => o.copy_from_slice(p),
                    None => o.copy_from_slice(&bv),
                }
                continue;
            }
            let (x0, y0) = (sx.floor(), sy.floor());
            let (tx, ty) = ((sx - x0) as f32, (sy - y0) as f32);
            let (x0, y0) = (x0 as i64, y0 as i64);
            let (wx, wy, n, off): ([f32; 4], [f32; 4], usize, i64) = if interp == Interp::Cubic {
                (cubic_w(tx), cubic_w(ty), 4, -1)
            } else {
                ([1.0 - tx, tx, 0.0, 0.0], [1.0 - ty, ty, 0.0, 0.0], 2, 0)
            };
            let mut acc = [0f32; 4];
            for (j, &wyj) in wy.iter().enumerate().take(n) {
                for (i, &wxi) in wx.iter().enumerate().take(n) {
                    let wgt = wxi * wyj;
                    let p = fetch(x0 + off + i as i64, y0 + off + j as i64).unwrap_or(&bv);
                    if premul {
                        let a = p[3] as f32 / 255.0;
                        acc[0] += wgt * p[0] as f32 * a;
                        acc[1] += wgt * p[1] as f32 * a;
                        acc[2] += wgt * p[2] as f32 * a;
                        acc[3] += wgt * p[3] as f32;
                    } else {
                        acc[0] += wgt * p[0] as f32;
                    }
                }
            }
            if premul && acc[3] > 0.0 {
                let k = 255.0 / acc[3];
                for v in acc.iter_mut().take(3) {
                    *v *= k;
                }
            }
            for c in 0..ch {
                o[c] = (acc[c] + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

fn interp_or_linear(i: Option<Interp>) -> Interp {
    match i {
        Some(Interp::Area) | None => Interp::Linear,
        Some(i) => i,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Rotate {
    #[serde(alias = "mask")]
    img: String,
    angle: f64,
    #[serde(default)]
    expand: bool,
    #[serde(default)]
    crop: bool,
    #[serde(default)]
    interpolation: Option<Interp>,
    #[serde(default)]
    border_mode: Border,
    #[serde(default)]
    border_value: Option<[u8; 4]>,
}

/// Exact quarter turns (counterclockwise `quarters`).
pub fn rotate90_buf(src: &[u8], w: usize, h: usize, ch: usize, quarters: u32) -> (Vec<u8>, usize, usize) {
    let q = quarters % 4;
    let (ow, oh) = if q % 2 == 1 { (h, w) } else { (w, h) };
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(ow * ch).enumerate().for_each(|(y, orow)| {
        for x in 0..ow {
            let (sx, sy) = match q {
                1 => (w - 1 - y, x),
                2 => (w - 1 - x, h - 1 - y),
                3 => (y, h - 1 - x),
                _ => (x, y),
            };
            let i = (sy * w + sx) * ch;
            orow[x * ch..(x + 1) * ch].copy_from_slice(&src[i..i + ch]);
        }
    });
    (out, ow, oh)
}

/// The largest centered rectangle of the original aspect inside a `w`×`h` image turned by `deg`.
pub fn inscribed_size(w: f64, h: f64, deg: f64) -> (f64, f64) {
    let t = deg.to_radians();
    let (c, s) = (t.cos().abs(), t.sin().abs());
    let k = (w / (w * c + h * s)).min(h / (w * s + h * c));
    ((w * k).floor().max(1.0), (h * k).floor().max(1.0))
}

pub fn rotate(ctx: &Ctx, p: Rotate) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let (w, h, ch) = (src.w(), src.h(), src.ch());
    if !p.angle.is_finite() {
        return Err(ApiError::bad_params("angle must be a finite number of degrees"));
    }
    let norm = p.angle.rem_euclid(360.0);
    let quarter = (norm / 90.0).round();
    if (norm - quarter * 90.0).abs() < 1e-9 {
        let (out, ow, oh) = rotate90_buf(src.data(), w, h, ch, quarter as u32);
        return put(ctx, ow, oh, ch, out);
    }
    let t = p.angle.to_radians();
    let (c, s) = (t.cos(), t.sin());
    let (fw, fh) = (w as f64, h as f64);
    let (ow, oh) = if p.crop {
        inscribed_size(fw, fh, p.angle)
    } else if p.expand {
        ((fw * c.abs() + fh * s.abs()).ceil(), (fw * s.abs() + fh * c.abs()).ceil())
    } else {
        (fw, fh)
    };
    let (ow, oh) = check_out_size(ow, oh)?;
    let (cx, cy) = ((fw - 1.0) / 2.0, (fh - 1.0) / 2.0);
    let (ocx, ocy) = ((ow as f64 - 1.0) / 2.0, (oh as f64 - 1.0) / 2.0);
    // Forward (OpenCV getRotationMatrix2D): d = [c s; -s c](p - center); inverse is the transpose.
    let inv = move |x: f64, y: f64| {
        let (dx, dy) = (x - ocx, y - ocy);
        Some((c * dx - s * dy + cx, s * dx + c * dy + cy))
    };
    let out = warp_buf(src.data(), w, h, ch, ow, oh, &inv, interp_or_linear(p.interpolation), p.border_mode, p.border_value.unwrap_or([0; 4]));
    put(ctx, ow, oh, ch, out)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct WarpAffine {
    #[serde(alias = "mask")]
    img: String,
    matrix: Vec<f64>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    inverse_map: bool,
    #[serde(default)]
    interpolation: Option<Interp>,
    #[serde(default)]
    border_mode: Border,
    #[serde(default)]
    border_value: Option<[u8; 4]>,
}

pub fn warp_affine(ctx: &Ctx, p: WarpAffine) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let m: [f64; 6] = p.matrix.as_slice().try_into().map_err(|_| ApiError::bad_params("matrix must have 6 numbers [a, b, tx, c, d, ty] (2x3, row-major)"))?;
    if m.iter().any(|v| !v.is_finite()) {
        return Err(ApiError::bad_params("matrix has a non-finite number"));
    }
    let inv = if p.inverse_map {
        m
    } else {
        let det = m[0] * m[4] - m[1] * m[3];
        if det.abs() < 1e-12 {
            return Err(ApiError::bad_params("matrix is singular"));
        }
        let (a, b, c, d) = (m[4] / det, -m[1] / det, -m[3] / det, m[0] / det);
        [a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5])]
    };
    let (ow, oh) = check_out_size(p.width.unwrap_or(src.w() as u32) as f64, p.height.unwrap_or(src.h() as u32) as f64)?;
    let f = move |x: f64, y: f64| Some((inv[0] * x + inv[1] * y + inv[2], inv[3] * x + inv[4] * y + inv[5]));
    let out = warp_buf(src.data(), src.w(), src.h(), src.ch(), ow, oh, &f, interp_or_linear(p.interpolation), p.border_mode, p.border_value.unwrap_or([0; 4]));
    put(ctx, ow, oh, src.ch(), out)
}

/// Solve `a x = b` (n ≤ 8) by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<[f64; 9]>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        for r in 0..n {
            if r != col {
                let f = a[r][col] / a[col][col];
                for k in col..=n {
                    a[r][k] -= f * a[col][k];
                }
            }
        }
    }
    Some((0..n).map(|i| a[i][n] / a[i][i]).collect())
}

/// Homography mapping `src[i]` to `dst[i]` (OpenCV getPerspectiveTransform).
pub fn perspective_transform(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<[f64; 9]> {
    let mut a = Vec::with_capacity(8);
    for i in 0..4 {
        let ([x, y], [u, v]) = (src[i], dst[i]);
        a.push([x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u]);
        a.push([0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v]);
    }
    let h = solve(a, 8)?;
    Some([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0])
}

fn invert3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-15 || !det.is_finite() {
        return None;
    }
    let d = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * d,
        (m[2] * m[7] - m[1] * m[8]) * d,
        (m[1] * m[5] - m[2] * m[4]) * d,
        (m[5] * m[6] - m[3] * m[8]) * d,
        (m[0] * m[8] - m[2] * m[6]) * d,
        (m[2] * m[3] - m[0] * m[5]) * d,
        (m[3] * m[7] - m[4] * m[6]) * d,
        (m[1] * m[6] - m[0] * m[7]) * d,
        (m[0] * m[4] - m[1] * m[3]) * d,
    ])
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct WarpPerspective {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default)]
    matrix: Option<Vec<f64>>,
    #[serde(default)]
    src_quad: Option<[[f64; 2]; 4]>,
    #[serde(default)]
    dst_quad: Option<[[f64; 2]; 4]>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    inverse_map: bool,
    #[serde(default)]
    interpolation: Option<Interp>,
    #[serde(default)]
    border_mode: Border,
    #[serde(default)]
    border_value: Option<[u8; 4]>,
}

pub fn warp_perspective(ctx: &Ctx, p: WarpPerspective) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let (mut ow, mut oh) = (p.width.map(f64::from), p.height.map(f64::from));
    let m: [f64; 9] = match (&p.matrix, &p.src_quad, &p.dst_quad) {
        (Some(m), None, None) => m.as_slice().try_into().map_err(|_| ApiError::bad_params("matrix must have 9 numbers (3x3, row-major)"))?,
        (None, Some(s), d) => {
            let d = match d {
                Some(d) => *d,
                None => {
                    // Rectify: the quad's mean edge lengths give the output size.
                    let dist = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
                    let qw = ((dist(s[0], s[1]) + dist(s[3], s[2])) / 2.0).round().max(1.0);
                    let qh = ((dist(s[0], s[3]) + dist(s[1], s[2])) / 2.0).round().max(1.0);
                    let (tw, th) = (ow.unwrap_or(qw), oh.unwrap_or(qh));
                    ow = Some(tw);
                    oh = Some(th);
                    [[0.0, 0.0], [tw - 1.0, 0.0], [tw - 1.0, th - 1.0], [0.0, th - 1.0]]
                }
            };
            if ow.is_none() || oh.is_none() {
                let maxx = d.iter().map(|q| q[0]).fold(0.0, f64::max);
                let maxy = d.iter().map(|q| q[1]).fold(0.0, f64::max);
                ow = ow.or(Some((maxx + 1.0).round()));
                oh = oh.or(Some((maxy + 1.0).round()));
            }
            perspective_transform(s, &d).ok_or_else(|| ApiError::bad_params("the quads are degenerate (three points on a line?)"))?
        }
        _ => return Err(ApiError::bad_params("give matrix (9 numbers) or srcQuad (+ optional dstQuad), not both")),
    };
    if m.iter().any(|v| !v.is_finite()) {
        return Err(ApiError::bad_params("matrix has a non-finite number"));
    }
    let inv = if p.inverse_map { m } else { invert3(&m).ok_or_else(|| ApiError::bad_params("matrix is singular"))? };
    let (ow, oh) = check_out_size(ow.unwrap_or(src.w() as f64), oh.unwrap_or(src.h() as f64))?;
    let f = move |x: f64, y: f64| {
        let z = inv[6] * x + inv[7] * y + inv[8];
        if z.abs() < 1e-12 {
            return None;
        }
        Some(((inv[0] * x + inv[1] * y + inv[2]) / z, (inv[3] * x + inv[4] * y + inv[5]) / z))
    };
    let out = warp_buf(src.data(), src.w(), src.h(), src.ch(), ow, oh, &f, interp_or_linear(p.interpolation), p.border_mode, p.border_value.unwrap_or([0; 4]));
    let mut r = put(ctx, ow, oh, src.ch(), out)?;
    r["matrix"] = serde_json::json!(m);
    Ok(r)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn resize_flat_and_area() {
        let src = vec![77u8; 10 * 8 * 4];
        for i in [Interp::Nearest, Interp::Linear, Interp::Area, Interp::Cubic] {
            assert_eq!(resize_buf(&src, 10, 8, 4, 5, 4, i), vec![77u8; 5 * 4 * 4]);
            assert_eq!(resize_buf(&src, 10, 8, 4, 23, 17, i), vec![77u8; 23 * 17 * 4]);
        }
        // 2x2 checker → 1x1 area = mean.
        let m = vec![0u8, 255, 255, 0];
        assert_eq!(resize_buf(&m, 2, 2, 1, 1, 1, Interp::Area), vec![128]);
    }

    #[test]
    fn quarter_turns() {
        // 3x2: 0 1 2 / 3 4 5
        let src: Vec<u8> = (0..6).collect();
        let (r, w, h) = rotate90_buf(&src, 3, 2, 1, 1);
        assert_eq!((w, h), (2, 3));
        assert_eq!(r, vec![2, 5, 1, 4, 0, 3]);
        let (r4, _, _) = rotate90_buf(&src, 3, 2, 1, 4);
        assert_eq!(r4, src);
        assert_eq!(flip_buf(&src, 3, 2, 1, true, false), vec![2, 1, 0, 5, 4, 3]);
        assert_eq!(flip_buf(&src, 3, 2, 1, false, true), vec![3, 4, 5, 0, 1, 2]);
    }

    #[test]
    fn homography_identity_and_square() {
        let q = [[0.0, 0.0], [9.0, 0.0], [9.0, 9.0], [0.0, 9.0]];
        let h = perspective_transform(&q, &q).unwrap();
        for (a, b) in h.iter().zip([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]) {
            assert!((a - b).abs() < 1e-9);
        }
        let d = [[0.0, 0.0], [18.0, 0.0], [18.0, 18.0], [0.0, 18.0]];
        let h = perspective_transform(&q, &d).unwrap();
        assert!((h[0] - 2.0).abs() < 1e-9 && (h[4] - 2.0).abs() < 1e-9);
        assert!(invert3(&[0.0; 9]).is_none());
    }

    #[test]
    fn inscribed() {
        assert_eq!(inscribed_size(100.0, 50.0, 0.0), (100.0, 50.0));
        let (w, h) = inscribed_size(100.0, 50.0, 10.0);
        assert!(w < 100.0 && h < 50.0 && (w / h - 2.0).abs() < 0.05);
    }
}
