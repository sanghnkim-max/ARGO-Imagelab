//! Morphology on masks: morphology (erode/dilate/open/close/gradient/tophat/blackhat),
//! growMask/shrinkMask (exact Euclidean distance), featherMask.

use imagelab_api::{ApiError, ApiResult, Ctx};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;

use crate::raster::{gaussian, load_gray, put_mask};

/// Sliding min/max of half-width `r` over a row (van Herk / Gil-Werman); out-of-row samples
/// are ignored (OpenCV's default morphology border).
fn row_extreme(row: &[u8], r: usize, is_max: bool, out: &mut [u8]) {
    let n = row.len();
    if r == 0 {
        out.copy_from_slice(row);
        return;
    }
    let id = if is_max { 0u8 } else { 255u8 };
    let op = |a: u8, b: u8| if is_max { a.max(b) } else { a.min(b) };
    let k = 2 * r + 1;
    let m = n + 2 * r;
    let mut p = vec![id; m];
    p[r..r + n].copy_from_slice(row);
    let mut g = vec![id; m];
    let mut h = vec![id; m];
    for i in 0..m {
        g[i] = if i % k == 0 { p[i] } else { op(g[i - 1], p[i]) };
    }
    for i in (0..m).rev() {
        h[i] = if i % k == k - 1 || i == m - 1 { p[i] } else { op(h[i + 1], p[i]) };
    }
    for (x, o) in out.iter_mut().enumerate() {
        *o = op(h[x], g[x + k - 1]);
    }
}

fn transpose(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
        for (y, v) in col.iter_mut().enumerate() {
            *v = src[y * w + x];
        }
    });
    out
}

fn rows_extreme(src: &[u8], w: usize, r: usize, is_max: bool) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(|(o, s)| row_extreme(s, r, is_max, o));
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Shape {
    #[default]
    #[serde(alias = "morph_rect", alias = "square")]
    Rect,
    #[serde(alias = "morph_ellipse", alias = "disk", alias = "circle")]
    Ellipse,
    #[serde(alias = "morph_cross")]
    Cross,
}

/// One erosion (`is_max` false) or dilation with a `k`×`k` structuring element.
pub fn morph_once(src: &[u8], w: usize, h: usize, k: usize, shape: Shape, is_max: bool) -> Vec<u8> {
    let r = k / 2;
    if r == 0 {
        return src.to_vec();
    }
    match shape {
        Shape::Rect => {
            let a = rows_extreme(src, w, r, is_max);
            let t = transpose(&a, w, h);
            let b = rows_extreme(&t, h, r, is_max);
            transpose(&b, h, w)
        }
        Shape::Ellipse | Shape::Cross => {
            let hw: Vec<usize> = (-(r as isize)..=r as isize)
                .map(|dy| match shape {
                    Shape::Cross => {
                        if dy == 0 {
                            r
                        } else {
                            0
                        }
                    }
                    _ => {
                        let f = dy as f64 / r as f64;
                        (r as f64 * (1.0 - f * f).max(0.0).sqrt()).round() as usize
                    }
                })
                .collect();
            let id = if is_max { 0u8 } else { 255u8 };
            let mut out = vec![id; src.len()];
            out.par_chunks_mut(w).enumerate().for_each(|(y, orow)| {
                let mut tmp = vec![0u8; w];
                for (j, &hwj) in hw.iter().enumerate() {
                    let sy = y as isize + j as isize - r as isize;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    let srow = &src[sy as usize * w..(sy as usize + 1) * w];
                    row_extreme(srow, hwj, is_max, &mut tmp);
                    for (o, &t) in orow.iter_mut().zip(&tmp) {
                        *o = if is_max { (*o).max(t) } else { (*o).min(t) };
                    }
                }
            });
            out
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MorphOp {
    #[serde(alias = "morph_erode")]
    Erode,
    #[serde(alias = "morph_dilate")]
    Dilate,
    #[serde(alias = "morph_open", alias = "opening")]
    Open,
    #[serde(alias = "morph_close", alias = "closing")]
    Close,
    #[serde(alias = "morph_gradient")]
    Gradient,
    #[serde(alias = "morph_tophat")]
    Tophat,
    #[serde(alias = "morph_blackhat")]
    Blackhat,
}

pub fn morph_buf(src: &[u8], w: usize, h: usize, op: MorphOp, k: usize, shape: Shape, iterations: usize) -> Vec<u8> {
    let rep = |mut b: Vec<u8>, is_max: bool| {
        for _ in 0..iterations {
            b = morph_once(&b, w, h, k, shape, is_max);
        }
        b
    };
    let erode = |b: Vec<u8>| rep(b, false);
    let dilate = |b: Vec<u8>| rep(b, true);
    match op {
        MorphOp::Erode => erode(src.to_vec()),
        MorphOp::Dilate => dilate(src.to_vec()),
        MorphOp::Open => dilate(erode(src.to_vec())),
        MorphOp::Close => erode(dilate(src.to_vec())),
        MorphOp::Gradient => {
            let d = dilate(src.to_vec());
            let e = erode(src.to_vec());
            d.iter().zip(&e).map(|(a, b)| a.saturating_sub(*b)).collect()
        }
        MorphOp::Tophat => {
            let o = dilate(erode(src.to_vec()));
            src.iter().zip(&o).map(|(a, b)| a.saturating_sub(*b)).collect()
        }
        MorphOp::Blackhat => {
            let c = erode(dilate(src.to_vec()));
            c.iter().zip(src).map(|(a, b)| a.saturating_sub(*b)).collect()
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Morphology {
    #[serde(alias = "img")]
    mask: String,
    op: MorphOp,
    #[serde(default = "three")]
    ksize: usize,
    #[serde(default)]
    shape: Shape,
    #[serde(default = "one")]
    iterations: usize,
}

fn three() -> usize {
    3
}
fn one() -> usize {
    1
}

pub fn morphology(ctx: &Ctx, p: Morphology) -> ApiResult<Value> {
    let max_k = if p.shape == Shape::Rect { 255 } else { 101 };
    if p.ksize == 0 || p.ksize > max_k {
        return Err(ApiError::bad_params(format!("ksize must be in 1..={max_k} for this shape (odd; even sizes round up)")));
    }
    if !(1..=50).contains(&p.iterations) {
        return Err(ApiError::bad_params("iterations must be in 1..=50"));
    }
    let (w, h, g) = load_gray(ctx, &p.mask)?;
    let out = morph_buf(&g, w, h, p.op, p.ksize | 1, p.shape, p.iterations);
    put_mask(ctx, w, h, out)
}

/// 1-D squared Euclidean distance transform (Felzenszwalb & Huttenlocher).
fn dt1d(f: &[f64], d: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut v = vec![0usize; n];
    let mut z = vec![0f64; n + 1];
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        loop {
            let vk = v[k];
            let s = ((f[q] + (q * q) as f64) - (f[vk] + (vk * vk) as f64)) / (2.0 * (q as f64 - vk as f64));
            if s <= z[k] && k > 0 {
                k -= 1;
                continue;
            }
            if s <= z[k] {
                // k == 0: replace the first parabola.
                v[0] = q;
                z[0] = f64::NEG_INFINITY;
                z[1] = f64::INFINITY;
                break;
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f64::INFINITY;
            break;
        }
    }
    k = 0;
    for (q, dq) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let vk = v[k];
        *dq = (q as f64 - vk as f64).powi(2) + f[vk];
    }
}

const FAR: f64 = 1e20;

/// Euclidean distance from each pixel to the nearest pixel where `target` is true.
pub fn distance_to(target: &[bool], w: usize, h: usize) -> Vec<f32> {
    let mut rows = vec![0f64; w * h];
    rows.par_chunks_mut(w).enumerate().for_each(|(y, r)| {
        let f: Vec<f64> = (0..w).map(|x| if target[y * w + x] { 0.0 } else { FAR }).collect();
        dt1d(&f, r);
    });
    let mut out = vec![0f32; w * h];
    let cols: Vec<Vec<f64>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let f: Vec<f64> = (0..h).map(|y| rows[y * w + x]).collect();
            let mut d = vec![0f64; h];
            dt1d(&f, &mut d);
            d
        })
        .collect();
    for (x, col) in cols.iter().enumerate() {
        for (y, &d) in col.iter().enumerate() {
            out[y * w + x] = if d >= FAR / 2.0 { f32::INFINITY } else { d.sqrt() as f32 };
        }
    }
    out
}

/// Grow (`radius` > 0) or shrink (< 0) a mask by a disc, with a one-pixel antialiased edge.
pub fn grow_buf(g: &[u8], w: usize, h: usize, radius: f32) -> Vec<u8> {
    if radius == 0.0 {
        return g.to_vec();
    }
    if radius > 0.0 {
        let fg: Vec<bool> = g.iter().map(|&v| v >= 128).collect();
        let d = distance_to(&fg, w, h);
        g.par_iter().zip(d.par_iter()).map(|(&v, &d)| v.max(((radius + 1.0 - d) * 255.0).clamp(0.0, 255.0) as u8)).collect()
    } else {
        let r = -radius;
        let bg: Vec<bool> = g.iter().map(|&v| v < 128).collect();
        let d = distance_to(&bg, w, h);
        g.par_iter().zip(d.par_iter()).map(|(&v, &d)| v.min(((d - r) * 255.0).clamp(0.0, 255.0) as u8)).collect()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Radius {
    #[serde(alias = "img")]
    mask: String,
    radius: f32,
}

fn check_radius(r: f32) -> ApiResult<()> {
    if !(0.0..=1000.0).contains(&r) {
        return Err(ApiError::bad_params("radius must be in 0..=1000 px"));
    }
    Ok(())
}

pub fn grow_mask(ctx: &Ctx, p: Radius) -> ApiResult<Value> {
    check_radius(p.radius)?;
    let (w, h, g) = load_gray(ctx, &p.mask)?;
    put_mask(ctx, w, h, grow_buf(&g, w, h, p.radius))
}

pub fn shrink_mask(ctx: &Ctx, p: Radius) -> ApiResult<Value> {
    check_radius(p.radius)?;
    let (w, h, g) = load_gray(ctx, &p.mask)?;
    put_mask(ctx, w, h, grow_buf(&g, w, h, -p.radius))
}

pub fn feather_mask(ctx: &Ctx, p: Radius) -> ApiResult<Value> {
    check_radius(p.radius)?;
    let (w, h, g) = load_gray(ctx, &p.mask)?;
    if p.radius == 0.0 {
        return put_mask(ctx, w, h, g);
    }
    let out = gaussian(&g, w, h, 1, None, (p.radius as f64 / 2.0).max(0.3));
    put_mask(ctx, w, h, out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn square(n: usize, a: usize, b: usize) -> Vec<u8> {
        (0..n * n).map(|i| if (a..b).contains(&(i % n)) && (a..b).contains(&(i / n)) { 255 } else { 0 }).collect()
    }
    fn count(m: &[u8]) -> usize {
        m.iter().filter(|&&v| v >= 128).count()
    }

    #[test]
    fn erode_dilate_open_close() {
        let s = square(32, 8, 24); // 16x16 = 256
        assert_eq!(count(&morph_buf(&s, 32, 32, MorphOp::Erode, 3, Shape::Rect, 1)), 14 * 14);
        assert_eq!(count(&morph_buf(&s, 32, 32, MorphOp::Dilate, 3, Shape::Rect, 1)), 18 * 18);
        assert_eq!(count(&morph_buf(&s, 32, 32, MorphOp::Dilate, 3, Shape::Rect, 2)), 20 * 20);
        assert_eq!(morph_buf(&s, 32, 32, MorphOp::Open, 3, Shape::Rect, 1), s);
        assert_eq!(morph_buf(&s, 32, 32, MorphOp::Close, 5, Shape::Ellipse, 1), s);
        let mut speck = s.clone();
        speck[2 * 32 + 2] = 255;
        assert_eq!(morph_buf(&speck, 32, 32, MorphOp::Open, 3, Shape::Rect, 1), s);
        let g = morph_buf(&s, 32, 32, MorphOp::Gradient, 3, Shape::Cross, 1);
        assert_eq!(g[16 * 32 + 16], 0);
        assert_eq!(g[16 * 32 + 8], 255);
        // Ellipse dilation of a point is a disc.
        let mut p = vec![0u8; 21 * 21];
        p[10 * 21 + 10] = 255;
        let d = morph_buf(&p, 21, 21, MorphOp::Dilate, 9, Shape::Ellipse, 1);
        assert_eq!(d[10 * 21 + 14], 255);
        assert_eq!(d[6 * 21 + 6], 0);
        // Edges are not eroded from outside the image.
        let full = vec![255u8; 100];
        assert_eq!(morph_buf(&full, 10, 10, MorphOp::Erode, 3, Shape::Rect, 1), full);
    }

    #[test]
    fn grow_shrink_by_distance() {
        let mut p = vec![0u8; 41 * 41];
        p[20 * 41 + 20] = 255;
        let g = grow_buf(&p, 41, 41, 10.0);
        assert_eq!(g[20 * 41 + 30], 255);
        assert_eq!(g[20 * 41 + 32], 0);
        let area = count(&g) as f64;
        assert!((area - std::f64::consts::PI * 100.0).abs() < 40.0, "{area}");
        let s = square(40, 10, 30);
        let sh = grow_buf(&s, 40, 40, -3.0);
        assert_eq!(count(&sh), 14 * 14);
        let none = grow_buf(&[0u8; 16], 4, 4, 5.0);
        assert!(none.iter().all(|&v| v == 0));
    }
}
