//! Filters: gaussianBlur, medianBlur, bilateralFilter, sobel, canny, laplacianVariance, denoise.

use imagelab_api::{ApiError, ApiResult, Ctx};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::raster::{Src, gaussian, gray_of, load, load_mask, put, put_img, put_mask, round2, same_size, shrink_gray};

pub const MAX_KSIZE: usize = 255;

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct GaussianBlur {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default)]
    ksize: Option<usize>,
    #[serde(default, alias = "sigmaX")]
    sigma: Option<f64>,
}

pub fn gaussian_blur(ctx: &Ctx, p: GaussianBlur) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let sigma = p.sigma.unwrap_or(0.0);
    if !(0.0..=500.0).contains(&sigma) {
        return Err(ApiError::bad_params("sigma must be in 0..=500"));
    }
    if let Some(k) = p.ksize
        && (k.is_multiple_of(2) || k > MAX_KSIZE)
    {
        return Err(ApiError::bad_params(format!("ksize must be odd and at most {MAX_KSIZE}")));
    }
    if p.ksize.is_none() && sigma <= 0.0 {
        return Err(ApiError::bad_params("give sigma (> 0) and/or ksize (odd)"));
    }
    let out = gaussian(src.data(), src.w(), src.h(), src.ch(), p.ksize, sigma);
    put(ctx, src.w(), src.h(), src.ch(), out)
}

/// Median of each channel over a `k`×`k` window (Huang's sliding histogram, coarse + fine bins).
pub fn median_buf(src: &[u8], w: usize, h: usize, ch: usize, k: usize) -> Vec<u8> {
    let r = (k / 2) as isize;
    let half = (k * k / 2) as u32;
    let stride = w * ch;
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(stride).enumerate().for_each(|(y, orow)| {
        let rows: Vec<usize> = (-r..=r).map(|d| crate::raster::reflect(y as isize + d, h)).collect();
        for c in 0..ch {
            let mut fine = [0u32; 256];
            let mut coarse = [0u32; 16];
            let col = |x: isize, fine: &mut [u32; 256], coarse: &mut [u32; 16], add: bool| {
                let sx = crate::raster::reflect(x, w);
                for &sy in &rows {
                    let v = src[sy * stride + sx * ch + c] as usize;
                    if add {
                        fine[v] += 1;
                        coarse[v >> 4] += 1;
                    } else {
                        fine[v] -= 1;
                        coarse[v >> 4] -= 1;
                    }
                }
            };
            for x in -r..=r {
                col(x, &mut fine, &mut coarse, true);
            }
            for x in 0..w {
                // Find the (half+1)-th smallest.
                let mut acc = 0u32;
                let mut bin = 0usize;
                while bin < 15 && acc + coarse[bin] <= half {
                    acc += coarse[bin];
                    bin += 1;
                }
                let mut v = bin << 4;
                while v < 255 && acc + fine[v] <= half {
                    acc += fine[v];
                    v += 1;
                }
                orow[x * ch + c] = v as u8;
                if x + 1 < w {
                    col(x as isize - r, &mut fine, &mut coarse, false);
                    col(x as isize + r + 1, &mut fine, &mut coarse, true);
                }
            }
        }
    });
    out
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MedianBlur {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default = "three")]
    ksize: usize,
}

fn three() -> usize {
    3
}

pub fn median_blur(ctx: &Ctx, p: MedianBlur) -> ApiResult<Value> {
    if p.ksize.is_multiple_of(2) || !(3..=31).contains(&p.ksize) {
        return Err(ApiError::bad_params("ksize must be odd, 3..=31"));
    }
    let src = load(ctx, &p.img)?;
    let out = median_buf(src.data(), src.w(), src.h(), src.ch(), p.ksize);
    put(ctx, src.w(), src.h(), src.ch(), out)
}

/// Bilateral filter (OpenCV semantics: colour distance is the L1 sum over RGB, circular window).
/// Alpha passes through.
pub fn bilateral_buf(src: &[u8], w: usize, h: usize, ch: usize, d: usize, sigma_color: f64, sigma_space: f64) -> Vec<u8> {
    let r = (d / 2).max(1) as isize;
    let sc = sigma_color.max(0.01);
    let ss = sigma_space.max(0.01);
    let mut offs: Vec<(isize, isize, f32)> = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            let dist2 = (dx * dx + dy * dy) as f64;
            if dist2.sqrt() <= r as f64 {
                offs.push((dx, dy, (-dist2 / (2.0 * ss * ss)).exp() as f32));
            }
        }
    }
    let cn = if ch == 4 { 3 } else { 1 };
    let color_w: Vec<f32> = (0..=255 * cn).map(|i| (-((i * i) as f64) / (2.0 * sc * sc)).exp() as f32).collect();
    let stride = w * ch;
    let mut out = src.to_vec();
    out.par_chunks_mut(stride).enumerate().for_each(|(y, orow)| {
        for x in 0..w {
            let c0 = &src[y * stride + x * ch..y * stride + x * ch + cn];
            let mut acc = [0f32; 3];
            let mut wsum = 0f32;
            for &(dx, dy, ws) in &offs {
                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                let p = &src[sy * stride + sx * ch..sy * stride + sx * ch + cn];
                let diff: usize = p.iter().zip(c0).map(|(&a, &b)| a.abs_diff(b) as usize).sum();
                let wgt = ws * color_w[diff];
                for k in 0..cn {
                    acc[k] += wgt * p[k] as f32;
                }
                wsum += wgt;
            }
            for k in 0..cn {
                orow[x * ch + k] = (acc[k] / wsum + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Bilateral {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default = "nine")]
    d: usize,
    #[serde(default = "seventy_five")]
    sigma_color: f64,
    #[serde(default = "seventy_five")]
    sigma_space: f64,
}

fn nine() -> usize {
    9
}
fn seventy_five() -> f64 {
    75.0
}

pub fn bilateral_filter(ctx: &Ctx, p: Bilateral) -> ApiResult<Value> {
    if !(1..=25).contains(&p.d) {
        return Err(ApiError::bad_params("d (diameter) must be in 1..=25"));
    }
    if !(p.sigma_color > 0.0 && p.sigma_space > 0.0 && p.sigma_color <= 1000.0 && p.sigma_space <= 1000.0) {
        return Err(ApiError::bad_params("sigmaColor and sigmaSpace must be in (0, 1000]"));
    }
    let src = load(ctx, &p.img)?;
    let out = bilateral_buf(src.data(), src.w(), src.h(), src.ch(), p.d, p.sigma_color, p.sigma_space);
    put(ctx, src.w(), src.h(), src.ch(), out)
}

/// 3×3 Sobel derivatives of a gray plane (reflect-101 borders).
pub fn sobel_xy(g: &[u8], w: usize, h: usize) -> (Vec<i16>, Vec<i16>) {
    let mut gx = vec![0i16; w * h];
    let mut gy = vec![0i16; w * h];
    gx.par_chunks_mut(w).zip(gy.par_chunks_mut(w)).enumerate().for_each(|(y, (rx, ry))| {
        let ym = crate::raster::reflect(y as isize - 1, h);
        let yp = crate::raster::reflect(y as isize + 1, h);
        let (a, b, c) = (&g[ym * w..ym * w + w], &g[y * w..y * w + w], &g[yp * w..yp * w + w]);
        for x in 0..w {
            let xm = crate::raster::reflect(x as isize - 1, w);
            let xp = crate::raster::reflect(x as isize + 1, w);
            let v = |r: &[u8], i: usize| r[i] as i16;
            rx[x] = (v(a, xp) + 2 * v(b, xp) + v(c, xp)) - (v(a, xm) + 2 * v(b, xm) + v(c, xm));
            ry[x] = (v(c, xm) + 2 * v(c, x) + v(c, xp)) - (v(a, xm) + 2 * v(a, x) + v(a, xp));
        }
    });
    (gx, gy)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sobel {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default)]
    dx: Option<u8>,
    #[serde(default)]
    dy: Option<u8>,
    #[serde(default)]
    scale: Option<f32>,
}

pub fn sobel(ctx: &Ctx, p: Sobel) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let (w, h) = (src.w(), src.h());
    let g = gray_of(&src);
    let (gx, gy) = sobel_xy(&g, w, h);
    let scale = p.scale.unwrap_or(1.0);
    if !(scale > 0.0 && scale <= 100.0) {
        return Err(ApiError::bad_params("scale must be in (0, 100]"));
    }
    let mode = match (p.dx.unwrap_or(0), p.dy.unwrap_or(0)) {
        (0, 0) | (1, 1) => 2,
        (1, 0) => 0,
        (0, 1) => 1,
        _ => return Err(ApiError::bad_params("dx/dy: {dx:1} (vertical edges), {dy:1} (horizontal edges), or neither / both for the magnitude")),
    };
    let mut max = 0f32;
    let mag: Vec<u8> = (0..w * h)
        .map(|i| {
            let v = match mode {
                0 => (gx[i] as f32).abs(),
                1 => (gy[i] as f32).abs(),
                _ => (gx[i] as f32).hypot(gy[i] as f32),
            };
            max = max.max(v);
            (v * scale / 4.0).round().clamp(0.0, 255.0) as u8
        })
        .collect();
    let mut r = if src.is_mask() {
        put_mask(ctx, w, h, mag)?
    } else {
        put_img(ctx, w, h, mag.iter().flat_map(|&v| [v, v, v, 255]).collect())?
    };
    r["note"] = json!("values are |gradient| / 4 * scale, clamped to 0..255");
    r["maxGradient"] = json!(round2(max as f64));
    Ok(r)
}

/// Canny edges of a gray plane: Sobel 3×3, non-maximum suppression, hysteresis. 255 = edge.
pub fn canny_buf(g: &[u8], w: usize, h: usize, low: f32, high: f32, l2: bool) -> Vec<u8> {
    let (gx, gy) = sobel_xy(g, w, h);
    let mag: Vec<f32> = gx
        .par_iter()
        .zip(gy.par_iter())
        .map(|(&x, &y)| if l2 { (x as f32).hypot(y as f32) } else { (x.unsigned_abs() + y.unsigned_abs()) as f32 })
        .collect();
    // 0 = none, 1 = weak, 2 = strong.
    let tan22 = 0.414_213_57f32;
    let mut cls = vec![0u8; w * h];
    cls.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y == 0 || y + 1 >= h {
            return;
        }
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            let m = mag[i];
            if m <= low {
                continue;
            }
            let (ax, ay) = ((gx[i] as f32).abs(), (gy[i] as f32).abs());
            let (a, b) = if ay <= ax * tan22 {
                (mag[i - 1], mag[i + 1])
            } else if ay >= ax / tan22 {
                (mag[i - w], mag[i + w])
            } else if (gx[i] > 0) == (gy[i] > 0) {
                (mag[i - w - 1], mag[i + w + 1])
            } else {
                (mag[i - w + 1], mag[i + w - 1])
            };
            if m > a && m >= b {
                row[x] = if m > high { 2 } else { 1 };
            }
        }
    });
    let mut out = vec![0u8; w * h];
    let mut stack: Vec<usize> = cls.iter().enumerate().filter(|&(_, &c)| c == 2).map(|(i, _)| i).collect();
    for &i in &stack {
        out[i] = 255;
    }
    while let Some(i) = stack.pop() {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if cls[j] == 1 && out[j] == 0 {
                    out[j] = 255;
                    stack.push(j);
                }
            }
        }
    }
    out
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Canny {
    #[serde(alias = "mask")]
    img: String,
    #[serde(alias = "threshold1")]
    low: f32,
    #[serde(alias = "threshold2")]
    high: f32,
    #[serde(default)]
    sigma: f64,
    #[serde(default, alias = "L2gradient")]
    l2gradient: bool,
}

pub fn canny(ctx: &Ctx, p: Canny) -> ApiResult<Value> {
    if !(p.low >= 0.0 && p.high >= 0.0 && p.low.is_finite() && p.high.is_finite()) {
        return Err(ApiError::bad_params("low and high must be >= 0"));
    }
    if !(0.0..=50.0).contains(&p.sigma) {
        return Err(ApiError::bad_params("sigma must be in 0..=50"));
    }
    let (low, high) = if p.low <= p.high { (p.low, p.high) } else { (p.high, p.low) };
    let src = load(ctx, &p.img)?;
    let (w, h) = (src.w(), src.h());
    let mut g = gray_of(&src);
    if p.sigma > 0.0 {
        g = gaussian(&g, w, h, 1, None, p.sigma);
    }
    ctx.check_cancel()?;
    let edges = canny_buf(&g, w, h, low, high, p.l2gradient);
    put_mask(ctx, w, h, edges)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct LaplacianVariance {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default)]
    max_side: Option<usize>,
    #[serde(default, rename = "roi")]
    roi_mask: Option<String>,
}

/// Variance of the 4-neighbour Laplacian (OpenCV `Laplacian(ksize=1)`), optionally under a mask.
pub fn laplacian_variance_buf(g: &[u8], w: usize, h: usize, mask: Option<&[u8]>) -> (f64, f64) {
    let (s, q, n) = (0..h)
        .into_par_iter()
        .map(|y| {
            let ym = crate::raster::reflect(y as isize - 1, h);
            let yp = crate::raster::reflect(y as isize + 1, h);
            let (mut s, mut q, mut n) = (0f64, 0f64, 0u64);
            for x in 0..w {
                if mask.is_some_and(|m| m[y * w + x] < 128) {
                    continue;
                }
                let xm = crate::raster::reflect(x as isize - 1, w);
                let xp = crate::raster::reflect(x as isize + 1, w);
                let v = g[ym * w + x] as f64 + g[yp * w + x] as f64 + g[y * w + xm] as f64 + g[y * w + xp] as f64 - 4.0 * g[y * w + x] as f64;
                s += v;
                q += v * v;
                n += 1;
            }
            (s, q, n)
        })
        .reduce(|| (0.0, 0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = s / n as f64;
    (q / n as f64 - mean * mean, n as f64)
}

pub fn laplacian_variance(ctx: &Ctx, p: LaplacianVariance) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let (w, h) = (src.w(), src.h());
    let g = gray_of(&src);
    let mask = match &p.roi_mask {
        Some(m) => {
            let m = load_mask(ctx, m)?;
            same_size((w, h), (m.width as usize, m.height as usize), "laplacianVariance roi")?;
            Some(m)
        }
        None => None,
    };
    let (g, gw, gh, mk) = match (p.max_side, &mask) {
        (Some(ms), _) if ms > 0 && w.max(h) > ms => {
            let (g2, nw, nh, _) = shrink_gray(&g, w, h, ms);
            let m2 = mask.as_ref().map(|m| crate::geom::resize_buf(&m.alpha, w, h, 1, nw, nh, crate::geom::Interp::Area));
            (g2, nw, nh, m2)
        }
        _ => (g, w, h, mask.as_ref().map(|m| m.alpha.clone())),
    };
    let (var, n) = laplacian_variance_buf(&g, gw, gh, mk.as_deref());
    Ok(json!({"variance": round2(var), "pixels": n, "measuredAt": [gw, gh]}))
}

/// Non-local means on RGB(A) (3×3 patches of luma, `search`×`search` window); alpha passes through.
pub fn nl_means_buf(src: &[u8], w: usize, h: usize, ch: usize, hparam: f32, search: usize) -> Vec<u8> {
    let luma: Vec<f32> = if ch == 4 {
        src.par_chunks(4).map(|p| crate::raster::luma(p[0], p[1], p[2]) as f32).collect()
    } else {
        src.iter().map(|&v| v as f32).collect()
    };
    let sr = (search / 2) as isize;
    let cn = if ch == 4 { 3 } else { 1 };
    let h2 = (hparam * hparam).max(1e-3);
    let stride = w * ch;
    let at = |x: isize, y: isize| luma[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    let mut out = src.to_vec();
    out.par_chunks_mut(stride).enumerate().for_each(|(y, orow)| {
        let y = y as isize;
        for x in 0..w as isize {
            let mut acc = [0f32; 3];
            let mut wsum = 0f32;
            for dy in -sr..=sr {
                for dx in -sr..=sr {
                    let (qx, qy) = (x + dx, y + dy);
                    let mut d = 0f32;
                    for py in -1..=1 {
                        for px in -1..=1 {
                            let t = at(x + px, y + py) - at(qx + px, qy + py);
                            d += t * t;
                        }
                    }
                    let wgt = (-(d / 9.0) / h2).exp();
                    let sx = qx.clamp(0, w as isize - 1) as usize;
                    let sy = qy.clamp(0, h as isize - 1) as usize;
                    let p = &src[sy * stride + sx * ch..];
                    for k in 0..cn {
                        acc[k] += wgt * p[k] as f32;
                    }
                    wsum += wgt;
                }
            }
            for k in 0..cn {
                orow[x as usize * ch + k] = (acc[k] / wsum + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Denoise {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default = "thirty")]
    strength: f32,
    #[serde(default)]
    method: Option<String>,
}

fn thirty() -> f32 {
    30.0
}

/// Largest image (pixels) non-local means runs on.
pub const NLM_MAX_PIXELS: usize = 16_000_000;

pub fn denoise(ctx: &Ctx, p: Denoise) -> ApiResult<Value> {
    if !(0.0..=100.0).contains(&p.strength) {
        return Err(ApiError::bad_params("strength must be in 0..=100"));
    }
    let src = load(ctx, &p.img)?;
    let (w, h, ch) = (src.w(), src.h(), src.ch());
    if p.strength == 0.0 {
        return put(ctx, w, h, ch, src.data().to_vec());
    }
    let method = p.method.as_deref().unwrap_or("bilateral");
    let out = match method {
        "bilateral" => {
            let s = p.strength as f64;
            let d = if s > 60.0 { 9 } else if s > 25.0 { 7 } else { 5 };
            let cn = if matches!(src, Src::Img(_)) { 3.0 } else { 1.0 };
            bilateral_buf(src.data(), w, h, ch, d, (4.0 + s * 0.6) * cn, d as f64 / 2.0)
        }
        "nlmeans" | "nlm" | "fastNlMeans" => {
            if w * h > NLM_MAX_PIXELS {
                return Err(ApiError::too_large(format!("nlmeans runs on up to {NLM_MAX_PIXELS} px; resize first or use method bilateral")));
            }
            nl_means_buf(src.data(), w, h, ch, 2.0 + p.strength * 0.3, 7)
        }
        _ => return Err(ApiError::bad_params("method must be bilateral or nlmeans")),
    };
    put(ctx, w, h, ch, out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn square(n: usize, a: usize, b: usize) -> Vec<u8> {
        (0..n * n).map(|i| if (a..b).contains(&(i % n)) && (a..b).contains(&(i / n)) { 255 } else { 0 }).collect()
    }

    #[test]
    fn canny_finds_square_edges() {
        let g = square(64, 16, 48);
        let e = canny_buf(&g, 64, 64, 50.0, 150.0, false);
        // Edges along the square's sides, nothing inside or far outside.
        let at = |x: usize, y: usize| e[y * 64 + x];
        assert!((15..=16).any(|x| at(x, 32) == 255), "left edge");
        assert!((47..=48).any(|x| at(x, 32) == 255), "right edge");
        assert!((15..=16).any(|y| at(32, y) == 255), "top edge");
        assert_eq!(at(32, 32), 0);
        assert_eq!(at(5, 5), 0);
        let n = e.iter().filter(|&&v| v == 255).count();
        assert!((100..=200).contains(&n), "edge pixels {n}");
    }

    #[test]
    fn median_removes_salt() {
        let mut g = vec![100u8; 32 * 32];
        g[10 * 32 + 10] = 255;
        g[20 * 32 + 5] = 0;
        let m = median_buf(&g, 32, 32, 1, 3);
        assert!(m.iter().all(|&v| v == 100));
        let rgba: Vec<u8> = g.iter().flat_map(|&v| [v, v, v, 255]).collect();
        let m4 = median_buf(&rgba, 32, 32, 4, 5);
        assert!(m4.chunks(4).all(|p| p == [100, 100, 100, 255]));
    }

    #[test]
    fn bilateral_and_nlm_smooth_noise_keep_edges() {
        let n = 32;
        let mut g: Vec<u8> = (0..n * n).map(|i| if i % n < 16 { 60 } else { 200 }).collect();
        for (i, v) in g.iter_mut().enumerate() {
            let noise = ((i * 7919) % 13) as i32 - 6;
            *v = (*v as i32 + noise).clamp(0, 255) as u8;
        }
        for out in [bilateral_buf(&g, n, n, 1, 7, 40.0, 3.0), nl_means_buf(&g, n, n, 1, 10.0, 7)] {
            let var = |o: &[u8]| {
                let vals: Vec<f64> = (0..n).flat_map(|y| (2..12).map(move |x| (y, x))).map(|(y, x)| o[y * n + x] as f64).collect();
                let m = vals.iter().sum::<f64>() / vals.len() as f64;
                vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / vals.len() as f64
            };
            assert!(var(&out) < var(&g) * 0.6, "noise reduced");
            assert!(out[16 * n + 13] < 100 && out[16 * n + 18] > 160, "edge kept");
        }
    }

    #[test]
    fn laplacian_sharp_vs_blurred() {
        let g = square(64, 16, 48);
        let (sharp, _) = laplacian_variance_buf(&g, 64, 64, None);
        let b = gaussian(&g, 64, 64, 1, None, 3.0);
        let (soft, _) = laplacian_variance_buf(&b, 64, 64, None);
        assert!(sharp > soft * 4.0, "{sharp} vs {soft}");
    }
}
