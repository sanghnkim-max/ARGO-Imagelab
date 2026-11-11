//! Shared plumbing: loading handles, storing results, gray/channel conversion, borders and
//! separable filtering on interleaved 8-bit buffers (`ch` = 4 for RGBA images, 1 for masks).

use std::sync::Arc;

use imagelab_api::{ApiError, ApiResult, Ctx, HandleId, ImageBuf, MaskBuf, parse_handle};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

/// An image or a mask taken from the arena.
#[derive(Clone)]
pub enum Src {
    Img(Arc<ImageBuf>),
    Mask(Arc<MaskBuf>),
}

impl Src {
    pub fn w(&self) -> usize {
        match self {
            Src::Img(i) => i.width as usize,
            Src::Mask(m) => m.width as usize,
        }
    }
    pub fn h(&self) -> usize {
        match self {
            Src::Img(i) => i.height as usize,
            Src::Mask(m) => m.height as usize,
        }
    }
    pub fn ch(&self) -> usize {
        match self {
            Src::Img(_) => 4,
            Src::Mask(_) => 1,
        }
    }
    pub fn data(&self) -> &[u8] {
        match self {
            Src::Img(i) => &i.rgba,
            Src::Mask(m) => &m.alpha,
        }
    }
    pub fn is_mask(&self) -> bool {
        matches!(self, Src::Mask(_))
    }
}

/// Load an `img:N` or `mask:N` handle.
pub fn load(ctx: &Ctx, s: &str) -> ApiResult<Src> {
    let id = parse_handle(s)?;
    match id {
        HandleId::Image(_) => Ok(Src::Img(ctx.shared.handles.image(id)?)),
        HandleId::Mask(_) => Ok(Src::Mask(ctx.shared.handles.mask(id)?)),
    }
}

pub fn load_img(ctx: &Ctx, s: &str) -> ApiResult<Arc<ImageBuf>> {
    ctx.shared.handles.image_str(s)
}

pub fn load_mask(ctx: &Ctx, s: &str) -> ApiResult<Arc<MaskBuf>> {
    ctx.shared.handles.mask_str(s)
}

/// Load a mask handle, or an image handle reduced to gray (so mask functions also take images).
pub fn load_gray(ctx: &Ctx, s: &str) -> ApiResult<(usize, usize, Vec<u8>)> {
    let src = load(ctx, s)?;
    Ok((src.w(), src.h(), gray_of(&src)))
}

/// Store a buffer as an image (`ch` = 4) or a mask (`ch` = 1).
pub fn put(ctx: &Ctx, w: usize, h: usize, ch: usize, data: Vec<u8>) -> ApiResult<Value> {
    if ch == 1 { put_mask(ctx, w, h, data) } else { put_img(ctx, w, h, data) }
}

pub fn put_img(ctx: &Ctx, w: usize, h: usize, rgba: Vec<u8>) -> ApiResult<Value> {
    let img = ImageBuf::new(dim(w)?, dim(h)?, rgba)?;
    let id = ctx.shared.handles.put_image(img)?;
    Ok(json!({"img": id.to_string(), "width": w, "height": h}))
}

pub fn put_mask(ctx: &Ctx, w: usize, h: usize, alpha: Vec<u8>) -> ApiResult<Value> {
    let coverage = coverage(&alpha);
    let m = MaskBuf::new(dim(w)?, dim(h)?, alpha)?;
    let id = ctx.shared.handles.put_mask(m)?;
    Ok(json!({"mask": id.to_string(), "width": w, "height": h, "coverage": round4(coverage)}))
}

pub fn dim(v: usize) -> ApiResult<u32> {
    u32::try_from(v).map_err(|_| ApiError::too_large(format!("{v} px is too large")))
}

/// Mean coverage of a mask, 0..1.
pub fn coverage(alpha: &[u8]) -> f64 {
    if alpha.is_empty() {
        return 0.0;
    }
    let sum: u64 = alpha.par_chunks(1 << 16).map(|c| c.iter().map(|&v| v as u64).sum::<u64>()).sum();
    sum as f64 / (alpha.len() as f64 * 255.0)
}

pub fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

pub fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// BT.601 luma with OpenCV's fixed-point coefficients.
#[inline]
pub fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 4899 + g as u32 * 9617 + b as u32 * 1868 + 8192) >> 14) as u8
}

/// Gray plane: luma of an image, the coverage of a mask.
pub fn gray_of(src: &Src) -> Vec<u8> {
    match src {
        Src::Img(i) => i.rgba.par_chunks(4).map(|p| luma(p[0], p[1], p[2])).collect(),
        Src::Mask(m) => m.alpha.clone(),
    }
}

/// A single channel to work on (threshold, histogram, extractChannel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    #[default]
    Gray,
    R,
    G,
    B,
    A,
    /// HSV hue scaled to 0..255 (0..360°).
    H,
    /// HSV saturation.
    S,
    /// HSV value (max of R, G, B).
    V,
    /// CIE L* scaled to 0..255.
    L,
}

pub fn channel_of(src: &Src, c: Channel) -> ApiResult<Vec<u8>> {
    let img = match src {
        Src::Mask(m) => {
            return match c {
                Channel::Gray | Channel::A | Channel::V | Channel::L => Ok(m.alpha.clone()),
                _ => Err(ApiError::bad_params(format!("a mask has one channel; `{c:?}` needs an image"))),
            };
        }
        Src::Img(i) => i,
    };
    let f = move |p: &[u8]| -> u8 {
        match c {
            Channel::Gray => luma(p[0], p[1], p[2]),
            Channel::R => p[0],
            Channel::G => p[1],
            Channel::B => p[2],
            Channel::A => p[3],
            Channel::H => {
                let (h, _, _) = crate::color::rgb_to_hsv(p[0], p[1], p[2]);
                (h / 360.0 * 255.0).round().clamp(0.0, 255.0) as u8
            }
            Channel::S => crate::color::rgb_to_hsv(p[0], p[1], p[2]).1.mul_add(255.0, 0.5).clamp(0.0, 255.0) as u8,
            Channel::V => p[0].max(p[1]).max(p[2]),
            Channel::L => {
                let (l, _, _) = crate::color::rgb_to_lab(p[0], p[1], p[2]);
                (l * 2.55).round().clamp(0.0, 255.0) as u8
            }
        }
    };
    Ok(img.rgba.par_chunks(4).map(f).collect())
}

/// Index under BORDER_REFLECT_101 (`gfedcb|abcdefgh|gfedcba`).
#[inline]
pub fn reflect(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    if i >= 0 && (i as usize) < n {
        return i as usize;
    }
    let period = 2 * n as isize - 2;
    let m = i.rem_euclid(period);
    if m >= n as isize { (period - m) as usize } else { m as usize }
}

/// Normalized 1-D Gaussian of odd length `ksize`.
pub fn gaussian_kernel(ksize: usize, sigma: f64) -> Vec<f32> {
    let ksize = ksize.max(1) | 1;
    let sigma = if sigma > 0.0 { sigma } else { 0.3 * ((ksize as f64 - 1.0) * 0.5 - 1.0) + 0.8 };
    let r = (ksize / 2) as f64;
    let mut k: Vec<f64> = (0..ksize).map(|i| (-((i as f64 - r).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f64 = k.iter().sum();
    for v in &mut k {
        *v /= s;
    }
    k.into_iter().map(|v| v as f32).collect()
}

/// OpenCV's kernel size for a sigma when none is given (8-bit images).
pub fn ksize_for_sigma(sigma: f64) -> usize {
    (((sigma * 3.0) * 2.0 + 1.0).round() as usize) | 1
}

/// Separable convolution with reflect-101 borders, rounded and clamped to 8 bits.
/// Each output row blends its source rows vertically into a float row, then horizontally.
pub fn conv_sep(src: &[u8], w: usize, h: usize, ch: usize, kx: &[f32], ky: &[f32]) -> Vec<u8> {
    let rx = kx.len() / 2;
    let ry = ky.len() / 2;
    let stride = w * ch;
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(stride).enumerate().for_each(|(y, orow)| {
        // Vertical pass into a padded float row.
        let mut row = vec![0f32; (w + 2 * rx) * ch];
        for (k, &kv) in ky.iter().enumerate() {
            let sy = reflect(y as isize + k as isize - ry as isize, h);
            let srow = &src[sy * stride..(sy + 1) * stride];
            let dst = &mut row[rx * ch..(rx + w) * ch];
            for (d, &s) in dst.iter_mut().zip(srow) {
                *d += kv * s as f32;
            }
        }
        for p in 0..rx {
            let l = reflect(p as isize - rx as isize, w);
            let r = reflect((w + p) as isize, w);
            for c in 0..ch {
                row[p * ch + c] = row[(l + rx) * ch + c];
                row[(rx + w + p) * ch + c] = row[(r + rx) * ch + c];
            }
        }
        for x in 0..w {
            let mut acc = [0f32; 4];
            for (k, &kv) in kx.iter().enumerate() {
                let base = (x + k) * ch;
                for c in 0..ch {
                    acc[c] += kv * row[base + c];
                }
            }
            for c in 0..ch {
                orow[x * ch + c] = (acc[c] + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

/// Box blur of radius `r` (window `2r+1`) with replicated borders; O(1) per sample.
pub fn box_blur(src: &[u8], w: usize, h: usize, ch: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return src.to_vec();
    }
    let n = (2 * r + 1) as u32;
    let stride = w * ch;
    // Horizontal.
    let mut tmp = vec![0u8; src.len()];
    tmp.par_chunks_mut(stride).zip(src.par_chunks(stride)).for_each(|(o, s)| {
        for c in 0..ch {
            let at = |x: isize| s[(x.clamp(0, w as isize - 1) as usize) * ch + c] as u32;
            let mut sum: u32 = (-(r as isize)..=r as isize).map(at).sum();
            for x in 0..w {
                o[x * ch + c] = ((sum + n / 2) / n) as u8;
                sum = sum + at(x as isize + r as isize + 1) - at(x as isize - r as isize);
            }
        }
    });
    // Vertical, in bands of rows with running column sums.
    let mut out = vec![0u8; src.len()];
    let band = 64usize;
    out.par_chunks_mut(stride * band).enumerate().for_each(|(bi, ob)| {
        let y0 = bi * band;
        let rows = ob.len() / stride;
        let at = |y: isize, i: usize| tmp[(y.clamp(0, h as isize - 1) as usize) * stride + i] as u32;
        let mut sums = vec![0u32; stride];
        for (i, s) in sums.iter_mut().enumerate() {
            *s = (y0 as isize - r as isize..=y0 as isize + r as isize).map(|y| at(y, i)).sum();
        }
        for ry in 0..rows {
            let y = (y0 + ry) as isize;
            let orow = &mut ob[ry * stride..(ry + 1) * stride];
            for i in 0..stride {
                orow[i] = ((sums[i] + n / 2) / n) as u8;
                sums[i] = sums[i] + at(y + r as isize + 1, i) - at(y - r as isize, i);
            }
        }
    });
    out
}

/// Gaussian blur: the exact separable kernel for small sigmas, three box passes beyond.
pub fn gaussian(src: &[u8], w: usize, h: usize, ch: usize, ksize: Option<usize>, sigma: f64) -> Vec<u8> {
    let ksize = ksize.unwrap_or_else(|| ksize_for_sigma(sigma)).max(1) | 1;
    if ksize <= 1 {
        return src.to_vec();
    }
    if ksize <= 31 {
        let k = gaussian_kernel(ksize, sigma);
        return conv_sep(src, w, h, ch, &k, &k);
    }
    let sigma = if sigma > 0.0 { sigma } else { 0.3 * ((ksize as f64 - 1.0) * 0.5 - 1.0) + 0.8 };
    // Box sizes approximating a Gaussian (Kovesi, "Fast almost-Gaussian filtering").
    let n = 3.0;
    let wi = ((12.0 * sigma * sigma / n) + 1.0).sqrt().floor();
    let wl = if (wi as i64) % 2 == 0 { wi - 1.0 } else { wi };
    let wu = wl + 2.0;
    let m = ((12.0 * sigma * sigma - n * wl * wl - 4.0 * n * wl - 3.0 * n) / (-4.0 * wl - 4.0)).round();
    let mut buf = src.to_vec();
    for i in 0..3 {
        let size = if (i as f64) < m { wl } else { wu };
        let r = ((size.max(1.0) as usize) - 1) / 2;
        buf = box_blur(&buf, w, h, ch, r);
    }
    buf
}

/// Downscale a gray plane by area averaging so the long side is at most `max_side`.
pub fn shrink_gray(g: &[u8], w: usize, h: usize, max_side: usize) -> (Vec<u8>, usize, usize, f64) {
    let long = w.max(h);
    if long <= max_side || max_side == 0 {
        return (g.to_vec(), w, h, 1.0);
    }
    let s = max_side as f64 / long as f64;
    let nw = ((w as f64 * s).round() as usize).max(1);
    let nh = ((h as f64 * s).round() as usize).max(1);
    let out = crate::geom::resize_buf(g, w, h, 1, nw, nh, crate::geom::Interp::Area);
    (out, nw, nh, w as f64 / nw as f64)
}

/// Require two rasters to share a size.
pub fn same_size(a: (usize, usize), b: (usize, usize), what: &str) -> ApiResult<()> {
    if a != b {
        return Err(ApiError::bad_params(format!("{what}: sizes differ ({}x{} vs {}x{}); resize one first", a.0, a.1, b.0, b.1)));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn reflect_101() {
        assert_eq!(reflect(-1, 5), 1);
        assert_eq!(reflect(-2, 5), 2);
        assert_eq!(reflect(5, 5), 3);
        assert_eq!(reflect(6, 5), 2);
        assert_eq!(reflect(-100, 5), reflect(-100 + 8 * 12, 5));
        assert_eq!(reflect(3, 1), 0);
    }

    #[test]
    fn blur_keeps_flat_and_mean() {
        let src = vec![100u8; 20 * 10 * 4];
        assert_eq!(gaussian(&src, 20, 10, 4, None, 2.0), src);
        assert_eq!(gaussian(&src, 20, 10, 4, None, 20.0), src);
        let mut spot = vec![0u8; 21 * 21];
        spot[10 * 21 + 10] = 255;
        let b = gaussian(&spot, 21, 21, 1, Some(5), 1.0);
        assert!(b[10 * 21 + 10] > b[10 * 21 + 12] && b[10 * 21 + 12] > 0);
    }
}
