//! Mask arithmetic shared by both engines: model logits → full-size coverage, resampling,
//! combining with a selection, feathering, bounds and coverage. Pure functions over `u8`
//! coverage (0 = outside, 255 = inside), so they are tested without a document or a model.

/// Combine modes of `select.fromMask` (PhotoCraft's selection modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Mode {
    #[default]
    Replace,
    Add,
    Subtract,
    Intersect,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Replace => "replace",
            Mode::Add => "add",
            Mode::Subtract => "subtract",
            Mode::Intersect => "intersect",
        }
    }
}

/// `base` ∘ `new` per pixel. `base` is the current selection (`None` = nothing selected).
pub fn combine(base: Option<&[u8]>, new: &[u8], mode: Mode) -> Vec<u8> {
    match (mode, base) {
        (Mode::Replace, _) => new.to_vec(),
        (Mode::Add, None) => new.to_vec(),
        (Mode::Subtract | Mode::Intersect, None) => {
            if mode == Mode::Subtract {
                vec![0; new.len()]
            } else {
                // Photoshop: intersecting with no selection intersects with the whole canvas.
                new.to_vec()
            }
        }
        (Mode::Add, Some(b)) => b.iter().zip(new).map(|(a, n)| (*a).max(*n)).collect(),
        (Mode::Subtract, Some(b)) => b.iter().zip(new).map(|(a, n)| mul255(*a, 255 - *n)).collect(),
        (Mode::Intersect, Some(b)) => b.iter().zip(new).map(|(a, n)| mul255(*a, *n)).collect(),
    }
}

fn mul255(a: u8, b: u8) -> u8 {
    ((a as u32 * b as u32 + 127) / 255) as u8
}

/// Bilinear resample of a coverage mask (half-pixel centres).
pub fn resample(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut out = vec![0u8; dw * dh];
    if sw == 0 || sh == 0 || src.len() < sw * sh {
        return out;
    }
    let at = |x: usize, y: usize| src[y.min(sh - 1) * sw + x.min(sw - 1)] as f32;
    let (sx, sy) = (sw as f32 / dw.max(1) as f32, sh as f32 / dh.max(1) as f32);
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let (y0, ty) = (fy as usize, fy.fract());
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let (x0, tx) = (fx as usize, fx.fract());
            let top = at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx;
            let bot = at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx;
            out[y * dw + x] = (top * (1.0 - ty) + bot * ty).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Model logits (`side × side` over the image stretched to a square) → coverage at `w × h`:
/// the logits are interpolated bilinearly *before* the sigmoid, so the edge stays about a pixel
/// wide however far the mask is enlarged (interpolating probabilities would smear it over a
/// whole grid cell), then the edge band gets one light 3×3 smoothing pass against stair-steps.
pub fn logits_to_mask(logits: &[f32], side: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if side == 0 || logits.len() < side * side || w == 0 || h == 0 {
        return out;
    }
    let sample = |x: usize, y: usize| {
        let v = logits[y.min(side - 1) * side + x.min(side - 1)];
        if v.is_finite() { v } else { -20.0 }
    };
    let (sx, sy) = (side as f32 / w as f32, side as f32 / h as f32);
    // Enlargement factor: logits change ~this much slower per output pixel than per cell, so
    // steepen the sigmoid by it (bounded) to keep the ramp near one output pixel.
    let gain = (1.0 / sx.max(sy)).clamp(1.0, 4.0);
    for y in 0..h {
        let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let (y0, ty) = (fy as usize, fy.fract());
        for x in 0..w {
            let fx = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let (x0, tx) = (fx as usize, fx.fract());
            let top = sample(x0, y0) * (1.0 - tx) + sample(x0 + 1, y0) * tx;
            let bot = sample(x0, y0 + 1) * (1.0 - tx) + sample(x0 + 1, y0 + 1) * tx;
            let l = (top * (1.0 - ty) + bot * ty) * gain;
            out[y * w + x] = (255.0 / (1.0 + (-l).exp())).round() as u8;
        }
    }
    smooth_edges(&mut out, w, h);
    out
}

/// One 3×3 box pass over pixels that are neither fully in nor fully out.
pub fn smooth_edges(m: &mut [u8], w: usize, h: usize) {
    if w < 3 || h < 3 || m.len() < w * h {
        return;
    }
    let src = m.to_vec();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let (mut lo, mut hi, mut sum) = (255u8, 0u8, 0u32);
            for dy in 0..3 {
                for dx in 0..3 {
                    let v = src[(y + dy - 1) * w + x + dx - 1];
                    lo = lo.min(v);
                    hi = hi.max(v);
                    sum += v as u32;
                }
            }
            if hi > lo && hi - lo > 8 {
                m[i] = ((sum + 4) / 9) as u8;
            }
        }
    }
}

/// Approximate Gaussian feather of `radius` px (three box passes, separable). Bounded: at most
/// 250 px.
pub fn feather(m: &mut [u8], w: usize, h: usize, radius: f32) {
    if !(radius.is_finite() && radius >= 0.5) || m.len() < w * h || w == 0 || h == 0 {
        return;
    }
    // Three boxes of half-width r ≈ a Gaussian of sigma ≈ radius / 2.
    let sigma = radius.min(250.0) / 2.0;
    let r = (((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() - 1.0) / 2.0).round().max(1.0) as usize;
    let mut buf: Vec<u32> = m.iter().map(|v| *v as u32).collect();
    let mut tmp = vec![0u32; buf.len()];
    for _ in 0..3 {
        box_pass(&buf, &mut tmp, w, h, r, true);
        box_pass(&tmp, &mut buf, w, h, r, false);
    }
    for (o, v) in m.iter_mut().zip(&buf) {
        *o = (*v).min(255) as u8;
    }
}

fn box_pass(src: &[u32], dst: &mut [u32], w: usize, h: usize, r: usize, horizontal: bool) {
    let (len, lines) = if horizontal { (w, h) } else { (h, w) };
    let idx = |line: usize, i: usize| if horizontal { line * w + i } else { i * w + line };
    let n = (2 * r + 1) as u32;
    for line in 0..lines {
        // Edge pixels extend outward (clamped), so a mask touching the canvas edge stays solid.
        let get = |i: isize| src[idx(line, i.clamp(0, len as isize - 1) as usize)];
        let mut acc: u32 = (-(r as isize)..=r as isize).map(get).sum();
        for i in 0..len {
            dst[idx(line, i)] = (acc + n / 2) / n;
            acc = acc + get(i as isize + r as isize + 1) - get(i as isize - r as isize);
        }
    }
}

/// `[x, y, w, h]` of pixels ≥ 128, or `None` when nothing is in.
pub fn bbox(m: &[u8], w: usize, h: usize) -> Option<[u32; 4]> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..h {
        let row = m.get(y * w..(y + 1) * w)?;
        if let Some(first) = row.iter().position(|v| *v >= 128) {
            let last = row.iter().rposition(|v| *v >= 128).unwrap_or(first);
            x0 = x0.min(first);
            x1 = x1.max(last);
            y0 = y0.min(y);
            y1 = y.max(y1);
        }
    }
    (x0 != usize::MAX).then(|| [x0 as u32, y0 as u32, (x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32])
}

/// Mean coverage 0..1.
pub fn coverage(m: &[u8]) -> f64 {
    if m.is_empty() {
        return 0.0;
    }
    m.iter().map(|v| *v as u64).sum::<u64>() as f64 / (m.len() as f64 * 255.0)
}

/// Zero everything outside `[x, y, w, h]` grown by `pad` px.
pub fn clip_to_box(m: &mut [u8], w: usize, h: usize, b: [f32; 4], pad: f32) {
    let x0 = (b[0] - pad).floor().max(0.0) as usize;
    let y0 = (b[1] - pad).floor().max(0.0) as usize;
    let x1 = ((b[0] + b[2] + pad).ceil().max(0.0) as usize).min(w);
    let y1 = ((b[1] + b[3] + pad).ceil().max(0.0) as usize).min(h);
    for y in 0..h {
        for x in 0..w {
            if !(x >= x0 && x < x1 && y >= y0 && y < y1) {
                m[y * w + x] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_modes() {
        let base = [255u8, 255, 0, 0];
        let new = [255u8, 0, 255, 0];
        assert_eq!(combine(Some(&base), &new, Mode::Add), vec![255, 255, 255, 0]);
        assert_eq!(combine(Some(&base), &new, Mode::Subtract), vec![0, 255, 0, 0]);
        assert_eq!(combine(Some(&base), &new, Mode::Intersect), vec![255, 0, 0, 0]);
        assert_eq!(combine(Some(&base), &new, Mode::Replace), new.to_vec());
        assert_eq!(combine(None, &new, Mode::Subtract), vec![0; 4]);
        assert_eq!(combine(None, &new, Mode::Add), new.to_vec());
    }

    #[test]
    fn logits_upsample_keeps_a_sharp_edge() {
        // 4×4 logits: left half in, right half out; enlarged to 64×64
        let mut l = vec![-10.0f32; 16];
        for y in 0..4 {
            l[y * 4] = 10.0;
            l[y * 4 + 1] = 10.0;
        }
        let m = logits_to_mask(&l, 4, 64, 64);
        assert_eq!(m[32 * 64 + 4], 255);
        assert_eq!(m[32 * 64 + 60], 0);
        let partial = m[32 * 64..33 * 64].iter().filter(|v| **v > 10 && **v < 245).count();
        assert!(partial <= 8, "edge ramp {partial} px");
        assert!((coverage(&m) - 0.5).abs() < 0.05);
    }

    #[test]
    fn bbox_resample_feather_clip() {
        let mut m = vec![0u8; 100];
        for y in 2..5 {
            for x in 3..8 {
                m[y * 10 + x] = 255;
            }
        }
        assert_eq!(bbox(&m, 10, 10), Some([3, 2, 5, 3]));
        assert_eq!(bbox(&[0; 4], 2, 2), None);
        let big = resample(&m, 10, 10, 20, 20);
        assert_eq!(bbox(&big, 20, 20).map(|b| b[2]), Some(10));
        let before = coverage(&m);
        let mut f = m.clone();
        feather(&mut f, 10, 10, 2.0);
        assert!((coverage(&f) - before).abs() < 0.02);
        assert!(f[2 * 10 + 2] > 0 && f[2 * 10 + 2] < 255);
        clip_to_box(&mut m, 10, 10, [0.0, 0.0, 5.0, 10.0], 0.0);
        assert_eq!(bbox(&m, 10, 10), Some([3, 2, 2, 3]));
    }
}
