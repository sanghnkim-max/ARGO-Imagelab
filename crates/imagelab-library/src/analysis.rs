//! Measurements on a ≤1024 px proxy: sharpness, exposure, contrast, noise, faces, a 64-bit
//! dHash and histograms. Results are cached per content hash in `.imagelab/cache/`.
//!
//! ## Scores (all 0..1, higher is better)
//!
//! * **sharpness** — variance of the 4-neighbour Laplacian of luma (0..255), per tile on an 8×8
//!   grid; the 90th-percentile tile is used so a sharp subject on a soft background still counts
//!   as sharp. Mapped logarithmically: `clamp(log10(var / 8) / 2, 0, 1)` (var 8 → 0, 800 → 1).
//! * **exposure** — `meanTerm × clipTerm`: `meanTerm = 1 − smoothstep(0.15, 0.45, |mean − 0.5|)`
//!   with mean luma in 0..1, `clipTerm = 1 − clamp((clipLow + clipHigh − 0.01) / 0.15, 0, 1)`
//!   where clipLow/High are the fractions of pixels with luma ≤ 2 / ≥ 253.
//! * **contrast** — `smoothstep(0.04, 0.20, σ)`, σ = luma standard deviation in 0..1.
//! * **noise** — Immerkær's fast noise estimate (the `[1 −2 1; −2 4 −2; 1 −2 1]` kernel) over
//!   the flattest 30 % of 16×16 blocks (excluding clipped ones), σ in 0..255 levels;
//!   `1 − smoothstep(1.0, 8.0, σ)`.
//! * **faces** (optional) — PhotoCraft's face detector (libfacedetection CNN) on the proxy;
//!   when faces are found, the Laplacian variance inside the face boxes gives `faces.sharpness`.
//! * **overall** = `0.45·S + 0.25·E + 0.15·N + 0.15·C`, where `S` is sharpness, or
//!   `0.4·sharpness + 0.6·faces.sharpness` when faces were found (focus on faces matters most).

use std::fs;

use imagelab_api::{ApiError, ApiResult};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::paths::{Workspace, write_atomic};
use crate::pixels::{self, Rgba};

/// Bump when any measurement changes: old cache entries are recomputed.
pub const ALGO: u32 = 1;
/// Longest side of the analysis proxy.
pub const PROXY_SIDE: u32 = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Quality {
    pub sharpness: f32,
    /// 90th-percentile tile Laplacian variance.
    pub laplacian_var: f32,
    /// Whole-image Laplacian variance.
    pub laplacian_var_global: f32,
    pub exposure: f32,
    pub mean_luma: f32,
    pub clip_low: f32,
    pub clip_high: f32,
    pub contrast: f32,
    pub luma_std: f32,
    pub noise: f32,
    pub noise_sigma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FacesInfo {
    pub count: usize,
    pub max_score: f32,
    /// Largest face box as a fraction of the image area.
    pub largest_area: f32,
    /// Sharpness (same mapping as the global one) inside the face boxes; 0 without faces.
    pub sharpness: f32,
    /// Whether the best face has both eye landmarks inside its box (a frontal-ish face).
    pub eyes: bool,
    /// Face boxes `[x, y, w, h]` as fractions of the image, best first (at most 10).
    pub boxes: Vec<[f32; 4]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Facts {
    pub algo: u32,
    pub width: u32,
    pub height: u32,
    /// 64-bit dHash, 16 hex digits.
    pub dhash: String,
    pub quality: Quality,
    #[serde(default)]
    pub faces: Option<FacesInfo>,
}

impl Facts {
    pub fn dhash_bits(&self) -> u64 {
        u64::from_str_radix(&self.dhash, 16).unwrap_or(0)
    }

    /// `(overall, notes)` per the module docs.
    pub fn overall(&self) -> (f32, Vec<String>) {
        let q = &self.quality;
        let s = match &self.faces {
            Some(f) if f.count > 0 => 0.4 * q.sharpness + 0.6 * f.sharpness,
            _ => q.sharpness,
        };
        let overall = (0.45 * s + 0.25 * q.exposure + 0.15 * q.noise + 0.15 * q.contrast).clamp(0.0, 1.0);
        let mut notes = Vec::new();
        if s < 0.3 {
            notes.push("blurry".to_string());
        } else if s < 0.5 {
            notes.push("slightly soft".to_string());
        }
        if let Some(f) = &self.faces
            && f.count > 0
        {
            notes.push(format!("{} face{}", f.count, if f.count == 1 { "" } else { "s" }));
            if f.sharpness < 0.4 && q.sharpness >= 0.5 {
                notes.push("faces out of focus".to_string());
            }
        }
        if q.mean_luma < 0.22 {
            notes.push("underexposed".to_string());
        } else if q.mean_luma > 0.78 {
            notes.push("overexposed".to_string());
        }
        if q.clip_high > 0.02 {
            notes.push(format!("highlights clipped {:.0}%", q.clip_high * 100.0));
        }
        if q.clip_low > 0.05 {
            notes.push(format!("shadows crushed {:.0}%", q.clip_low * 100.0));
        }
        if q.noise < 0.5 {
            notes.push("noisy".to_string());
        }
        if q.contrast < 0.4 {
            notes.push("low contrast".to_string());
        }
        (overall, notes)
    }

    /// The per-photo result of `analyze.quality`.
    pub fn to_json(&self) -> Value {
        let (overall, notes) = self.overall();
        let q = &self.quality;
        json!({
            "sharpness": r3(q.sharpness),
            "exposure": r3(q.exposure),
            "contrast": r3(q.contrast),
            "noise": r3(q.noise),
            "faces": self.faces,
            "overall": r3(overall),
            "notes": notes,
            "width": self.width,
            "height": self.height,
            "details": {
                "laplacianVar": r3(q.laplacian_var), "laplacianVarGlobal": r3(q.laplacian_var_global),
                "meanLuma": r3(q.mean_luma), "clipLow": r3(q.clip_low), "clipHigh": r3(q.clip_high),
                "lumaStd": r3(q.luma_std), "noiseSigma": r3(q.noise_sigma),
            },
        })
    }
}

fn r3(v: f32) -> f64 {
    (f64::from(v) * 1000.0).round() / 1000.0
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn sharp_map(var: f32) -> f32 {
    ((var.max(1e-6) / 8.0).log10() / 2.0).clamp(0.0, 1.0)
}

/// Laplacian variance over `[x0, x1) × [y0, y1)` (interior pixels only).
fn laplacian_var(l: &[f32], w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f32 {
    let (x0, y0, x1, y1) = (x0.max(1), y0.max(1), x1.min(w.saturating_sub(1)), y1.min(h.saturating_sub(1)));
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let (mut s, mut s2, mut n) = (0f64, 0f64, 0f64);
    for y in y0..y1 {
        let r = y * w;
        for x in x0..x1 {
            let i = r + x;
            let v = f64::from(4.0 * l[i] - l[i - 1] - l[i + 1] - l[i - w] - l[i + w]);
            s += v;
            s2 += v * v;
            n += 1.0;
        }
    }
    let m = s / n;
    ((s2 / n - m * m).max(0.0)) as f32
}

pub fn measure(img: &Rgba) -> Quality {
    let (w, h) = (img.w as usize, img.h as usize);
    let l = img.luma();
    // Sharpness: tiles.
    let grid = 8usize.min(w / 16).min(h / 16).max(1);
    let mut tiles = Vec::with_capacity(grid * grid);
    for ty in 0..grid {
        for tx in 0..grid {
            tiles.push(laplacian_var(&l, w, h, tx * w / grid, ty * h / grid, (tx + 1) * w / grid, (ty + 1) * h / grid));
        }
    }
    tiles.sort_by(f32::total_cmp);
    let p90 = tiles.get(((tiles.len() as f32 * 0.9) as usize).min(tiles.len().saturating_sub(1))).copied().unwrap_or(0.0);
    let global = laplacian_var(&l, w, h, 0, 0, w, h);
    // Exposure + contrast.
    let n = l.len().max(1) as f64;
    let (mut s, mut s2, mut lo, mut hi) = (0f64, 0f64, 0usize, 0usize);
    for &v in &l {
        s += f64::from(v);
        s2 += f64::from(v) * f64::from(v);
        if v <= 2.5 {
            lo += 1;
        }
        if v >= 252.5 {
            hi += 1;
        }
    }
    let mean = (s / n) as f32 / 255.0;
    let std = ((s2 / n - (s / n).powi(2)).max(0.0).sqrt()) as f32 / 255.0;
    let (clip_low, clip_high) = (lo as f32 / n as f32, hi as f32 / n as f32);
    let mean_term = 1.0 - smoothstep(0.15, 0.45, (mean - 0.5).abs());
    let clip_term = 1.0 - ((clip_low + clip_high - 0.01) / 0.15).clamp(0.0, 1.0);
    let sigma = noise_sigma(&l, w, h);
    Quality {
        sharpness: sharp_map(p90),
        laplacian_var: p90,
        laplacian_var_global: global,
        exposure: mean_term * clip_term,
        mean_luma: mean,
        clip_low,
        clip_high,
        contrast: smoothstep(0.04, 0.20, std),
        luma_std: std,
        noise: 1.0 - smoothstep(1.0, 8.0, sigma),
        noise_sigma: sigma,
    }
}

/// Immerkær noise σ over the flattest blocks.
fn noise_sigma(l: &[f32], w: usize, h: usize) -> f32 {
    const B: usize = 16;
    if w < B + 2 || h < B + 2 {
        return 0.0;
    }
    let mut blocks: Vec<(f32, usize, usize)> = Vec::new();
    let mut by = 1;
    while by + B < h {
        let mut bx = 1;
        while bx + B < w {
            let (mut s, mut s2) = (0f32, 0f32);
            for y in by..by + B {
                for x in bx..bx + B {
                    let v = l[y * w + x];
                    s += v;
                    s2 += v * v;
                }
            }
            let nn = (B * B) as f32;
            let m = s / nn;
            if (8.0..=247.0).contains(&m) {
                blocks.push(((s2 / nn - m * m).max(0.0), bx, by));
            }
            bx += B;
        }
        by += B;
    }
    if blocks.is_empty() {
        return 0.0;
    }
    blocks.sort_by(|a, b| a.0.total_cmp(&b.0));
    let keep = (blocks.len() * 3 / 10).max(1);
    let (mut acc, mut count) = (0f64, 0f64);
    for &(_, bx, by) in &blocks[..keep] {
        for y in by..by + B {
            for x in bx..bx + B {
                let i = y * w + x;
                let v = l[i - w - 1] - 2.0 * l[i - w] + l[i - w + 1] - 2.0 * l[i - 1] + 4.0 * l[i] - 2.0 * l[i + 1] + l[i + w - 1] - 2.0 * l[i + w] + l[i + w + 1];
                acc += f64::from(v.abs());
                count += 1.0;
            }
        }
    }
    ((std::f64::consts::FRAC_PI_2).sqrt() * acc / (6.0 * count.max(1.0))) as f32
}

/// 64-bit difference hash: luma area-averaged to 9×8; bit = left < right.
pub fn dhash(img: &Rgba) -> u64 {
    let (w, h) = (img.w as usize, img.h as usize);
    let l = img.luma();
    let mut cells = [[0f32; 9]; 8];
    for (cy, row) in cells.iter_mut().enumerate() {
        let (y0, y1) = (cy * h / 8, ((cy + 1) * h / 8).max(cy * h / 8 + 1).min(h));
        for (cx, cell) in row.iter_mut().enumerate() {
            let (x0, x1) = (cx * w / 9, ((cx + 1) * w / 9).max(cx * w / 9 + 1).min(w));
            let mut s = 0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    s += l[y * w + x];
                }
            }
            *cell = s / ((y1 - y0) * (x1 - x0)).max(1) as f32;
        }
    }
    let mut bits = 0u64;
    for row in &cells {
        for x in 0..8 {
            bits = (bits << 1) | u64::from(row[x] < row[x + 1]);
        }
    }
    bits
}

/// Faces on the proxy, bounded to the detector's own working size.
pub fn faces(img: &Rgba) -> FacesInfo {
    let none = FacesInfo { count: 0, max_score: 0.0, largest_area: 0.0, sharpness: 0.0, eyes: false, boxes: Vec::new() };
    if !photocraft_algo::faces::AVAILABLE || img.w < 32 || img.h < 32 {
        return none;
    }
    let src = match pixels::shrink(img, photocraft_algo::faces::MAX_SIDE as u32) {
        Ok(s) => s,
        Err(_) => return none,
    };
    let px: Vec<[f32; 4]> = src.px.as_chunks::<4>().0.iter().map(|p| [f32::from(p[0]) / 255.0, f32::from(p[1]) / 255.0, f32::from(p[2]) / 255.0, 1.0]).collect();
    let (w, h) = (src.w as usize, src.h as usize);
    let found = photocraft_algo::faces::detect(&px, w, h, 0.6).unwrap_or_default();
    if found.is_empty() {
        return none;
    }
    let l = src.luma();
    let area = (w * h) as f32;
    let mut sharp = 0f32;
    let mut largest = 0f32;
    let mut boxes = Vec::new();
    for f in found.iter().take(10) {
        let [x, y, fw, fh] = f.rect;
        let (x0, y0) = (x.max(0.0) as usize, y.max(0.0) as usize);
        let (x1, y1) = (((x + fw).max(0.0) as usize).min(w), ((y + fh).max(0.0) as usize).min(h));
        sharp = sharp.max(sharp_map(laplacian_var(&l, w, h, x0, y0, x1, y1)));
        largest = largest.max((fw * fh) / area);
        boxes.push([x / w as f32, y / h as f32, fw / w as f32, fh / h as f32].map(|v| (v * 1000.0).round() / 1000.0));
    }
    let best = &found[0];
    let inside = |p: [f32; 2]| p[0] >= best.rect[0] && p[0] <= best.rect[0] + best.rect[2] && p[1] >= best.rect[1] && p[1] <= best.rect[1] + best.rect[3];
    FacesInfo {
        count: found.len(),
        max_score: best.score,
        largest_area: largest,
        sharpness: sharp,
        eyes: inside(best.landmarks[0]) && inside(best.landmarks[1]),
        boxes,
    }
}

fn facts_path(ws: &Workspace, hash: &str) -> std::path::PathBuf {
    let shard = hash.get(..2).unwrap_or("00");
    ws.cache_dir().join(shard).join(format!("{hash}.facts.json"))
}

/// Cached facts, when present and current (`want_faces` requires the faces to be cached too).
pub fn cached_facts(ws: &Workspace, hash: &str, want_faces: bool) -> Option<Facts> {
    let bytes = fs::read(facts_path(ws, hash)).ok()?;
    let f: Facts = serde_json::from_slice(&bytes).ok()?;
    (f.algo == ALGO && (!want_faces || f.faces.is_some())).then_some(f)
}

/// Facts for a photo: from the cache, else one decode that also fills the thumbnail cache.
pub fn facts(ws: &Workspace, hash: &str, original: &std::path::Path, want_faces: bool) -> ApiResult<Facts> {
    if let Some(f) = cached_facts(ws, hash, want_faces) {
        return Ok(f);
    }
    let bytes = pixels::read_file(original)?;
    let (proxy, (w, h)) = pixels::decode_scaled(&bytes, PROXY_SIDE)?;
    drop(bytes);
    let mut f = cached_facts(ws, hash, false).unwrap_or_else(|| Facts {
        algo: ALGO,
        width: w,
        height: h,
        dhash: format!("{:016x}", dhash(&proxy)),
        quality: measure(&proxy),
        faces: None,
    });
    if want_faces {
        f.faces = Some(faces(&proxy));
    }
    if !pixels::thumb_path(ws, hash).exists() {
        let _ = pixels::store_thumb(ws, hash, &proxy);
    }
    let bytes = serde_json::to_vec(&f).map_err(|e| ApiError::internal(e.to_string()))?;
    let _ = write_atomic(&facts_path(ws, hash), &bytes);
    Ok(f)
}

/// `{r, g, b, luma: [256], clipping, mean, pixels}` of an RGBA image (alpha ignored).
pub fn histogram(px: &[u8]) -> Value {
    let mut r = [0u64; 256];
    let mut g = [0u64; 256];
    let mut b = [0u64; 256];
    let mut l = [0u64; 256];
    let mut sum = 0f64;
    for p in px.as_chunks::<4>().0 {
        r[p[0] as usize] += 1;
        g[p[1] as usize] += 1;
        b[p[2] as usize] += 1;
        let y = (0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2])).round().clamp(0.0, 255.0) as usize;
        l[y] += 1;
        sum += y as f64;
    }
    let n = (px.len() / 4).max(1) as f64;
    let frac = |h: &[u64; 256], lo: bool| -> f64 {
        let c: u64 = if lo { h[..3].iter().sum() } else { h[253..].iter().sum() };
        ((c as f64 / n) * 10000.0).round() / 10000.0
    };
    json!({
        "r": r.to_vec(), "g": g.to_vec(), "b": b.to_vec(), "luma": l.to_vec(),
        "pixels": px.len() / 4,
        "mean": ((sum / n) * 10.0).round() / 10.0,
        "clipping": {
            "shadows": frac(&l, true), "highlights": frac(&l, false),
            "r": frac(&r, false), "g": frac(&g, false), "b": frac(&b, false),
        },
    })
}

/// Union-find clusters of hashes within `threshold` Hamming distance; indices into `hashes`.
pub fn clusters(hashes: &[Option<u64>], threshold: u32) -> Vec<Vec<usize>> {
    let n = hashes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for i in 0..n {
        let Some(a) = hashes[i] else { continue };
        for (j, h) in hashes.iter().enumerate().skip(i + 1) {
            let Some(b) = *h else { continue };
            if (a ^ b).count_ones() <= threshold {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().collect();
    out.sort_by_key(|g| g[0]);
    out
}
