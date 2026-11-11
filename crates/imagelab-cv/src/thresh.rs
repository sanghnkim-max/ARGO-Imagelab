//! threshold and adaptiveThreshold: images (or masks) to masks.

use imagelab_api::{ApiError, ApiResult, Ctx};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::raster::{Channel, box_blur, channel_of, gaussian, load, put_mask};

/// Otsu's threshold of a 256-bin histogram: the level maximizing between-class variance.
pub fn otsu(hist: &[u64; 256]) -> u8 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0;
    }
    let sum_all: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let (mut wb, mut sb, mut best, mut best_t) = (0f64, 0f64, -1f64, 0u8);
    for (t, &c) in hist.iter().enumerate() {
        wb += c as f64;
        if wb == 0.0 {
            continue;
        }
        let wf = total as f64 - wb;
        if wf == 0.0 {
            break;
        }
        sb += t as f64 * c as f64;
        let (mb, mf) = (sb / wb, (sum_all - sb) / wf);
        let between = wb * wf * (mb - mf) * (mb - mf);
        if between > best {
            best = between;
            best_t = t as u8;
        }
    }
    best_t
}

/// The triangle method (Zack 1977): for unimodal histograms with a long tail.
pub fn triangle(hist: &[u64; 256]) -> u8 {
    let Some(lo) = hist.iter().position(|&c| c > 0) else { return 0 };
    let hi = hist.iter().rposition(|&c| c > 0).unwrap_or(255);
    let (peak, &pc) = hist.iter().enumerate().max_by_key(|&(_, c)| *c).unwrap_or((0, &0));
    // Use the longer side of the peak.
    let (end, flip) = if peak - lo > hi - peak { (lo, true) } else { (hi, false) };
    let (x1, y1, x2, y2) = (peak as f64, pc as f64, end as f64, hist[end] as f64);
    let (a, b) = (y2 - y1, x1 - x2);
    let norm = (a * a + b * b).sqrt().max(1e-9);
    let range: Box<dyn Iterator<Item = usize>> = if flip { Box::new(end..=peak) } else { Box::new(peak..=end) };
    let mut best = (0f64, peak);
    for i in range {
        let d = (a * (i as f64 - x1) + b * (hist[i] as f64 - y1)).abs() / norm;
        if d > best.0 {
            best = (d, i);
        }
    }
    best.1 as u8
}

pub fn hist256(plane: &[u8]) -> [u64; 256] {
    let mut h = [0u64; 256];
    for &v in plane {
        h[v as usize] += 1;
    }
    h
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThreshType {
    #[serde(alias = "thresh_binary")]
    Binary,
    #[serde(alias = "binaryInv", alias = "thresh_binary_inv", rename = "binary_inv")]
    BinaryInv,
    #[serde(alias = "thresh_trunc")]
    Trunc,
    #[serde(alias = "toZero", alias = "thresh_tozero")]
    Tozero,
    #[serde(alias = "toZeroInv", alias = "thresh_tozero_inv", rename = "tozero_inv")]
    TozeroInv,
    /// Binary with Otsu's automatic level.
    #[serde(alias = "thresh_otsu")]
    Otsu,
    /// Binary with the triangle method's automatic level.
    #[serde(alias = "thresh_triangle")]
    Triangle,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Threshold {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default)]
    thresh: Option<f64>,
    #[serde(default = "max255", alias = "maxval")]
    max_value: f64,
    #[serde(rename = "type", default = "binary")]
    kind: ThreshType,
    /// Combine with an automatic level (OpenCV `THRESH_BINARY | THRESH_OTSU`).
    #[serde(default)]
    otsu: bool,
    #[serde(default)]
    channel: Channel,
}

fn max255() -> f64 {
    255.0
}
fn binary() -> ThreshType {
    ThreshType::Binary
}

pub fn threshold_buf(plane: &[u8], t: u8, maxv: u8, kind: ThreshType) -> Vec<u8> {
    plane
        .par_iter()
        .map(|&v| match kind {
            ThreshType::Binary | ThreshType::Otsu | ThreshType::Triangle => {
                if v > t {
                    maxv
                } else {
                    0
                }
            }
            ThreshType::BinaryInv => {
                if v > t {
                    0
                } else {
                    maxv
                }
            }
            ThreshType::Trunc => v.min(t),
            ThreshType::Tozero => {
                if v > t {
                    v
                } else {
                    0
                }
            }
            ThreshType::TozeroInv => {
                if v > t {
                    0
                } else {
                    v
                }
            }
        })
        .collect()
}

pub fn threshold(ctx: &Ctx, p: Threshold) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let plane = channel_of(&src, p.channel)?;
    let auto = match p.kind {
        ThreshType::Otsu => Some(otsu(&hist256(&plane))),
        ThreshType::Triangle => Some(triangle(&hist256(&plane))),
        _ if p.otsu => Some(otsu(&hist256(&plane))),
        _ => None,
    };
    let t = match (auto, p.thresh) {
        (Some(a), _) => a,
        (None, Some(t)) if (0.0..=255.0).contains(&t) => t.floor() as u8,
        (None, Some(_)) => return Err(ApiError::bad_params("thresh must be in 0..=255")),
        (None, None) => return Err(ApiError::bad_params("give thresh, or type otsu/triangle")),
    };
    if !(0.0..=255.0).contains(&p.max_value) {
        return Err(ApiError::bad_params("maxValue must be in 0..=255"));
    }
    let out = threshold_buf(&plane, t, p.max_value.round() as u8, p.kind);
    let mut r = put_mask(ctx, src.w(), src.h(), out)?;
    r["thresh"] = json!(t);
    Ok(r)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct AdaptiveThreshold {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default = "eleven")]
    block_size: usize,
    #[serde(default = "two", rename = "C", alias = "c")]
    c: f64,
    #[serde(default, alias = "adaptiveMethod")]
    method: Option<String>,
    #[serde(rename = "type", default = "binary")]
    kind: ThreshType,
    #[serde(default = "max255", alias = "maxval")]
    max_value: f64,
    #[serde(default)]
    channel: Channel,
}

fn eleven() -> usize {
    11
}
fn two() -> f64 {
    2.0
}

pub fn adaptive_threshold(ctx: &Ctx, p: AdaptiveThreshold) -> ApiResult<Value> {
    if p.block_size.is_multiple_of(2) || !(3..=255).contains(&p.block_size) {
        return Err(ApiError::bad_params("blockSize must be odd, 3..=255"));
    }
    if !(-255.0..=255.0).contains(&p.c) || !(0.0..=255.0).contains(&p.max_value) {
        return Err(ApiError::bad_params("C must be in -255..=255 and maxValue in 0..=255"));
    }
    let inv = match p.kind {
        ThreshType::Binary => false,
        ThreshType::BinaryInv => true,
        _ => return Err(ApiError::bad_params("type must be binary or binary_inv")),
    };
    let src = load(ctx, &p.img)?;
    let (w, h) = (src.w(), src.h());
    let plane = channel_of(&src, p.channel)?;
    let local = match p.method.as_deref().unwrap_or("mean") {
        "mean" | "adaptive_thresh_mean_c" => box_blur(&plane, w, h, 1, p.block_size / 2),
        "gaussian" | "adaptive_thresh_gaussian_c" => gaussian(&plane, w, h, 1, Some(p.block_size), 0.0),
        _ => return Err(ApiError::bad_params("method must be mean or gaussian")),
    };
    let maxv = p.max_value.round() as u8;
    let c = p.c;
    let out: Vec<u8> = plane
        .par_iter()
        .zip(local.par_iter())
        .map(|(&v, &m)| {
            let above = v as f64 > m as f64 - c;
            if above != inv { maxv } else { 0 }
        })
        .collect();
    put_mask(ctx, w, h, out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn otsu_splits_bimodal() {
        let mut plane = vec![40u8; 500];
        plane.extend(vec![200u8; 300]);
        for (i, v) in plane.iter_mut().enumerate() {
            *v = v.saturating_add((i % 7) as u8);
        }
        let t = otsu(&hist256(&plane));
        assert!((46..200).contains(&t), "otsu {t}");
        let m = threshold_buf(&plane, t, 255, ThreshType::Binary);
        assert_eq!(m.iter().filter(|&&v| v == 255).count(), 300);
    }

    #[test]
    fn triangle_on_skewed() {
        let mut h = [0u64; 256];
        for (i, v) in h.iter_mut().enumerate().take(120).skip(10) {
            *v = (1000.0 * (-(i as f64 - 20.0).abs() / 15.0).exp()) as u64 + 1;
        }
        let t = triangle(&h);
        assert!(t > 20 && t < 120, "triangle {t}");
    }

    #[test]
    fn threshold_types() {
        let p = [10u8, 100, 200];
        assert_eq!(threshold_buf(&p, 100, 255, ThreshType::Binary), vec![0, 0, 255]);
        assert_eq!(threshold_buf(&p, 100, 255, ThreshType::BinaryInv), vec![255, 255, 0]);
        assert_eq!(threshold_buf(&p, 100, 255, ThreshType::Trunc), vec![10, 100, 100]);
        assert_eq!(threshold_buf(&p, 100, 255, ThreshType::Tozero), vec![0, 0, 200]);
        assert_eq!(threshold_buf(&p, 100, 255, ThreshType::TozeroInv), vec![10, 100, 0]);
    }
}
