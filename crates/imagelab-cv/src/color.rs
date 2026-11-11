//! Colour: cvtColor, adjust, histogram, meanStd, extractChannel.

use std::sync::OnceLock;

use imagelab_api::{ApiError, ApiResult, Ctx};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::raster::{Channel, Src, channel_of, load, load_img, load_mask, put_img, put_mask, round2, round4, same_size};

/// RGB → (hue 0..360, saturation 0..1, value 0..1).
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d <= 0.0 {
        0.0
    } else if max == r {
        60.0 * ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    let s = if max <= 0.0 { 0.0 } else { d / max };
    (h, s, max)
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    let q = |t: f32| ((t + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    (q(r), q(g), q(b))
}

fn srgb_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let c = i as f32 / 255.0;
            *v = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        }
        t
    })
}

fn to_srgb8(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let c = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
    (c * 255.0).round().clamp(0.0, 255.0) as u8
}

const WX: f32 = 0.950_456;
const WZ: f32 = 1.088_754;

/// sRGB → CIE L*a*b* (D65): L 0..100, a and b roughly −128..127.
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let t = srgb_lut();
    let (r, g, b) = (t[r as usize], t[g as usize], t[b as usize]);
    let x = (0.412_453 * r + 0.357_580 * g + 0.180_423 * b) / WX;
    let y = 0.212_671 * r + 0.715_160 * g + 0.072_169 * b;
    let z = (0.019_334 * r + 0.119_193 * g + 0.950_227 * b) / WZ;
    let f = |t: f32| if t > 0.008_856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

pub fn lab_to_rgb(l: f32, a: f32, b: f32) -> (u8, u8, u8) {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let inv = |t: f32| if t > 0.206_893 { t * t * t } else { (t - 16.0 / 116.0) / 7.787 };
    let (x, y, z) = (inv(fx) * WX, inv(fy), inv(fz) * WZ);
    let r = 3.240_48 * x - 1.537_15 * y - 0.498_54 * z;
    let g = -0.969_26 * x + 1.875_99 * y + 0.041_56 * z;
    let bl = 0.055_65 * x - 0.204_04 * y + 1.057_31 * z;
    (to_srgb8(r), to_srgb8(g), to_srgb8(bl))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CvtColor {
    img: String,
    code: String,
}

/// The conversions, accepting OpenCV spellings (`COLOR_RGB2HSV`, `rgb2hsv`, `hsv`).
fn parse_code(code: &str) -> ApiResult<&'static str> {
    let c = code.trim().to_ascii_lowercase().replace("color_", "").replace(['_', ' '], "");
    let c = c.trim_start_matches("rgba2").trim_start_matches("rgb2").trim_start_matches("bgr2").trim_start_matches("bgra2");
    Ok(match c {
        "gray" | "grey" => "gray",
        "hsv" | "hsvfull" => "hsv",
        "lab" => "lab",
        "ycrcb" | "ycc" => "ycrcb",
        "bgr" | "rgb" | "swaprb" => "bgr",
        "hsv2rgb" | "hsv2bgr" | "hsvfull2rgb" | "fromhsv" => "hsv2rgb",
        "lab2rgb" | "lab2bgr" | "fromlab" => "lab2rgb",
        "ycrcb2rgb" | "ycrcb2bgr" | "fromycrcb" => "ycrcb2rgb",
        _ => return Err(ApiError::bad_params(format!("unknown code `{code}`; use gray, hsv, lab, ycrcb, bgr, hsv2rgb, lab2rgb, ycrcb2rgb"))),
    })
}

pub fn cvt_color_buf(rgba: &[u8], code: &str) -> ApiResult<Vec<u8>> {
    let code = parse_code(code)?;
    let mut out = rgba.to_vec();
    out.par_chunks_mut(4).for_each(|p| {
        let (r, g, b) = (p[0], p[1], p[2]);
        let (x, y, z) = match code {
            "gray" => {
                let l = crate::raster::luma(r, g, b);
                (l, l, l)
            }
            "hsv" => {
                let (h, s, v) = rgb_to_hsv(r, g, b);
                ((h / 2.0).round().min(179.0) as u8, (s * 255.0).round() as u8, (v * 255.0).round() as u8)
            }
            "hsv2rgb" => hsv_to_rgb(r as f32 * 2.0, g as f32 / 255.0, b as f32 / 255.0),
            "lab" => {
                let (l, a, bb) = rgb_to_lab(r, g, b);
                let q = |v: f32| v.round().clamp(0.0, 255.0) as u8;
                (q(l * 255.0 / 100.0), q(a + 128.0), q(bb + 128.0))
            }
            "lab2rgb" => lab_to_rgb(r as f32 * 100.0 / 255.0, g as f32 - 128.0, b as f32 - 128.0),
            "ycrcb" => {
                let (rf, gf, bf) = (r as f32, g as f32, b as f32);
                let yy = 0.299 * rf + 0.587 * gf + 0.114 * bf;
                let q = |v: f32| v.round().clamp(0.0, 255.0) as u8;
                (q(yy), q((rf - yy) * 0.713 + 128.0), q((bf - yy) * 0.564 + 128.0))
            }
            "ycrcb2rgb" => {
                let (yy, cr, cb) = (r as f32, g as f32 - 128.0, b as f32 - 128.0);
                let q = |v: f32| v.round().clamp(0.0, 255.0) as u8;
                (q(yy + 1.403 * cr), q(yy - 0.714 * cr - 0.344 * cb), q(yy + 1.773 * cb))
            }
            _ => (b, g, r),
        };
        p[0] = x;
        p[1] = y;
        p[2] = z;
    });
    Ok(out)
}

pub fn cvt_color(ctx: &Ctx, p: CvtColor) -> ApiResult<Value> {
    let img = load_img(ctx, &p.img)?;
    let out = cvt_color_buf(&img.rgba, &p.code)?;
    put_img(ctx, img.width as usize, img.height as usize, out)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractChannel {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default)]
    channel: Channel,
}

pub fn extract_channel(ctx: &Ctx, p: ExtractChannel) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let plane = channel_of(&src, p.channel)?;
    put_mask(ctx, src.w(), src.h(), plane)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Adjust {
    img: String,
    #[serde(default)]
    brightness: f32,
    #[serde(default)]
    contrast: f32,
    #[serde(default)]
    saturation: f32,
    #[serde(default = "one")]
    gamma: f32,
}

fn one() -> f32 {
    1.0
}

pub fn adjust_buf(rgba: &[u8], brightness: f32, contrast: f32, saturation: f32, gamma: f32) -> ApiResult<Vec<u8>> {
    for (n, v) in [("brightness", brightness), ("contrast", contrast), ("saturation", saturation)] {
        if !(-100.0..=100.0).contains(&v) {
            return Err(ApiError::bad_params(format!("{n} must be in -100..=100")));
        }
    }
    if !(0.1..=10.0).contains(&gamma) {
        return Err(ApiError::bad_params("gamma must be in 0.1..=10"));
    }
    let cf = if contrast >= 0.0 { 1.0 + contrast / 100.0 * 2.0 } else { 1.0 + contrast / 100.0 };
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        let mut x = i as f32 / 255.0;
        x = x.powf(1.0 / gamma);
        x = (x - 0.5) * cf + 0.5 + brightness / 100.0;
        *v = (x * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    let sf = 1.0 + saturation / 100.0;
    let mut out = rgba.to_vec();
    out.par_chunks_mut(4).for_each(|p| {
        let (r, g, b) = (lut[p[0] as usize] as f32, lut[p[1] as usize] as f32, lut[p[2] as usize] as f32);
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let q = |c: f32| (y + (c - y) * sf).round().clamp(0.0, 255.0) as u8;
        p[0] = q(r);
        p[1] = q(g);
        p[2] = q(b);
    });
    Ok(out)
}

pub fn adjust(ctx: &Ctx, p: Adjust) -> ApiResult<Value> {
    let img = load_img(ctx, &p.img)?;
    let out = adjust_buf(&img.rgba, p.brightness, p.contrast, p.saturation, p.gamma)?;
    put_img(ctx, img.width as usize, img.height as usize, out)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Histogram {
    img: String,
    #[serde(default)]
    channel: Option<Channel>,
    #[serde(default)]
    channels: Option<Vec<Channel>>,
    #[serde(default)]
    bins: Option<usize>,
    #[serde(default)]
    mask: Option<String>,
}

fn stats_of(hist: &[u64; 256]) -> Value {
    let n: u64 = hist.iter().sum();
    if n == 0 {
        return json!({"count": 0});
    }
    let pct = |q: f64| {
        let target = (q * n as f64).ceil().max(1.0) as u64;
        let mut acc = 0;
        for (i, &c) in hist.iter().enumerate() {
            acc += c;
            if acc >= target {
                return i;
            }
        }
        255
    };
    let mean = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum::<f64>() / n as f64;
    let var = hist.iter().enumerate().map(|(i, &c)| (i as f64 - mean).powi(2) * c as f64).sum::<f64>() / n as f64;
    let min = hist.iter().position(|&c| c > 0).unwrap_or(0);
    let max = hist.iter().rposition(|&c| c > 0).unwrap_or(255);
    let low = hist[..3].iter().sum::<u64>() as f64 / n as f64;
    let high = hist[253..].iter().sum::<u64>() as f64 / n as f64;
    json!({"count": n, "mean": round2(mean), "std": round2(var.sqrt()), "min": min, "max": max,
           "p1": pct(0.01), "p5": pct(0.05), "median": pct(0.5), "p95": pct(0.95), "p99": pct(0.99),
           "clippedLow": round4(low), "clippedHigh": round4(high)})
}

fn hist_of(plane: &[u8], mask: Option<&[u8]>) -> [u64; 256] {
    let chunk = 1 << 16;
    let parts: Vec<[u64; 256]> = plane
        .par_chunks(chunk)
        .enumerate()
        .map(|(ci, c)| {
            let mut h = [0u64; 256];
            match mask {
                Some(m) => {
                    let m = &m[ci * chunk..ci * chunk + c.len()];
                    for (&v, &mv) in c.iter().zip(m) {
                        if mv >= 128 {
                            h[v as usize] += 1;
                        }
                    }
                }
                None => {
                    for &v in c {
                        h[v as usize] += 1;
                    }
                }
            }
            h
        })
        .collect();
    let mut h = [0u64; 256];
    for p in parts {
        for (a, b) in h.iter_mut().zip(p) {
            *a += b;
        }
    }
    h
}

pub fn histogram(ctx: &Ctx, p: Histogram) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let bins = p.bins.unwrap_or(256);
    if !(1..=256).contains(&bins) {
        return Err(ApiError::bad_params("bins must be in 1..=256"));
    }
    let mask = match &p.mask {
        Some(m) => {
            let m = load_mask(ctx, m)?;
            same_size((src.w(), src.h()), (m.width as usize, m.height as usize), "histogram mask")?;
            Some(m)
        }
        None => None,
    };
    let chans = match (p.channels, p.channel) {
        (Some(c), _) if !c.is_empty() => c,
        (_, Some(c)) => vec![c],
        _ => vec![Channel::Gray],
    };
    if chans.len() > 9 {
        return Err(ApiError::bad_params("at most 9 channels"));
    }
    let mut out = serde_json::Map::new();
    for c in chans {
        let plane = channel_of(&src, c)?;
        let h = hist_of(&plane, mask.as_ref().map(|m| m.alpha.as_slice()));
        let mut binned = vec![0u64; bins];
        for (i, &v) in h.iter().enumerate() {
            binned[i * bins / 256] += v;
        }
        let name = format!("{c:?}").to_ascii_lowercase();
        out.insert(name, json!({"hist": binned, "stats": stats_of(&h)}));
    }
    Ok(json!({"bins": bins, "channels": out}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeanStd {
    img: String,
    #[serde(default)]
    mask: Option<String>,
}

pub fn mean_std(ctx: &Ctx, p: MeanStd) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let mask = match &p.mask {
        Some(m) => {
            let m = load_mask(ctx, m)?;
            same_size((src.w(), src.h()), (m.width as usize, m.height as usize), "meanStd mask")?;
            Some(m)
        }
        None => None,
    };
    let ch = src.ch();
    let data = src.data();
    let n = src.w() * src.h();
    let (sum, sq, count) = (0..n)
        .into_par_iter()
        .fold(
            || ([0f64; 4], [0f64; 4], 0u64),
            |(mut s, mut q, mut c), i| {
                if mask.as_ref().is_none_or(|m| m.alpha[i] >= 128) {
                    for k in 0..ch {
                        let v = data[i * ch + k] as f64;
                        s[k] += v;
                        q[k] += v * v;
                    }
                    c += 1;
                }
                (s, q, c)
            },
        )
        .reduce(
            || ([0f64; 4], [0f64; 4], 0u64),
            |a, b| {
                let mut s = a.0;
                let mut q = a.1;
                for k in 0..4 {
                    s[k] += b.0[k];
                    q[k] += b.1[k];
                }
                (s, q, a.2 + b.2)
            },
        );
    if count == 0 {
        return Ok(json!({"count": 0}));
    }
    let c = count as f64;
    let mean: Vec<f64> = (0..ch).map(|k| round2(sum[k] / c)).collect();
    let std: Vec<f64> = (0..ch).map(|k| round2((sq[k] / c - (sum[k] / c).powi(2)).max(0.0).sqrt())).collect();
    let mut r = json!({"count": count, "mean": mean, "std": std});
    if let Src::Img(_) = src {
        r["channels"] = json!(["r", "g", "b", "a"]);
    }
    Ok(r)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn color_roundtrips() {
        for &(r, g, b) in &[(255u8, 0u8, 0u8), (12, 200, 99), (0, 0, 0), (255, 255, 255), (128, 64, 200)] {
            let (h, s, v) = rgb_to_hsv(r, g, b);
            let (r2, g2, b2) = hsv_to_rgb(h, s, v);
            assert!((r as i32 - r2 as i32).abs() <= 1 && (g as i32 - g2 as i32).abs() <= 1 && (b as i32 - b2 as i32).abs() <= 1);
            let (l, a, bb) = rgb_to_lab(r, g, b);
            let (r3, g3, b3) = lab_to_rgb(l, a, bb);
            assert!((r as i32 - r3 as i32).abs() <= 1 && (g as i32 - g3 as i32).abs() <= 1 && (b as i32 - b3 as i32).abs() <= 1, "{r},{g},{b}");
        }
        let (l, _, _) = rgb_to_lab(255, 255, 255);
        assert!((l - 100.0).abs() < 0.1);
        let px = [10u8, 120, 240, 255];
        for (to, back) in [("hsv", "hsv2rgb"), ("COLOR_RGB2Lab", "lab2rgb"), ("ycrcb", "ycrcb2rgb")] {
            let f = cvt_color_buf(&px, to).unwrap();
            let b = cvt_color_buf(&f, back).unwrap();
            for i in 0..3 {
                // 8-bit Lab quantizes coarsely near the gamut edge (as in OpenCV).
                let tol = if to.contains("Lab") { 12 } else { 3 };
                assert!((px[i] as i32 - b[i] as i32).abs() <= tol, "{to}: {px:?} -> {f:?} -> {b:?}");
            }
        }
        assert!(cvt_color_buf(&px, "nope").is_err());
    }

    #[test]
    fn adjust_identity() {
        let px: Vec<u8> = (0..=255u8).flat_map(|v| [v, 255 - v, v / 2, 255]).collect();
        assert_eq!(adjust_buf(&px, 0.0, 0.0, 0.0, 1.0).unwrap(), px);
        let gray = adjust_buf(&px, 0.0, 0.0, -100.0, 1.0).unwrap();
        assert!(gray.chunks(4).all(|p| p[0].abs_diff(p[1]) <= 1 && p[1].abs_diff(p[2]) <= 1));
    }
}
