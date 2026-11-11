//! Decoding (PhotoCraft codecs, upright by EXIF orientation), area-filter downscaling and the
//! JPEG thumbnail cache.

use std::fs;
use std::path::{Path, PathBuf};

use imagelab_api::{ApiError, ApiResult};
use photocraft_codecs::{ChannelLayout, DecodeOptions, EncodeOptions, Format, Image, Limits, SampleType};

use crate::paths::{Workspace, write_atomic};

/// Largest source file read into memory.
pub const MAX_FILE_BYTES: u64 = 512 << 20;
/// Largest decoded image (pixels): about a 200 MP sensor.
const MAX_PIXELS: u64 = 210_000_000;
/// Side of the cached thumbnail.
pub const THUMB_SIDE: u32 = 512;

/// Straight 8-bit RGBA, alpha always 255 after [`decode_scaled`] (transparency is flattened
/// over white).
#[derive(Debug, Clone)]
pub struct Rgba {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
}

impl Rgba {
    /// Rec.601 luma, 0..255.
    pub fn luma(&self) -> Vec<f32> {
        self.px.as_chunks::<4>().0.iter().map(|p| 0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2])).collect()
    }
}

pub fn read_file(path: &Path) -> ApiResult<Vec<u8>> {
    let meta = fs::metadata(path).map_err(|e| ApiError::not_found(format!("{}: {e}", path.display())))?;
    if !meta.is_file() {
        return Err(ApiError::bad_params(format!("{} is not a file", path.display())));
    }
    if meta.len() > MAX_FILE_BYTES {
        return Err(ApiError::too_large(format!("{} is {} bytes (max {MAX_FILE_BYTES})", path.display(), meta.len())));
    }
    fs::read(path).map_err(|e| ApiError::internal(format!("read {}: {e}", path.display())))
}

/// `(w, h)` scaled to fit `max_side` (never enlarged).
pub fn fit(w: u32, h: u32, max_side: u32) -> (u32, u32) {
    let max_side = max_side.max(1);
    let long = w.max(h).max(1);
    if long <= max_side {
        return (w.max(1), h.max(1));
    }
    let s = f64::from(max_side) / f64::from(long);
    (((f64::from(w) * s).round() as u32).clamp(1, max_side), ((f64::from(h) * s).round() as u32).clamp(1, max_side))
}

fn decode(bytes: &[u8]) -> ApiResult<Image> {
    let limits = Limits { max_width: 65_535, max_height: 65_535, max_pixels: MAX_PIXELS, max_alloc: 2 << 30 };
    let opts = DecodeOptions { limits, ..Default::default() };
    photocraft_codecs::decode_with(bytes, &opts).map_err(|e| ApiError::new("decode", e.to_string()))
}

/// Decode and downscale to fit `max_side`; returns the proxy and the full upright size.
pub fn decode_scaled(bytes: &[u8], max_side: u32) -> ApiResult<(Rgba, (u32, u32))> {
    let img = decode(bytes)?;
    let (w, h) = img.dimensions();
    let (tw, th) = fit(w, h, max_side);
    let channels = img.layout().channels();
    let out = match (img.sample_type(), img.layout()) {
        (SampleType::U8, ChannelLayout::Gray | ChannelLayout::GrayA | ChannelLayout::Rgb | ChannelLayout::Rgba) => {
            box_resize(img.data(), w, h, channels, img.layout(), tw, th)
        }
        _ => {
            let rgba = img.to_rgba8();
            drop(img);
            box_resize(&rgba, w, h, 4, ChannelLayout::Rgba, tw, th)
        }
    }?;
    Ok((out, (w, h)))
}

/// Area-average downscale (each target pixel averages its source box) of 8-bit samples into
/// RGBA; alpha is flattened over white. Also accepts `tw/th` equal to the source (a copy).
pub fn box_resize(src: &[u8], w: u32, h: u32, channels: usize, layout: ChannelLayout, tw: u32, th: u32) -> ApiResult<Rgba> {
    let (wu, hu, twu, thu) = (w as usize, h as usize, tw as usize, th as usize);
    if w == 0 || h == 0 || tw == 0 || th == 0 || tw > w || th > h {
        return Err(ApiError::internal(format!("bad resize {w}x{h} → {tw}x{th}")));
    }
    if src.len() < wu * hu * channels {
        return Err(ApiError::internal("pixel buffer shorter than its size"));
    }
    let has_alpha = layout.has_alpha();
    let gray = layout.is_gray();
    // Source x → target column boundaries.
    let xb: Vec<usize> = (0..=twu).map(|x| x * wu / twu).collect();
    let mut out = vec![255u8; twu * thu * 4];
    let mut col = vec![0u32; wu * 3];
    for ty in 0..thu {
        let (y0, y1) = (ty * hu / thu, ((ty + 1) * hu / thu).max(ty * hu / thu + 1).min(hu));
        col.iter_mut().for_each(|c| *c = 0);
        for y in y0..y1 {
            let row = &src[y * wu * channels..(y + 1) * wu * channels];
            for x in 0..wu {
                let p = &row[x * channels..(x + 1) * channels];
                let (mut r, mut g, mut b) = if gray { (p[0], p[0], p[0]) } else { (p[0], p[1], p[2]) };
                if has_alpha {
                    let a = u32::from(p[channels - 1]);
                    let over = |c: u8| ((u32::from(c) * a + 255 * (255 - a) + 127) / 255) as u8;
                    (r, g, b) = (over(r), over(g), over(b));
                }
                col[x * 3] += u32::from(r);
                col[x * 3 + 1] += u32::from(g);
                col[x * 3 + 2] += u32::from(b);
            }
        }
        let rows = (y1 - y0) as u64;
        for tx in 0..twu {
            let (x0, x1) = (xb[tx], xb[tx + 1].max(xb[tx] + 1).min(wu));
            let mut s = [0u64; 3];
            for x in x0..x1 {
                s[0] += u64::from(col[x * 3]);
                s[1] += u64::from(col[x * 3 + 1]);
                s[2] += u64::from(col[x * 3 + 2]);
            }
            let n = rows * (x1 - x0) as u64;
            let o = (ty * twu + tx) * 4;
            for c in 0..3 {
                out[o + c] = ((s[c] + n / 2) / n.max(1)).min(255) as u8;
            }
        }
    }
    Ok(Rgba { w: tw, h: th, px: out })
}

/// Downscale an RGBA image to fit `max_side`.
pub fn shrink(img: &Rgba, max_side: u32) -> ApiResult<Rgba> {
    let (tw, th) = fit(img.w, img.h, max_side);
    if (tw, th) == (img.w, img.h) {
        return Ok(img.clone());
    }
    box_resize(&img.px, img.w, img.h, 4, ChannelLayout::Rgba, tw, th)
}

pub fn thumb_path(ws: &Workspace, hash: &str) -> PathBuf {
    let shard = hash.get(..2).unwrap_or("00");
    ws.cache_dir().join(shard).join(format!("{hash}.thumb.jpg"))
}

/// Write the cached thumbnail (≤ [`THUMB_SIDE`]) from any proxy at least that large.
pub fn store_thumb(ws: &Workspace, hash: &str, proxy: &Rgba) -> ApiResult<()> {
    let t = shrink(proxy, THUMB_SIDE)?;
    let rgb: Vec<u8> = t.px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    let img = Image::from_u8(t.w, t.h, ChannelLayout::Rgb, rgb).map_err(|e| ApiError::internal(e.to_string()))?;
    let opts = EncodeOptions { jpeg_quality: 85, ..Default::default() };
    let bytes = photocraft_codecs::encode(&img, Format::Jpeg, &opts).map_err(|e| ApiError::internal(e.to_string()))?;
    write_atomic(&thumb_path(ws, hash), &bytes)
}

/// A preview of the photo fitting `max_side`: from the thumbnail cache when it is big enough,
/// else decoded from the original (and the cache filled on the way).
pub fn preview(ws: &Workspace, hash: &str, original: &Path, max_side: u32) -> ApiResult<Rgba> {
    let tp = thumb_path(ws, hash);
    if max_side <= THUMB_SIDE
        && let Ok(bytes) = fs::read(&tp)
        && let Ok((img, _)) = decode_scaled(&bytes, max_side)
    {
        return Ok(img);
    }
    let bytes = read_file(original)?;
    let (img, _) = decode_scaled(&bytes, max_side.max(THUMB_SIDE))?;
    if !tp.exists() {
        let _ = store_thumb(ws, hash, &img);
    }
    shrink(&img, max_side)
}
