//! What `segment.*` looks at: an `image` handle, a workspace `photo`, or (default) the open
//! document. Each becomes a ≤ [`PROXY_SIDE`] RGB proxy for the encoder, plus the full size the
//! caller's coordinates and the returned masks are in.

use std::path::{Component, Path};
use std::sync::Arc;

use imagelab_api::{ApiError, ApiResult, Ctx, ImageBuf, handles::MAX_SIDE};
use serde::Deserialize;

/// Longest side of the encoder's proxy (SAM 3's input is 1008² ; never upsampled).
pub const PROXY_SIDE: u32 = 1008;
/// Largest photo file read.
const MAX_FILE_BYTES: u64 = 512 << 20;
/// Largest decoded photo (pixels).
const MAX_PIXELS: u64 = 200_000_000;

/// The `image?` / `photo?` part of every `segment.*` call.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    /// `img:N` handle.
    pub image: Option<String>,
    /// Workspace-relative photo path.
    pub photo: Option<String>,
}

impl Source {
    pub fn is_document(&self) -> bool {
        self.image.is_none() && self.photo.is_none()
    }
}

/// Where an input's full-resolution pixels come from, for crops ([`crop`]).
pub enum Full {
    /// Decoded pixels (an `img:N` handle or a photo file).
    Pixels(Arc<ImageBuf>),
    /// The open document: regions are rendered from it.
    Document,
}

/// A resolved input.
pub struct Input {
    /// Full size: coordinates and masks are in these pixels.
    pub width: u32,
    pub height: u32,
    /// RGB proxy, long side ≤ PROXY_SIDE.
    pub rgb: Vec<u8>,
    pub pw: u32,
    pub ph: u32,
    pub kind: &'static str,
    pub full: Full,
}



pub fn resolve(ctx: &Ctx, src: &Source) -> ApiResult<Input> {
    if src.image.is_some() && src.photo.is_some() {
        return Err(ApiError::bad_params("give `image` or `photo`, not both (neither = the open document)"));
    }
    if let Some(h) = &src.image {
        let img = ctx.shared.handles.image_str(h)?;
        return Ok(from_image(img, "image"));
    }
    if let Some(p) = &src.photo {
        let img = decode_photo(ctx, p)?;
        return Ok(from_image(Arc::new(img), "photo"));
    }
    // The open document: its size, and a proxy render (never full size).
    let (size, png) = ctx.with_session(|h| {
        let size = h.session.active().map(|d| d.doc.size).ok_or_else(|| ApiError::not_found("no document is open"))?;
        let png = h.render_png(None, PROXY_SIDE)?;
        Ok((size, png))
    })?;
    let img = decode_bytes(&png)?;
    let rgb = rgba_to_rgb(&img.rgba);
    Ok(Input { width: size.width, height: size.height, rgb, pw: img.width, ph: img.height, kind: "document", full: Full::Document })
}

fn from_image(img: Arc<ImageBuf>, kind: &'static str) -> Input {
    let (pw, ph, rgba) = downscale(&img, PROXY_SIDE);
    Input { width: img.width, height: img.height, rgb: rgba_to_rgb(&rgba), pw, ph, kind, full: Full::Pixels(img) }
}

/// `w × h` scaled to fit `max` on its long side (never up).
fn fit(w: u32, h: u32, max: u32) -> (u32, u32) {
    let long = w.max(h);
    if long <= max {
        return (w, h);
    }
    let s = f64::from(max) / f64::from(long);
    (((f64::from(w) * s).round() as u32).max(1), ((f64::from(h) * s).round() as u32).max(1))
}

/// The region `[x, y, w, h]` (full-size pixels) of `input` as an input of its own: a
/// ≤ [`PROXY_SIDE`] proxy of just that region, from the full-resolution pixels (resampled from
/// decoded pixels, or rendered from the open document). Masks for it are `w × h`.
pub fn crop(ctx: &Ctx, input: &Input, r: [u32; 4]) -> ApiResult<Input> {
    let [x, y, w, h] = r;
    if w == 0 || h == 0 || x.saturating_add(w) > input.width || y.saturating_add(h) > input.height {
        return Err(ApiError::bad_params(format!("crop {r:?} is outside the {}x{} image", input.width, input.height)));
    }
    let (pw, ph, rgba) = match &input.full {
        Full::Pixels(img) => downscale_region(img, r, PROXY_SIDE),
        Full::Document => {
            let (pw, ph) = fit(w, h, PROXY_SIDE);
            let rgba = ctx.with_session(|hs| {
                let d = hs.session.active().ok_or_else(|| ApiError::not_found("no document is open"))?;
                let rect = photocraft_geom::Rect::from_xywh(x as i32, y as i32, w, h);
                Ok(photocraft_compose::render_reduced_rect(&d.doc, rect, pw, ph).to_rgba8().pixels)
            })?;
            (pw, ph, rgba)
        }
    };
    if rgba.len() != pw as usize * ph as usize * 4 {
        return Err(ApiError::internal(format!("crop render is {} bytes for {pw}x{ph}", rgba.len())));
    }
    let full = match &input.full {
        Full::Pixels(img) => Full::Pixels(img.clone()),
        Full::Document => Full::Document,
    };
    Ok(Input { width: w, height: h, rgb: rgba_to_rgb(&rgba), pw, ph, kind: input.kind, full })
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    // Composite over white so transparent areas don't read as black objects.
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            let a = p[3] as u32;
            let c = |v: u8| ((v as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
            [c(p[0]), c(p[1]), c(p[2])]
        })
        .collect()
}

/// Area-average downscale to long side ≤ `max` (no upscale).
pub fn downscale(img: &ImageBuf, max: u32) -> (u32, u32, Vec<u8>) {
    if img.width.max(img.height) <= max {
        return (img.width, img.height, img.rgba.clone());
    }
    downscale_region(img, [0, 0, img.width, img.height], max)
}

/// Area-average downscale of the region `[x, y, w, h]` to long side ≤ `max` (no upscale).
pub fn downscale_region(img: &ImageBuf, r: [u32; 4], max: u32) -> (u32, u32, Vec<u8>) {
    let [ox, oy, w, h] = r;
    let (dw, dh) = fit(w, h, max);
    let stride = img.width as usize;
    let (ox, oy) = (ox as usize, oy as usize);
    let mut out = vec![0u8; dw as usize * dh as usize * 4];
    for y in 0..dh as usize {
        let sy0 = y * h as usize / dh as usize;
        let sy1 = ((y + 1) * h as usize / dh as usize).max(sy0 + 1).min(h as usize);
        for x in 0..dw as usize {
            let sx0 = x * w as usize / dw as usize;
            let sx1 = ((x + 1) * w as usize / dw as usize).max(sx0 + 1).min(w as usize);
            let mut acc = [0u64; 4];
            for sy in sy0..sy1 {
                let row = (oy + sy) * stride * 4;
                for sx in sx0..sx1 {
                    let i = row + (ox + sx) * 4;
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += img.rgba.get(i + c).copied().unwrap_or(0) as u64;
                    }
                }
            }
            let n = ((sy1 - sy0) * (sx1 - sx0)).max(1) as u64;
            let o = (y * dw as usize + x) * 4;
            for c in 0..4 {
                out[o + c] = ((acc[c] + n / 2) / n) as u8;
            }
        }
    }
    (dw, dh, out)
}

/// A workspace-relative path, refused if it escapes the workspace.
pub fn workspace_path(ctx: &Ctx, rel: &str) -> ApiResult<std::path::PathBuf> {
    let p = Path::new(rel);
    if p.is_absolute() || p.components().any(|c| matches!(c, Component::ParentDir | Component::Prefix(_) | Component::RootDir)) {
        return Err(ApiError::bad_params(format!("`{rel}` must be a path inside the workspace (relative, no `..`)")));
    }
    let full = ctx.shared.root.join(p);
    // Symlinks must not lead out either.
    if let (Ok(c), Ok(root)) = (full.canonicalize(), ctx.shared.root.canonicalize())
        && !c.starts_with(&root)
    {
        return Err(ApiError::bad_params(format!("`{rel}` leads outside the workspace")));
    }
    Ok(full)
}

fn decode_photo(ctx: &Ctx, rel: &str) -> ApiResult<ImageBuf> {
    let path = workspace_path(ctx, rel)?;
    let meta = std::fs::metadata(&path).map_err(|e| ApiError::not_found(format!("{rel}: {e}")))?;
    if meta.len() > MAX_FILE_BYTES {
        return Err(ApiError::too_large(format!("{rel} is {} bytes (limit {MAX_FILE_BYTES})", meta.len())));
    }
    let bytes = std::fs::read(&path).map_err(|e| ApiError::not_found(format!("{rel}: {e}")))?;
    decode_bytes(&bytes)
}

/// Decode an encoded image (PNG, JPEG, …) to RGBA8, bounded.
pub fn decode_bytes(bytes: &[u8]) -> ApiResult<ImageBuf> {
    let mut opts = photocraft_codecs::DecodeOptions::default();
    opts.limits.max_width = opts.limits.max_width.min(MAX_SIDE);
    opts.limits.max_height = opts.limits.max_height.min(MAX_SIDE);
    opts.limits.max_pixels = opts.limits.max_pixels.min(MAX_PIXELS);
    let img = photocraft_codecs::decode_with(bytes, &opts).map_err(|e| ApiError::new("decode", e.to_string()))?;
    let (w, h) = img.dimensions();
    ImageBuf::new(w, h, img.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downscale_bounds_long_side_and_averages() {
        let img = ImageBuf::new(4, 2, [10u8, 20, 30, 255].repeat(8)).unwrap();
        assert_eq!(downscale(&img, 8).0, 4);
        let (w, h, px) = downscale(&img, 2);
        assert_eq!((w, h), (2, 1));
        assert_eq!(&px[..4], &[10, 20, 30, 255]);
        assert_eq!(rgba_to_rgb(&[0, 0, 0, 0, 255, 0, 0, 255]), vec![255, 255, 255, 255, 0, 0]);
    }
}
