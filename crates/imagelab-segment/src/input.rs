//! What `segment.*` looks at: an `image` handle, a workspace `photo`, or (default) the open
//! document. Each becomes a ≤ [`PROXY_SIDE`] RGB proxy for the encoder, plus the full size the
//! caller's coordinates and the returned masks are in.

use std::path::{Component, Path};

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
}

pub fn resolve(ctx: &Ctx, src: &Source) -> ApiResult<Input> {
    if src.image.is_some() && src.photo.is_some() {
        return Err(ApiError::bad_params("give `image` or `photo`, not both (neither = the open document)"));
    }
    if let Some(h) = &src.image {
        let img = ctx.shared.handles.image_str(h)?;
        return Ok(from_image(&img, "image"));
    }
    if let Some(p) = &src.photo {
        let img = decode_photo(ctx, p)?;
        return Ok(from_image(&img, "photo"));
    }
    // The open document: its size, and a proxy render (never full size).
    let (size, png) = ctx.with_session(|h| {
        let size = h.session.active().map(|d| d.doc.size).ok_or_else(|| ApiError::not_found("no document is open"))?;
        let png = h.render_png(None, PROXY_SIDE)?;
        Ok((size, png))
    })?;
    let img = decode_bytes(&png)?;
    let rgb = rgba_to_rgb(&img.rgba);
    Ok(Input { width: size.width, height: size.height, rgb, pw: img.width, ph: img.height, kind: "document" })
}

fn from_image(img: &ImageBuf, kind: &'static str) -> Input {
    let (pw, ph, rgba) = downscale(img, PROXY_SIDE);
    Input { width: img.width, height: img.height, rgb: rgba_to_rgb(&rgba), pw, ph, kind }
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
    let (w, h) = (img.width, img.height);
    let long = w.max(h);
    if long <= max {
        return (w, h, img.rgba.clone());
    }
    let s = max as f64 / long as f64;
    let (dw, dh) = (((w as f64 * s).round() as u32).max(1), ((h as f64 * s).round() as u32).max(1));
    let mut out = vec![0u8; dw as usize * dh as usize * 4];
    for y in 0..dh as usize {
        let sy0 = y * h as usize / dh as usize;
        let sy1 = ((y + 1) * h as usize / dh as usize).max(sy0 + 1).min(h as usize);
        for x in 0..dw as usize {
            let sx0 = x * w as usize / dw as usize;
            let sx1 = ((x + 1) * w as usize / dw as usize).max(sx0 + 1).min(w as usize);
            let mut acc = [0u64; 4];
            for sy in sy0..sy1 {
                let row = sy * w as usize * 4;
                for sx in sx0..sx1 {
                    let i = row + sx * 4;
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
