//! Handles to PNG: what a script returns as `{image: "img:N"}` becomes an image block, and the
//! app draws mask overlays from the same bytes (`renderHandle`).

use imagelab_api::{ApiError, ApiResult, HandleId, Handles};
use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image};

/// Encode the image or mask behind `id` as PNG, scaled down (box filter) so its longest side
/// is at most `max_side` (0: full size). Images are RGBA, masks 8-bit grayscale.
pub fn encode_handle_png(handles: &Handles, id: HandleId, max_side: u32) -> ApiResult<Vec<u8>> {
    let (width, height, channels, data) = match id {
        HandleId::Image(_) => {
            let img = handles.image(id)?;
            (img.width, img.height, 4usize, img.rgba.clone())
        }
        HandleId::Mask(_) => {
            let mask = handles.mask(id)?;
            (mask.width, mask.height, 1usize, mask.alpha.clone())
        }
    };
    let (w, h, data) = downscale(width, height, channels, data, max_side);
    let layout = if channels == 4 { ChannelLayout::Rgba } else { ChannelLayout::Gray };
    let image = Image::from_u8(w, h, layout, data).map_err(|e| ApiError::internal(format!("png: {e}")))?;
    photocraft_codecs::encode(&image, Format::Png, &EncodeOptions::default()).map_err(|e| ApiError::internal(format!("png: {e}")))
}

/// Box-filter `data` (`channels` interleaved u8 samples) so the longest side is `max_side`.
fn downscale(width: u32, height: u32, channels: usize, data: Vec<u8>, max_side: u32) -> (u32, u32, Vec<u8>) {
    let longest = width.max(height);
    if max_side == 0 || longest <= max_side {
        return (width, height, data);
    }
    let scale = f64::from(max_side) / f64::from(longest);
    let dw = ((f64::from(width) * scale).round() as u32).max(1);
    let dh = ((f64::from(height) * scale).round() as u32).max(1);
    let (sw, sh) = (width as usize, height as usize);
    let mut out = vec![0u8; dw as usize * dh as usize * channels];
    for dy in 0..dh as usize {
        let y0 = dy * sh / dh as usize;
        let y1 = ((dy + 1) * sh / dh as usize).max(y0 + 1).min(sh);
        for dx in 0..dw as usize {
            let x0 = dx * sw / dw as usize;
            let x1 = ((dx + 1) * sw / dw as usize).max(x0 + 1).min(sw);
            let mut acc = [0u64; 4];
            for y in y0..y1 {
                let row = y * sw;
                for x in x0..x1 {
                    let at = (row + x) * channels;
                    for (c, slot) in acc.iter_mut().enumerate().take(channels) {
                        *slot += u64::from(data.get(at + c).copied().unwrap_or(0));
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)).max(1) as u64;
            let at = (dy * dw as usize + dx) * channels;
            for (c, sum) in acc.iter().enumerate().take(channels) {
                if let Some(slot) = out.get_mut(at + c) {
                    *slot = ((sum + n / 2) / n) as u8;
                }
            }
        }
    }
    (dw, dh, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use imagelab_api::{ImageBuf, MaskBuf};

    #[test]
    fn encodes_images_and_masks() {
        let h = Handles::default();
        let img = h.put_image(ImageBuf::new(8, 4, vec![200; 8 * 4 * 4]).unwrap()).unwrap();
        let mask = h.put_mask(MaskBuf::new(8, 4, vec![255; 32]).unwrap()).unwrap();
        let png = encode_handle_png(&h, img, 0).unwrap();
        assert!(png.starts_with(b"\x89PNG"));
        let small = encode_handle_png(&h, mask, 4).unwrap();
        assert!(small.starts_with(b"\x89PNG"));
        let decoded = photocraft_codecs::decode(&small).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (4, 2));
    }

    #[test]
    fn downscale_averages() {
        let (w, h, d) = downscale(2, 2, 1, vec![0, 255, 255, 0], 1);
        assert_eq!((w, h), (1, 1));
        assert_eq!(d, vec![128]);
    }
}
