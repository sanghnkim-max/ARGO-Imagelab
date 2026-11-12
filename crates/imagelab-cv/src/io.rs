//! Files and the document: read, write, fromDocument, toDocumentLayer, info, free; plus the
//! plain encode/decode helpers other crates may reuse.

use std::path::{Component, Path, PathBuf};

use imagelab_api::handles::MAX_SIDE;
use imagelab_api::{ApiError, ApiResult, Ctx, HandleId, ImageBuf, MaskBuf, parse_handle};
use photocraft_codecs::{ChannelLayout, DecodeOptions, EncodeOptions, Format, Image, Limits};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::raster::{Src, coverage, load, put_img, round4};

/// Largest file `cv.read` opens.
pub const MAX_FILE_BYTES: u64 = 1 << 30;

/// Decode any format PhotoCraft reads (PNG, JPEG, WebP, TIFF, …) to 8-bit RGBA, turned upright
/// by its EXIF/TIFF orientation.
pub fn decode_image(bytes: &[u8]) -> ApiResult<ImageBuf> {
    let opts = DecodeOptions {
        limits: Limits { max_width: MAX_SIDE, max_height: MAX_SIDE, max_alloc: 4 << 30, ..Limits::default() },
        keep_orientation: false,
    };
    let img = photocraft_codecs::decode_with(bytes, &opts).map_err(|e| ApiError::new("decode", e.to_string()))?;
    let (w, h) = img.dimensions();
    ImageBuf::new(w, h, img.to_rgba8())
}

fn encode(width: u32, height: u32, layout: ChannelLayout, data: Vec<u8>, format: Format, quality: Option<u8>) -> ApiResult<Vec<u8>> {
    let img = Image::from_u8(width, height, layout, data).map_err(|e| ApiError::internal(e.to_string()))?;
    let mut opts = EncodeOptions::default();
    if let Some(q) = quality {
        opts.jpeg_quality = q.clamp(1, 100);
        opts.webp_quality = q.clamp(1, 100);
        opts.webp_lossless = false;
    }
    photocraft_codecs::encode(&img, format, &opts).map_err(|e| ApiError::new("encode", e.to_string()))
}

/// Encode an image as PNG.
pub fn image_to_png(img: &ImageBuf) -> ApiResult<Vec<u8>> {
    encode(img.width, img.height, ChannelLayout::Rgba, img.rgba.clone(), Format::Png, None)
}

/// Encode a mask as an 8-bit grayscale PNG (255 = inside).
pub fn mask_to_png(mask: &MaskBuf) -> ApiResult<Vec<u8>> {
    encode(mask.width, mask.height, ChannelLayout::Gray, mask.alpha.clone(), Format::Png, None)
}

/// Encode an image or mask in `format`.
pub fn encode_src(src: &Src, format: Format, quality: Option<u8>) -> ApiResult<Vec<u8>> {
    match src {
        Src::Img(i) => encode(i.width, i.height, ChannelLayout::Rgba, i.rgba.clone(), format, quality),
        Src::Mask(m) => encode(m.width, m.height, ChannelLayout::Gray, m.alpha.clone(), format, quality),
    }
}

/// A workspace-relative path with only normal components.
fn relative(path: &str) -> ApiResult<PathBuf> {
    let p = Path::new(path.trim());
    if path.trim().is_empty() {
        return Err(ApiError::bad_params("path is empty"));
    }
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::Normal(s) => out.push(s),
            Component::CurDir => {}
            _ => return Err(ApiError::new("forbidden", format!("`{path}` must be relative to the workspace, without `..`"))),
        }
    }
    if out.as_os_str().is_empty() {
        return Err(ApiError::bad_params("path names no file"));
    }
    Ok(out)
}

fn canonical_root(ctx: &Ctx) -> PathBuf {
    ctx.shared.root.canonicalize().unwrap_or_else(|_| ctx.shared.root.clone())
}

/// Resolve an existing workspace file for reading.
pub fn resolve_read(ctx: &Ctx, path: &str) -> ApiResult<PathBuf> {
    let root = canonical_root(ctx);
    let full = root.join(relative(path)?);
    let canon = full.canonicalize().map_err(|_| ApiError::not_found(format!("no file `{path}` in the workspace")))?;
    if !canon.starts_with(&root) {
        return Err(ApiError::new("forbidden", format!("`{path}` resolves outside the workspace")));
    }
    Ok(canon)
}

/// Where writes may go: `exports/` (the default) or `.imagelab/tmp/`.
const WRITE_DIRS: [&str; 2] = ["exports", ".imagelab/tmp"];

/// Resolve a write target: (absolute path, workspace-relative display path).
pub fn resolve_write(ctx: &Ctx, path: &str) -> ApiResult<(PathBuf, String)> {
    let mut rel = relative(path)?;
    if !WRITE_DIRS.iter().any(|d| rel.starts_with(d)) {
        rel = Path::new("exports").join(rel);
    }
    let root = canonical_root(ctx);
    let file_name = rel.file_name().map(|s| s.to_os_string()).ok_or_else(|| ApiError::bad_params("path names no file"))?;
    let parent_rel = rel.parent().map(Path::to_path_buf).unwrap_or_default();
    let parent = root.join(&parent_rel);
    std::fs::create_dir_all(&parent).map_err(|e| ApiError::new("io", format!("cannot create {}: {e}", parent_rel.display())))?;
    let parent = parent.canonicalize().map_err(|e| ApiError::new("io", e.to_string()))?;
    if !parent.starts_with(&root) {
        return Err(ApiError::new("forbidden", format!("`{path}` resolves outside the workspace")));
    }
    let full = parent.join(&file_name);
    if let Ok(meta) = std::fs::symlink_metadata(&full)
        && meta.file_type().is_symlink()
    {
        return Err(ApiError::new("forbidden", format!("`{}` is a symbolic link", rel.display())));
    }
    Ok((full, rel.to_string_lossy().replace('\\', "/")))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Read {
    path: String,
}

pub fn read(ctx: &Ctx, p: Read) -> ApiResult<Value> {
    let full = resolve_read(ctx, &p.path)?;
    let meta = std::fs::metadata(&full).map_err(|e| ApiError::new("io", e.to_string()))?;
    if !meta.is_file() {
        return Err(ApiError::bad_params(format!("`{}` is not a file", p.path)));
    }
    if meta.len() > MAX_FILE_BYTES {
        return Err(ApiError::too_large(format!("`{}` is {} bytes; the limit is {MAX_FILE_BYTES}", p.path, meta.len())));
    }
    let bytes = std::fs::read(&full).map_err(|e| ApiError::new("io", e.to_string()))?;
    let format = photocraft_codecs::detect(&bytes).map(|f| format!("{f:?}").to_ascii_lowercase());
    let img = decode_image(&bytes)?;
    let (w, h) = (img.width as usize, img.height as usize);
    let mut r = put_img(ctx, w, h, img.rgba)?;
    r["format"] = json!(format);
    Ok(r)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Write {
    #[serde(alias = "mask", alias = "handle")]
    img: String,
    path: String,
    #[serde(default)]
    quality: Option<u8>,
    #[serde(default)]
    overwrite: bool,
}

fn write_impl(ctx: &Ctx, p: Write, allow_overwrite: bool) -> ApiResult<Value> {
    let src = load(ctx, &p.img)?;
    let (full, rel) = resolve_write(ctx, &p.path)?;
    let ext = full.extension().and_then(|e| e.to_str()).unwrap_or("");
    let format = photocraft_codecs::from_extension(ext).ok_or_else(|| ApiError::bad_params(format!("unknown file type `.{ext}`; use .png, .jpg, .webp or .tif")))?;
    if !photocraft_codecs::caps(format).write {
        return Err(ApiError::bad_params(format!("cannot write {format:?} files")));
    }
    let exists = full.exists();
    if exists && !allow_overwrite {
        let hint = if p.overwrite { "overwriting is Destructive: call cv.overwrite with the same params" } else { "pick another name, or call cv.overwrite (Destructive)" };
        return Err(ApiError::new("exists", format!("`{rel}` already exists; {hint}")));
    }
    if let Some(q) = p.quality
        && !(1..=100).contains(&q)
    {
        return Err(ApiError::bad_params("quality must be in 1..=100"));
    }
    let bytes = encode_src(&src, format, p.quality)?;
    // Write next to the target, then rename, so a failed write never leaves half a file.
    let tmp = full.with_extension(format!("{ext}.part"));
    std::fs::write(&tmp, &bytes).map_err(|e| ApiError::new("io", format!("cannot write `{rel}`: {e}")))?;
    std::fs::rename(&tmp, &full).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        ApiError::new("io", format!("cannot write `{rel}`: {e}"))
    })?;
    Ok(json!({"path": rel, "bytes": bytes.len(), "format": format!("{format:?}").to_ascii_lowercase(), "overwrote": exists}))
}

pub fn write(ctx: &Ctx, p: Write) -> ApiResult<Value> {
    write_impl(ctx, p, false)
}

pub fn overwrite(ctx: &Ctx, p: Write) -> ApiResult<Value> {
    write_impl(ctx, p, true)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct FromDocument {
    #[serde(default)]
    max_side: Option<u32>,
    #[serde(default)]
    document: Option<usize>,
}

pub fn from_document(ctx: &Ctx, p: FromDocument) -> ApiResult<Value> {
    let max_side = p.max_side.unwrap_or(2048);
    if max_side > MAX_SIDE {
        return Err(ApiError::bad_params(format!("maxSide must be at most {MAX_SIDE} (0 = full size)")));
    }
    // The headless compositor renders at most PREVIEW_SIDE a side; past that (and for full
    // size) the document is written to a temporary PNG and decoded, as an export would.
    let png = ctx.with_session(|h| {
        let (w, hgt) = match p.document {
            Some(i) => h.session.documents().get(i).map(|d| (d.doc.size.width, d.doc.size.height)),
            None => h.session.active().map(|d| (d.doc.size.width, d.doc.size.height)),
        }
        .ok_or_else(|| ApiError::not_found("no open document"))?;
        let side = if max_side == 0 { w.max(hgt) } else { max_side.min(w.max(hgt)) };
        if side <= PREVIEW_SIDE {
            return Ok(h.render_png(p.document, max_side)?);
        }
        full_size_png(ctx, h, p.document)
    })?;
    let img = decode_image(&png)?;
    let (w, h) = (img.width as usize, img.height as usize);
    if max_side != 0 && w.max(h) > max_side as usize {
        let scale = max_side as f64 / w.max(h) as f64;
        let (nw, nh) = (((w as f64 * scale).round() as usize).max(1), ((h as f64 * scale).round() as usize).max(1));
        let rgba = crate::geom::resize_buf(&img.rgba, w, h, 4, nw, nh, crate::geom::Interp::Area);
        return put_img(ctx, nw, nh, rgba);
    }
    put_img(ctx, w, h, img.rgba)
}

/// The largest side PhotoCraft's headless compositor renders.
pub const PREVIEW_SIDE: u32 = 2048;

/// The active document flattened at full size as RGBA8 (straight alpha): `(width, height, rgba)`.
/// For callers already inside `with_session`.
pub fn document_rgba(ctx: &Ctx, h: &mut imagelab_api::Headless) -> ApiResult<(usize, usize, Vec<u8>)> {
    let (w, hgt) = h.session.active().map(|d| (d.doc.size.width, d.doc.size.height)).ok_or_else(|| ApiError::not_found("no open document"))?;
    let png = if w.max(hgt) <= PREVIEW_SIDE { h.render_png(None, 0)? } else { full_size_png(ctx, h, None)? };
    let img = decode_image(&png)?;
    Ok((img.width as usize, img.height as usize, img.rgba))
}

/// The document flattened at full size, through a temporary PNG under `.imagelab/tmp/`.
pub fn full_size_png(ctx: &Ctx, h: &mut imagelab_api::Headless, document: Option<usize>) -> ApiResult<Vec<u8>> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let rel = format!(".imagelab/tmp/cv-document-{}-{n}.png", std::process::id());
    let abs = ctx.shared.root.join(&rel);
    if let Some(dir) = abs.parent() {
        std::fs::create_dir_all(dir).map_err(|e| ApiError::internal(format!("cannot make the temp folder: {e}")))?;
    }
    let saved = h.save(document, Some(Path::new(&rel)), Some("png"), &photocraft_io::ExportOptions::default());
    let bytes = saved.map_err(ApiError::from).and_then(|_| std::fs::read(&abs).map_err(|e| ApiError::internal(format!("cannot read the render: {e}"))));
    let _ = std::fs::remove_file(&abs);
    bytes
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ToDocumentLayer {
    img: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    smart_object: bool,
    #[serde(default)]
    document: Option<usize>,
}

fn sanitize(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.') { c } else { '_' }).take(64).collect();
    let s = s.trim().trim_matches('.').to_string();
    if s.is_empty() { "cv".into() } else { s }
}

pub fn to_document_layer(ctx: &Ctx, p: ToDocumentLayer) -> ApiResult<Value> {
    let img = crate::raster::load_img(ctx, &p.img)?;
    let png = image_to_png(&img)?;
    // The layer takes the file's name, so the file is named after the layer, in its own folder.
    let name = sanitize(p.name.as_deref().unwrap_or(&p.img.replace(':', "-")));
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let rel = format!(".imagelab/tmp/place-{stamp}/{name}.png");
    let (full, rel) = resolve_write(ctx, &rel)?;
    std::fs::write(&full, &png).map_err(|e| ApiError::new("io", format!("cannot write `{rel}`: {e}")))?;
    let params = json!({"fit": false, "smartObject": p.smart_object, "frame": false});
    let result = ctx.with_session(|h| Ok(h.place(p.document, &rel, &params)?));
    // In a workspace session the bytes are read before placing; drop the temporary file.
    if result.is_ok() {
        let _ = std::fs::remove_file(&full);
        if let Some(dir) = full.parent() {
            let _ = std::fs::remove_dir(dir);
        }
    }
    let placed = result?;
    Ok(json!({"placed": placed, "name": name, "width": img.width, "height": img.height}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Info {
    #[serde(default, alias = "mask", alias = "handle")]
    img: Option<String>,
}

pub fn info(ctx: &Ctx, p: Info) -> ApiResult<Value> {
    let arena = json!({"handles": ctx.shared.handles.len(), "usedBytes": ctx.shared.handles.used_bytes()});
    let Some(h) = p.img else { return Ok(json!({"arena": arena})) };
    let src = load(ctx, &h)?;
    let mut r = json!({"handle": h, "width": src.w(), "height": src.h(), "channels": src.ch(), "bytes": src.data().len(), "arena": arena});
    match &src {
        Src::Img(i) => {
            r["kind"] = json!("image");
            r["opaque"] = json!(i.rgba.chunks(4).all(|p| p[3] == 255));
        }
        Src::Mask(m) => {
            r["kind"] = json!("mask");
            r["coverage"] = json!(round4(coverage(&m.alpha)));
        }
    }
    Ok(r)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Free {
    handles: Vec<String>,
}

pub fn free(ctx: &Ctx, p: Free) -> ApiResult<Value> {
    if p.handles.len() > 10_000 {
        return Err(ApiError::bad_params("at most 10000 handles per call"));
    }
    let mut freed = 0;
    let mut missing = Vec::new();
    for s in &p.handles {
        let id: HandleId = parse_handle(s)?;
        if ctx.shared.handles.free(id) {
            freed += 1;
        } else {
            missing.push(s.clone());
        }
    }
    Ok(json!({"freed": freed, "missing": missing, "usedBytes": ctx.shared.handles.used_bytes()}))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn png_roundtrip() {
        let img = ImageBuf::new(3, 2, (0..24).map(|v| (v * 10) as u8).collect()).unwrap();
        let png = image_to_png(&img).unwrap();
        assert_eq!(decode_image(&png).unwrap(), img);
        let m = MaskBuf::new(2, 2, vec![0, 64, 128, 255]).unwrap();
        let back = decode_image(&mask_to_png(&m).unwrap()).unwrap();
        assert_eq!(back.rgba.chunks(4).map(|p| p[0]).collect::<Vec<_>>(), m.alpha);
        assert!(decode_image(b"not an image").is_err());
    }

    #[test]
    fn relative_paths_only() {
        assert!(relative("../x.png").is_err());
        assert!(relative("/etc/passwd").is_err());
        assert!(relative("a/../../b").is_err());
        assert!(relative("").is_err());
        assert_eq!(relative("./exports/a.png").unwrap(), PathBuf::from("exports/a.png"));
        assert_eq!(sanitize("a/b:c"), "a_b_c");
        assert_eq!(sanitize("..."), "cv");
    }
}
