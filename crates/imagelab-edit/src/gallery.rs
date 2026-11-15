//! Gallery photos as files: a program that edits or exports a gallery photo works on a copy of
//! the original placed in the workspace (`imports/gallery/`). The gallery itself is only read.

use imagelab_api::{ApiError, ApiResult, GallerySlot};

use crate::paths;

/// Where gallery copies live, relative to the workspace.
pub const STAGING_DIR: &str = "imports/gallery";

/// A gallery original copied into the workspace.
pub struct Staged {
    /// Workspace-relative path of the copy.
    pub file: String,
    /// The photo's display name (`IMG_0012.jpg`).
    pub name: String,
    /// The name without extension, safe for a file name (what exports are named after).
    pub stem: String,
}

fn safe(s: &str) -> String {
    let mut out: String = s.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') { c } else { '_' }).collect();
    out.truncate(80);
    if out.is_empty() { "photo".to_string() } else { out }
}

/// Copy the original of gallery photo `id` into the workspace (never writes to the gallery).
/// The copy's name carries the id, so the same photo always stages to the same file.
pub fn stage(gallery: &GallerySlot, root: &std::path::Path, id: i64) -> ApiResult<Staged> {
    let photo = gallery.read(id, 0)?;
    let stem = safe(photo.name.rsplit_once('.').map_or(photo.name.as_str(), |(s, _)| s));
    let ext = photo.name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).filter(|e| (1..=5).contains(&e.len()) && e.chars().all(|c| c.is_ascii_alphanumeric())).unwrap_or_else(|| "jpg".to_string());
    let file = format!("{STAGING_DIR}/{id}-{stem}.{ext}");
    paths::relative(&file)?;
    std::fs::create_dir_all(root.join(STAGING_DIR)).map_err(|e| ApiError::internal(format!("create {STAGING_DIR}/: {e}")))?;
    std::fs::write(root.join(&file), &photo.bytes).map_err(|e| ApiError::internal(format!("write the copy of gallery photo {id}: {e}")))?;
    Ok(Staged { file, name: photo.name, stem })
}
