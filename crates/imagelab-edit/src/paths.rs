//! Workspace paths: photo ids, sidecars, recipes, exports.
//!
//! Every path an area takes is workspace-relative (forward slashes, no absolute paths, no `..`).
//! PhotoCraft's `AuthorizedWorkspace` checks again (and catches symlinks); this gives a clear
//! early error and keeps originals under `library/` read-only.

use std::path::{Component, Path};

use imagelab_api::{ApiError, ApiResult};

/// Photos imported into the workspace. Never written.
pub const LIBRARY_DIR: &str = "library";
/// Edit sidecars: `edits/<photo id, slashes replaced>.pcraft`.
pub const EDITS_DIR: &str = "edits";
/// Recipes: `recipes/<name>.json`.
pub const RECIPES_DIR: &str = "recipes";
/// Default export folder.
pub const EXPORTS_DIR: &str = "exports";

/// Longest path accepted.
const MAX_PATH_CHARS: usize = 512;

/// A workspace-relative path, normalised to forward slashes, or `bad_params`.
pub fn relative(path: &str) -> ApiResult<String> {
    let trimmed = path.trim().replace('\\', "/");
    if trimmed.is_empty() {
        return Err(ApiError::bad_params("empty path"));
    }
    if trimmed.chars().count() > MAX_PATH_CHARS {
        return Err(ApiError::bad_params(format!("path longer than {MAX_PATH_CHARS} characters")));
    }
    let p = Path::new(&trimmed);
    if p.is_absolute() || trimmed.starts_with('/') || p.components().any(|c| !matches!(c, Component::Normal(_)))
        || trimmed.split('/').any(|seg| seg.is_empty() || seg == "." || seg == "..") {
        return Err(ApiError::bad_params(format!("path must be workspace-relative, without `..` or `.`: {trimmed}")));
    }
    if trimmed.chars().any(char::is_control) {
        return Err(ApiError::bad_params("path contains control characters"));
    }
    Ok(trimmed)
}

/// A path ImageLab may write: workspace-relative and not under `library/` (originals are never
/// modified).
pub fn writable(path: &str) -> ApiResult<String> {
    let p = relative(path)?;
    if p == LIBRARY_DIR || p.starts_with(&format!("{LIBRARY_DIR}/")) {
        return Err(ApiError::new("read_only_path", format!("`{p}` is under {LIBRARY_DIR}/: originals are never modified; write to {EDITS_DIR}/ or {EXPORTS_DIR}/")));
    }
    Ok(p)
}

/// True for an edit sidecar path (`edits/….pcraft`).
pub fn is_sidecar(path: &str) -> bool {
    path.starts_with(&format!("{EDITS_DIR}/")) && path.ends_with(".pcraft")
}

/// The sidecar of a photo: `edits/<id with "/" replaced by "__">.pcraft`, e.g.
/// `library/2026-11/IMG_0012.jpg` → `edits/library__2026-11__IMG_0012.jpg.pcraft`.
/// A sidecar path maps to itself.
pub fn sidecar_for(photo: &str) -> String {
    if is_sidecar(photo) {
        return photo.to_string();
    }
    format!("{EDITS_DIR}/{}.pcraft", photo.replace('/', "__"))
}

/// The file name of `path` without its extension.
pub fn stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    let name = name.strip_suffix(".pcraft").unwrap_or(name);
    match name.rsplit_once('.') {
        Some((s, _)) if !s.is_empty() => s.to_string(),
        _ => name.to_string(),
    }
}

/// The lower-case extension of `path`, if any.
pub fn extension(path: &str) -> Option<String> {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.rsplit_once('.').filter(|(s, _)| !s.is_empty()).map(|(_, e)| e.to_ascii_lowercase())
}

/// A recipe name: 1–64 of letters, digits, space, `-`, `_`, `.`; not starting with `.`.
pub fn recipe_name(name: &str) -> ApiResult<String> {
    let n = name.trim();
    let ok = !n.is_empty()
        && n.chars().count() <= 64
        && !n.starts_with('.')
        && n.chars().all(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.'));
    if !ok {
        return Err(ApiError::bad_params(format!("recipe name `{n}` must be 1-64 letters, digits, spaces, `-`, `_` or `.`, not starting with `.`")));
    }
    Ok(n.to_string())
}

/// `dir/name.ext`, or `dir/name (2).ext`, … — the first that does not exist under `root`.
pub fn unique(root: &Path, dir: &str, name: &str, ext: &str) -> ApiResult<String> {
    let first = format!("{dir}/{name}.{ext}");
    if !root.join(&first).exists() {
        return Ok(first);
    }
    for n in 2..10_000 {
        let candidate = format!("{dir}/{name} ({n}).{ext}");
        if !root.join(&candidate).exists() {
            return Ok(candidate);
        }
    }
    Err(ApiError::new("exists", format!("too many files named {name}.{ext} in {dir}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn relative_and_writable() {
        assert_eq!(relative("library/a.jpg").unwrap(), "library/a.jpg");
        for bad in ["", "/etc/passwd", "../x", "a/../../x", "./x", "a/./b"] {
            assert!(relative(bad).is_err(), "{bad}");
        }
        assert!(writable("library/a.jpg").is_err());
        assert!(writable("exports/a.jpg").is_ok());
    }

    #[test]
    fn sidecars_and_names() {
        assert_eq!(sidecar_for("library/2026-11/IMG_0012.jpg"), "edits/library__2026-11__IMG_0012.jpg.pcraft");
        assert_eq!(sidecar_for("edits/x.pcraft"), "edits/x.pcraft");
        assert_eq!(stem("library/2026-11/IMG_0012.jpg"), "IMG_0012");
        assert_eq!(stem("edits/library__a.jpg.pcraft"), "library__a");
        assert_eq!(extension("a/b.JPG").as_deref(), Some("jpg"));
        assert!(recipe_name("warm look").is_ok());
        assert!(recipe_name("../x").is_err());
        assert!(recipe_name(".hidden").is_err());
    }
}
