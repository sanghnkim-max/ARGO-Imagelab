//! Workspace paths: every path an agent passes is workspace-relative, normalised and checked to
//! stay inside the root after symlinks are resolved.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use imagelab_api::{ApiError, ApiResult};

/// Longest workspace-relative path accepted.
const MAX_REL_LEN: usize = 1024;

/// The canonical workspace root and the directories this crate owns under it.
#[derive(Debug, Clone)]
pub struct Workspace {
    pub root: PathBuf,
}

impl Workspace {
    pub fn new(root: &Path) -> ApiResult<Self> {
        let root = root.canonicalize().map_err(|e| ApiError::unavailable(format!("workspace root {}: {e}", root.display())))?;
        Ok(Self { root })
    }

    pub fn catalog_dir(&self) -> PathBuf {
        self.root.join(".imagelab").join("catalog")
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.root.join(".imagelab").join("cache")
    }

    pub fn trash_dir(&self) -> PathBuf {
        self.root.join(".imagelab").join("trash")
    }

    /// `a/./b//c` → `a/b/c`; refuses absolute paths, `..`, drive letters, NUL and empty paths.
    pub fn normalize(rel: &str) -> ApiResult<String> {
        let rel = rel.trim();
        if rel.is_empty() {
            return Err(ApiError::bad_params("empty path"));
        }
        if rel.len() > MAX_REL_LEN {
            return Err(ApiError::bad_params(format!("path longer than {MAX_REL_LEN} bytes")));
        }
        let b = rel.as_bytes();
        if rel.starts_with('/') || rel.starts_with('\\') || rel.contains('\0') || (b.len() > 1 && b[1] == b':') {
            return Err(ApiError::bad_params(format!("`{rel}` must be relative to the workspace")));
        }
        let mut parts = Vec::new();
        for p in rel.split(['/', '\\']) {
            match p {
                "" | "." => {}
                ".." => return Err(ApiError::bad_params(format!("`{rel}` may not contain `..`"))),
                p => parts.push(p),
            }
        }
        if parts.is_empty() {
            return Err(ApiError::bad_params("empty path"));
        }
        Ok(parts.join("/"))
    }

    /// An existing path, symlinks resolved; returns the canonical workspace-relative path and
    /// the absolute path. Fails when the target lies outside the workspace.
    pub fn resolve_existing(&self, rel: &str) -> ApiResult<(String, PathBuf)> {
        let n = Self::normalize(rel)?;
        let abs = self.root.join(&n).canonicalize().map_err(|_| ApiError::not_found(format!("`{n}` does not exist")))?;
        let canon_rel = self.relative(&abs).ok_or_else(|| ApiError::bad_params(format!("`{n}` resolves outside the workspace")))?;
        Ok((canon_rel, abs))
    }

    /// `abs` as a `/`-separated workspace-relative path, when it is inside the root.
    pub fn relative(&self, abs: &Path) -> Option<String> {
        let r = abs.strip_prefix(&self.root).ok()?;
        let parts: Vec<String> = r.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
        if parts.is_empty() { None } else { Some(parts.join("/")) }
    }

    /// The absolute path of a photo id (`library/...`), which must exist inside the workspace.
    pub fn photo_path(&self, id: &str) -> ApiResult<PathBuf> {
        let n = Self::normalize(id)?;
        if !n.starts_with("library/") {
            return Err(ApiError::bad_params(format!("`{id}` is not a photo id (ids start with `library/`)")));
        }
        let (canon, abs) = self.resolve_existing(&n)?;
        if !canon.starts_with("library/") {
            return Err(ApiError::bad_params(format!("`{id}` resolves outside library/")));
        }
        Ok(abs)
    }
}

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn io_err(what: &str, path: &Path, e: std::io::Error) -> ApiError {
    ApiError::internal(format!("{what} {}: {e}", path.display()))
}

/// Write `bytes` to `path` atomically: a temp file in the same directory, synced, then renamed.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> ApiResult<()> {
    let dir = path.parent().ok_or_else(|| ApiError::internal(format!("{} has no parent", path.display())))?;
    fs::create_dir_all(dir).map_err(|e| io_err("create", dir, e))?;
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dir.join(format!(".{name}.tmp.{}.{n}", std::process::id()));
    let result = (|| {
        let mut f = File::create(&tmp).map_err(|e| io_err("create", &tmp, e))?;
        f.write_all(bytes).map_err(|e| io_err("write", &tmp, e))?;
        f.sync_all().map_err(|e| io_err("sync", &tmp, e))?;
        fs::rename(&tmp, path).map_err(|e| io_err("rename", path, e))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Copy `src` to `dst` (which must not exist) through a temp file + rename.
pub fn copy_atomic(src: &Path, dst: &Path) -> ApiResult<()> {
    let dir = dst.parent().ok_or_else(|| ApiError::internal(format!("{} has no parent", dst.display())))?;
    fs::create_dir_all(dir).map_err(|e| io_err("create", dir, e))?;
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!(".import.tmp.{}.{n}", std::process::id()));
    let result = (|| {
        fs::copy(src, &tmp).map_err(|e| io_err("copy", src, e))?;
        if let Ok(f) = File::open(&tmp) {
            let _ = f.sync_all();
        }
        fs::rename(&tmp, dst).map_err(|e| io_err("rename", dst, e))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Move `src` to `dst`: a rename, or copy + remove across filesystems.
pub fn move_file(src: &Path, dst: &Path) -> ApiResult<()> {
    if let Some(dir) = dst.parent() {
        fs::create_dir_all(dir).map_err(|e| io_err("create", dir, e))?;
    }
    if fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    copy_atomic(src, dst)?;
    fs::remove_file(src).map_err(|e| io_err("remove", src, e))
}

/// A file name safe on every filesystem: ASCII letters, digits, `._-`; others become `_`.
pub fn sanitize_file_name(name: &str) -> String {
    let mut s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' }).collect();
    while s.starts_with('.') {
        s.remove(0);
    }
    if s.len() > 100 {
        // Keep the extension.
        let ext = s.rfind('.').map(|i| s[i..].to_string()).filter(|e| e.len() <= 10).unwrap_or_default();
        s.truncate(100 - ext.len());
        s.push_str(&ext);
    }
    if s.is_empty() { "photo".to_string() } else { s }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_and_refuses_escapes() {
        assert_eq!(Workspace::normalize("library/./2026-11//a.jpg").unwrap(), "library/2026-11/a.jpg");
        for bad in ["", "/etc/passwd", "../x", "a/../../b", "C:\\x", "\\\\server\\x", "a\0b"] {
            assert!(Workspace::normalize(bad).is_err(), "{bad:?}");
        }
        assert_eq!(sanitize_file_name("..my photo (1).JPG"), "my_photo__1_.JPG");
        assert_eq!(sanitize_file_name("日本"), "__");
    }
}
