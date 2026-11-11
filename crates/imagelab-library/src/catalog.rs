//! The catalog: one JSON file (`.imagelab/catalog/catalog.json`) written atomically under an
//! exclusive file lock, re-read on every call so the app, the CLI and MCP see one truth.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use imagelab_api::{ApiError, ApiResult};
use serde::{Deserialize, Serialize};

use crate::paths::{Workspace, write_atomic};

pub const CATALOG_VERSION: u32 = 1;
/// A catalog larger than this is refused rather than read into memory.
const MAX_CATALOG_BYTES: u64 = 256 << 20;
/// How long a call waits for another process's lock.
const LOCK_WAIT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Flag {
    #[default]
    None,
    Pick,
    Reject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhotoRecord {
    pub id: String,
    /// SHA-256 of the file's bytes (hex): dedupe key and cache key.
    pub hash: String,
    pub size: u64,
    pub format: String,
    /// Unix seconds when imported.
    pub added: i64,
    /// `YYYY-MM-DDTHH:MM:SS`, camera-local from EXIF, else the file's mtime in UTC.
    pub taken: String,
    /// `exif` or `file`.
    pub taken_source: String,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub rating: u8,
    #[serde(default)]
    pub flag: Flag,
    #[serde(default)]
    pub tags: BTreeSet<String>,
    #[serde(default)]
    pub original_name: String,
    #[serde(default)]
    pub camera: Option<String>,
    #[serde(default)]
    pub lens: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlbumRecord {
    pub name: String,
    pub created: i64,
    /// Photo ids in album order.
    pub photos: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub version: u32,
    #[serde(default)]
    pub photos: BTreeMap<String, PhotoRecord>,
    #[serde(default)]
    pub albums: BTreeMap<String, AlbumRecord>,
}

impl Default for Catalog {
    fn default() -> Self {
        Self { version: CATALOG_VERSION, photos: BTreeMap::new(), albums: BTreeMap::new() }
    }
}

impl Catalog {
    /// Photo id → the albums it is in.
    pub fn album_index(&self) -> BTreeMap<&str, Vec<String>> {
        let mut m: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for a in self.albums.values() {
            for p in &a.photos {
                m.entry(p.as_str()).or_default().push(a.name.clone());
            }
        }
        m
    }

    pub fn photo(&self, id: &str) -> ApiResult<&PhotoRecord> {
        self.photos.get(id).ok_or_else(|| ApiError::not_found(format!("no photo `{id}` in the library")))
    }

    pub fn photo_mut(&mut self, id: &str) -> ApiResult<&mut PhotoRecord> {
        self.photos.get_mut(id).ok_or_else(|| ApiError::not_found(format!("no photo `{id}` in the library")))
    }
}

/// Load/modify/save with a lock file next to the catalog.
pub struct Store {
    dir: PathBuf,
}

impl Store {
    pub fn new(ws: &Workspace) -> Self {
        Self { dir: ws.catalog_dir() }
    }

    fn file(&self) -> PathBuf {
        self.dir.join("catalog.json")
    }

    fn lock(&self, exclusive: bool) -> ApiResult<File> {
        fs::create_dir_all(&self.dir).map_err(|e| ApiError::internal(format!("create {}: {e}", self.dir.display())))?;
        let path = self.dir.join("catalog.lock");
        let f = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| ApiError::internal(format!("open {}: {e}", path.display())))?;
        let start = Instant::now();
        loop {
            let r = if exclusive { f.try_lock() } else { f.try_lock_shared() };
            match r {
                Ok(()) => return Ok(f),
                Err(fs::TryLockError::WouldBlock) if start.elapsed() < LOCK_WAIT => std::thread::sleep(Duration::from_millis(20)),
                Err(fs::TryLockError::WouldBlock) => return Err(ApiError::unavailable("the catalog is locked by another process")),
                Err(fs::TryLockError::Error(e)) => return Err(ApiError::internal(format!("lock {}: {e}", path.display()))),
            }
        }
    }

    fn load(&self) -> ApiResult<Catalog> {
        let path = self.file();
        let meta = match fs::metadata(&path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Catalog::default()),
            Err(e) => return Err(ApiError::internal(format!("stat {}: {e}", path.display()))),
        };
        if meta.len() > MAX_CATALOG_BYTES {
            return Err(ApiError::too_large(format!("catalog is {} bytes", meta.len())));
        }
        let bytes = fs::read(&path).map_err(|e| ApiError::internal(format!("read {}: {e}", path.display())))?;
        let c: Catalog = serde_json::from_slice(&bytes).map_err(|e| ApiError::new("catalog_corrupt", format!("{}: {e}", path.display())))?;
        if c.version > CATALOG_VERSION {
            return Err(ApiError::new("catalog_newer", format!("catalog version {} is newer than this build ({CATALOG_VERSION})", c.version)));
        }
        Ok(c)
    }

    /// A snapshot under a shared lock.
    pub fn read(&self) -> ApiResult<Catalog> {
        let lock = self.lock(false)?;
        let c = self.load();
        drop(lock);
        c
    }

    /// Load under an exclusive lock, apply `f`, save when it succeeds.
    pub fn update<R>(&self, f: impl FnOnce(&mut Catalog) -> ApiResult<R>) -> ApiResult<R> {
        let lock = self.lock(true)?;
        let mut c = self.load()?;
        let r = f(&mut c)?;
        c.version = CATALOG_VERSION;
        let bytes = serde_json::to_vec(&c).map_err(|e| ApiError::internal(e.to_string()))?;
        write_atomic(&self.file(), &bytes)?;
        drop(lock);
        Ok(r)
    }
}

pub fn now_unix() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Days since 1970-01-01 → (year, month, day) (H. Hinnant's civil_from_days).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Unix seconds → `YYYY-MM-DDTHH:MM:SS` (UTC).
pub fn iso_from_unix(secs: i64) -> String {
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let s = secs.rem_euclid(86_400);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// EXIF `YYYY:MM:DD HH:MM:SS` → `YYYY-MM-DDTHH:MM:SS`; `None` when malformed or all zeros.
pub fn iso_from_exif(s: &str) -> Option<String> {
    let b = s.trim().as_bytes();
    if b.len() < 19 {
        return None;
    }
    let digits = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18];
    if !digits.iter().all(|&i| b[i].is_ascii_digit()) {
        return None;
    }
    let num = |r: std::ops::Range<usize>| std::str::from_utf8(&b[r]).ok().and_then(|t| t.parse::<u32>().ok());
    let (y, mo, d, h, mi, se) = (num(0..4)?, num(5..7)?, num(8..10)?, num(11..13)?, num(14..16)?, num(17..19)?);
    if y < 1800 || !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || se > 60 {
        return None;
    }
    Some(format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{se:02}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn dates() {
        assert_eq!(iso_from_unix(0), "1970-01-01T00:00:00");
        assert_eq!(iso_from_unix(1_762_819_200), "2025-11-11T00:00:00");
        assert_eq!(iso_from_exif("2026:11:10 14:03:22").as_deref(), Some("2026-11-10T14:03:22"));
        assert_eq!(iso_from_exif("0000:00:00 00:00:00"), None);
        assert_eq!(iso_from_exif("garbage"), None);
    }
}
