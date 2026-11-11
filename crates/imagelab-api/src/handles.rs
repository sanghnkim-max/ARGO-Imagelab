//! The handle arena: images and masks referenced as `img:N` / `mask:N`.
//!
//! Bounded by total bytes, so a program that keeps allocating gets `too_large` instead of
//! taking the phone's memory.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use serde::Serialize;

use crate::{ApiError, ApiResult};

/// Largest side of any image or mask in the arena.
pub const MAX_SIDE: u32 = 16_384;
/// Total bytes the arena may hold.
pub const DEFAULT_BUDGET_BYTES: usize = 768 << 20;

/// 8-bit RGBA, row-major, unpremultiplied.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageBuf {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl ImageBuf {
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> ApiResult<Self> {
        check_dims(width, height)?;
        if rgba.len() != width as usize * height as usize * 4 {
            return Err(ApiError::bad_params(format!("rgba length {} does not match {width}x{height}", rgba.len())));
        }
        Ok(Self { width, height, rgba })
    }
    pub fn bytes(&self) -> usize {
        self.rgba.len()
    }
}

/// 8-bit coverage (0 = outside, 255 = inside), row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskBuf {
    pub width: u32,
    pub height: u32,
    pub alpha: Vec<u8>,
}

impl MaskBuf {
    pub fn new(width: u32, height: u32, alpha: Vec<u8>) -> ApiResult<Self> {
        check_dims(width, height)?;
        if alpha.len() != width as usize * height as usize {
            return Err(ApiError::bad_params(format!("mask length {} does not match {width}x{height}", alpha.len())));
        }
        Ok(Self { width, height, alpha })
    }
    pub fn bytes(&self) -> usize {
        self.alpha.len()
    }
}

fn check_dims(width: u32, height: u32) -> ApiResult<()> {
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(ApiError::too_large(format!("{width}x{height} is outside 1..={MAX_SIDE}")));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum HandleId {
    Image(u64),
    Mask(u64),
}

impl HandleId {
    pub fn parse(s: &str) -> Option<Self> {
        let (kind, n) = s.trim().split_once(':')?;
        let n: u64 = n.parse().ok()?;
        match kind {
            "img" => Some(HandleId::Image(n)),
            "mask" => Some(HandleId::Mask(n)),
            _ => None,
        }
    }
}

impl std::fmt::Display for HandleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandleId::Image(n) => write!(f, "img:{n}"),
            HandleId::Mask(n) => write!(f, "mask:{n}"),
        }
    }
}

enum Entry {
    Image(Arc<ImageBuf>),
    Mask(Arc<MaskBuf>),
}

impl Entry {
    fn bytes(&self) -> usize {
        match self {
            Entry::Image(i) => i.bytes(),
            Entry::Mask(m) => m.bytes(),
        }
    }
}

pub struct Handles {
    next: AtomicU64,
    budget: usize,
    inner: Mutex<(HashMap<HandleId, Entry>, usize)>,
}

impl Default for Handles {
    fn default() -> Self {
        Self::with_budget(DEFAULT_BUDGET_BYTES)
    }
}

impl Handles {
    pub fn with_budget(budget: usize) -> Self {
        Self { next: AtomicU64::new(1), budget, inner: Mutex::new((HashMap::new(), 0)) }
    }

    fn insert(&self, make: impl FnOnce(u64) -> HandleId, entry: Entry) -> ApiResult<HandleId> {
        let mut g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let bytes = entry.bytes();
        if g.1.saturating_add(bytes) > self.budget {
            return Err(ApiError::too_large(format!("handle arena full ({} of {} bytes); free handles with cv.free", g.1, self.budget)));
        }
        let id = make(self.next.fetch_add(1, Ordering::Relaxed));
        g.0.insert(id, entry);
        g.1 += bytes;
        Ok(id)
    }

    pub fn put_image(&self, image: ImageBuf) -> ApiResult<HandleId> {
        self.insert(HandleId::Image, Entry::Image(Arc::new(image)))
    }

    pub fn put_mask(&self, mask: MaskBuf) -> ApiResult<HandleId> {
        self.insert(HandleId::Mask, Entry::Mask(Arc::new(mask)))
    }

    pub fn image(&self, id: HandleId) -> ApiResult<Arc<ImageBuf>> {
        let g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        match g.0.get(&id) {
            Some(Entry::Image(i)) => Ok(i.clone()),
            Some(Entry::Mask(_)) => Err(ApiError::bad_params(format!("{id} is a mask, not an image"))),
            None => Err(ApiError::not_found(format!("{id} does not exist (freed or never made)"))),
        }
    }

    pub fn mask(&self, id: HandleId) -> ApiResult<Arc<MaskBuf>> {
        let g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        match g.0.get(&id) {
            Some(Entry::Mask(m)) => Ok(m.clone()),
            Some(Entry::Image(_)) => Err(ApiError::bad_params(format!("{id} is an image, not a mask"))),
            None => Err(ApiError::not_found(format!("{id} does not exist (freed or never made)"))),
        }
    }

    /// Parse and fetch an image handle string.
    pub fn image_str(&self, s: &str) -> ApiResult<Arc<ImageBuf>> {
        self.image(crate::parse_handle(s)?)
    }

    /// Parse and fetch a mask handle string.
    pub fn mask_str(&self, s: &str) -> ApiResult<Arc<MaskBuf>> {
        self.mask(crate::parse_handle(s)?)
    }

    pub fn free(&self, id: HandleId) -> bool {
        let mut g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        match g.0.remove(&id) {
            Some(e) => {
                g.1 = g.1.saturating_sub(e.bytes());
                true
            }
            None => false,
        }
    }

    /// Free every handle in `ids` (end of a script run).
    pub fn free_all(&self, ids: &[HandleId]) {
        for id in ids {
            self.free(*id);
        }
    }

    /// Every live handle (unordered). The script runtime snapshots this before a run so it
    /// frees only handles the run itself created.
    pub fn ids(&self) -> Vec<HandleId> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).0.keys().copied().collect()
    }

    /// True if `id` is live.
    pub fn contains(&self, id: HandleId) -> bool {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).0.contains_key(&id)
    }

    pub fn used_bytes(&self) -> usize {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).1
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_free_and_budget() {
        let h = Handles::with_budget(100);
        let img = h.put_image(ImageBuf::new(2, 2, vec![0; 16]).unwrap()).unwrap();
        assert_eq!(img.to_string(), "img:1");
        assert_eq!(HandleId::parse("img:1"), Some(img));
        assert!(h.image(img).is_ok());
        assert_eq!(h.mask(img).unwrap_err().code, "bad_params");
        let m = h.put_mask(MaskBuf::new(4, 4, vec![255; 16]).unwrap()).unwrap();
        assert_eq!(h.used_bytes(), 32);
        assert_eq!(h.put_image(ImageBuf::new(6, 4, vec![0; 96]).unwrap()).unwrap_err().code, "too_large");
        assert_eq!(h.ids().len(), 2);
        assert!(h.contains(m));
        assert!(h.free(m));
        assert!(!h.contains(m));
        assert_eq!(h.used_bytes(), 16);
        assert_eq!(h.mask(m).unwrap_err().code, "not_found");
    }

    #[test]
    fn rejects_bad_buffers() {
        assert!(ImageBuf::new(2, 2, vec![0; 3]).is_err());
        assert!(ImageBuf::new(0, 2, vec![]).is_err());
        assert!(MaskBuf::new(MAX_SIDE + 1, 1, vec![0; (MAX_SIDE + 1) as usize]).is_err());
        assert_eq!(HandleId::parse("foo:1"), None);
        assert_eq!(HandleId::parse("img:x"), None);
    }
}
