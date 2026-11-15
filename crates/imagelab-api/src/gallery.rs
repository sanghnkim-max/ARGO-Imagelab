//! The phone's gallery as seen from a program (Android only).
//!
//! Gallery photos are MediaStore ids, as ARGO's device tools return them. They are not in
//! ImageLab's library, so a function that takes photos takes a [`PhotoRef`]: a library id, or a
//! gallery photo (`{gallery: 61}`, or the string `"gallery:61"` that results use for it).
//!
//! The host (the Android app) provides a [`GalleryReader`]; where there is none (CLI, desktop,
//! tests) every read fails with `unsupported`. Reading never writes to the gallery.

use std::fmt;
use std::sync::{Arc, RwLock};

use serde::de::{Deserialize, Deserializer, Error as _};
use serde_json::{Value, json};

use crate::{ApiError, ApiResult};

/// Most gallery photos one call reads (each is decoded one at a time, downscaled by the host).
pub const MAX_GALLERY_PHOTOS: usize = 200;
/// Largest original a host returns for `max_side == 0`.
pub const MAX_GALLERY_BYTES: usize = 64 << 20;

/// One gallery photo as the host returns it.
#[derive(Debug, Clone)]
pub struct GalleryPhoto {
    pub id: i64,
    /// The display name (`IMG_0012.jpg`).
    pub name: String,
    /// The upright size of the original, in pixels.
    pub width: u32,
    pub height: u32,
    /// An encoded image (JPEG/PNG/WebP...): upright, longest side at most the `max_side` asked
    /// for, or the original file's bytes when `max_side` was 0.
    pub bytes: Vec<u8>,
}

/// What the host provides. Errors use the stable codes: `not_found` (no such photo, or it was
/// deleted), `permission` (no photo access), `unsupported`.
pub trait GalleryReader: Send + Sync {
    /// Read gallery photo `id`, downscaled to fit `max_side` (0: the original, untouched).
    fn read(&self, id: i64, max_side: u32) -> ApiResult<GalleryPhoto>;
}

/// Where the host's reader is kept; shared by every call. Empty until the host sets one.
#[derive(Default)]
pub struct GallerySlot(RwLock<Option<Arc<dyn GalleryReader>>>);

impl GallerySlot {
    pub fn set(&self, reader: Arc<dyn GalleryReader>) {
        *self.0.write().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(reader);
    }

    pub fn available(&self) -> bool {
        self.0.read().unwrap_or_else(std::sync::PoisonError::into_inner).is_some()
    }

    pub fn read(&self, id: i64, max_side: u32) -> ApiResult<GalleryPhoto> {
        let reader = self.0.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        let reader = reader.ok_or_else(|| ApiError::unsupported("there is no photo gallery here: gallery photos are only available in the Android app"))?;
        let photo = reader.read(id, max_side)?;
        if photo.bytes.len() > MAX_GALLERY_BYTES {
            return Err(ApiError::too_large(format!("gallery photo {id} is larger than {} MB", MAX_GALLERY_BYTES >> 20)));
        }
        Ok(photo)
    }
}

/// A photo a function takes: in ImageLab's library, or in the phone's gallery.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PhotoRef {
    /// A library id (`library/2026-11/IMG_0012.jpg`), not yet normalised.
    Library(String),
    /// A MediaStore id.
    Gallery(i64),
}

impl PhotoRef {
    /// The id results use: the library id, or `gallery:<id>`.
    pub fn id(&self) -> String {
        match self {
            PhotoRef::Library(id) => id.clone(),
            PhotoRef::Gallery(id) => format!("gallery:{id}"),
        }
    }

    pub fn gallery_id(&self) -> Option<i64> {
        match self {
            PhotoRef::Gallery(id) => Some(*id),
            PhotoRef::Library(_) => None,
        }
    }

    /// From a string: `gallery:61` is a gallery photo, anything else a library id.
    pub fn parse_str(s: &str) -> ApiResult<PhotoRef> {
        match s.strip_prefix("gallery:") {
            Some(n) => n.trim().parse::<i64>().ok().filter(|n| *n >= 0).map(PhotoRef::Gallery).ok_or_else(|| ApiError::bad_params(format!("`{s}` is not a gallery photo: use {{gallery: <id>}}"))),
            None => Ok(PhotoRef::Library(s.to_string())),
        }
    }

    /// From JSON: `"library/..."`, `"gallery:61"` or `{gallery: 61}`.
    pub fn from_value(v: &Value) -> ApiResult<PhotoRef> {
        match v {
            Value::String(s) => Self::parse_str(s),
            Value::Object(m) => match (m.get("gallery"), m.len()) {
                (Some(g), 1) => g.as_i64().filter(|n| *n >= 0).map(PhotoRef::Gallery).ok_or_else(|| ApiError::bad_params("`gallery` is a photo's gallery id (a non-negative integer)")),
                _ => Err(ApiError::bad_params("a photo is a library id string or {gallery: <id>}")),
            },
            Value::Number(_) => Err(ApiError::bad_params("give a gallery photo as {gallery: <id>}, not a bare number")),
            _ => Err(ApiError::bad_params("a photo is a library id string or {gallery: <id>}")),
        }
    }

    /// The JSON schema of one photo.
    pub fn schema() -> Value {
        json!({"oneOf": [
            {"type": "string", "description": "library id (library/...) or \"gallery:<id>\""},
            {"type": "object", "required": ["gallery"], "properties": {"gallery": {"type": "integer", "minimum": 0, "description": "the photo's gallery (MediaStore) id"}}, "additionalProperties": false}
        ]})
    }

    /// The JSON schema of a list of photos.
    pub fn list_schema(max: usize) -> Value {
        json!({"type": "array", "items": Self::schema(), "maxItems": max, "description": "photos: library ids (library/...) and gallery photos ({gallery: id})"})
    }
}

impl fmt::Display for PhotoRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.id())
    }
}

impl<'de> Deserialize<'de> for PhotoRef {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        PhotoRef::from_value(&v).map_err(|e| D::Error::custom(e.message))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn photo_refs_parse_every_shape_and_refuse_a_bare_number() {
        let r: Vec<PhotoRef> = serde_json::from_value(json!(["library/a.jpg", {"gallery": 61}, "gallery:60"])).unwrap();
        assert_eq!(r, vec![PhotoRef::Library("library/a.jpg".into()), PhotoRef::Gallery(61), PhotoRef::Gallery(60)]);
        assert_eq!(r[1].id(), "gallery:61");
        let e = serde_json::from_value::<Vec<PhotoRef>>(json!([61])).unwrap_err().to_string();
        assert!(e.contains("{gallery: <id>}"), "{e}");
        assert!(serde_json::from_value::<PhotoRef>(json!({"gallery": "x"})).is_err());
        assert!(serde_json::from_value::<PhotoRef>(json!({"gallery": 1, "x": 2})).is_err());
        assert!(serde_json::from_value::<PhotoRef>(json!("gallery:abc")).is_err());
    }

    #[test]
    fn no_reader_is_unsupported() {
        let e = GallerySlot::default().read(1, 256).unwrap_err();
        assert_eq!(e.code, "unsupported");
    }
}
