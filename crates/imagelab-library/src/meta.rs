//! Metadata without decoding pixels: EXIF (kamadak-exif) and header-only dimensions.
//!
//! EXIF text comes from the file, so it is untrusted: control characters are stripped and
//! strings are capped.

use std::io::Cursor;

use exif::{Exif, In, Tag, Value};
use serde_json::{Map, Value as Json, json};

const MAX_TEXT: usize = 128;

pub fn read_exif(bytes: &[u8]) -> Option<Exif> {
    exif::Reader::new().read_from_container(&mut Cursor::new(bytes)).ok()
}

fn clean(s: &str) -> Option<String> {
    let s: String = s.chars().filter(|c| !c.is_control()).take(MAX_TEXT).collect();
    let s = s.trim().trim_matches('"').trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn text(ex: &Exif, tag: Tag) -> Option<String> {
    let f = ex.get_field(tag, In::PRIMARY)?;
    match &f.value {
        Value::Ascii(v) => v.first().and_then(|b| clean(&String::from_utf8_lossy(b))),
        _ => clean(&f.display_value().to_string()),
    }
}

fn rational(ex: &Exif, tag: Tag) -> Option<f64> {
    let f = ex.get_field(tag, In::PRIMARY)?;
    let v = match &f.value {
        Value::Rational(v) => v.first().map(|r| r.to_f64()),
        Value::SRational(v) => v.first().map(|r| r.to_f64()),
        _ => f.value.get_uint(0).map(f64::from),
    }?;
    v.is_finite().then_some(v)
}

fn uint(ex: &Exif, tag: Tag) -> Option<u32> {
    ex.get_field(tag, In::PRIMARY)?.value.get_uint(0)
}

fn gps_coord(ex: &Exif, tag: Tag, reference: Tag) -> Option<f64> {
    let f = ex.get_field(tag, In::PRIMARY)?;
    let Value::Rational(v) = &f.value else { return None };
    if v.len() < 3 {
        return None;
    }
    let deg = v[0].to_f64() + v[1].to_f64() / 60.0 + v[2].to_f64() / 3600.0;
    if !deg.is_finite() {
        return None;
    }
    let r = text(ex, reference).unwrap_or_default();
    let sign = if r.starts_with('S') || r.starts_with('W') { -1.0 } else { 1.0 };
    Some((sign * deg * 1e6).round() / 1e6)
}

/// The capture time as `YYYY-MM-DDTHH:MM:SS` (camera-local), from DateTimeOriginal, else DateTime.
pub fn taken(ex: &Exif) -> Option<String> {
    [Tag::DateTimeOriginal, Tag::DateTimeDigitized, Tag::DateTime].into_iter().find_map(|t| text(ex, t).and_then(|s| crate::catalog::iso_from_exif(&s)))
}

pub fn orientation(ex: &Exif) -> u16 {
    uint(ex, Tag::Orientation).and_then(|o| u16::try_from(o).ok()).filter(|o| (1..=8).contains(o)).unwrap_or(1)
}

/// "Make Model", without the make repeated when the model already starts with it.
pub fn camera(ex: &Exif) -> Option<String> {
    let make = text(ex, Tag::Make);
    let model = text(ex, Tag::Model);
    match (make, model) {
        (Some(mk), Some(md)) if md.to_lowercase().starts_with(&mk.to_lowercase()) => Some(md),
        (Some(mk), Some(md)) => Some(format!("{mk} {md}")),
        (a, b) => a.or(b),
    }
}

pub fn lens(ex: &Exif) -> Option<String> {
    text(ex, Tag::LensModel)
}

/// The EXIF facts `library.get` reports.
pub fn exif_json(ex: &Exif) -> Json {
    let mut m = Map::new();
    let mut put = |k: &str, v: Option<Json>| {
        if let Some(v) = v {
            m.insert(k.to_string(), v);
        }
    };
    put("camera", camera(ex).map(Json::from));
    put("make", text(ex, Tag::Make).map(Json::from));
    put("model", text(ex, Tag::Model).map(Json::from));
    put("lens", lens(ex).map(Json::from));
    put("focalLength", rational(ex, Tag::FocalLength).map(|v| json!((v * 10.0).round() / 10.0)));
    put("focalLength35mm", uint(ex, Tag::FocalLengthIn35mmFilm).map(Json::from));
    put("fNumber", rational(ex, Tag::FNumber).map(|v| json!((v * 10.0).round() / 10.0)));
    put("iso", uint(ex, Tag::PhotographicSensitivity).map(Json::from));
    if let Some(t) = rational(ex, Tag::ExposureTime) {
        let shown = if t > 0.0 && t < 0.5 { format!("1/{}", (1.0 / t).round()) } else { format!("{t}s") };
        put("exposureTime", Some(json!(t)));
        put("exposure", Some(json!(shown)));
    }
    put("exposureBias", rational(ex, Tag::ExposureBiasValue).map(Json::from));
    put("flash", uint(ex, Tag::Flash).map(|f| json!(f & 1 == 1)));
    put("dateTimeOriginal", taken(ex).map(Json::from));
    put("offsetTime", text(ex, Tag::OffsetTimeOriginal).map(Json::from));
    put("orientation", Some(json!(orientation(ex))));
    put("software", text(ex, Tag::Software).map(Json::from));
    let lat = gps_coord(ex, Tag::GPSLatitude, Tag::GPSLatitudeRef);
    let lon = gps_coord(ex, Tag::GPSLongitude, Tag::GPSLongitudeRef);
    if let (Some(lat), Some(lon)) = (lat, lon) {
        let mut g = json!({"latitude": lat, "longitude": lon});
        if let Some(alt) = rational(ex, Tag::GPSAltitude) {
            g["altitude"] = json!((alt * 10.0).round() / 10.0);
        }
        put("gps", Some(g));
    }
    Json::Object(m)
}

/// Upright dimensions from the header alone (JPEG SOF, PNG IHDR, or PhotoCraft's probe), with
/// EXIF orientation 5–8 swapping them. `None` when only a decode can tell.
pub fn header_dims(bytes: &[u8], orientation: u16) -> Option<(u32, u32)> {
    let raw = jpeg_dims(bytes).or_else(|| png_dims(bytes));
    match raw {
        Some((w, h)) if (5..=8).contains(&orientation) => Some((h, w)),
        Some(d) => Some(d),
        None => photocraft_codecs::probe_size(bytes).ok(),
    }
}

fn be16(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from(*b.get(i)?) << 8 | u32::from(*b.get(i + 1)?))
}

fn jpeg_dims(b: &[u8]) -> Option<(u32, u32)> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 3 < b.len() {
        if b[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = b[i + 1];
        if m == 0xFF {
            i += 1;
            continue;
        }
        if m == 0xD8 || m == 0x01 || (0xD0..=0xD7).contains(&m) {
            i += 2;
            continue;
        }
        let len = be16(b, i + 2)? as usize;
        if matches!(m, 0xC0..=0xCF) && !matches!(m, 0xC4 | 0xC8 | 0xCC) {
            let h = be16(b, i + 5)?;
            let w = be16(b, i + 7)?;
            return (w > 0 && h > 0).then_some((w, h));
        }
        if m == 0xDA || m == 0xD9 || len < 2 {
            return None;
        }
        i = i.checked_add(2 + len)?;
    }
    None
}

fn png_dims(b: &[u8]) -> Option<(u32, u32)> {
    if b.len() < 24 || &b[..8] != b"\x89PNG\r\n\x1a\n" || &b[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]);
    let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]);
    (w > 0 && h > 0).then_some((w, h))
}
