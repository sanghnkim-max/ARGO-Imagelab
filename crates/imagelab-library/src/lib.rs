//! # imagelab-library
//!
//! Library + analyze areas (docs/INTERFACES.md §1 Cull/Organize, §2, §4):
//!
//! * `library.*` — photos under `<root>/library/<yyyy-mm>/`, imported by move/copy from inside
//!   the workspace (the app drops picked photos into `imports/`), deduped by content hash;
//!   ratings, flags, tags, albums and EXIF metadata. A photo id is its workspace-relative path,
//!   e.g. `library/2026-11/IMG_0012.jpg`. Originals are never modified.
//! * `analyze.*` — quality scores, near-duplicate clusters, histograms, thumbnails and contact
//!   sheets, measured on a ≤1024 px proxy and cached per content hash.
//!
//! ## Storage
//!
//! * `.imagelab/catalog/catalog.json` — the catalog (photos, albums), written atomically
//!   (temp + rename) under `.imagelab/catalog/catalog.lock` (an OS file lock).
//! * `.imagelab/cache/<h2>/<sha256>.facts.json` — measurements; `<sha256>.thumb.jpg` — a 512 px
//!   thumbnail.
//! * `.imagelab/trash/<unix>-<n>/` — deleted photos (same relative paths) + `manifest.json`.
//!
//! LightCraft's `lightcraft-catalog` was not reused: it links `lightcraft-develop`,
//! `lightcraft-raw` and `lightcraft-meta` (a second raw/develop engine in the APK) and keys
//! photos by numeric ids, while ImageLab's contract keys them by workspace path.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod analysis;
mod catalog;
mod meta;
mod paths;
mod pixels;
mod sheet;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, UNIX_EPOCH};

use imagelab_api::{ApiError, ApiResult, Ctx, Effects, Event, ImageBuf, Module, Registry};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::analysis::Facts;
use crate::catalog::{AlbumRecord, Catalog, Flag, PhotoRecord, Store, iso_from_unix, now_unix};
use crate::paths::{Workspace, copy_atomic, move_file, sanitize_file_name, write_atomic};

/// Most photos one call takes.
pub const MAX_PHOTOS: usize = 2000;
/// Most photos on one contact sheet.
pub const MAX_SHEET: usize = 64;
/// Most files `importDir` looks at, and how deep it goes.
const MAX_DIR_FILES: usize = 2000;
const MAX_DIR_DEPTH: usize = 4;
const MAX_TAGS: usize = 100;
const DEFAULT_TIME_BUDGET_MS: u64 = 300_000;
const MAX_TIME_BUDGET_MS: u64 = 1_800_000;

/// Registers the `library` and `analyze` areas.
pub struct LibraryModule;

// ---------------------------------------------------------------------------------------------
// Output types

/// A photo as every surface sees it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Photo {
    pub id: String,
    pub name: String,
    pub hash: String,
    pub size: u64,
    pub format: String,
    /// When imported (UTC, `YYYY-MM-DDTHH:MM:SS`).
    pub added: String,
    /// Capture time: camera-local from EXIF, else the file time in UTC (`takenSource`).
    pub taken: String,
    pub taken_source: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub rating: u8,
    pub flag: Flag,
    pub tags: Vec<String>,
    pub albums: Vec<String>,
    pub camera: Option<String>,
    pub lens: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_name: Option<String>,
    /// EXIF details (`library.get` only); untrusted text from the file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exif: Option<Value>,
    /// Cached quality scores (`library.get` only, when analysed before).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<Value>,
}

fn photo_of(rec: &PhotoRecord, albums: &BTreeMap<&str, Vec<String>>) -> Photo {
    Photo {
        id: rec.id.clone(),
        name: rec.id.rsplit('/').next().unwrap_or(&rec.id).to_string(),
        hash: rec.hash.clone(),
        size: rec.size,
        format: rec.format.clone(),
        added: iso_from_unix(rec.added),
        taken: rec.taken.clone(),
        taken_source: rec.taken_source.clone(),
        width: rec.width,
        height: rec.height,
        rating: rec.rating,
        flag: rec.flag,
        tags: rec.tags.iter().cloned().collect(),
        albums: albums.get(rec.id.as_str()).cloned().unwrap_or_default(),
        camera: rec.camera.clone(),
        lens: rec.lens.clone(),
        original_name: None,
        exif: None,
        quality: None,
    }
}

// ---------------------------------------------------------------------------------------------
// Params

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ImportParams {
    path: String,
    /// Move instead of copy. Default: move when the source is under `imports/`, else copy.
    #[serde(default, rename = "move")]
    move_file: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum SortKey {
    Taken,
    Added,
    Rating,
    Quality,
    Name,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Order {
    Asc,
    Desc,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct QueryParams {
    album: Option<String>,
    min_rating: Option<u8>,
    flag: Option<Flag>,
    tags: Vec<String>,
    taken_after: Option<String>,
    taken_before: Option<String>,
    text: Option<String>,
    sort: Option<SortKey>,
    order: Option<Order>,
    limit: Option<usize>,
    offset: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PhotoParam {
    photo: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RatingParams {
    photos: Vec<String>,
    rating: u8,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FlagParams {
    photos: Vec<String>,
    flag: Flag,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TagParams {
    photos: Vec<String>,
    tags: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AlbumParam {
    name: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AlbumPhotosParams {
    album: String,
    photos: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PhotosParam {
    photos: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct QualityParams {
    photos: Vec<String>,
    #[serde(default)]
    faces: Option<bool>,
    #[serde(default)]
    time_budget_ms: Option<u64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DuplicatesParams {
    photos: Vec<String>,
    #[serde(default)]
    threshold: Option<u32>,
    #[serde(default)]
    include_singletons: Option<bool>,
    #[serde(default)]
    time_budget_ms: Option<u64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct HistogramParams {
    #[serde(default)]
    photo: Option<String>,
    #[serde(default)]
    image: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ThumbnailParams {
    photo: String,
    #[serde(default)]
    max_side: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SheetParams {
    photos: Vec<String>,
    #[serde(default)]
    columns: Option<u32>,
    #[serde(default)]
    cell_size: Option<u32>,
}

// ---------------------------------------------------------------------------------------------
// Helpers

fn workspace(ctx: &Ctx) -> ApiResult<Workspace> {
    Workspace::new(&ctx.shared.root)
}

fn changed(ctx: &Ctx, photos: Vec<String>) {
    if !photos.is_empty() {
        ctx.emit(Event::LibraryChanged { photos, source: ctx.source });
    }
}

/// Deduplicated photo ids (order kept), normalised, at most `max`.
fn photo_ids(ids: &[String], max: usize) -> ApiResult<Vec<String>> {
    if ids.len() > max {
        return Err(ApiError::too_large(format!("{} photos; at most {max} per call", ids.len())));
    }
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let n = Workspace::normalize(id)?;
        if !n.starts_with("library/") {
            return Err(ApiError::bad_params(format!("`{id}` is not a photo id (ids start with `library/`)")));
        }
        if seen.insert(n.clone()) {
            out.push(n);
        }
    }
    Ok(out)
}

fn clean_tag(t: &str) -> ApiResult<String> {
    let t: String = t.chars().filter(|c| !c.is_control()).collect::<String>().trim().to_lowercase();
    if t.is_empty() || t.chars().count() > 64 {
        return Err(ApiError::bad_params(format!("tag `{t}` must be 1..64 characters")));
    }
    Ok(t)
}

fn clean_album(name: &str) -> ApiResult<String> {
    let n: String = name.chars().filter(|c| !c.is_control()).collect::<String>().trim().to_string();
    if n.is_empty() || n.chars().count() > 100 {
        return Err(ApiError::bad_params("album names are 1..100 characters"));
    }
    Ok(n)
}

/// `2026-11-10`, `2026-11-10 14:00` or `2026-11-10T14:00:00` → comparable ISO prefix.
fn clean_date(s: &str) -> ApiResult<String> {
    let s = s.trim().replacen(' ', "T", 1);
    let b = s.as_bytes();
    let ok = b.len() >= 4 && b.len() <= 25 && b[..4].iter().all(u8::is_ascii_digit) && b.iter().all(|c| c.is_ascii_digit() || matches!(c, b'-' | b':' | b'T' | b'.' | b'Z' | b'+'));
    if !ok {
        return Err(ApiError::bad_params(format!("`{s}` is not a date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")));
    }
    Ok(s)
}

fn time_budget(ms: Option<u64>) -> Duration {
    Duration::from_millis(ms.unwrap_or(DEFAULT_TIME_BUDGET_MS).clamp(1, MAX_TIME_BUDGET_MS))
}

fn err_json(e: &ApiError) -> Value {
    json!({"error": {"code": e.code, "message": e.message}})
}

fn put_image(ctx: &Ctx, img: pixels::Rgba) -> ApiResult<(String, u32, u32)> {
    let (w, h) = (img.w, img.h);
    let id = ctx.shared.handles.put_image(ImageBuf::new(w, h, img.px)?)?;
    Ok((id.to_string(), w, h))
}

// ---------------------------------------------------------------------------------------------
// Import

struct Prepared {
    rel: String,
    abs: PathBuf,
    hash: String,
    size: u64,
    format: photocraft_codecs::Format,
    taken: String,
    taken_source: &'static str,
    dims: Option<(u32, u32)>,
    camera: Option<String>,
    lens: Option<String>,
    name: String,
}

fn prepare(ws: &Workspace, rel: &str) -> ApiResult<Prepared> {
    let (rel, abs) = ws.resolve_existing(rel)?;
    if rel == ".imagelab" || rel.starts_with(".imagelab/") {
        return Err(ApiError::bad_params("cannot import from .imagelab/"));
    }
    let bytes = pixels::read_file(&abs)?;
    let format = photocraft_codecs::detect(&bytes).ok_or_else(|| ApiError::new("unsupported", format!("`{rel}` is not a recognised image")))?;
    if !photocraft_codecs::caps(format).read {
        return Err(ApiError::new("unsupported", format!("`{rel}`: {} cannot be decoded by this build", format.name())));
    }
    let hash: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    let ex = meta::read_exif(&bytes);
    let orientation = ex.as_ref().map_or(1, meta::orientation);
    let (taken, taken_source) = match ex.as_ref().and_then(meta::taken) {
        Some(t) => (t, "exif"),
        None => {
            let mtime = fs::metadata(&abs).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or_else(now_unix, |d| d.as_secs() as i64);
            (iso_from_unix(mtime), "file")
        }
    };
    let dims = meta::header_dims(&bytes, orientation);
    let name = abs.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "photo".to_string());
    Ok(Prepared {
        rel,
        abs,
        hash,
        size: bytes.len() as u64,
        format,
        taken,
        taken_source,
        dims,
        camera: ex.as_ref().and_then(meta::camera),
        lens: ex.as_ref().and_then(meta::lens),
        name,
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Imported {
    #[serde(flatten)]
    photo: Photo,
    /// `imported` (new file in library/), `registered` (already in library/, now catalogued),
    /// `duplicate` (same content already in the library: the existing photo is returned).
    import_status: &'static str,
}

/// Import one prepared file into the (locked) catalog.
fn commit(ws: &Workspace, cat: &mut Catalog, by_hash: &mut HashMap<String, String>, p: Prepared, move_opt: Option<bool>) -> ApiResult<(String, &'static str)> {
    let in_imports = p.rel.starts_with("imports/");
    let do_move = move_opt.unwrap_or(in_imports);
    if let Some(existing) = by_hash.get(&p.hash) {
        // Same bytes already catalogued: drop the redundant copy only from the app's drop area.
        if do_move && in_imports && *existing != p.rel {
            let _ = fs::remove_file(&p.abs);
        }
        return Ok((existing.clone(), "duplicate"));
    }
    let (id, status) = if p.rel.starts_with("library/") {
        (p.rel.clone(), "registered")
    } else {
        let month = p.taken.get(..7).filter(|m| m.len() == 7).unwrap_or("undated");
        let mut name = sanitize_file_name(&p.name);
        let exts = p.format.extensions();
        let has_ext = name.rsplit_once('.').is_some_and(|(_, e)| exts.iter().any(|x| x.eq_ignore_ascii_case(e)));
        if !has_ext && let Some(e) = exts.first() {
            name = format!("{name}.{e}");
        }
        let (stem, ext) = name.rsplit_once('.').map_or((name.clone(), String::new()), |(s, e)| (s.to_string(), format!(".{e}")));
        let mut chosen = None;
        for i in 0..1000 {
            let cand = if i == 0 { format!("library/{month}/{stem}{ext}") } else { format!("library/{month}/{stem}_{i}{ext}") };
            if !cat.photos.contains_key(&cand) && fs::symlink_metadata(ws.root.join(&cand)).is_err() {
                chosen = Some(cand);
                break;
            }
        }
        let id = chosen.ok_or_else(|| ApiError::internal(format!("no free name for {name} in library/{month}")))?;
        let dst = ws.root.join(&id);
        if do_move { move_file(&p.abs, &dst)? } else { copy_atomic(&p.abs, &dst)? }
        (id, "imported")
    };
    let rec = PhotoRecord {
        id: id.clone(),
        hash: p.hash.clone(),
        size: p.size,
        format: p.format.name().to_string(),
        added: now_unix(),
        taken: p.taken,
        taken_source: p.taken_source.to_string(),
        width: p.dims.map(|d| d.0),
        height: p.dims.map(|d| d.1),
        rating: 0,
        flag: Flag::None,
        tags: BTreeSet::new(),
        original_name: p.name,
        camera: p.camera,
        lens: p.lens,
    };
    cat.photos.insert(id.clone(), rec);
    by_hash.insert(p.hash, id.clone());
    Ok((id, status))
}

fn hash_index(cat: &Catalog) -> HashMap<String, String> {
    cat.photos.values().map(|r| (r.hash.clone(), r.id.clone())).collect()
}

fn import_one(ctx: &Ctx, p: ImportParams) -> ApiResult<Imported> {
    let ws = workspace(ctx)?;
    let prepared = prepare(&ws, &p.path)?;
    let store = Store::new(&ws);
    let (id, status, photo) = store.update(|cat| {
        let mut by_hash = hash_index(cat);
        let (id, status) = commit(&ws, cat, &mut by_hash, prepared, p.move_file)?;
        let rec = cat.photo(&id)?;
        Ok((id.clone(), status, photo_of(rec, &cat.album_index())))
    })?;
    if status != "duplicate" {
        changed(ctx, vec![id]);
    }
    Ok(Imported { photo, import_status: status })
}

fn collect_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > MAX_DIR_DEPTH || out.len() >= MAX_DIR_FILES {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        if out.len() >= MAX_DIR_FILES {
            return;
        }
        if e.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            collect_files(&e.path(), depth + 1, out);
        } else if ft.is_file() {
            out.push(e.path());
        }
    }
}

fn import_dir(ctx: &Ctx, p: ImportParams) -> ApiResult<Value> {
    let ws = workspace(ctx)?;
    let (rel, abs) = ws.resolve_existing(&p.path)?;
    if !abs.is_dir() {
        return Err(ApiError::bad_params(format!("`{rel}` is not a folder")));
    }
    // library/ may be registered in place (photos placed there by hand); .imagelab/ never.
    if rel == ".imagelab" || rel.starts_with(".imagelab/") {
        return Err(ApiError::bad_params("cannot import from .imagelab/"));
    }
    let move_opt = p.move_file;
    let mut files = Vec::new();
    collect_files(&abs, 0, &mut files);
    let truncated = files.len() >= MAX_DIR_FILES;
    let mut skipped = Vec::new();
    let mut prepared = Vec::new();
    let deadline = Instant::now() + time_budget(None);
    for f in files {
        ctx.check_cancel()?;
        let r = ws.relative(&f).unwrap_or_else(|| f.display().to_string());
        if Instant::now() > deadline {
            skipped.push(json!({"path": r, "reason": "time budget"}));
            continue;
        }
        match prepare(&ws, &r) {
            Ok(p) => prepared.push(p),
            Err(e) => skipped.push(json!({"path": r, "reason": e.message, "code": e.code})),
        }
    }
    let store = Store::new(&ws);
    let (imported, duplicates, errors) = store.update(|cat| {
        let mut by_hash = hash_index(cat);
        let (mut imported, mut duplicates, mut errors) = (Vec::new(), Vec::new(), Vec::new());
        for p in prepared {
            let src = p.rel.clone();
            match commit(&ws, cat, &mut by_hash, p, move_opt) {
                Ok((id, "duplicate")) => duplicates.push((src, id)),
                Ok((id, _)) => imported.push(id),
                Err(e) => errors.push(json!({"path": src, "reason": e.message, "code": e.code})),
            }
        }
        let albums = cat.album_index();
        let imported: Vec<Photo> = imported.iter().filter_map(|id| cat.photos.get(id)).map(|r| photo_of(r, &albums)).collect();
        let duplicates: Vec<Value> = duplicates.iter().map(|(src, id)| json!({"path": src, "existing": id})).collect();
        Ok((imported, duplicates, errors))
    })?;
    skipped.extend(errors);
    changed(ctx, imported.iter().map(|p| p.id.clone()).collect());
    Ok(json!({"imported": imported, "duplicates": duplicates, "skipped": skipped, "truncated": truncated}))
}

// ---------------------------------------------------------------------------------------------
// Query

fn query(ctx: &Ctx, q: QueryParams) -> ApiResult<Vec<Photo>> {
    let ws = workspace(ctx)?;
    let cat = Store::new(&ws).read()?;
    let limit = q.limit.unwrap_or(100).clamp(1, MAX_PHOTOS);
    if q.min_rating.is_some_and(|r| r > 5) {
        return Err(ApiError::bad_params("minRating is 0..5"));
    }
    let album: Option<BTreeSet<&str>> = match &q.album {
        Some(a) => {
            let a = clean_album(a)?;
            let rec = cat.albums.get(&a).ok_or_else(|| ApiError::not_found(format!("no album `{a}`")))?;
            Some(rec.photos.iter().map(String::as_str).collect())
        }
        None => None,
    };
    let tags: Vec<String> = q.tags.iter().map(|t| clean_tag(t)).collect::<ApiResult<_>>()?;
    let after = q.taken_after.as_deref().map(clean_date).transpose()?;
    let before = q.taken_before.as_deref().map(clean_date).transpose()?;
    let text = q.text.as_deref().map(|t| t.trim().to_lowercase()).filter(|t| !t.is_empty());
    let albums = cat.album_index();
    let mut hits: Vec<&PhotoRecord> = cat
        .photos
        .values()
        .filter(|r| album.as_ref().is_none_or(|a| a.contains(r.id.as_str())))
        .filter(|r| q.min_rating.is_none_or(|m| r.rating >= m))
        .filter(|r| q.flag.is_none_or(|f| r.flag == f))
        .filter(|r| tags.iter().all(|t| r.tags.contains(t)))
        .filter(|r| after.as_ref().is_none_or(|a| r.taken.as_str() >= a.as_str()))
        .filter(|r| before.as_ref().is_none_or(|b| r.taken.as_str() < b.as_str()))
        .filter(|r| {
            text.as_ref().is_none_or(|t| {
                let in_opt = |o: &Option<String>| o.as_ref().is_some_and(|s| s.to_lowercase().contains(t));
                r.id.to_lowercase().contains(t)
                    || r.original_name.to_lowercase().contains(t)
                    || r.tags.iter().any(|g| g.contains(t))
                    || in_opt(&r.camera)
                    || in_opt(&r.lens)
                    || albums.get(r.id.as_str()).is_some_and(|a| a.iter().any(|n| n.to_lowercase().contains(t)))
            })
        })
        .collect();
    let sort = q.sort.unwrap_or(SortKey::Taken);
    let desc = q.order.unwrap_or(if sort == SortKey::Name { Order::Asc } else { Order::Desc }) == Order::Desc;
    match sort {
        SortKey::Taken => hits.sort_by(|a, b| a.taken.cmp(&b.taken).then(a.id.cmp(&b.id))),
        SortKey::Added => hits.sort_by(|a, b| a.added.cmp(&b.added).then(a.id.cmp(&b.id))),
        SortKey::Rating => hits.sort_by(|a, b| a.rating.cmp(&b.rating).then(a.taken.cmp(&b.taken))),
        SortKey::Name => hits.sort_by(|a, b| a.id.cmp(&b.id)),
        SortKey::Quality => {
            let score: HashMap<&str, f32> = hits.iter().map(|r| (r.id.as_str(), analysis::cached_facts(&ws, &r.hash, false).map_or(-1.0, |f| f.overall().0))).collect();
            let s = |id: &str| score.get(id).copied().unwrap_or(-1.0);
            hits.sort_by(|a, b| s(&a.id).total_cmp(&s(&b.id)).then(a.id.cmp(&b.id)));
        }
    }
    if desc {
        hits.reverse();
    }
    Ok(hits.into_iter().skip(q.offset).take(limit).map(|r| photo_of(r, &albums)).collect())
}

fn get(ctx: &Ctx, p: PhotoParam) -> ApiResult<Photo> {
    let ws = workspace(ctx)?;
    let id = Workspace::normalize(&p.photo)?;
    let cat = Store::new(&ws).read()?;
    let rec = cat.photo(&id)?;
    let mut photo = photo_of(rec, &cat.album_index());
    photo.original_name = Some(rec.original_name.clone());
    if let Ok(path) = ws.photo_path(&id)
        && let Ok(bytes) = pixels::read_file(&path)
    {
        photo.exif = Some(meta::read_exif(&bytes).map_or_else(|| json!({}), |ex| meta::exif_json(&ex)));
    }
    if let Some(f) = analysis::cached_facts(&ws, &rec.hash, false) {
        photo.width = photo.width.or(Some(f.width));
        photo.height = photo.height.or(Some(f.height));
        photo.quality = Some(f.to_json());
    }
    Ok(photo)
}

// ---------------------------------------------------------------------------------------------
// Ratings, flags, tags, albums, delete

fn update_photos<T: Serialize>(ctx: &Ctx, ids: &[String], mut f: impl FnMut(&mut PhotoRecord) -> Option<T>) -> ApiResult<Value> {
    let ids = photo_ids(ids, MAX_PHOTOS)?;
    let ws = workspace(ctx)?;
    let (updated, previous) = Store::new(&ws).update(|cat| {
        for id in &ids {
            cat.photo(id)?;
        }
        let mut previous = Map::new();
        let mut updated = Vec::new();
        for id in &ids {
            let rec = cat.photo_mut(id)?;
            if let Some(prev) = f(rec) {
                previous.insert(id.clone(), serde_json::to_value(prev).unwrap_or(Value::Null));
                updated.push(id.clone());
            }
        }
        Ok((updated, previous))
    })?;
    changed(ctx, updated.clone());
    Ok(json!({"updated": updated, "previous": previous}))
}

fn album_summary(a: &AlbumRecord) -> Value {
    json!({"name": a.name, "count": a.photos.len(), "cover": a.photos.first(), "created": iso_from_unix(a.created)})
}

fn album_edit(ctx: &Ctx, p: AlbumPhotosParams, add: bool) -> ApiResult<Value> {
    let name = clean_album(&p.album)?;
    let ids = photo_ids(&p.photos, MAX_PHOTOS)?;
    let ws = workspace(ctx)?;
    let (summary, touched) = Store::new(&ws).update(|cat| {
        if add {
            for id in &ids {
                cat.photo(id)?;
            }
        }
        let a = cat.albums.get_mut(&name).ok_or_else(|| ApiError::not_found(format!("no album `{name}`; create it with library.albumCreate")))?;
        let mut touched = Vec::new();
        for id in &ids {
            let has = a.photos.contains(id);
            if add && !has {
                a.photos.push(id.clone());
                touched.push(id.clone());
            } else if !add && has {
                a.photos.retain(|x| x != id);
                touched.push(id.clone());
            }
        }
        Ok((album_summary(a), touched))
    })?;
    changed(ctx, touched.clone());
    Ok(json!({"album": summary, "changed": touched}))
}

fn delete(ctx: &Ctx, p: PhotosParam) -> ApiResult<Value> {
    let ids = photo_ids(&p.photos, MAX_PHOTOS)?;
    let ws = workspace(ctx)?;
    let stamp = now_unix();
    let mut n = 0u32;
    let trash = loop {
        let t = ws.trash_dir().join(format!("{stamp}-{n}"));
        if !t.exists() || n > 1000 {
            break t;
        }
        n += 1;
    };
    let (deleted, missing) = Store::new(&ws).update(|cat| {
        let mut deleted = Vec::new();
        let mut missing = Vec::new();
        let mut records = Vec::new();
        for id in &ids {
            let Some(rec) = cat.photos.get(id).cloned() else {
                missing.push(id.clone());
                continue;
            };
            if let Ok(src) = ws.photo_path(id) {
                move_file(&src, &trash.join(id))?;
            }
            cat.photos.remove(id);
            for a in cat.albums.values_mut() {
                a.photos.retain(|x| x != id);
            }
            records.push(rec);
            deleted.push(id.clone());
        }
        if !records.is_empty() {
            let manifest = serde_json::to_vec_pretty(&json!({"deleted": iso_from_unix(stamp), "photos": records})).map_err(|e| ApiError::internal(e.to_string()))?;
            write_atomic(&trash.join("manifest.json"), &manifest)?;
        }
        Ok((deleted, missing))
    })?;
    changed(ctx, deleted.clone());
    Ok(json!({"deleted": deleted, "missing": missing, "trash": ws.relative(&trash)}))
}

// ---------------------------------------------------------------------------------------------
// Analyze

/// Catalog records for `ids`, each or an error.
fn records(ws: &Workspace, ids: &[String]) -> ApiResult<Vec<(String, ApiResult<PhotoRecord>)>> {
    let cat = Store::new(ws).read()?;
    Ok(ids.iter().map(|id| (id.clone(), cat.photo(id).cloned())).collect())
}

fn facts_for(ws: &Workspace, rec: &PhotoRecord, faces: bool) -> ApiResult<Facts> {
    let path = ws.photo_path(&rec.id)?;
    analysis::facts(ws, &rec.hash, &path, faces)
}

fn progress(ctx: &Ctx, job: &str, done: usize, total: usize) {
    if total > 1 && (done.is_multiple_of(10) || done == total) {
        ctx.emit(Event::JobProgress { job: job.to_string(), done: done as u64, total: total as u64, message: format!("{done}/{total} photos") });
    }
}

fn quality(ctx: &Ctx, p: QualityParams) -> ApiResult<Value> {
    let ids = photo_ids(&p.photos, MAX_PHOTOS)?;
    let ws = workspace(ctx)?;
    let want_faces = p.faces.unwrap_or(true);
    let deadline = Instant::now() + time_budget(p.time_budget_ms);
    let mut out = Map::new();
    let recs = records(&ws, &ids)?;
    let total = recs.len();
    for (i, (id, rec)) in recs.into_iter().enumerate() {
        ctx.check_cancel()?;
        let v = if Instant::now() > deadline && analysis::cached_facts(&ws, rec.as_ref().map(|r| r.hash.as_str()).unwrap_or(""), want_faces).is_none() {
            err_json(&ApiError::new("time_budget", "not analysed within the time budget; call again (finished photos are cached)"))
        } else {
            match rec.and_then(|r| facts_for(&ws, &r, want_faces)) {
                Ok(f) => f.to_json(),
                Err(e) => err_json(&e),
            }
        };
        out.insert(id, v);
        progress(ctx, "analyze.quality", i + 1, total);
    }
    Ok(Value::Object(out))
}

fn duplicates(ctx: &Ctx, p: DuplicatesParams) -> ApiResult<Value> {
    let ids = photo_ids(&p.photos, MAX_PHOTOS)?;
    let threshold = p.threshold.unwrap_or(10);
    if threshold > 32 {
        return Err(ApiError::bad_params("threshold is a Hamming distance 0..32 (of 64 bits)"));
    }
    let ws = workspace(ctx)?;
    let deadline = Instant::now() + time_budget(p.time_budget_ms);
    let recs = records(&ws, &ids)?;
    let total = recs.len();
    let mut facts: Vec<Option<Facts>> = Vec::with_capacity(total);
    let mut failed: HashMap<usize, Value> = HashMap::new();
    for (i, (_, rec)) in recs.iter().enumerate() {
        ctx.check_cancel()?;
        let f = match rec {
            Ok(r) if Instant::now() <= deadline || analysis::cached_facts(&ws, &r.hash, false).is_some() => facts_for(&ws, r, false),
            Ok(_) => Err(ApiError::new("time_budget", "not analysed within the time budget")),
            Err(e) => Err(e.clone()),
        };
        match f {
            Ok(f) => facts.push(Some(f)),
            Err(e) => {
                failed.insert(i, json!({"code": e.code, "message": e.message}));
                facts.push(None);
            }
        }
        progress(ctx, "analyze.duplicates", i + 1, total);
    }
    let hashes: Vec<Option<u64>> = facts.iter().map(|f| f.as_ref().map(Facts::dhash_bits)).collect();
    let include_singletons = p.include_singletons.unwrap_or(true);
    let mut out = Vec::new();
    for group in analysis::clusters(&hashes, threshold) {
        if let Some(e) = failed.get(&group[0]) {
            // Unreadable photos stay visible as singletons carrying their error.
            let id = recs[group[0]].0.as_str();
            out.push(json!({"members": [id], "representative": id, "error": e}));
            continue;
        }
        if group.len() < 2 && !include_singletons {
            continue;
        }
        // Representative: best overall score, then most pixels, then first.
        let key = |i: &usize| {
            facts[*i].as_ref().map_or((-1.0f32, 0u64), |f| (f.overall().0, u64::from(f.width) * u64::from(f.height)))
        };
        let rep = group.iter().copied().max_by(|a, b| {
            let (ka, kb) = (key(a), key(b));
            ka.0.total_cmp(&kb.0).then(ka.1.cmp(&kb.1)).then(b.cmp(a))
        });
        let members: Vec<&str> = group.iter().map(|&i| recs[i].0.as_str()).collect();
        out.push(json!({"members": members, "representative": rep.map(|i| recs[i].0.as_str())}));
    }
    Ok(Value::Array(out))
}

fn histogram(ctx: &Ctx, p: HistogramParams) -> ApiResult<Value> {
    match (p.photo, p.image) {
        (Some(photo), None) => {
            let ws = workspace(ctx)?;
            let id = Workspace::normalize(&photo)?;
            let path = ws.photo_path(&id)?;
            let bytes = pixels::read_file(&path)?;
            let (img, (w, h)) = pixels::decode_scaled(&bytes, analysis::PROXY_SIDE)?;
            let mut v = analysis::histogram(&img.px);
            v["source"] = json!({"photo": id, "width": w, "height": h, "sampled": format!("{}x{} proxy", img.w, img.h)});
            Ok(v)
        }
        (None, Some(handle)) => {
            let img = ctx.shared.handles.image_str(&handle)?;
            let mut v = analysis::histogram(&img.rgba);
            v["source"] = json!({"image": handle, "width": img.width, "height": img.height});
            Ok(v)
        }
        _ => Err(ApiError::bad_params("give exactly one of `photo` (id) or `image` (img:N)")),
    }
}

fn thumbnail(ctx: &Ctx, p: ThumbnailParams) -> ApiResult<Value> {
    let max_side = p.max_side.unwrap_or(pixels::THUMB_SIDE);
    if !(16..=2048).contains(&max_side) {
        return Err(ApiError::bad_params("maxSide is 16..2048"));
    }
    let ws = workspace(ctx)?;
    let id = Workspace::normalize(&p.photo)?;
    let cat = Store::new(&ws).read()?;
    let rec = cat.photo(&id)?;
    let path = ws.photo_path(&id)?;
    let img = pixels::preview(&ws, &rec.hash, &path, max_side)?;
    let (handle, w, h) = put_image(ctx, img)?;
    Ok(json!({"image": handle, "width": w, "height": h, "photo": id}))
}

fn contact_sheet(ctx: &Ctx, p: SheetParams) -> ApiResult<Value> {
    let ids = photo_ids(&p.photos, MAX_SHEET)?;
    if ids.is_empty() {
        return Err(ApiError::bad_params("no photos"));
    }
    let n = ids.len() as u32;
    let cols = p.columns.unwrap_or_else(|| (f64::from(n).sqrt().ceil() as u32).max(1)).clamp(1, 16).min(n);
    let rows = n.div_ceil(cols);
    let gap = 6u32;
    let mut cell = p.cell_size.unwrap_or(256).clamp(64, 512);
    // Keep the sheet within 4096 px a side.
    while cell > 64 && (cols * (cell + gap) + gap > 4096 || rows * (cell + gap) + gap > 4096) {
        cell -= 16;
    }
    let (sw, sh) = (cols * (cell + gap) + gap, rows * (cell + gap) + gap);
    let ws = workspace(ctx)?;
    let recs = records(&ws, &ids)?;
    let mut canvas = sheet::Canvas::new(sw, sh);
    let scale = (cell / 64).clamp(2, 6);
    let mut layout = Vec::new();
    for (i, (id, rec)) in recs.into_iter().enumerate() {
        ctx.check_cancel()?;
        let (c, r) = (i as u32 % cols, i as u32 / cols);
        let (cx, cy) = (gap + c * (cell + gap), gap + r * (cell + gap));
        let img = rec.and_then(|rec| {
            let path = ws.photo_path(&rec.id)?;
            pixels::preview(&ws, &rec.hash, &path, cell)
        });
        let index = i + 1;
        let entry = match img {
            Ok(img) => {
                let (x, y) = (cx + (cell - img.w.min(cell)) / 2, cy + (cell - img.h.min(cell)) / 2);
                canvas.blit(&img, x, y);
                json!({"id": id, "index": index, "rect": [x, y, img.w, img.h]})
            }
            Err(e) => {
                canvas.cross(cx, cy, cell, cell);
                json!({"id": id, "index": index, "rect": [cx, cy, cell, cell], "error": {"code": e.code, "message": e.message}})
            }
        };
        canvas.label(index, cx, cy, scale);
        layout.push(entry);
    }
    let (handle, w, h) = put_image(ctx, canvas.img)?;
    Ok(json!({"image": handle, "width": w, "height": h, "columns": cols, "cellSize": cell, "layout": layout}))
}

// ---------------------------------------------------------------------------------------------
// Registration

fn photos_schema() -> Value {
    json!({"type": "array", "items": {"type": "string"}, "maxItems": MAX_PHOTOS, "description": "photo ids (library/...)"})
}

impl Module for LibraryModule {
    fn register(&self, registry: &mut Registry) {
        registry.area("library", "Photos in the workspace: import, query, albums, ratings, flags, tags, EXIF metadata");
        registry.area("analyze", "Measure photos: quality scores (sharpness, exposure, noise, faces), near-duplicates, histograms, thumbnails, numbered contact sheets");

        // ---- library
        registry
            .function("library", "import", "Import one image file from the workspace (e.g. imports/x.jpg) into library/<yyyy-mm>/")
            .description("Moves (default for files under imports/) or copies a workspace file into library/<yyyy-mm>/ by EXIF capture date (else file time). Deduplicates by content hash: importing the same bytes again returns the existing photo with importStatus \"duplicate\". A file already under library/ is catalogued in place (\"registered\"). Returns the Photo plus importStatus. Undo with library.delete.")
            .params(json!({"type": "object", "required": ["path"], "properties": {
                "path": {"type": "string", "description": "workspace-relative file path"},
                "move": {"type": "boolean", "description": "move instead of copy (default: true under imports/, else false)"}}}))
            .effects(Effects::Revertible)
            .typed(import_one)
            .register();
        registry
            .function("library", "importDir", "Import every image in a workspace folder (recursive, ≤2000 files)")
            .description("Like library.import for each image file under `path` (depth ≤ 4, hidden files skipped, non-images reported in `skipped`). Returns {imported:[Photo], duplicates:[{path, existing}], skipped:[{path, reason}], truncated}.")
            .params(json!({"type": "object", "required": ["path"], "properties": {
                "path": {"type": "string"}, "move": {"type": "boolean"}}}))
            .effects(Effects::Revertible)
            .typed(import_dir)
            .register();
        let query_schema = json!({"type": "object", "properties": {
            "album": {"type": "string"},
            "minRating": {"type": "integer", "minimum": 0, "maximum": 5},
            "flag": {"enum": ["pick", "reject", "none"]},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "all must match"},
            "takenAfter": {"type": "string", "description": "inclusive, YYYY-MM-DD[THH:MM:SS]"},
            "takenBefore": {"type": "string", "description": "exclusive"},
            "text": {"type": "string", "description": "substring of name, tags, camera, lens or album"},
            "sort": {"enum": ["taken", "added", "rating", "quality", "name"], "default": "taken"},
            "order": {"enum": ["asc", "desc"], "description": "default desc (asc for name)"},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_PHOTOS, "default": 100},
            "offset": {"type": "integer", "minimum": 0}}});
        registry
            .function("library", "query", "Find photos by album, rating, flag, tags, capture date, text; sorted")
            .description("Returns [Photo]. sort=quality uses cached analyze.quality scores (unanalysed photos last). Dates compare as ISO strings: takenAfter inclusive, takenBefore exclusive.")
            .params(query_schema.clone())
            .typed(query)
            .register();
        registry
            .function("library", "list", "List photos (same filters as library.query; newest first)")
            .description("Alias of library.query.")
            .params(query_schema)
            .typed(query)
            .register();
        registry
            .function("library", "get", "One photo with EXIF (camera, lens, focal, ISO, exposure, date, GPS) and cached scores")
            .params(json!({"type": "object", "required": ["photo"], "properties": {"photo": {"type": "string"}}}))
            .typed(get)
            .register();
        registry
            .function("library", "setRating", "Set the star rating (0..5) of photos")
            .description("Returns {updated:[id], previous:{id: rating}} so the change can be reverted.")
            .params(json!({"type": "object", "required": ["photos", "rating"], "properties": {"photos": photos_schema(), "rating": {"type": "integer", "minimum": 0, "maximum": 5}}}))
            .effects(Effects::Revertible)
            .typed(|ctx, p: RatingParams| {
                if p.rating > 5 {
                    return Err(ApiError::bad_params("rating is 0..5"));
                }
                update_photos(ctx, &p.photos, |r| (r.rating != p.rating).then(|| std::mem::replace(&mut r.rating, p.rating)))
            })
            .register();
        registry
            .function("library", "setFlag", "Flag photos as pick, reject or none")
            .description("Returns {updated:[id], previous:{id: flag}}.")
            .params(json!({"type": "object", "required": ["photos", "flag"], "properties": {"photos": photos_schema(), "flag": {"enum": ["pick", "reject", "none"]}}}))
            .effects(Effects::Revertible)
            .typed(|ctx, p: FlagParams| update_photos(ctx, &p.photos, |r| (r.flag != p.flag).then(|| std::mem::replace(&mut r.flag, p.flag))))
            .register();
        let tag_schema = json!({"type": "object", "required": ["photos", "tags"], "properties": {"photos": photos_schema(), "tags": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_TAGS}}});
        registry
            .function("library", "tag", "Add tags (lower-cased) to photos")
            .description("Returns {updated:[id], previous:{id: [tags before]}}.")
            .params(tag_schema.clone())
            .effects(Effects::Revertible)
            .typed(|ctx, p: TagParams| {
                let tags: Vec<String> = p.tags.iter().map(|t| clean_tag(t)).collect::<ApiResult<_>>()?;
                update_photos(ctx, &p.photos, |r| {
                    let before = r.tags.clone();
                    for t in &tags {
                        if r.tags.len() < MAX_TAGS {
                            r.tags.insert(t.clone());
                        }
                    }
                    (r.tags != before).then_some(before)
                })
            })
            .register();
        registry
            .function("library", "untag", "Remove tags from photos")
            .description("Returns {updated:[id], previous:{id: [tags before]}}.")
            .params(tag_schema)
            .effects(Effects::Revertible)
            .typed(|ctx, p: TagParams| {
                let tags: Vec<String> = p.tags.iter().map(|t| clean_tag(t)).collect::<ApiResult<_>>()?;
                update_photos(ctx, &p.photos, |r| {
                    let before = r.tags.clone();
                    for t in &tags {
                        r.tags.remove(t);
                    }
                    (r.tags != before).then_some(before)
                })
            })
            .register();
        registry
            .function("library", "albumCreate", "Create an album (idempotent)")
            .params(json!({"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}))
            .effects(Effects::Revertible)
            .typed(|ctx, p: AlbumParam| {
                let name = clean_album(&p.name)?;
                let ws = workspace(ctx)?;
                let (summary, created) = Store::new(&ws).update(|cat| {
                    let created = !cat.albums.contains_key(&name);
                    let a = cat.albums.entry(name.clone()).or_insert_with(|| AlbumRecord { name: name.clone(), created: now_unix(), photos: Vec::new() });
                    Ok((album_summary(a), created))
                })?;
                Ok(json!({"album": summary, "created": created}))
            })
            .register();
        registry
            .function("library", "albumAdd", "Add photos to an album")
            .params(json!({"type": "object", "required": ["album", "photos"], "properties": {"album": {"type": "string"}, "photos": photos_schema()}}))
            .effects(Effects::Revertible)
            .typed(|ctx, p: AlbumPhotosParams| album_edit(ctx, p, true))
            .register();
        registry
            .function("library", "albumRemove", "Remove photos from an album (the photos stay in the library)")
            .params(json!({"type": "object", "required": ["album", "photos"], "properties": {"album": {"type": "string"}, "photos": photos_schema()}}))
            .effects(Effects::Revertible)
            .typed(|ctx, p: AlbumPhotosParams| album_edit(ctx, p, false))
            .register();
        registry
            .function("library", "albumDelete", "Delete an album (not its photos)")
            .description("Returns {deleted, photos:[ids it held]} so it can be recreated.")
            .params(json!({"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}))
            .effects(Effects::Revertible)
            .typed(|ctx, p: AlbumParam| {
                let name = clean_album(&p.name)?;
                let ws = workspace(ctx)?;
                let removed = Store::new(&ws).update(|cat| cat.albums.remove(&name).ok_or_else(|| ApiError::not_found(format!("no album `{name}`"))))?;
                changed(ctx, removed.photos.clone());
                Ok(json!({"deleted": removed.name, "photos": removed.photos}))
            })
            .register();
        registry
            .function("library", "albumList", "List albums with photo counts")
            .typed(|ctx, _: Value| {
                let ws = workspace(ctx)?;
                let cat = Store::new(&ws).read()?;
                Ok(cat.albums.values().map(album_summary).collect::<Vec<_>>())
            })
            .register();
        registry
            .function("library", "delete", "Move photos to .imagelab/trash (removes them from the library and albums)")
            .description("The files move to .imagelab/trash/<time>/ with a manifest.json of their catalog records. Returns {deleted, missing, trash}.")
            .params(json!({"type": "object", "required": ["photos"], "properties": {"photos": photos_schema()}}))
            .effects(Effects::Destructive)
            .typed(delete)
            .register();

        // ---- analyze
        registry
            .function("analyze", "quality", "Quality scores per photo: sharpness, exposure, contrast, noise, faces, overall (0..1) + notes")
            .description("Returns {id: {sharpness, exposure, contrast, noise, faces:{count, maxScore, largestArea, sharpness, eyes, boxes}|null, overall, notes:[..], width, height, details}} or {id: {error}}. Measured on a ≤1024 px proxy; overall = 0.45·sharpness (blended 40/60 with face sharpness when faces are found) + 0.25·exposure + 0.15·noise + 0.15·contrast. Cached per content hash, so repeat calls are instant. faces (default true) runs the on-device face detector. Photos left when timeBudgetMs (default 300000) runs out get {error:{code:\"time_budget\"}}.")
            .params(json!({"type": "object", "required": ["photos"], "properties": {"photos": photos_schema(), "faces": {"type": "boolean", "default": true}, "timeBudgetMs": {"type": "integer", "minimum": 1, "maximum": MAX_TIME_BUDGET_MS}}}))
            .typed(quality)
            .register();
        registry
            .function("analyze", "duplicates", "Cluster near-duplicate photos (64-bit dHash, Hamming distance ≤ threshold)")
            .description("Returns [{members:[id], representative}] covering every photo (singletons included unless includeSingletons=false); representative = best cached overall quality, then most pixels. threshold default 10 (of 64 bits); 0-4 = near-identical, 10 = same shot resized/recompressed, 16+ = loose. Photos that could not be read appear as singletons with an `error`.")
            .params(json!({"type": "object", "required": ["photos"], "properties": {"photos": photos_schema(), "threshold": {"type": "integer", "minimum": 0, "maximum": 32, "default": 10}, "includeSingletons": {"type": "boolean", "default": true}, "timeBudgetMs": {"type": "integer"}}}))
            .typed(duplicates)
            .register();
        registry
            .function("analyze", "histogram", "RGB + luma histograms (256 bins) with clipping fractions, of a photo or an img:N handle")
            .description("Give `photo` (sampled on a ≤1024 px proxy) or `image` (full handle). Returns {r, g, b, luma: [256 counts], pixels, mean, clipping:{shadows, highlights, r, g, b}, source}.")
            .params(json!({"type": "object", "properties": {"photo": {"type": "string"}, "image": {"type": "string", "description": "img:N"}}}))
            .typed(histogram)
            .register();
        registry
            .function("analyze", "thumbnail", "A preview of a photo as an image handle (img:N)")
            .params(json!({"type": "object", "required": ["photo"], "properties": {"photo": {"type": "string"}, "maxSide": {"type": "integer", "minimum": 16, "maximum": 2048, "default": 512}}}))
            .typed(thumbnail)
            .register();
        registry
            .function("analyze", "contactSheet", "A numbered grid of up to 64 photos as one image (img:N), to look at and refer to by number")
            .description("Each cell shows its 1-based index in the top-left corner; `layout` maps index → id and the image rect [x, y, w, h] on the sheet. columns default ⌈√n⌉ (≤16), cellSize default 256 (64..512); the sheet stays within 4096 px.")
            .params(json!({"type": "object", "required": ["photos"], "properties": {"photos": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_SHEET}, "columns": {"type": "integer", "minimum": 1, "maximum": 16}, "cellSize": {"type": "integer", "minimum": 64, "maximum": 512}}}))
            .typed(contact_sheet)
            .register();
    }
}
