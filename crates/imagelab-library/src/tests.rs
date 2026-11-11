#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use imagelab_api::{CollectEvents, Ctx, Event, Handles, Headless, Registry, Shared, Source};
use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image};
use serde_json::{Value, json};

use crate::LibraryModule;
use crate::pixels::{Rgba, box_resize};

static N: AtomicU64 = AtomicU64::new(0);

struct Env {
    root: PathBuf,
    reg: Registry,
    ctx: Ctx,
    events: Arc<CollectEvents>,
}

impl Env {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("imagelab-library-test-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("imports")).unwrap();
        let events = Arc::new(CollectEvents::default());
        let shared = Shared { root: root.clone(), session: Arc::new(Mutex::new(Headless::new())), handles: Arc::new(Handles::default()), events: events.clone() };
        let reg = Registry::new().with(&LibraryModule);
        Self { root, reg, ctx: Ctx::new(shared, Source::Agent), events }
    }

    fn call(&self, id: &str, params: Value) -> Value {
        match self.reg.call(&self.ctx, id, params.clone()) {
            Ok(v) => v,
            Err(e) => panic!("{id} {params}: {e}"),
        }
    }

    fn err(&self, id: &str, params: Value) -> String {
        self.reg.call(&self.ctx, id, params).unwrap_err().code
    }

    fn write(&self, rel: &str, bytes: &[u8]) {
        let p = self.root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }

    fn import(&self, rel: &str, bytes: &[u8]) -> String {
        self.write(rel, bytes);
        self.call("library.import", json!({"path": rel}))["id"].as_str().unwrap().to_string()
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A detailed test scene: edges, text-like stripes, gradients and discs.
fn scene(w: u32, h: u32, seed: u32) -> Rgba {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let check = ((x / 24 + y / 24 + seed) % 2) as u8 * 120;
            let stripes = if (x / 3 + seed).is_multiple_of(5) && y > h / 2 { 90 } else { 0 };
            let (cx, cy) = (w as i64 / 3 + seed as i64 * 7, h as i64 / 3);
            let d = (x as i64 - cx).pow(2) + (y as i64 - cy).pow(2);
            let disc = if d < (h as i64 / 5).pow(2) { 60u8 } else { 0 };
            let r = check.saturating_add(stripes).saturating_add(disc).saturating_add((x * 40 / w) as u8);
            let g = check.saturating_add(disc / 2).saturating_add((y * 60 / h) as u8);
            let b = (120u8).saturating_sub(check / 2).saturating_add(stripes / 2);
            px.extend_from_slice(&[r, g, b, 255]);
        }
    }
    Rgba { w, h, px }
}

fn box_blur(img: &Rgba, radius: i64, passes: usize) -> Rgba {
    let (w, h) = (img.w as i64, img.h as i64);
    let mut cur = img.px.clone();
    for _ in 0..passes {
        for horizontal in [true, false] {
            let mut next = cur.clone();
            for y in 0..h {
                for x in 0..w {
                    let mut s = [0u32; 3];
                    let mut n = 0;
                    for k in -radius..=radius {
                        let (xx, yy) = if horizontal { ((x + k).clamp(0, w - 1), y) } else { (x, (y + k).clamp(0, h - 1)) };
                        let o = ((yy * w + xx) * 4) as usize;
                        for c in 0..3 {
                            s[c] += u32::from(cur[o + c]);
                        }
                        n += 1;
                    }
                    let o = ((y * w + x) * 4) as usize;
                    for c in 0..3 {
                        next[o + c] = (s[c] / n) as u8;
                    }
                }
            }
            cur = next;
        }
    }
    Rgba { w: img.w, h: img.h, px: cur }
}

fn encode(img: &Rgba, format: Format, exif: Option<Vec<u8>>) -> Vec<u8> {
    let rgb: Vec<u8> = img.px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    let mut im = Image::from_u8(img.w, img.h, ChannelLayout::Rgb, rgb).unwrap();
    im.meta.exif = exif;
    let opts = EncodeOptions { jpeg_quality: 92, embed_metadata: true, ..Default::default() };
    photocraft_codecs::encode(&im, format, &opts).unwrap()
}

/// A minimal big-endian TIFF/EXIF block: IFD0 {Make, ExifIFD} and Exif IFD {DateTimeOriginal}.
fn exif_block(make: &str, date: &str) -> Vec<u8> {
    let mut make_b = make.as_bytes().to_vec();
    make_b.push(0);
    let mut date_b = date.as_bytes().to_vec();
    date_b.push(0);
    let ifd0 = 8u32;
    let ifd0_len = 2 + 2 * 12 + 4;
    let exif_ifd = ifd0 + ifd0_len;
    let exif_len = 2 + 12 + 4;
    let make_off = exif_ifd + exif_len;
    let date_off = make_off + make_b.len() as u32;
    let mut v = b"MM\0\x2a".to_vec();
    v.extend_from_slice(&ifd0.to_be_bytes());
    let entry = |v: &mut Vec<u8>, tag: u16, typ: u16, count: u32, val: u32| {
        v.extend_from_slice(&tag.to_be_bytes());
        v.extend_from_slice(&typ.to_be_bytes());
        v.extend_from_slice(&count.to_be_bytes());
        v.extend_from_slice(&val.to_be_bytes());
    };
    v.extend_from_slice(&2u16.to_be_bytes());
    entry(&mut v, 0x010F, 2, make_b.len() as u32, make_off);
    entry(&mut v, 0x8769, 4, 1, exif_ifd);
    v.extend_from_slice(&0u32.to_be_bytes());
    v.extend_from_slice(&1u16.to_be_bytes());
    entry(&mut v, 0x9003, 2, date_b.len() as u32, date_off);
    v.extend_from_slice(&0u32.to_be_bytes());
    v.extend_from_slice(&make_b);
    v.extend_from_slice(&date_b);
    v
}

#[test]
fn import_moves_dedupes_and_files_by_exif_date() {
    let env = Env::new();
    let img = scene(320, 240, 0);
    let jpg = encode(&img, Format::Jpeg, Some(exif_block("TestCam", "2024:07:15 09:30:00")));
    env.write("imports/My Shot.jpg", &jpg);
    let r = env.call("library.import", json!({"path": "imports/My Shot.jpg"}));
    assert_eq!(r["importStatus"], "imported");
    assert_eq!(r["id"], "library/2024-07/My_Shot.jpg");
    assert_eq!(r["taken"], "2024-07-15T09:30:00");
    assert_eq!(r["takenSource"], "exif");
    assert_eq!(r["camera"], "TestCam");
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(320), Some(240)));
    assert!(!env.root.join("imports/My Shot.jpg").exists(), "moved out of imports/");
    assert!(env.root.join("library/2024-07/My_Shot.jpg").exists());
    assert!(matches!(env.events.0.lock().unwrap().last(), Some(Event::LibraryChanged { .. })));

    // Same bytes again: the existing photo, and the redundant drop is cleaned up.
    env.write("imports/again.jpg", &jpg);
    let r2 = env.call("library.import", json!({"path": "imports/again.jpg"}));
    assert_eq!(r2["importStatus"], "duplicate");
    assert_eq!(r2["id"], r["id"]);
    assert!(!env.root.join("imports/again.jpg").exists());

    // Copy semantics outside imports/.
    let png = encode(&scene(64, 48, 3), Format::Png, None);
    env.write("elsewhere/a.png", &png);
    let r3 = env.call("library.import", json!({"path": "elsewhere/a.png"}));
    assert_eq!(r3["takenSource"], "file");
    assert!(env.root.join("elsewhere/a.png").exists(), "copied, not moved");

    // library.get reports EXIF.
    let g = env.call("library.get", json!({"photo": r["id"]}));
    assert_eq!(g["exif"]["make"], "TestCam");
    assert_eq!(g["exif"]["dateTimeOriginal"], "2024-07-15T09:30:00");
    assert_eq!(g["originalName"], "My Shot.jpg");
}

#[test]
fn import_dir_and_bad_paths() {
    let env = Env::new();
    for i in 0..3 {
        env.write(&format!("imports/batch/p{i}.png"), &encode(&scene(48, 40, i), Format::Png, None));
    }
    env.write("imports/batch/notes.txt", b"hello");
    env.write("imports/batch/sub/p0copy.png", &encode(&scene(48, 40, 0), Format::Png, None));
    let r = env.call("library.importDir", json!({"path": "imports/batch"}));
    assert_eq!(r["imported"].as_array().unwrap().len(), 3);
    assert_eq!(r["duplicates"].as_array().unwrap().len(), 1);
    assert_eq!(r["skipped"].as_array().unwrap().len(), 1);

    for bad in ["../etc/passwd", "/etc/passwd", "imports/../../x.jpg", ".imagelab/catalog/catalog.json"] {
        let code = env.err("library.import", json!({"path": bad}));
        assert!(code == "bad_params" || code == "not_found", "{bad}: {code}");
    }
    assert_eq!(env.err("library.import", json!({"path": "imports/batch/notes.txt"})), "unsupported");
    assert_eq!(env.err("library.import", json!({"path": "nope.jpg"})), "not_found");
    #[cfg(unix)]
    {
        let outside = std::env::temp_dir().join(format!("imagelab-outside-{}.png", std::process::id()));
        std::fs::write(&outside, encode(&scene(16, 16, 1), Format::Png, None)).unwrap();
        std::os::unix::fs::symlink(&outside, env.root.join("imports/link.png")).unwrap();
        assert_eq!(env.err("library.import", json!({"path": "imports/link.png"})), "bad_params");
        let _ = std::fs::remove_file(outside);
    }
    assert_eq!(env.err("library.get", json!({"photo": "library/x.jpg"})), "not_found");
    assert_eq!(env.err("library.setRating", json!({"photos": ["../x"], "rating": 3})), "bad_params");
}

#[test]
fn ratings_flags_tags_albums_query_and_persistence() {
    let env = Env::new();
    let ids: Vec<String> = (0..4).map(|i| env.import(&format!("imports/q{i}.png"), &encode(&scene(40, 30, i + 10), Format::Png, None))).collect();
    let r = env.call("library.setRating", json!({"photos": [ids[0], ids[1]], "rating": 4}));
    assert_eq!(r["previous"][&ids[0]], 0);
    env.call("library.setRating", json!({"photos": [ids[2]], "rating": 2}));
    env.call("library.setFlag", json!({"photos": [ids[0]], "flag": "pick"}));
    env.call("library.setFlag", json!({"photos": [ids[3]], "flag": "reject"}));
    env.call("library.tag", json!({"photos": [ids[1], ids[2]], "tags": ["Beach", "sunset"]}));
    env.call("library.untag", json!({"photos": [ids[2]], "tags": ["sunset"]}));
    env.call("library.albumCreate", json!({"name": "Summer"}));
    env.call("library.albumAdd", json!({"album": "Summer", "photos": [ids[1], ids[3]]}));
    env.call("library.albumRemove", json!({"album": "Summer", "photos": [ids[3]]}));
    assert_eq!(env.err("library.setRating", json!({"photos": [ids[0]], "rating": 6})), "bad_params");
    assert_eq!(env.err("library.albumAdd", json!({"album": "Nope", "photos": [ids[0]]})), "not_found");

    let q = |p: Value| -> Vec<String> { env.call("library.query", p).as_array().unwrap().iter().map(|v| v["id"].as_str().unwrap().to_string()).collect() };
    let mut four = q(json!({"minRating": 4}));
    four.sort();
    let mut want = vec![ids[0].clone(), ids[1].clone()];
    want.sort();
    assert_eq!(four, want);
    assert_eq!(q(json!({"flag": "pick"})), vec![ids[0].clone()]);
    assert_eq!(q(json!({"flag": "reject"})), vec![ids[3].clone()]);
    assert_eq!(q(json!({"tags": ["beach", "sunset"]})), vec![ids[1].clone()]);
    assert_eq!(q(json!({"album": "Summer"})), vec![ids[1].clone()]);
    assert_eq!(q(json!({"text": "summer"})), vec![ids[1].clone()]);
    let top = q(json!({"sort": "rating", "limit": 1}));
    assert!(top[0] == ids[0] || top[0] == ids[1]);
    assert_eq!(q(json!({"sort": "rating", "order": "asc"}))[0], ids[3]);
    assert_eq!(q(json!({"limit": 2, "offset": 3})).len(), 1);
    assert_eq!(q(json!({"takenAfter": "2999-01-01"})).len(), 0);
    assert_eq!(q(json!({"takenBefore": "2999-01-01"})).len(), 4);
    assert_eq!(env.err("library.query", json!({"bogus": 1})), "bad_params");
    let albums = env.call("library.albumList", json!({}));
    assert_eq!(albums[0]["count"], 1);

    // A fresh registry and context (another process, say) sees the same catalog on disk.
    let env2 = Env { root: env.root.clone(), reg: Registry::new().with(&LibraryModule), ctx: env.ctx.clone(), events: env.events.clone() };
    let g = env2.call("library.get", json!({"photo": ids[1]}));
    assert_eq!(g["rating"], 4);
    assert_eq!(g["tags"], json!(["beach", "sunset"]));
    assert_eq!(g["albums"], json!(["Summer"]));
    std::mem::forget(env2); // shares env's directory; env cleans up.
    let raw: Value = serde_json::from_slice(&std::fs::read(env.root.join(".imagelab/catalog/catalog.json")).unwrap()).unwrap();
    assert_eq!(raw["photos"][&ids[0]]["rating"], 4);

    // Read-only realm refuses edits.
    let ro = env.ctx.clone().read_only();
    assert_eq!(env.reg.call(&ro, "library.setRating", json!({"photos": [ids[0]], "rating": 1})).unwrap_err().code, "read_only");
    assert!(env.reg.call(&ro, "library.query", json!({})).is_ok());

    // Delete moves to trash and removes from albums.
    let d = env.call("library.delete", json!({"photos": [ids[1]]}));
    assert_eq!(d["deleted"], json!([ids[1]]));
    let trash = env.root.join(d["trash"].as_str().unwrap());
    assert!(trash.join(&ids[1]).exists());
    assert!(trash.join("manifest.json").exists());
    assert!(!env.root.join(&ids[1]).exists());
    assert_eq!(q(json!({})).len(), 3);
    assert_eq!(env.call("library.albumList", json!({}))[0]["count"], 0);
}

#[test]
fn quality_orders_sharp_above_blurred_and_caches() {
    let env = Env::new();
    let sharp = scene(640, 480, 1);
    let blurred = box_blur(&sharp, 4, 2);
    let a = env.import("imports/sharp.png", &encode(&sharp, Format::Png, None));
    let b = env.import("imports/blur.png", &encode(&blurred, Format::Png, None));
    let t = std::time::Instant::now();
    let q = env.call("analyze.quality", json!({"photos": [a, b]}));
    let first = t.elapsed();
    let (qa, qb) = (&q[&a], &q[&b]);
    assert!(qa["sharpness"].as_f64().unwrap() > qb["sharpness"].as_f64().unwrap() + 0.2, "{qa} vs {qb}");
    assert!(qa["overall"].as_f64().unwrap() > qb["overall"].as_f64().unwrap());
    assert!(qb["notes"].as_array().unwrap().iter().any(|n| n.as_str().unwrap().contains("blur") || n.as_str().unwrap().contains("soft")), "{qb}");
    assert!(qa["faces"].is_object());
    for k in ["exposure", "contrast", "noise"] {
        let v = qa[k].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&v), "{k}={v}");
    }
    // Cached: the second call does not decode.
    let t = std::time::Instant::now();
    let q2 = env.call("analyze.quality", json!({"photos": [a, b]}));
    assert_eq!(q, q2);
    assert!(t.elapsed() <= first, "cached call slower than the first");
    // sort=quality uses the cache.
    let sorted = env.call("library.query", json!({"sort": "quality"}));
    assert_eq!(sorted[0]["id"], a);
    // get includes the cached scores.
    assert!(env.call("library.get", json!({"photo": a}))["quality"]["overall"].is_number());
    let missing = env.call("analyze.quality", json!({"photos": ["library/none.jpg"]}));
    assert_eq!(missing["library/none.jpg"]["error"]["code"], "not_found");
}

#[test]
fn exposure_and_noise_respond() {
    use crate::analysis::measure;
    let base = scene(256, 192, 2);
    let dark = Rgba { px: base.px.iter().enumerate().map(|(i, &v)| if i % 4 == 3 { v } else { v / 6 }).collect(), ..base.clone() };
    let (qb, qd) = (measure(&base), measure(&dark));
    assert!(qd.exposure < qb.exposure, "{} vs {}", qd.exposure, qb.exposure);
    // Flat grey + pseudo-random noise.
    let mut seed = 12345u32;
    let noisy = Rgba {
        w: 256,
        h: 192,
        px: (0..256 * 192)
            .flat_map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                let n = ((seed >> 16) % 41) as i32 - 20;
                let v = (128 + n) as u8;
                [v, v, v, 255]
            })
            .collect(),
    };
    let flat = Rgba { w: 256, h: 192, px: [128u8, 128, 128, 255].repeat(256 * 192) };
    assert!(measure(&noisy).noise < measure(&flat).noise);
}

#[test]
fn duplicates_cluster_resized_copy() {
    let env = Env::new();
    let img = scene(600, 400, 4);
    let small = box_resize(&img.px, 600, 400, 4, ChannelLayout::Rgba, 300, 200).unwrap();
    let other = scene(600, 400, 9);
    let a = env.import("imports/a.png", &encode(&img, Format::Png, None));
    let b = env.import("imports/a_small.jpg", &encode(&small, Format::Jpeg, None));
    let c = env.import("imports/c.png", &encode(&other, Format::Png, None));
    let r = env.call("analyze.duplicates", json!({"photos": [a, b, c]}));
    let clusters = r.as_array().unwrap();
    assert_eq!(clusters.len(), 2, "{r}");
    let pair = clusters.iter().find(|c| c["members"].as_array().unwrap().len() == 2).unwrap();
    let members: Vec<&str> = pair["members"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(members.contains(&a.as_str()) && members.contains(&b.as_str()), "{r}");
    let only = env.call("analyze.duplicates", json!({"photos": [a, b, c], "includeSingletons": false}));
    assert_eq!(only.as_array().unwrap().len(), 1);
    assert_eq!(env.err("analyze.duplicates", json!({"photos": [a], "threshold": 99})), "bad_params");
}

#[test]
fn thumbnail_histogram_and_contact_sheet() {
    let env = Env::new();
    let ids: Vec<String> = (0..5).map(|i| env.import(&format!("imports/s{i}.png"), &encode(&scene(400, 300, i + 20), Format::Png, None))).collect();
    let t = env.call("analyze.thumbnail", json!({"photo": ids[0], "maxSide": 128}));
    assert_eq!((t["width"].as_u64(), t["height"].as_u64()), (Some(128), Some(96)));
    let handle = t["image"].as_str().unwrap();
    assert!(env.ctx.shared.handles.image_str(handle).is_ok());
    let h = env.call("analyze.histogram", json!({"image": handle}));
    assert_eq!(h["luma"].as_array().unwrap().len(), 256);
    assert_eq!(h["luma"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).sum::<u64>(), 128 * 96);
    let hp = env.call("analyze.histogram", json!({"photo": ids[1]}));
    assert_eq!(hp["pixels"], 400 * 300);
    assert_eq!(env.err("analyze.histogram", json!({})), "bad_params");

    let mut photos: Vec<Value> = ids.iter().map(|s| json!(s)).collect();
    photos.push(json!("library/missing.jpg"));
    let s = env.call("analyze.contactSheet", json!({"photos": photos, "cellSize": 128}));
    let layout = s["layout"].as_array().unwrap();
    assert_eq!(layout.len(), 6);
    assert_eq!(layout[0]["index"], 1);
    assert_eq!(layout[0]["id"], ids[0]);
    assert!(layout[5]["error"].is_object());
    assert_eq!(s["columns"], 3);
    let img = env.ctx.shared.handles.image_str(s["image"].as_str().unwrap()).unwrap();
    assert_eq!((img.width, img.height), (3 * (128 + 6) + 6, 2 * (128 + 6) + 6));
    // Index labels are drawn: pure white pixels exist in the first cell's corner.
    let corner_white = (6..30u32).flat_map(|y| (6..30u32).map(move |x| (x, y))).any(|(x, y)| {
        let o = ((y * img.width + x) * 4) as usize;
        img.rgba[o..o + 3] == [255, 255, 255]
    });
    assert!(corner_white);
    let too_many: Vec<String> = (0..65).map(|i| format!("library/x{i}.jpg")).collect();
    assert_eq!(env.err("analyze.contactSheet", json!({"photos": too_many})), "too_large");
}

#[test]
fn registry_lists_both_areas() {
    let reg = Registry::new().with(&LibraryModule);
    let help = reg.help(None).unwrap();
    let areas: Vec<&str> = help.as_array().unwrap().iter().map(|a| a["area"].as_str().unwrap()).collect();
    assert_eq!(areas, vec!["analyze", "library"]);
    assert!(reg.get("library.delete").is_some_and(|f| f.effects == imagelab_api::Effects::Destructive));
    assert!(reg.get("analyze.quality").is_some_and(|f| f.effects == imagelab_api::Effects::ReadOnly));
}
