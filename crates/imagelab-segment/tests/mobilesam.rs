//! MobileSAM with the real weights. Runs only when `IMAGELAB_MOBILESAM_WEIGHTS` names the
//! downloaded `mobile_sam-tiny-vitt.safetensors` (40.6 MB; URL in segment.models), so CI without
//! the file skips it:
//!
//! ```sh
//! IMAGELAB_MOBILESAM_WEIGHTS=/path/mobile_sam-tiny-vitt.safetensors \
//!   cargo test -p imagelab-segment --release --test mobilesam -- --nocapture
//! ```
#![cfg(feature = "mobile-sam")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};
use std::time::Instant;

use imagelab_api::{Ctx, Handles, Headless, NoEvents, Registry, Shared, Source};
use imagelab_segment::SegmentModule;
use serde_json::json;

const W: u32 = 640;
const H: u32 = 480;
/// Disk centre and radius; square [x, y, side].
const DISK: (f64, f64, f64) = (220.0, 240.0, 110.0);
const SQUARE: (u32, u32, u32) = (420, 140, 150);

fn inside_disk(x: u32, y: u32) -> bool {
    let (cx, cy, r) = DISK;
    (f64::from(x) - cx).powi(2) + (f64::from(y) - cy).powi(2) <= r * r
}

fn inside_square(x: u32, y: u32) -> bool {
    let (sx, sy, s) = SQUARE;
    x >= sx && x < sx + s && y >= sy && y < sy + s
}

/// Orange disk and a green square on a blue background, with a little texture.
fn scene() -> Vec<u8> {
    let mut px = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            let n = ((x * 7 + y * 13) % 17) as u8;
            let (r, g, b): (u8, u8, u8) = if inside_disk(x, y) {
                (235, 120, 30)
            } else if inside_square(x, y) {
                (40, 170, 70)
            } else {
                (40, 70, 160)
            };
            px.extend_from_slice(&[r.saturating_add(n), g.saturating_add(n), b.saturating_add(n), 255]);
        }
    }
    px
}

/// Intersection over union of a mask (threshold 128) against a shape.
fn iou(mask: &[u8], truth: impl Fn(u32, u32) -> bool) -> f64 {
    let (mut inter, mut uni) = (0u64, 0u64);
    for y in 0..H {
        for x in 0..W {
            let m = mask[(y * W + x) as usize] >= 128;
            let t = truth(x, y);
            inter += u64::from(m && t);
            uni += u64::from(m || t);
        }
    }
    inter as f64 / uni.max(1) as f64
}

#[test]
fn mobilesam_segments_points_and_boxes() {
    let Ok(weights) = std::env::var("IMAGELAB_MOBILESAM_WEIGHTS") else {
        eprintln!("skipped: set IMAGELAB_MOBILESAM_WEIGHTS to the MobileSAM weights");
        return;
    };
    let root = tempfile::tempdir().unwrap();
    std::fs::copy(&weights, root.path().join("weights.safetensors")).unwrap();
    let shared = Shared {
        root: root.path().canonicalize().unwrap(),
        session: Arc::new(Mutex::new(Headless::new())),
        handles: Arc::new(Handles::default()),
        events: Arc::new(NoEvents),
    };
    let ctx = Ctx::new(shared.clone(), Source::Agent);
    let reg = Registry::new().with(&SegmentModule);

    // Install (size + SHA-256 checked), then the status says ready.
    let installed = reg.call(&ctx, "segment.installModel", json!({"model": "mobilesam", "path": "weights.safetensors"})).unwrap();
    assert_eq!(installed["mobilesam"], "ready", "{installed}");
    let models = reg.call(&ctx, "segment.models", json!({})).unwrap();
    assert_eq!(models["mobilesam"]["state"], "ready");

    let img = shared.handles.put_image(imagelab_api::ImageBuf::new(W, H, scene()).unwrap()).unwrap().to_string();

    // One point on the disk: first call encodes, second only decodes.
    let t = Instant::now();
    let a = reg.call(&ctx, "segment.point", json!({"image": img, "points": [[220, 240, 1]]})).unwrap();
    let first = t.elapsed();
    assert_eq!(a["engine"], "mobilesam", "{a}");
    assert_eq!(a["cached"], false);
    let t = Instant::now();
    let b = reg.call(&ctx, "segment.point", json!({"image": img, "points": [[480, 210, 1]]})).unwrap();
    let second = t.elapsed();
    assert_eq!(b["cached"], true, "the embedding is reused");

    let mask_a = shared.handles.mask_str(a["mask"].as_str().unwrap()).unwrap();
    let mask_b = shared.handles.mask_str(b["mask"].as_str().unwrap()).unwrap();
    let iou_disk = iou(&mask_a.alpha, inside_disk);
    let iou_square = iou(&mask_b.alpha, inside_square);

    // A box around the square.
    let c = reg.call(&ctx, "segment.box", json!({"image": img, "box": [410, 130, 170, 170]})).unwrap();
    assert_eq!(c["engine"], "mobilesam", "{c}");
    let mask_c = shared.handles.mask_str(c["mask"].as_str().unwrap()).unwrap();
    let iou_box = iou(&mask_c.alpha, inside_square);

    eprintln!("encode+decode {first:?}, decode only {second:?}; IoU disk {iou_disk:.3}, square(point) {iou_square:.3}, square(box) {iou_box:.3}");
    assert!(iou_disk > 0.85, "disk IoU {iou_disk}");
    assert!(iou_square > 0.85, "square IoU {iou_square}");
    assert!(iou_box > 0.85, "box IoU {iou_box}");

    // A second image: encode + decode without the model load.
    let mut px2 = scene();
    px2.reverse();
    let img2 = shared.handles.put_image(imagelab_api::ImageBuf::new(W, H, px2).unwrap()).unwrap().to_string();
    let t = Instant::now();
    reg.call(&ctx, "segment.point", json!({"image": img2, "points": [[100, 100, 1]]})).unwrap();
    eprintln!("encode+decode without load {:?}", t.elapsed());

    // A bad file is refused.
    std::fs::write(root.path().join("bad.safetensors"), b"nope").unwrap();
    let bad = reg.call(&ctx, "segment.installModel", json!({"model": "mobilesam", "path": "bad.safetensors"})).unwrap_err();
    assert_eq!(bad.code, "bad_params");
}

/// The detail pass on a large image: a disk whose edge ripples (period ~25 px) are finer than
/// the whole-image proxy can show (a 3200 px image is seen at ~1/3 scale, and SAM's mask grid at
/// a quarter of that). `refine: true` re-segments a crop at full resolution and gets the edge.
#[test]
fn mobilesam_detail_pass_sharpens_a_small_object() {
    let Ok(weights) = std::env::var("IMAGELAB_MOBILESAM_WEIGHTS") else {
        eprintln!("skipped: set IMAGELAB_MOBILESAM_WEIGHTS to the MobileSAM weights");
        return;
    };
    let root = tempfile::tempdir().unwrap();
    std::fs::copy(&weights, root.path().join("weights.safetensors")).unwrap();
    let shared = Shared { root: root.path().canonicalize().unwrap(), session: Arc::new(Mutex::new(Headless::new())), handles: Arc::new(Handles::default()), events: Arc::new(NoEvents) };
    let ctx = Ctx::new(shared.clone(), Source::Agent);
    let reg = Registry::new().with(&SegmentModule);
    reg.call(&ctx, "segment.installModel", json!({"model": "mobilesam", "path": "weights.safetensors"})).unwrap();

    let (w, h) = (3200u32, 2400u32);
    let (cx, cy) = (1100.0f64, 900.0f64);
    let inside = |x: u32, y: u32| {
        let (dx, dy) = (f64::from(x) - cx, f64::from(y) - cy);
        let r = 160.0 + 9.0 * (40.0 * dy.atan2(dx)).sin();
        dx * dx + dy * dy <= r * r
    };
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let n = ((x * 7 + y * 13) % 17) as u8;
            let (r, g, b): (u8, u8, u8) = if inside(x, y) { (235, 120, 30) } else { (40, 70, 160) };
            px.extend_from_slice(&[r.saturating_add(n), g.saturating_add(n), b.saturating_add(n), 255]);
        }
    }
    let img = shared.handles.put_image(imagelab_api::ImageBuf::new(w, h, px).unwrap()).unwrap().to_string();
    let score = |v: &serde_json::Value| {
        let m = shared.handles.mask_str(v["mask"].as_str().unwrap()).unwrap();
        let (mut inter, mut uni) = (0u64, 0u64);
        // Over the object's neighbourhood only (the rest is empty in both and would inflate IoU).
        for y in 650..1150 {
            for x in 850..1350 {
                let (a, t) = (m.alpha[(y * w + x) as usize] >= 128, inside(x, y));
                inter += u64::from(a && t);
                uni += u64::from(a || t);
            }
        }
        inter as f64 / uni.max(1) as f64
    };
    for (name, params) in [("point", json!({"points": [[1100, 900, 1]]})), ("box", json!({"box": [920, 720, 360, 360]}))] {
        let mut plain = params.clone();
        plain["image"] = json!(img);
        let a = reg.call(&ctx, &format!("segment.{name}"), plain.clone()).unwrap();
        let mut detailed = plain;
        detailed["refine"] = json!(true);
        let t = Instant::now();
        let b = reg.call(&ctx, &format!("segment.{name}"), detailed).unwrap();
        let dt = t.elapsed();
        let (ia, ib) = (score(&a), score(&b));
        eprintln!("{name}: IoU {ia:.4} -> {ib:.4} with the detail pass ({dt:?}); detail {}", b["detail"]);
        assert_eq!(b["detail"]["refined"], true, "{}", b["detail"]);
        assert!(ib > ia + 0.005, "{name}: refine did not help ({ia} -> {ib})");
        assert!(ib > 0.95, "{name}: refined IoU {ib}");
    }
}

/// 24-bit BMP of RGBA pixels (alpha dropped), so a test can open a document from pixels.
fn bmp(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let row = (w * 3).div_ceil(4) * 4;
    let size = 54 + row * h;
    let mut out = Vec::with_capacity(size as usize);
    out.extend_from_slice(b"BM");
    for v in [size, 0, 54, 40] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    for v in [0u32, row * h, 2835, 2835, 0, 0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for y in (0..h).rev() {
        let start = out.len();
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            out.extend_from_slice(&[rgba[i + 2], rgba[i + 1], rgba[i]]);
        }
        out.resize(start + row as usize, 0);
    }
    out
}

/// On the open document, the detail pass is followed by PhotoCraft's Refine Edge, and the
/// document's selection and history are left as they were.
#[test]
fn mobilesam_detail_pass_runs_refine_edge_on_the_document() {
    let Ok(weights) = std::env::var("IMAGELAB_MOBILESAM_WEIGHTS") else {
        eprintln!("skipped: set IMAGELAB_MOBILESAM_WEIGHTS to the MobileSAM weights");
        return;
    };
    let root = tempfile::tempdir().unwrap();
    std::fs::copy(&weights, root.path().join("weights.safetensors")).unwrap();
    let shared = Shared { root: root.path().canonicalize().unwrap(), session: Arc::new(Mutex::new(authorized(root.path()))), handles: Arc::new(Handles::default()), events: Arc::new(NoEvents) };
    let ctx = Ctx::new(shared.clone(), Source::Agent);
    let reg = Registry::new().with(&SegmentModule);
    reg.call(&ctx, "segment.installModel", json!({"model": "mobilesam", "path": "weights.safetensors"})).unwrap();

    let (w, h) = (3200u32, 2400u32);
    let (cx, cy) = (1100.0f64, 900.0f64);
    let inside = |x: u32, y: u32| {
        let (dx, dy) = (f64::from(x) - cx, f64::from(y) - cy);
        let r = 160.0 + 9.0 * (40.0 * dy.atan2(dx)).sin();
        dx * dx + dy * dy <= r * r
    };
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let n = ((x * 7 + y * 13) % 17) as u8;
            let (r, g, b): (u8, u8, u8) = if inside(x, y) { (235, 120, 30) } else { (40, 70, 160) };
            px.extend_from_slice(&[r.saturating_add(n), g.saturating_add(n), b.saturating_add(n), 255]);
        }
    }
    let path = root.path().join("scene.bmp");
    std::fs::write(&path, bmp(w, h, &px)).unwrap();
    ctx.with_session(|hs| Ok(hs.open(std::path::Path::new("scene.bmp"))?)).unwrap();
    let _ = &path;
    let undo_before = ctx.with_session(|hs| Ok(hs.session.active().unwrap().history.undo_label().map(str::to_string))).unwrap();

    let t = Instant::now();
    let v = reg.call(&ctx, "segment.box", json!({"box": [920, 720, 360, 360], "refine": true})).unwrap();
    let dt = t.elapsed();
    let m = shared.handles.mask_str(v["mask"].as_str().unwrap()).unwrap();
    let (mut inter, mut uni) = (0u64, 0u64);
    for y in 650..1150 {
        for x in 850..1350 {
            let (a, t) = (m.alpha[(y * w + x) as usize] >= 128, inside(x, y));
            inter += u64::from(a && t);
            uni += u64::from(a || t);
        }
    }
    let iou = inter as f64 / uni.max(1) as f64;
    eprintln!("document box + detail + Refine Edge: IoU {iou:.4} in {dt:?}; detail {}", v["detail"]);
    assert_eq!(v["detail"]["refined"], true, "{}", v["detail"]);
    assert_eq!(v["detail"]["edge"]["ran"], true, "{}", v["detail"]);
    assert!(iou > 0.95, "IoU {iou}");
    // Read-only: no selection, no new history step.
    let (sel, undo_after) = ctx.with_session(|hs| {
        let d = hs.session.active().unwrap();
        Ok((d.doc.selection.is_some(), d.history.undo_label().map(str::to_string)))
    }).unwrap();
    assert!(!sel, "segment.box left a selection");
    assert_eq!(undo_after, undo_before, "segment.box left a history step");
}

/// A headless session that may read and write files under `root` (as the app's core grants).
fn authorized(root: &std::path::Path) -> Headless {
    let root = root.canonicalize().unwrap();
    Headless::with_workspace(photocraft_automation::AuthorizedWorkspace::new(Some(&root), Some(&root)).unwrap())
}

/// Calibration of the Refine Edge settings used after the detail pass (run explicitly:
/// `cargo test --release -p imagelab-segment --test mobilesam calibrate -- --ignored --nocapture`).
/// Two scenes: a high-contrast rippled edge and a low-contrast, noisy one; each setting is scored
/// by IoU against the exact shape, starting from the same detail-pass mask.
#[test]
#[ignore]
fn calibrate_refine_edge() {
    let Ok(weights) = std::env::var("IMAGELAB_MOBILESAM_WEIGHTS") else { return };
    let (w, h) = (3200u32, 2400u32);
    let (cx, cy) = (1100.0f64, 900.0f64);
    let ripple = move |x: u32, y: u32| {
        let (dx, dy) = (f64::from(x) - cx, f64::from(y) - cy);
        let r = 160.0 + 9.0 * (40.0 * dy.atan2(dx)).sin();
        dx * dx + dy * dy <= r * r
    };
    // A disk with eight 6 px spokes out to 230 px: thin structure a wide radius could erase.
    let spokes = move |x: u32, y: u32| {
        let (dx, dy) = (f64::from(x) - cx, f64::from(y) - cy);
        let d = (dx * dx + dy * dy).sqrt();
        if d <= 110.0 {
            return true;
        }
        if d > 230.0 {
            return false;
        }
        (0..8).any(|k| {
            let a = f64::from(k) * std::f64::consts::PI / 4.0;
            let (ux, uy) = (a.cos(), a.sin());
            let along = dx * ux + dy * uy;
            along > 0.0 && (dx * uy - dy * ux).abs() <= 3.0
        })
    };
    type Shape = Box<dyn Fn(u32, u32) -> bool>;
    let scenes: [(&str, [u8; 3], [u8; 3], u32, Shape); 3] = [
        ("high contrast", [235, 120, 30], [40, 70, 160], 17, Box::new(ripple)),
        ("low contrast, noisy", [130, 115, 100], [100, 105, 115], 41, Box::new(ripple)),
        ("thin spokes", [220, 220, 225], [60, 110, 60], 23, Box::new(spokes)),
    ];
    for (name, fg, bg, noise, inside) in scenes {
        let root = tempfile::tempdir().unwrap();
        std::fs::copy(&weights, root.path().join("weights.safetensors")).unwrap();
        let shared = Shared { root: root.path().canonicalize().unwrap(), session: Arc::new(Mutex::new(authorized(root.path()))), handles: Arc::new(Handles::default()), events: Arc::new(NoEvents) };
        let ctx = Ctx::new(shared.clone(), Source::Agent);
        let reg = Registry::new().with(&SegmentModule);
        reg.call(&ctx, "segment.installModel", json!({"model": "mobilesam", "path": "weights.safetensors"})).unwrap();
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let n = ((x * 7919 + y * 104_729) % noise) as u8;
                let c = if inside(x, y) { fg } else { bg };
                px.extend_from_slice(&[c[0].saturating_add(n), c[1].saturating_add(n), c[2].saturating_add(n), 255]);
            }
        }
        std::fs::write(root.path().join("scene.bmp"), bmp(w, h, &px)).unwrap();
        ctx.with_session(|hs| Ok(hs.open(std::path::Path::new("scene.bmp"))?)).unwrap();
        let img = shared.handles.put_image(imagelab_api::ImageBuf::new(w, h, px).unwrap()).unwrap().to_string();
        let score = |m: &[u8]| {
            let (mut inter, mut uni) = (0u64, 0u64);
            for y in 650..1150 {
                for x in 850..1350 {
                    let (a, t) = (m[(y * w + x) as usize] >= 128, inside(x, y));
                    inter += u64::from(a && t);
                    uni += u64::from(a || t);
                }
            }
            inter as f64 / uni.max(1) as f64
        };
        // The detail-pass mask (an image handle: no Refine Edge yet).
        let v = reg.call(&ctx, "segment.box", json!({"image": img, "box": [860, 660, 480, 480], "refine": true})).unwrap();
        let base = shared.handles.mask_str(v["mask"].as_str().unwrap()).unwrap();
        eprintln!("{name}: detail pass alone IoU {:.4}", score(&base.alpha));
        let mut rows = Vec::new();
        for radius in [0.0f64, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0] {
            for smooth in [0.0f64, 2.0] {
                for contrast in [0.0f64] {
                    let params = json!({"radius": radius, "smartRadius": true, "smooth": smooth, "contrast": contrast, "output": "selection"});
                    let t = Instant::now();
                    let (_, _, sel) = ctx.with_session(|hs| imagelab_segment::select::probe_from(hs, w, h, &base.alpha, "select.refineEdge", params.clone())).unwrap();
                    let dt = t.elapsed();
                    let s = score(&sel.unwrap());
                    rows.push((s, radius, smooth, contrast, dt));
                }
            }
        }
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        rows.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)));
        for (s, r, sm, c, dt) in rows.iter() {
            eprintln!("  IoU {s:.4}  radius {r:>4}  smooth {sm}  contrast {c:>4}  ({dt:?})");
        }
    }
}
