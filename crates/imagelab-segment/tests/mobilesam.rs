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
