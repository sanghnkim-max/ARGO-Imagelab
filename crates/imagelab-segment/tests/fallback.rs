//! End to end on a synthetic document, without the SAM 3 model: PhotoCraft selections, the
//! selection ↔ mask bridges, and segment.* falling back to PhotoCraft without touching the
//! document's selection.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use imagelab_api::{CollectEvents, Ctx, Event, Handles, Headless, Registry, Shared, Source};
use imagelab_segment::SegmentModule;
use serde_json::{Value, json};

struct Fixture {
    reg: Registry,
    ctx: Ctx,
    events: Arc<CollectEvents>,
    _root: tempfile::TempDir,
}

/// 200×100 white document with a black 60×40 rectangle at (50, 20), nothing selected.
fn fixture() -> Fixture {
    let root = tempfile::tempdir().unwrap();
    let events = Arc::new(CollectEvents::default());
    let shared = Shared {
        root: root.path().to_path_buf(),
        session: Arc::new(Mutex::new(Headless::new())),
        handles: Arc::new(Handles::default()),
        events: events.clone(),
    };
    let ctx = Ctx::new(shared, Source::Agent);
    ctx.with_session(|h| {
        h.command_run("file.new", json!({"width": 200, "height": 100, "background": "white"}))?;
        h.command_run("select.rect", json!({"x": 50, "y": 20, "width": 60, "height": 40, "antiAlias": false}))?;
        h.command_run("edit.fill", json!({"contents": "color", "color": "#000000"}))?;
        h.command_run("select.deselect", json!({}))?;
        Ok(())
    })
    .unwrap();
    Fixture { reg: Registry::new().with(&SegmentModule), ctx, events, _root: root }
}

impl Fixture {
    fn call(&self, id: &str, p: Value) -> Value {
        self.reg.call(&self.ctx, id, p).unwrap_or_else(|e| panic!("{id}: {e}"))
    }
    fn err(&self, id: &str, p: Value) -> String {
        self.reg.call(&self.ctx, id, p).expect_err(id).code
    }
    fn has_selection(&self) -> bool {
        self.ctx.with_session(|h| Ok(h.session.active().unwrap().doc.selection.is_some())).unwrap()
    }
    fn mask_alpha(&self, handle: &Value) -> Arc<imagelab_api::MaskBuf> {
        self.ctx.shared.handles.mask_str(handle.as_str().unwrap()).unwrap()
    }
}

const RECT_COVERAGE: f64 = 60.0 * 40.0 / (200.0 * 100.0);

#[test]
fn magic_wand_to_mask_has_the_rectangle() {
    let f = fixture();
    let r = f.call("select.magicWand", json!({"x": 80, "y": 40, "tolerance": 10, "antiAlias": false}));
    assert_eq!(r["selected"], true);
    assert!((r["coverage"].as_f64().unwrap() - RECT_COVERAGE).abs() < 0.005, "{r}");
    assert!(f.events.0.lock().unwrap().iter().any(|e| matches!(e, Event::DocChanged { .. })));

    let m = f.call("select.toMask", json!({}));
    assert_eq!((m["width"].as_u64(), m["height"].as_u64()), (Some(200), Some(100)));
    assert_eq!(m["bounds"], json!([50, 20, 60, 40]));
    let mask = f.mask_alpha(&m["mask"]);
    assert_eq!(mask.alpha[40 * 200 + 80], 255);
    assert_eq!(mask.alpha[5 * 200 + 5], 0);

    // read-only realm: toMask works, changing the selection does not
    let ro = f.ctx.clone().read_only();
    assert!(f.reg.call(&ro, "select.toMask", json!({})).is_ok());
    assert_eq!(f.reg.call(&ro, "select.fromMask", json!({"mask": m["mask"]})).unwrap_err().code, "read_only");
}

#[test]
fn from_mask_round_trips_and_undoes() {
    let f = fixture();
    f.call("select.magicWand", json!({"x": 80, "y": 40, "tolerance": 10, "antiAlias": false}));
    let m = f.call("select.toMask", json!({}));
    f.call("select.none", json!({}));
    assert!(!f.has_selection());
    assert_eq!(f.err("select.toMask", json!({})), "not_found");

    let r = f.call("select.fromMask", json!({"mask": m["mask"]}));
    assert_eq!(r["bounds"], json!([50, 20, 60, 40]));
    let back = f.call("select.toMask", json!({}));
    assert_eq!(f.mask_alpha(&back["mask"]).alpha, f.mask_alpha(&m["mask"]).alpha);

    // one history step: undo leaves nothing selected again
    let label = f.ctx.with_session(|h| Ok(h.session.active().unwrap().history.undo_label().map(str::to_string))).unwrap();
    assert_eq!(label.as_deref(), Some("Load Selection"));
    f.ctx.with_session(|h| Ok(h.session.undo())).unwrap();
    assert!(!f.has_selection());

    // modes: all minus the rectangle, then intersect with the rectangle = nothing
    f.call("select.all", json!({}));
    let r = f.call("select.fromMask", json!({"mask": m["mask"], "mode": "subtract"}));
    assert!((r["coverage"].as_f64().unwrap() - (1.0 - RECT_COVERAGE)).abs() < 0.005, "{r}");
    let r = f.call("select.fromMask", json!({"mask": m["mask"], "mode": "intersect"}));
    assert_eq!(r["selected"], false, "{r}");

    // feather softens the edge and keeps the area
    let r = f.call("select.fromMask", json!({"mask": m["mask"], "feather": 4}));
    assert!((r["coverage"].as_f64().unwrap() - RECT_COVERAGE).abs() < 0.01, "{r}");
    let soft = f.call("select.toMask", json!({}));
    let a = f.mask_alpha(&soft["mask"]);
    let edge = a.alpha[40 * 200 + 50];
    assert!(edge > 0 && edge < 255, "edge {edge}");

    // a smaller mask is resampled to the document
    let small = f.ctx.shared.handles.put_mask(imagelab_api::MaskBuf::new(20, 10, vec![255; 200]).unwrap()).unwrap();
    let r = f.call("select.fromMask", json!({"mask": small.to_string()}));
    assert!((r["coverage"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(f.err("select.fromMask", json!({"mask": "mask:999"})), "not_found");
    assert_eq!(f.err("select.fromMask", json!({"mask": small.to_string(), "feather": -1})), "bad_params");
}

#[test]
fn segment_falls_back_to_photocraft_without_touching_the_selection() {
    let f = fixture();
    let st = f.call("segment.models", json!({}));
    assert!(st["sam3"] == "missing" || st["sam3"] == "unavailable", "{st}");

    let p = f.call("segment.point", json!({"points": [[80, 40, 1]]}));
    assert_eq!(p["engine"], "photocraft", "{p}");
    assert!(p["samError"].is_string());
    assert_eq!((p["width"].as_u64(), p["height"].as_u64()), (Some(200), Some(100)));
    assert!(!f.has_selection(), "segment.point left a selection");

    let b = f.call("segment.box", json!({"box": [45, 15, 70, 50]}));
    assert_eq!(b["engine"], "photocraft");
    assert!(!f.has_selection());

    let s = f.call("segment.subject", json!({}));
    assert_eq!(s["engine"], "photocraft");

    // a selection the person made survives a fallback probe, and so does its history
    f.call("select.magicWand", json!({"x": 80, "y": 40, "tolerance": 10, "antiAlias": false}));
    let before = f.call("select.toMask", json!({}));
    let undo = f.ctx.with_session(|h| Ok(h.session.active().unwrap().history.undo_label().map(str::to_string))).unwrap();
    f.call("segment.subject", json!({}));
    f.call("segment.text", json!({"prompt": "sky"}));
    let after = f.call("select.toMask", json!({}));
    assert_eq!(f.mask_alpha(&before["mask"]).alpha, f.mask_alpha(&after["mask"]).alpha);
    let undo_after = f.ctx.with_session(|h| Ok(h.session.active().unwrap().history.undo_label().map(str::to_string))).unwrap();
    assert_eq!(undo, undo_after);

    // read-only realm allows segment.*
    let ro = f.ctx.clone().read_only();
    assert!(f.reg.call(&ro, "segment.point", json!({"points": [[80, 40, 1]]})).is_ok());

    // text concepts PhotoCraft can't stand in for need the model
    assert_eq!(f.err("segment.text", json!({"prompt": "red car"})), "unavailable");
    // bad input
    assert_eq!(f.err("segment.point", json!({"points": [[500, 40, 1]]})), "bad_params");
    assert_eq!(f.err("segment.point", json!({"points": [[80, 40, 0]]})), "bad_params");
    assert_eq!(f.err("segment.point", json!({"points": []})), "bad_params");
    // an image handle has no PhotoCraft fallback
    let img = f.ctx.shared.handles.put_image(imagelab_api::ImageBuf::new(4, 4, vec![128; 64]).unwrap()).unwrap();
    assert_eq!(f.err("segment.point", json!({"image": img.to_string(), "points": [[2, 2, 1]]})), "unavailable");
}

#[test]
fn ensure_model_asks_first_and_needs_a_download_location() {
    let f = fixture();
    let code = f.err("segment.ensureModel", json!({}));
    if code == "unavailable" {
        return; // built without sam3
    }
    assert_eq!(code, "confirm_required");
    let msg = f.reg.call(&f.ctx, "segment.ensureModel", json!({})).unwrap_err().message;
    assert!(msg.contains("GB") && msg.contains("SAM License"), "{msg}");
    if std::env::var("IMAGELAB_SAM3_MIRRORS").is_err() {
        assert_eq!(f.err("segment.ensureModel", json!({"confirm": true})), "unavailable");
    }
    // Destructive: refused in a read-only realm
    let ro = f.ctx.clone().read_only();
    assert_eq!(f.reg.call(&ro, "segment.ensureModel", json!({"confirm": true})).unwrap_err().code, "read_only");
    // install checks the files before touching the model folder
    std::fs::create_dir_all(f.ctx.shared.root.join("dl")).unwrap();
    std::fs::write(f.ctx.shared.root.join("dl/model.safetensors"), b"nope").unwrap();
    std::fs::write(f.ctx.shared.root.join("dl/vocab.json"), b"{}").unwrap();
    std::fs::write(f.ctx.shared.root.join("dl/merges.txt"), b"").unwrap();
    assert_eq!(f.err("segment.installModel", json!({"path": "dl"})), "bad_params");
    assert_eq!(f.err("segment.installModel", json!({"path": "../etc"})), "bad_params");
    assert_eq!(f.call("segment.models", json!({}))["sam3"], "missing");
}
