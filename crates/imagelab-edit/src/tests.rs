//! End-to-end tests through the registry, on generated photos in a temporary workspace.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use imagelab_api::{CollectEvents, Ctx, Event, EventSink, Handles, Headless, Registry, Shared, Source};
use photocraft_automation::AuthorizedWorkspace;
use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image};
use serde_json::{Value, json};

use crate::EditModule;

fn write_png(root: &Path, rel: &str, w: u32, h: u32, seed: u8) {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            px.extend_from_slice(&[(x * 255 / w) as u8 ^ seed, (y * 255 / h) as u8, seed, 255]);
        }
    }
    let img = Image::from_u8(w, h, ChannelLayout::Rgba, px).unwrap();
    let bytes = photocraft_codecs::encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

struct Env {
    _dir: tempfile::TempDir,
    root: std::path::PathBuf,
    registry: Registry,
    events: Arc<CollectEvents>,
    shared: Shared,
}

fn env() -> Env {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let ws = AuthorizedWorkspace::new(Some(&root), Some(&root)).unwrap();
    let events = Arc::new(CollectEvents::default());
    let shared = Shared { root: root.clone(), session: Arc::new(Mutex::new(Headless::with_workspace(ws))), handles: Arc::new(Handles::default()), events: events.clone() };
    Env { _dir: dir, root, registry: Registry::new().with(&EditModule), events, shared }
}

impl Env {
    fn ctx(&self) -> Ctx {
        Ctx::new(self.shared.clone(), Source::Agent)
    }
    fn call(&self, id: &str, params: Value) -> Result<Value, imagelab_api::ApiError> {
        self.registry.call(&self.ctx(), id, params)
    }
    fn ok(&self, id: &str, params: Value) -> Value {
        self.call(id, params.clone()).unwrap_or_else(|e| panic!("{id}({params}) failed: {e}"))
    }
    fn events(&self) -> Vec<Event> {
        self.events.0.lock().unwrap().clone()
    }
    fn layer_count(&self) -> usize {
        self.ok("doc.inspect", json!({}))["layers"].as_array().unwrap().len()
    }
}

#[test]
fn registers_all_functions() {
    let e = env();
    for id in [
        "doc.open", "doc.new", "doc.list", "doc.select", "doc.inspect", "doc.preview", "doc.save", "doc.export", "doc.exportOverwrite", "doc.close", "doc.undo", "doc.redo",
        "doc.history", "edit.commands", "edit.run", "edit.steps", "edit.adjust", "edit.auto", "edit.straighten", "edit.cropTo", "edit.saveRecipe", "edit.listRecipes",
        "edit.getRecipe", "edit.applyRecipe", "edit.deleteRecipe", "batch.apply", "batch.export", "batch.presets",
    ] {
        assert!(e.registry.get(id).is_some(), "{id} missing");
    }
    assert!(e.registry.warnings.is_empty(), "{:?}", e.registry.warnings);
}

#[test]
fn doc_open_preview_save_round_trip() {
    let e = env();
    write_png(&e.root, "library/2026-11/IMG_0001.png", 96, 64, 7);
    let original = std::fs::read(e.root.join("library/2026-11/IMG_0001.png")).unwrap();
    let opened = e.ok("doc.open", json!({"photo": "library/2026-11/IMG_0001.png"}));
    assert_eq!(opened["fromSidecar"], false);
    assert!(matches!(e.events().last(), Some(Event::DocChanged { source: Source::Agent, .. })));

    let preview = e.ok("doc.preview", json!({"maxSide": 48}));
    assert_eq!(preview["width"], 48);
    assert_eq!(preview["height"], 32);
    let img = e.shared.handles.image_str(preview["image"].as_str().unwrap()).unwrap();
    assert_eq!((img.width, img.height), (48, 32));
    assert!(e.call("doc.preview", json!({"maxSide": 99999})).is_err());

    e.ok("edit.adjust", json!({"kind": "exposure", "params": {"exposure": 0.5}}));
    let saved = e.ok("doc.save", json!({}));
    assert_eq!(saved["path"], "edits/library__2026-11__IMG_0001.png.pcraft");
    assert!(e.root.join("edits/library__2026-11__IMG_0001.png.pcraft").is_file());
    // Originals are never written.
    assert_eq!(std::fs::read(e.root.join("library/2026-11/IMG_0001.png")).unwrap(), original);
    assert_eq!(e.call("doc.save", json!({"path": "library/2026-11/IMG_0001.png"})).unwrap_err().code, "read_only_path");
    assert_eq!(e.call("doc.save", json!({"path": "../x.png"})).unwrap_err().code, "bad_params");

    // Reopening continues from the sidecar, with the adjustment layer.
    e.ok("doc.close", json!({}));
    let again = e.ok("doc.open", json!({"photo": "library/2026-11/IMG_0001.png"}));
    assert_eq!(again["fromSidecar"], true);
    assert_eq!(e.layer_count(), 2);

    // Export: resized on a duplicate, open document unchanged, no overwrite.
    let ex = e.ok("doc.export", json!({"path": "exports/one.jpg", "maxSide": 32, "quality": 80}));
    assert_eq!(ex["width"], 32);
    assert!(e.root.join("exports/one.jpg").is_file());
    assert_eq!(e.ok("doc.list", json!({}))["documents"].as_array().unwrap().len(), 1);
    assert_eq!(e.ok("doc.inspect", json!({}))["width"], 96);
    assert_eq!(e.call("doc.export", json!({"path": "exports/one.jpg"})).unwrap_err().code, "exists");
    e.ok("doc.exportOverwrite", json!({"path": "exports/one.jpg"}));
    // Read-only contexts refuse writes but allow looking.
    let ro = e.ctx().read_only();
    assert_eq!(e.registry.call(&ro, "doc.save", json!({})).unwrap_err().code, "read_only");
    assert!(e.registry.call(&ro, "doc.inspect", json!({})).is_ok());
}

#[test]
fn run_refuses_non_allowlisted_commands() {
    let e = env();
    e.ok("doc.new", json!({"width": 32, "height": 32}));
    for bad in ["file.open", "file.save", "edit.preferences.general", "layer.exportAs", "plugin.run"] {
        let err = e.call("edit.run", json!({"id": bad, "params": {}})).unwrap_err();
        assert_eq!(err.code, "not_allowed", "{bad}: {err}");
        assert!(err.message.contains("edit.commands"), "{err}");
    }
    e.ok("edit.run", json!({"id": "select.all"}));
    let cmds = e.ok("edit.commands", json!({"filter": "gaussian"}));
    assert!(cmds["commands"].as_array().unwrap().iter().any(|c| c["id"] == "filter.blur.gaussianBlur"), "{cmds}");
    let files = e.ok("edit.commands", json!({"filter": "file.", "limit": 1000}));
    assert!(files["commands"].as_array().unwrap().iter().all(|c| !c["id"].as_str().unwrap().starts_with("file.")));
}

#[test]
fn steps_atomic_rolls_back() {
    let e = env();
    e.ok("doc.new", json!({"width": 64, "height": 48}));
    let before = e.ok("doc.history", json!({}));
    let layers = e.layer_count();
    let err = e
        .call(
            "edit.steps",
            json!({"steps": [
                {"id": "filter.blur.gaussianBlur", "params": {"radius": 2}},
                {"fn": "edit.adjust", "params": {"kind": "exposure", "params": {"exposure": 1.0}}},
                {"id": "image.crop", "params": {"x": 0, "y": 0, "width": 10, "height": 10, "contentAware": "nope"}}
            ]}),
        )
        .unwrap_err();
    assert!(err.message.contains("step 2") && err.message.contains("rolled back 2"), "{err}");
    assert_eq!(e.layer_count(), layers);
    let after = e.ok("doc.history", json!({}));
    assert_eq!(after["count"], before["count"], "{before} vs {after}");
    assert_eq!(e.ok("doc.inspect", json!({}))["width"], 64);
    // A step outside the allowlist is refused before anything runs.
    let err = e.call("edit.steps", json!({"steps": [{"id": "file.save", "params": {}}]})).unwrap_err();
    assert_eq!(err.code, "not_allowed");
    // Non-atomic keeps what was applied.
    let _ = e.call("edit.steps", json!({"atomic": false, "steps": [{"fn": "edit.adjust", "params": {"kind": "vibrance", "params": {"vibrance": 20}}}, {"id": "image.crop", "params": {"contentAware": 3}}]}));
    assert_eq!(e.layer_count(), layers + 1);
}

#[test]
fn adjust_adds_an_adjustment_layer() {
    let e = env();
    e.ok("doc.new", json!({"width": 40, "height": 30}));
    let n = e.layer_count();
    let r = e.ok("edit.adjust", json!({"kind": "hueSaturation", "params": {"saturation": 20}, "name": "Warm"}));
    assert!(r["layer"].is_u64(), "{r}");
    let info = e.ok("doc.inspect", json!({}));
    let layers = info["layers"].as_array().unwrap();
    assert_eq!(layers.len(), n + 1);
    assert_eq!(layers[0]["name"], "Warm");
    assert_eq!(e.call("edit.adjust", json!({"kind": "nope"})).unwrap_err().code, "bad_params");
    // Undo removes it.
    e.ok("doc.undo", json!({}));
    assert_eq!(e.layer_count(), n);
    e.ok("doc.redo", json!({}));
    assert_eq!(e.layer_count(), n + 1);
}

#[test]
fn straighten_and_crop() {
    let e = env();
    e.ok("doc.new", json!({"width": 400, "height": 300}));
    let r = e.ok("edit.straighten", json!({"angle": 5}));
    let info = e.ok("doc.inspect", json!({}));
    assert_eq!(info["width"], r["width"]);
    assert!(info["width"].as_u64().unwrap() < 400 && info["height"].as_u64().unwrap() < 300, "{r}");
    e.ok("edit.cropTo", json!({"aspect": "1:1"}));
    let info = e.ok("doc.inspect", json!({}));
    assert_eq!(info["width"], info["height"]);
    e.ok("edit.cropTo", json!({"x": 0, "y": 0, "w": 10, "h": 20}));
    let info = e.ok("doc.inspect", json!({}));
    assert_eq!((info["width"].as_u64(), info["height"].as_u64()), (Some(10), Some(20)));
    assert!(e.call("edit.cropTo", json!({"x": 0, "y": 0, "w": 100, "h": 20})).is_err());
    assert!(e.call("edit.straighten", json!({"angle": 80})).is_err());
}

#[test]
fn recipes_save_apply_list_delete() {
    let e = env();
    write_png(&e.root, "library/a.png", 50, 40, 1);
    let steps = json!([
        {"fn": "edit.adjust", "params": {"kind": "photoFilter", "params": {"filter": "warming85", "density": 30}}},
        {"id": "layer.newAdjustmentLayer.vibrance", "params": {"vibrance": 25}}
    ]);
    e.ok("edit.saveRecipe", json!({"name": "warm", "steps": steps, "description": "warm and vivid"}));
    assert_eq!(e.call("edit.saveRecipe", json!({"name": "warm", "steps": steps})).unwrap_err().code, "exists");
    assert_eq!(e.call("edit.saveRecipe", json!({"name": "bad", "steps": [{"id": "file.open"}]})).unwrap_err().code, "not_allowed");
    assert_eq!(e.call("edit.saveRecipe", json!({"name": "../x", "steps": steps})).unwrap_err().code, "bad_params");
    let list = e.ok("edit.listRecipes", json!({}));
    assert_eq!(list["recipes"][0]["name"], "warm");
    assert_eq!(e.ok("edit.getRecipe", json!({"name": "warm"}))["steps"].as_array().unwrap().len(), 2);
    let r = e.ok("edit.applyRecipe", json!({"name": "warm", "photo": "library/a.png"}));
    assert_eq!(r["applied"], 2);
    assert_eq!(e.layer_count(), 3);
    // On the open document again.
    e.ok("edit.applyRecipe", json!({"name": "warm"}));
    assert_eq!(e.layer_count(), 5);
    e.ok("edit.deleteRecipe", json!({"name": "warm"}));
    assert_eq!(e.call("edit.getRecipe", json!({"name": "warm"})).unwrap_err().code, "not_found");
}

#[test]
fn batch_apply_exports_and_reports_progress() {
    let e = env();
    let photos: Vec<String> = (0..3).map(|i| format!("library/2026-11/IMG_{i:04}.png")).collect();
    for (i, p) in photos.iter().enumerate() {
        write_png(&e.root, p, 120, 80, i as u8 * 40);
    }
    e.ok("edit.saveRecipe", json!({"name": "pop", "steps": [{"fn": "edit.adjust", "params": {"kind": "vibrance", "params": {"vibrance": 30}}}]}));
    // The UI's open document stays as it is.
    e.ok("doc.new", json!({"width": 10, "height": 10}));
    let r = e.ok(
        "batch.apply",
        json!({"photos": photos, "recipe": "pop", "saveSidecar": true, "export": {"format": "jpg", "maxSide": 60, "quality": 85, "suffix": "_web"}, "concurrency": 2}),
    );
    assert_eq!(r["ok"], 3, "{r}");
    assert_eq!(r["cancelled"], false);
    for res in r["results"].as_array().unwrap() {
        let out = res["output"].as_str().unwrap();
        assert!(out.starts_with("exports/IMG_") && out.ends_with("_web.jpg"), "{out}");
        let bytes = std::fs::read(e.root.join(out)).unwrap();
        let img = photocraft_codecs::decode(&bytes).unwrap();
        assert_eq!(img.dimensions(), (60, 40));
        assert!(e.root.join(res["sidecar"].as_str().unwrap()).is_file());
    }
    let events = e.events();
    let progress: Vec<u64> = events.iter().filter_map(|ev| if let Event::JobProgress { done, total: 3, .. } = ev { Some(*done) } else { None }).collect();
    assert_eq!(progress.len(), 3, "{events:?}");
    assert!(progress.contains(&3));
    assert!(matches!(events.last(), Some(Event::JobFinished { ok: true, .. })), "{events:?}");
    assert_eq!(e.ok("doc.list", json!({}))["documents"].as_array().unwrap().len(), 1);
    assert_eq!(e.ok("doc.inspect", json!({}))["width"], 10);

    // A second run never overwrites: names get " (2)". The preset way, 4:5 crop.
    let r = e.ok("batch.export", json!({"photos": [photos[0].clone()], "preset": "instagram", "crop": true}));
    let out = r["results"][0]["output"].as_str().unwrap().to_string();
    assert_eq!(out, "exports/IMG_0000_ig.jpg");
    let img = photocraft_codecs::decode(&std::fs::read(e.root.join(&out)).unwrap()).unwrap();
    assert_eq!(img.dimensions(), (64, 80)); // 4:5 of 120x80, never upscaled to 1080.
    let r = e.ok("batch.export", json!({"photos": [photos[0].clone()], "preset": "instagram", "crop": true}));
    assert_eq!(r["results"][0]["output"], "exports/IMG_0000_ig (2).jpg");
    // A missing photo fails alone.
    let r = e.ok("batch.apply", json!({"photos": ["library/nope.png", photos[1].clone()], "export": {"format": "png"}}));
    assert_eq!((r["ok"].as_u64(), r["failed"].as_u64()), (Some(1), Some(1)));
    assert!(e.call("batch.export", json!({"photos": photos, "preset": "nope"})).is_err());
}

/// Cancels the context after the first progress event.
struct CancelAfterFirst(Arc<AtomicBool>, Mutex<Vec<Event>>);
impl EventSink for CancelAfterFirst {
    fn emit(&self, event: Event) {
        if matches!(event, Event::JobProgress { .. }) {
            self.0.store(true, Ordering::Relaxed);
        }
        self.1.lock().unwrap().push(event);
    }
}

#[test]
fn batch_honours_cancel() {
    let e = env();
    let photos: Vec<String> = (0..4).map(|i| format!("library/c{i}.png")).collect();
    for p in &photos {
        write_png(&e.root, p, 32, 32, 3);
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let sink = Arc::new(CancelAfterFirst(cancel.clone(), Mutex::new(Vec::new())));
    let shared = Shared { events: sink.clone(), ..e.shared.clone() };
    let mut ctx = Ctx::new(shared, Source::Ui);
    ctx.cancel = cancel;
    let r = e.registry.call(&ctx, "batch.apply", json!({"photos": photos, "export": {"format": "png"}, "concurrency": 1})).unwrap();
    assert_eq!(r["cancelled"], true, "{r}");
    assert_eq!(r["results"].as_array().unwrap().len(), 1);
    let events = sink.1.lock().unwrap();
    assert!(matches!(events.last(), Some(Event::JobFinished { ok: false, .. })), "{events:?}");
}

#[test]
fn pixel_commands_retarget_from_an_adjustment_layer() {
    let e = env();
    e.ok("doc.new", json!({"width": 32, "height": 32}));
    e.ok("edit.adjust", json!({"kind": "vibrance", "params": {"vibrance": 10}}));
    let v = e.ok("edit.run", json!({"id": "image.autoColor", "params": {}}));
    assert!(v["retargetedLayer"].is_u64(), "{v}");
}

/// Renders every look (and template) on `IMAGELAB_LOOK_PHOTO` into `IMAGELAB_LOOK_OUT` for a
/// visual check, with timings; each look must be exactly one history step.
#[test]
#[ignore]
fn render_looks() {
    let (Ok(photo), Ok(out)) = (std::env::var("IMAGELAB_LOOK_PHOTO"), std::env::var("IMAGELAB_LOOK_OUT")) else { return };
    let e = env();
    std::fs::create_dir_all(e.root.join("library/x")).unwrap();
    std::fs::copy(&photo, e.root.join("library/x/p.jpg")).unwrap();
    e.ok("doc.open", json!({"photo": "library/x/p.jpg"}));
    println!("suggest {}", e.ok("edit.lookSuggest", json!({})));
    let mut runs: Vec<(String, Value)> = crate::looks::looks().iter().filter(|l| l.id != "template").map(|l| (l.id.to_string(), json!({"look": l.id}))).collect();
    for (id, ..) in crate::looks::TEMPLATES {
        runs.push((format!("t_{id}"), json!({"look": "template", "style": id})));
    }
    // Twice: the second run (a slider move) reuses the measurements and the scene.
    runs.push(("enhance_sky".into(), json!({"look": "enhance", "sky": 70})));
    runs.push(("enhance_sky2".into(), json!({"look": "enhance", "sky": 40})));
    for s in ["sunset", "dusk"] {
        runs.push((format!("sky_{s}"), json!({"look": "skyReplace", "style": s})));
    }
    for (name, p) in runs {
        let before = e.ok("doc.history", json!({}))["count"].as_u64().unwrap();
        let t = std::time::Instant::now();
        let r = e.call("edit.look", p);
        let ms = t.elapsed().as_millis();
        let after = e.ok("doc.history", json!({}))["count"].as_u64().unwrap();
        println!("{name:14} {ms:5} ms  history +{}  {}", after - before, r.as_ref().map_or_else(|e| e.to_string(), |v| v.to_string()));
        if after > before {
            let mut s = e.shared.session.lock().unwrap();
            std::fs::write(format!("{out}/{name}.png"), s.render_png(None, 600).unwrap()).unwrap();
            drop(s);
            e.ok("doc.undo", json!({}));
        }
    }
}

/// Face liquify strokes on a grid with a stand-in face, rendered to `IMAGELAB_LOOK_OUT`.
#[test]
#[ignore]
fn render_face_liquify_grid() {
    let Ok(out) = std::env::var("IMAGELAB_LOOK_OUT") else { return };
    let e = env();
    let (w, h) = (600u32, 600u32);
    let mut px = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let line = x % 20 == 0 || y % 20 == 0;
            px.extend_from_slice(if line { &[0, 0, 0, 255] } else { &[235, 220, 200, 255] });
        }
    }
    let img = Image::from_u8(w, h, ChannelLayout::Rgba, px).unwrap();
    let bytes = photocraft_codecs::encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    std::fs::create_dir_all(e.root.join("library/x")).unwrap();
    std::fs::write(e.root.join("library/x/g.png"), bytes).unwrap();
    e.ok("doc.open", json!({"photo": "library/x/g.png"}));
    let face = crate::looks::Face { rect: [150.0, 120.0, 300.0, 360.0], eyes: vec![[240.0, 255.0], [360.0, 255.0]], nose: Some([300.0, 330.0]), mouth: vec![[250.0, 400.0], [350.0, 400.0]] };
    let cases = [
        ("eyes", json!({"eyes": 100})),
        ("slim", json!({"faceWidth": -100})),
        ("smile", json!({"smile": 100})),
        ("chin", json!({"chin": 100})),
        ("nose", json!({"nose": -100})),
    ];
    for (name, p) in cases {
        let p: crate::face::FaceParams = serde_json::from_value(p).unwrap();
        let strokes = crate::face::strokes(&face, &p);
        let mut s = e.shared.session.lock().unwrap();
        let t = std::time::Instant::now();
        s.command_run("filter.liquify", json!({"strokes": strokes})).unwrap();
        println!("{name} {:?}", t.elapsed());
        std::fs::write(format!("{out}/liq_{name}.png"), s.render_png(None, 600).unwrap()).unwrap();
        s.command_run("edit.undo", json!({})).unwrap();
    }
}
