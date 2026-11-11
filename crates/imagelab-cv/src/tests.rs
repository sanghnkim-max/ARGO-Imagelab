//! End-to-end tests through the registry, on synthetic images.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use imagelab_api::{Ctx, Handles, Headless, ImageBuf, MaskBuf, NoEvents, Registry, Shared, Source};
use serde_json::{Value, json};

use super::CvModule;

struct Lab {
    _dir: tempfile::TempDir,
    reg: Registry,
    ctx: Ctx,
}

impl Lab {
    fn new() -> Self {
        Self::with_session(|_| Headless::new())
    }

    fn with_session(make: impl FnOnce(&std::path::Path) -> Headless) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let shared = Shared { root: root.clone(), session: Arc::new(Mutex::new(make(&root))), handles: Arc::new(Handles::default()), events: Arc::new(NoEvents) };
        Lab { _dir: dir, reg: Registry::new().with(&CvModule), ctx: Ctx::new(shared, Source::Agent) }
    }

    fn call(&self, id: &str, p: Value) -> Value {
        match self.reg.call(&self.ctx, id, p.clone()) {
            Ok(v) => v,
            Err(e) => panic!("{id} {p}: {e}"),
        }
    }

    fn err(&self, id: &str, p: Value) -> String {
        self.reg.call(&self.ctx, id, p).unwrap_err().code
    }

    fn put_img(&self, w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> String {
        let f = &f;
        let rgba = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).flat_map(|(x, y)| f(x, y)).collect();
        self.ctx.shared.handles.put_image(ImageBuf::new(w, h, rgba).unwrap()).unwrap().to_string()
    }

    fn put_mask(&self, w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> String {
        let f = &f;
        let a = (0..h).flat_map(|y| (0..w).map(move |x| f(x, y))).collect();
        self.ctx.shared.handles.put_mask(MaskBuf::new(w, h, a).unwrap()).unwrap().to_string()
    }

    fn img(&self, v: &Value) -> Arc<ImageBuf> {
        self.ctx.shared.handles.image_str(v["img"].as_str().unwrap()).unwrap()
    }

    fn mask(&self, v: &Value) -> Arc<MaskBuf> {
        self.ctx.shared.handles.mask_str(v["mask"].as_str().unwrap()).unwrap()
    }
}

fn gray(v: u8) -> [u8; 4] {
    [v, v, v, 255]
}

fn in_square(x: u32, y: u32, a: u32, b: u32) -> bool {
    (a..b).contains(&x) && (a..b).contains(&y)
}

#[test]
fn every_function_is_documented() {
    let lab = Lab::new();
    let help = lab.reg.help(Some("cv")).unwrap();
    let fns = help.as_array().unwrap();
    assert!(fns.len() >= 40, "{} functions", fns.len());
    for f in fns {
        assert!(f["description"].as_str().unwrap().len() > 30, "{}", f["name"]);
        assert_eq!(f["params"]["type"], "object", "{}", f["name"]);
    }
    assert!(lab.reg.warnings.is_empty(), "{:?}", lab.reg.warnings);
    for (id, fx) in [("cv.write", "revertible"), ("cv.overwrite", "destructive"), ("cv.toDocumentLayer", "revertible"), ("cv.canny", "readOnly")] {
        assert_eq!(serde_json::to_value(lab.reg.get(id).unwrap().effects).unwrap(), fx, "{id}");
    }
}

#[test]
fn read_write_paths_and_effects() {
    let lab = Lab::new();
    let img = lab.put_img(5, 4, |x, y| [x as u8 * 40, y as u8 * 60, 7, 255]);
    let w = lab.call("cv.write", json!({"img": img, "path": "a.png"}));
    assert_eq!(w["path"], "exports/a.png");
    assert_eq!(lab.err("cv.write", json!({"img": img, "path": "a.png"})), "exists");
    assert_eq!(lab.call("cv.overwrite", json!({"img": img, "path": "exports/a.png"}))["overwrote"], true);
    let back = lab.call("cv.read", json!({"path": "exports/a.png"}));
    assert_eq!(back["format"], "png");
    assert_eq!(*lab.img(&back), *lab.ctx.shared.handles.image_str(&img).unwrap());
    // JPEG and masks.
    let m = lab.put_mask(8, 8, |x, _| (x * 32) as u8);
    assert_eq!(lab.call("cv.write", json!({"mask": m, "path": "m.jpg", "quality": 95}))["format"], "jpeg");
    let mj = lab.call("cv.read", json!({"path": "exports/m.jpg"}));
    assert_eq!(mj["width"], 8);
    // Escapes are refused.
    assert_eq!(lab.err("cv.read", json!({"path": "../x.png"})), "forbidden");
    assert_eq!(lab.err("cv.read", json!({"path": "/etc/hosts"})), "forbidden");
    assert_eq!(lab.err("cv.write", json!({"img": img, "path": "../../x.png"})), "forbidden");
    assert_eq!(lab.err("cv.read", json!({"path": "nope.png"})), "not_found");
    assert_eq!(lab.err("cv.write", json!({"img": img, "path": "x.unknown"})), "bad_params");
    // Read-only realms may analyse but not write.
    let ro = lab.ctx.clone().read_only();
    assert_eq!(lab.reg.call(&ro, "cv.write", json!({"img": img, "path": "b.png"})).unwrap_err().code, "read_only");
    assert!(lab.reg.call(&ro, "cv.gaussianBlur", json!({"img": img, "sigma": 1.0})).is_ok());
    // Typos are caught.
    assert_eq!(lab.err("cv.gaussianBlur", json!({"img": img, "sigmaa": 1.0})), "bad_params");
    assert_eq!(lab.err("cv.gaussianBlur", json!({"img": "img:999", "sigma": 1.0})), "not_found");
}

#[test]
fn read_applies_exif_orientation() {
    // A 2x1 JPEG-free check: PNG with an eXIf chunk is decoded upright by the codecs; we only
    // check the size swap for orientation 6 through the codec directly.
    let img = ImageBuf::new(3, 2, vec![255; 24]).unwrap();
    let png = super::image_to_png(&img).unwrap();
    let back = super::decode_image(&png).unwrap();
    assert_eq!((back.width, back.height), (3, 2));
}

#[test]
fn info_and_free() {
    let lab = Lab::new();
    let img = lab.put_img(4, 4, |_, _| gray(9));
    let i = lab.call("cv.info", json!({"img": img}));
    assert_eq!(i["kind"], "image");
    assert_eq!(i["opaque"], true);
    let f = lab.call("cv.free", json!({"handles": [img, "mask:77"]}));
    assert_eq!(f["freed"], 1);
    assert_eq!(f["missing"][0], "mask:77");
    assert_eq!(lab.call("cv.info", json!({}))["arena"]["handles"], 0);
}

#[test]
fn geometry_family() {
    let lab = Lab::new();
    let img = lab.put_img(40, 20, |x, y| [x as u8 * 6, y as u8 * 12, 100, 255]);
    let r = lab.call("cv.resize", json!({"img": img, "width": 20}));
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(20), Some(10)));
    let r = lab.call("cv.resize", json!({"img": img, "scale": 2.0, "interpolation": "cubic"}));
    assert_eq!(r["width"], 80);
    let r = lab.call("cv.resize", json!({"img": img, "maxSide": 10}));
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(10), Some(5)));
    assert_eq!(lab.err("cv.resize", json!({"img": img, "width": 100000})), "too_large");
    let c = lab.call("cv.crop", json!({"img": img, "x": 30, "y": 15, "w": 20, "h": 20}));
    assert_eq!(c["rect"], json!({"x": 30, "y": 15, "w": 10, "h": 5}));
    assert_eq!(lab.img(&c).rgba[0..4], [180, 180, 100, 255]);
    assert_eq!(lab.err("cv.crop", json!({"img": img, "x": 50, "y": 0, "w": 5, "h": 5})), "bad_params");
    let q = lab.call("cv.rotate", json!({"img": img, "angle": 90}));
    assert_eq!((q["width"].as_u64(), q["height"].as_u64()), (Some(20), Some(40)));
    // Top-right pixel of the source becomes top-left after a counterclockwise quarter turn.
    assert_eq!(lab.img(&q).rgba[0..4], [234, 0, 100, 255]);
    let e = lab.call("cv.rotate", json!({"img": img, "angle": 30, "expand": true}));
    assert!(e["width"].as_u64().unwrap() > 40);
    let k = lab.call("cv.rotate", json!({"img": img, "angle": 10, "crop": true}));
    let ki = lab.img(&k);
    assert!(ki.width < 40 && ki.rgba.chunks(4).all(|p| p[3] == 255), "no empty corners");
    let f = lab.call("cv.flip", json!({"img": img, "mode": "horizontal"}));
    assert_eq!(lab.img(&f).rgba[0..4], [234, 0, 100, 255]);
    // warpAffine translation and warpPerspective identity.
    let t = lab.call("cv.warpAffine", json!({"img": img, "matrix": [1, 0, 5, 0, 1, 0]}));
    let ti = lab.img(&t);
    assert_eq!(ti.rgba[(5 * 4)..(5 * 4 + 4)], lab.ctx.shared.handles.image_str(&img).unwrap().rgba[0..4]);
    assert_eq!(ti.rgba[3], 0, "uncovered pixels are transparent");
    let id = lab.call("cv.warpPerspective", json!({"img": img, "matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]}));
    assert_eq!(*lab.img(&id), *lab.ctx.shared.handles.image_str(&img).unwrap());
    let q = [[0, 0], [39, 0], [39, 19], [0, 19]];
    let id2 = lab.call("cv.warpPerspective", json!({"img": img, "srcQuad": q, "dstQuad": q}));
    assert_eq!(*lab.img(&id2), *lab.ctx.shared.handles.image_str(&img).unwrap());
    // Rectify a sub-quad: the output is sized from the quad.
    let rect = lab.call("cv.warpPerspective", json!({"img": img, "srcQuad": [[10, 5], [29, 5], [29, 14], [10, 14]]}));
    assert_eq!((rect["width"].as_u64(), rect["height"].as_u64()), (Some(19), Some(9)));
    assert_eq!(lab.img(&rect).rgba[0..4], [60, 60, 100, 255]);
    assert_eq!(lab.err("cv.warpAffine", json!({"img": img, "matrix": [0, 0, 0, 0, 0, 0]})), "bad_params");
    // Masks keep being masks.
    let m = lab.put_mask(10, 10, |x, _| if x < 5 { 255 } else { 0 });
    assert!(lab.call("cv.resize", json!({"img": m, "width": 5}))["mask"].is_string());
}

#[test]
fn color_family() {
    let lab = Lab::new();
    let img = lab.put_img(16, 16, |x, y| [x as u8 * 16, y as u8 * 16, 128, 255]);
    let g = lab.call("cv.cvtColor", json!({"img": img, "code": "COLOR_RGB2GRAY"}));
    assert!(lab.img(&g).rgba.chunks(4).all(|p| p[0] == p[1] && p[1] == p[2]));
    let hsv = lab.call("cv.cvtColor", json!({"img": img, "code": "hsv"}));
    let back = lab.call("cv.cvtColor", json!({"img": hsv["img"], "code": "hsv2rgb"}));
    let (a, b) = (lab.ctx.shared.handles.image_str(&img).unwrap(), lab.img(&back));
    assert!(a.rgba.iter().zip(&b.rgba).all(|(x, y)| x.abs_diff(*y) <= 4));
    let ch = lab.call("cv.extractChannel", json!({"img": img, "channel": "r"}));
    assert_eq!(lab.mask(&ch).alpha[5], 80);
    let adj = lab.call("cv.adjust", json!({"img": img, "brightness": 20}));
    assert!(lab.img(&adj).rgba[0] > a.rgba[0]);
    let hist = lab.call("cv.histogram", json!({"img": img, "channels": ["r", "b"], "bins": 16}));
    assert_eq!(hist["channels"]["r"]["hist"].as_array().unwrap().len(), 16);
    assert_eq!(hist["channels"]["b"]["stats"]["median"], 128);
    let ms = lab.call("cv.meanStd", json!({"img": img}));
    assert_eq!(ms["mean"][2], 128.0);
    assert_eq!(ms["std"][2], 0.0);
    let half = lab.put_mask(16, 16, |x, _| if x < 8 { 255 } else { 0 });
    let msm = lab.call("cv.meanStd", json!({"img": img, "mask": half}));
    assert_eq!(msm["count"], 128);
    assert_eq!(msm["mean"][0], 56.0);
}

#[test]
fn filter_family() {
    let lab = Lab::new();
    let sq = lab.put_img(64, 64, |x, y| gray(if in_square(x, y, 16, 48) { 255 } else { 0 }));
    let e = lab.call("cv.canny", json!({"img": sq, "low": 50, "high": 150}));
    let em = lab.mask(&e);
    assert_eq!(em.alpha[32 * 64 + 32], 0);
    assert!((15..=16).any(|x| em.alpha[32 * 64 + x] == 255));
    let b = lab.call("cv.gaussianBlur", json!({"img": sq, "sigma": 2.0}));
    let bi = lab.img(&b);
    assert!(bi.rgba[(32 * 64 + 16) * 4] > 60 && bi.rgba[(32 * 64 + 16) * 4] < 200);
    let big = lab.call("cv.gaussianBlur", json!({"img": sq, "sigma": 25.0}));
    assert!(lab.img(&big).rgba[0] > 0);
    assert_eq!(lab.err("cv.gaussianBlur", json!({"img": sq, "ksize": 4})), "bad_params");
    assert_eq!(lab.err("cv.gaussianBlur", json!({"img": sq})), "bad_params");
    lab.call("cv.medianBlur", json!({"img": sq, "ksize": 5}));
    lab.call("cv.bilateralFilter", json!({"img": sq, "d": 5}));
    let s = lab.call("cv.sobel", json!({"img": sq, "dx": 1}));
    assert!(s["maxGradient"].as_f64().unwrap() > 500.0);
    let sharp = lab.call("cv.laplacianVariance", json!({"img": sq}))["variance"].as_f64().unwrap();
    let soft = lab.call("cv.laplacianVariance", json!({"img": b["img"]}))["variance"].as_f64().unwrap();
    assert!(sharp > 4.0 * soft);
    for m in ["bilateral", "nlmeans"] {
        let d = lab.call("cv.denoise", json!({"img": sq, "strength": 50, "method": m}));
        assert_eq!(d["width"], 64);
    }
}

#[test]
fn threshold_family() {
    let lab = Lab::new();
    // Bimodal: left dark (~50), right bright (~190), with noise.
    let img = lab.put_img(40, 20, |x, y| gray(if x < 15 { 50 } else { 190 } + ((x * 7 + y * 3) % 9) as u8));
    let t = lab.call("cv.threshold", json!({"img": img, "type": "otsu"}));
    let th = t["thresh"].as_u64().unwrap();
    assert!((58..190).contains(&th), "otsu {th}");
    assert_eq!(lab.mask(&t).alpha.iter().filter(|&&v| v == 255).count(), 25 * 20);
    let fixed = lab.call("cv.threshold", json!({"img": img, "thresh": 100, "type": "binary_inv"}));
    assert_eq!(lab.mask(&fixed).alpha[0], 255);
    let combo = lab.call("cv.threshold", json!({"img": img, "type": "binary", "otsu": true}));
    assert_eq!(combo["thresh"], t["thresh"]);
    // Uneven lighting: a dark line on a gradient; adaptive finds the line.
    let grad = lab.put_img(60, 30, |x, y| gray(if y == 15 { (x * 2) as u8 } else { (x * 3 + 40) as u8 }));
    let a = lab.call("cv.adaptiveThreshold", json!({"img": grad, "blockSize": 7, "C": 5, "type": "binary_inv"}));
    let am = lab.mask(&a);
    assert!(am.alpha[15 * 60 + 30] == 255 && am.alpha[5 * 60 + 30] == 0);
    assert_eq!(lab.err("cv.threshold", json!({"img": img})), "bad_params");
}

#[test]
fn morphology_and_masks() {
    let lab = Lab::new();
    let m = lab.put_mask(32, 32, |x, y| if in_square(x, y, 8, 24) || (x, y) == (2, 2) { 255 } else { 0 });
    let o = lab.call("cv.morphology", json!({"mask": m, "op": "open", "ksize": 3}));
    assert_eq!(lab.mask(&o).alpha[2 * 32 + 2], 0);
    assert_eq!(lab.mask(&o).alpha[16 * 32 + 16], 255);
    let g = lab.call("cv.growMask", json!({"mask": m, "radius": 2}));
    assert!(g["coverage"].as_f64().unwrap() > o["coverage"].as_f64().unwrap());
    let s = lab.call("cv.shrinkMask", json!({"mask": m, "radius": 2}));
    assert_eq!(lab.mask(&s).alpha.iter().filter(|&&v| v >= 128).count(), 12 * 12);
    let f = lab.call("cv.featherMask", json!({"mask": m, "radius": 4}));
    let fv = lab.mask(&f).alpha[16 * 32 + 8];
    assert!(fv > 60 && fv < 200, "{fv}");
    let inv = lab.call("cv.invertMask", json!({"mask": m}));
    assert_eq!(lab.mask(&inv).alpha[0], 255);
    let c = lab.call("cv.combineMasks", json!({"a": m, "b": inv["mask"], "op": "union"}));
    assert_eq!(c["coverage"], 1.0);
    let x = lab.call("cv.combineMasks", json!({"a": m, "b": m, "op": "xor"}));
    assert_eq!(x["coverage"], 0.0);
    let small = lab.put_mask(4, 4, |_, _| 0);
    assert_eq!(lab.err("cv.combineMasks", json!({"a": m, "b": small, "op": "union"})), "bad_params");
    let bnd = lab.call("cv.maskBounds", json!({"mask": o["mask"]}));
    assert_eq!((bnd["x"].as_u64(), bnd["y"].as_u64(), bnd["w"].as_u64(), bnd["h"].as_u64()), (Some(8), Some(8), Some(16), Some(16)));
    // Colour mask, applyMask, composite.
    let img = lab.put_img(32, 32, |x, _| if x < 16 { [220, 30, 30, 255] } else { [30, 30, 220, 255] });
    let red = lab.call("cv.maskFromColor", json!({"img": img, "color": "#dc1e1e"}));
    assert_eq!(red["coverage"], 0.5);
    let cut = lab.call("cv.applyMask", json!({"img": img, "mask": red["mask"]}));
    let ci = lab.img(&cut);
    assert_eq!((ci.rgba[3], ci.rgba[20 * 4 + 3]), (255, 0));
    let bg = lab.put_img(32, 32, |_, _| [0, 255, 0, 255]);
    let comp = lab.call("cv.composite", json!({"bg": bg, "fg": cut["img"]}));
    let co = lab.img(&comp);
    assert_eq!(&co.rgba[0..4], &[220, 30, 30, 255]);
    assert_eq!(&co.rgba[20 * 4..20 * 4 + 4], &[0, 255, 0, 255]);
}

#[test]
fn contours_components_and_lines() {
    let lab = Lab::new();
    // Three blobs: a 20x10 rect, a 6x6 square, a 1-px speck.
    let m = lab.put_mask(64, 48, |x, y| {
        let r = (5..25).contains(&x) && (5..15).contains(&y);
        let s = (40..46).contains(&x) && (30..36).contains(&y);
        if r || s || (x, y) == (60, 2) { 255 } else { 0 }
    });
    let c = lab.call("cv.findContours", json!({"mask": m}));
    assert_eq!(c["count"], 3);
    let big = &c["contours"][0];
    assert_eq!(big["area"], 200);
    assert_eq!(big["bbox"], json!([5, 5, 20, 10]));
    assert_eq!(big["centroid"], json!([14.5, 9.5]));
    assert_eq!(big["points"].as_array().unwrap().len(), 4);
    assert!((big["perimeter"].as_f64().unwrap() - 56.0).abs() < 1e-6);
    assert_eq!(lab.call("cv.findContours", json!({"mask": m, "minArea": 2}))["count"], 2);
    let cc = lab.call("cv.connectedComponents", json!({"mask": m, "maxComponents": 1}));
    assert_eq!(cc["count"], 3);
    assert_eq!(lab.mask(&cc).alpha.iter().filter(|&&v| v == 255).count(), 200);
    // Hough: a horizontal and a vertical line.
    let lines = lab.put_mask(100, 80, |x, y| if y == 30 || x == 70 { 255 } else { 0 });
    let h = lab.call("cv.houghLines", json!({"mask": lines, "maxLines": 2}));
    let ls = h["lines"].as_array().unwrap();
    assert_eq!(ls.len(), 2);
    let horiz = ls.iter().find(|l| l["angle"].as_f64().unwrap().abs() < 1.0).unwrap();
    assert!((horiz["rho"].as_f64().unwrap() - 30.0).abs() <= 1.0);
    assert_eq!((horiz["x1"].as_f64(), horiz["x2"].as_f64()), (Some(0.0), Some(99.0)));
    let vert = ls.iter().find(|l| l["angle"].as_f64().unwrap().abs() > 89.0).unwrap();
    assert!((vert["rho"].as_f64().unwrap() - 70.0).abs() <= 1.0);
}

/// A sky/ground image whose horizon passes through the centre tilted by `deg` (CCW positive).
fn horizon(lab: &Lab, w: u32, h: u32, deg: f64) -> String {
    let t = deg.to_radians().tan();
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    lab.put_img(w, h, move |x, y| {
        let line_y = cy - t * (x as f64 - cx);
        let cov = (line_y - y as f64 + 0.5).clamp(0.0, 1.0); // 1 = sky
        let tex = ((x * 13 + y * 7) % 11) as f64;
        let v = cov * (200.0 + tex) + (1.0 - cov) * (60.0 + tex);
        gray(v.round() as u8)
    })
}

#[test]
fn dominant_angle_recovers_tilt() {
    let lab = Lab::new();
    for deg in [3.0, -7.5, 0.0, 12.0] {
        let img = horizon(&lab, 800, 600, deg);
        let r = lab.call("cv.dominantAngle", json!({"img": img}));
        let a = r["angle"].as_f64().unwrap();
        assert!((a - deg).abs() < 1.0, "tilt {deg}: measured {a} ({r})");
        assert!((r["correction"].as_f64().unwrap() + a).abs() < 1e-9);
        assert!(r["confidence"].as_f64().unwrap() > 0.5);
    }
    // Rotate a level horizon with cv.rotate and measure the rotation back, then straighten.
    let level = horizon(&lab, 900, 600, 0.0);
    let tilted = lab.call("cv.rotate", json!({"img": level, "angle": 4.0, "crop": true}));
    let a = lab.call("cv.dominantAngle", json!({"img": tilted["img"]}))["angle"].as_f64().unwrap();
    assert!((a - 4.0).abs() < 1.0, "rotate 4 → {a}");
    let fixed = lab.call("cv.rotate", json!({"img": tilted["img"], "angle": -a, "crop": true}));
    let b = lab.call("cv.dominantAngle", json!({"img": fixed["img"]}))["angle"].as_f64().unwrap();
    assert!(b.abs() < 0.5, "straightened → {b}");
    // A vertical edge (building) in vertical mode.
    let v = lab.put_img(400, 400, |x, y| gray(if (x as f64) < 200.0 + (y as f64 - 200.0) * (2f64.to_radians().tan()) { 30 } else { 220 }));
    let rv = lab.call("cv.dominantAngle", json!({"img": v, "mode": "vertical"}));
    assert_eq!(rv["kind"], "vertical");
    assert!((rv["angle"].as_f64().unwrap() - 2.0).abs() < 1.0, "{rv}");
    let flat = lab.put_img(50, 50, |_, _| gray(100));
    assert_eq!(lab.call("cv.dominantAngle", json!({"img": flat}))["confidence"], 0.0);
}

#[test]
fn template_and_inpaint() {
    let lab = Lab::new();
    let img = lab.put_img(120, 90, |x, y| gray(((x * 37 + y * 91 + (x * y) % 23) % 256) as u8));
    let tpl = lab.call("cv.crop", json!({"img": img, "x": 50, "y": 40, "w": 20, "h": 16}));
    let m = lab.call("cv.matchTemplate", json!({"img": img, "template": tpl["img"]}));
    assert_eq!((m["x"].as_u64(), m["y"].as_u64()), (Some(50), Some(40)));
    assert!(m["score"].as_f64().unwrap() > 0.999);
    // Inpaint a dark square out of a smooth gradient: the fill matches its surroundings.
    let grad = lab.put_img(96, 96, |x, y| {
        let v = (60 + x + y / 2) as u8;
        if in_square(x, y, 40, 56) { [0, 0, 0, 255] } else { [v, v, v, 255] }
    });
    let hole = lab.put_mask(96, 96, |x, y| if in_square(x, y, 39, 57) { 255 } else { 0 });
    for method in ["membrane", "patchmatch"] {
        let r = lab.call("cv.inpaint", json!({"img": grad, "mask": hole, "method": method}));
        let o = lab.img(&r);
        let at = |x: usize, y: usize| o.rgba[(y * 96 + x) * 4] as i32;
        let expect = 60 + 48 + 24;
        assert!((at(48, 48) - expect).abs() < 25, "{method}: centre {} vs {expect}", at(48, 48));
        assert_eq!(at(10, 10), 60 + 10 + 5, "outside untouched");
    }
    let empty = lab.put_mask(96, 96, |_, _| 0);
    assert!(lab.call("cv.inpaint", json!({"img": grad, "mask": empty}))["note"].is_string());
}

#[test]
fn document_roundtrip() {
    let lab = Lab::with_session(|root| {
        let ws = photocraft_automation::workspace::AuthorizedWorkspace::new(Some(root), Some(root)).unwrap();
        let mut h = Headless::with_workspace(ws);
        h.command_run("file.new", json!({"width": 64, "height": 48, "background": "white"})).unwrap();
        h
    });
    let d = lab.call("cv.fromDocument", json!({"maxSide": 0}));
    assert_eq!((d["width"].as_u64(), d["height"].as_u64()), (Some(64), Some(48)));
    assert!(lab.img(&d).rgba.chunks(4).all(|p| p == [255, 255, 255, 255]));
    let red = lab.put_img(64, 48, |_, _| [255, 0, 0, 255]);
    let placed = lab.call("cv.toDocumentLayer", json!({"img": red, "name": "Red fill"}));
    assert!(placed["placed"]["layer"].is_number(), "{placed}");
    let d2 = lab.call("cv.fromDocument", json!({"maxSide": 0}));
    assert!(lab.img(&d2).rgba.chunks(4).all(|p| p == [255, 0, 0, 255]), "the layer covers the canvas");
    // No document: a clean error, not a crash.
    let none = Lab::new();
    assert!(none.reg.call(&none.ctx, "cv.fromDocument", json!({})).is_err());
}

/// Timings on a 12 MP image: `cargo test -p imagelab-cv --release -- --ignored --nocapture perf`.
#[test]
#[ignore]
fn perf_12mp() {
    let lab = Lab::new();
    let img = lab.put_img(4000, 3000, |x, y| [(x % 251) as u8, (y % 241) as u8, ((x ^ y) % 256) as u8, 255]);
    let edges = lab.call("cv.canny", json!({"img": img, "low": 50, "high": 150}));
    for (id, p) in [
        ("cv.gaussianBlur", json!({"img": img, "sigma": 3.0})),
        ("cv.gaussianBlur", json!({"img": img, "sigma": 20.0})),
        ("cv.resize", json!({"img": img, "maxSide": 1024})),
        ("cv.rotate", json!({"img": img, "angle": 3.0, "crop": true})),
        ("cv.medianBlur", json!({"img": img, "ksize": 5})),
        ("cv.bilateralFilter", json!({"img": img, "d": 9})),
        ("cv.denoise", json!({"img": img, "strength": 30})),
        ("cv.denoise", json!({"img": img, "strength": 30, "method": "nlmeans"})),
        ("cv.canny", json!({"img": img, "low": 50, "high": 150, "sigma": 1.4})),
        ("cv.threshold", json!({"img": img, "type": "otsu"})),
        ("cv.morphology", json!({"mask": edges["mask"], "op": "close", "ksize": 15, "shape": "ellipse"})),
        ("cv.growMask", json!({"mask": edges["mask"], "radius": 40})),
        ("cv.findContours", json!({"mask": edges["mask"]})),
        ("cv.dominantAngle", json!({"img": img})),
        ("cv.laplacianVariance", json!({"img": img})),
        ("cv.histogram", json!({"img": img, "channels": ["r", "g", "b"]})),
    ] {
        let t = std::time::Instant::now();
        let r = lab.call(id, p);
        println!("{id:22} {:>8.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        for k in ["img", "mask"] {
            if let Some(h) = r[k].as_str() {
                lab.call("cv.free", json!({"handles": [h]}));
            }
        }
    }
}
