//! One-slider looks: easy, photo-aware tools in the spirit of AI photo editors, built only from
//! PhotoCraft engine commands.
//!
//! A look is an `amount` (and sometimes a `style` or a second slider) that becomes a short plan:
//! one `filter.cameraRaw` pass for the whole photo, masked passes for regions PhotoCraft finds
//! on its own (sky, faces, eyes, skin, the subject's background), and, for sky replacement, a
//! gradient fill layer masked by the sky. The plan runs as a temporary action, so the whole
//! look is **one** history step ("Action: Enhance") that one undo removes; the action is
//! deleted afterwards.
//!
//! Photo-aware means:
//! * *Enhance* measures the photo (luminance percentiles, dark and bright share, saturation)
//!   and sets exposure, shadows, highlights, whites, blacks, contrast, vibrance and dehaze from
//!   what it finds, so the same amount brightens a dark photo and calms a bright one.
//! * Sky looks run only when PhotoCraft's Select Sky finds a sky; face looks only on faces
//!   that `image.detectFaces` finds, sized from each face.
//! * Templates combine looks; [`suggest`] picks templates for the scene (portrait, landscape).
//!
//! Measurements and the scene are cached per history state, so a slider that re-runs a
//! look (undo, run again) measures the photo once.

use std::sync::Mutex;

use imagelab_api::{ApiError, ApiResult, Headless};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::ops;

// ── catalog ──────────────────────────────────────────────────────────────────────────────────

/// One look as the UI and agents see it.
pub struct LookDef {
    pub id: &'static str,
    pub label: &'static str,
    pub group: &'static str,
    pub min: f64,
    pub max: f64,
    pub default: f64,
    /// Choices for `style`: (value, label); the first is the default.
    pub styles: &'static [(&'static str, &'static str)],
    pub about: &'static str,
}

const TONINGS: &[(&str, &str)] = &[
    ("tealOrange", "Teal & orange"),
    ("warmFade", "Warm fade"),
    ("coolBlue", "Cool blue"),
    ("cinematic", "Cinematic"),
    ("sepia", "Sepia"),
    ("rose", "Rose"),
];

const SKIES: &[(&str, &str)] = &[("blue", "Clear blue"), ("golden", "Golden"), ("sunset", "Sunset"), ("dusk", "Dusk"), ("storm", "Storm")];

/// Templates: (id, label, scene, recipe of (look, amount at full strength, style)).
type Recipe = &'static [(&'static str, f64, Option<&'static str>)];
pub const TEMPLATES: &[(&str, &str, &str, Recipe)] = &[
    ("auto", "Auto", "any", &[("enhance", 70.0, None), ("sky", 40.0, None)]),
    ("airy", "Bright & airy", "any", &[("enhance", 50.0, None), ("glow", 25.0, None), ("matte", 20.0, None), ("atmosphere", 10.0, None)]),
    ("crisp", "Crisp landscape", "landscape", &[("enhance", 60.0, None), ("sky", 60.0, None), ("structure", 40.0, None), ("details", 35.0, None), ("foliage", 30.0, None)]),
    ("golden", "Golden", "landscape", &[("enhance", 40.0, None), ("goldenHour", 70.0, None), ("glow", 20.0, None), ("vignette", 20.0, None)]),
    ("moody", "Moody", "any", &[("dramatic", 50.0, None), ("toning", 40.0, Some("cinematic")), ("vignette", 40.0, None), ("matte", 25.0, None)]),
    ("portrait", "Portrait glow", "portrait", &[("enhance", 40.0, None), ("faceLight", 45.0, None), ("skin", 40.0, None), ("eyes", 35.0, None), ("glow", 15.0, None), ("vignette", 15.0, None)]),
    ("film", "Film", "any", &[("matte", 45.0, None), ("grain", 40.0, None), ("toning", 40.0, Some("warmFade"))]),
    ("tealOrange", "Teal & orange", "any", &[("enhance", 40.0, None), ("toning", 70.0, Some("tealOrange")), ("structure", 15.0, None)]),
    ("noir", "Noir", "any", &[("bw", 100.0, None), ("dramatic", 40.0, None), ("grain", 25.0, None), ("vignette", 35.0, None)]),
];

fn template_styles() -> &'static [(&'static str, &'static str)] {
    const T: &[(&str, &str)] = &[
        ("auto", "Auto"),
        ("airy", "Bright & airy"),
        ("crisp", "Crisp landscape"),
        ("golden", "Golden"),
        ("moody", "Moody"),
        ("portrait", "Portrait glow"),
        ("film", "Film"),
        ("tealOrange", "Teal & orange"),
        ("noir", "Noir"),
    ];
    T
}

pub fn looks() -> Vec<LookDef> {
    vec![
        LookDef { id: "template", label: "Template", group: "Templates", min: 0.0, max: 100.0, default: 80.0, styles: template_styles(), about: "Several looks in one; amount is the strength" },
        LookDef { id: "enhance", label: "Enhance", group: "Essentials", min: 0.0, max: 100.0, default: 60.0, styles: &[], about: "Measures the photo and fixes exposure, shadows, highlights, contrast and colour; `sky` (0..100) also enhances a sky" },
        LookDef { id: "sky", label: "Sky enhancer", group: "Essentials", min: 0.0, max: 100.0, default: 60.0, styles: &[], about: "Deeper, clearer sky (PhotoCraft's Select Sky); nothing when there is no sky" },
        LookDef { id: "structure", label: "Structure", group: "Essentials", min: -100.0, max: 100.0, default: 40.0, styles: &[], about: "Detail and local contrast; negative softens" },
        LookDef { id: "details", label: "Details", group: "Essentials", min: 0.0, max: 100.0, default: 40.0, styles: &[], about: "Sharpening that skips flat areas" },
        LookDef { id: "denoise", label: "Denoise", group: "Essentials", min: 0.0, max: 100.0, default: 40.0, styles: &[], about: "Luminance and colour noise" },
        LookDef { id: "bw", label: "Black & white", group: "Essentials", min: 0.0, max: 100.0, default: 100.0, styles: &[], about: "Desaturates with a little punch" },
        LookDef { id: "vignette", label: "Vignette", group: "Essentials", min: -100.0, max: 100.0, default: 35.0, styles: &[], about: "Darker edges; negative lightens them" },
        LookDef { id: "goldenHour", label: "Golden hour", group: "Landscape", min: 0.0, max: 100.0, default: 50.0, styles: &[], about: "Warm, low-sun light" },
        LookDef { id: "foliage", label: "Foliage", group: "Landscape", min: 0.0, max: 100.0, default: 50.0, styles: &[], about: "Richer greens and yellows" },
        LookDef { id: "atmosphere", label: "Atmosphere", group: "Landscape", min: -100.0, max: 100.0, default: 35.0, styles: &[], about: "Adds haze; negative clears it" },
        LookDef { id: "skyReplace", label: "Sky replacement", group: "Landscape", min: 0.0, max: 100.0, default: 85.0, styles: SKIES, about: "A new sky as a gradient layer masked by the sky, with the land relit to match" },
        LookDef { id: "dramatic", label: "Dramatic", group: "Creative", min: 0.0, max: 100.0, default: 50.0, styles: &[], about: "Gritty local contrast, muted colour, dark edges" },
        LookDef { id: "glow", label: "Glow", group: "Creative", min: 0.0, max: 100.0, default: 40.0, styles: &[], about: "Soft, dreamy glow" },
        LookDef { id: "matte", label: "Matte", group: "Creative", min: 0.0, max: 100.0, default: 40.0, styles: &[], about: "Lifted blacks, softened whites" },
        LookDef { id: "toning", label: "Toning", group: "Creative", min: 0.0, max: 100.0, default: 50.0, styles: TONINGS, about: "Colour grading of shadows and highlights" },
        LookDef { id: "grain", label: "Film grain", group: "Creative", min: 0.0, max: 100.0, default: 35.0, styles: &[], about: "Film-like grain" },
        LookDef { id: "faceLight", label: "Face light", group: "Portrait", min: 0.0, max: 100.0, default: 50.0, styles: &[], about: "Brightens faces (image.detectFaces)" },
        LookDef { id: "skin", label: "Skin", group: "Portrait", min: 0.0, max: 100.0, default: 40.0, styles: &[], about: "Smooths skin on faces, keeping eyes, brows and lips" },
        LookDef { id: "eyes", label: "Eyes", group: "Portrait", min: 0.0, max: 100.0, default: 50.0, styles: &[], about: "Brighter, crisper eyes" },
        LookDef { id: "redEye", label: "Red eye", group: "Portrait", min: 0.0, max: 100.0, default: 60.0, styles: &[], about: "Removes red eye on detected faces" },
        LookDef { id: "bokeh", label: "Background blur", group: "Portrait", min: 0.0, max: 100.0, default: 50.0, styles: &[], about: "Blurs behind the subject (Select Subject)" },
    ]
}

fn find(id: &str) -> ApiResult<LookDef> {
    looks().into_iter().find(|l| l.id == id).ok_or_else(|| {
        let ids: Vec<&str> = looks().iter().map(|l| l.id).collect();
        ApiError::bad_params(format!("unknown look `{id}`; one of {}", ids.join(", ")))
    })
}

pub fn catalog() -> Value {
    let looks: Vec<Value> = looks()
        .iter()
        .map(|l| {
            json!({"id": l.id, "label": l.label, "group": l.group, "min": l.min, "max": l.max, "default": l.default,
                   "styles": l.styles.iter().map(|(v, n)| json!({"value": v, "label": n})).collect::<Vec<_>>(), "about": l.about})
        })
        .collect();
    let templates: Vec<Value> = TEMPLATES
        .iter()
        .map(|(id, label, scene, recipe)| json!({"id": id, "label": label, "scene": scene, "looks": recipe.iter().map(|(l, a, s)| json!({"look": l, "amount": a, "style": s})).collect::<Vec<_>>()}))
        .collect();
    json!({"looks": looks, "templates": templates})
}

// ── params ───────────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookParams {
    pub look: String,
    #[serde(default)]
    pub amount: Option<f64>,
    /// Enhance: also enhance the sky (0..100).
    #[serde(default)]
    pub sky: Option<f64>,
    /// Toning colours, sky replacement sky, or the template.
    #[serde(default)]
    pub style: Option<String>,
}

// ── measuring the photo ──────────────────────────────────────────────────────────────────────

/// What Enhance needs to know about the photo (all 0..1).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub p01: f64,
    pub p05: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    /// Share of pixels darker than 15% / brighter than 90%.
    pub dark: f64,
    pub bright: f64,
    /// Mean HSV saturation.
    pub sat: f64,
}

impl Stats {
    /// From RGBA8 pixels.
    pub fn of_rgba(px: &[u8]) -> Stats {
        let mut hist = [0u64; 256];
        let mut sat = 0.0;
        let mut n = 0u64;
        for p in px.as_chunks::<4>().0 {
            let (r, g, b) = (f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
            let y = (0.2126 * r + 0.7152 * g + 0.0722 * b).round().clamp(0.0, 255.0) as usize;
            hist[y] += 1;
            let mx = r.max(g).max(b);
            let mn = r.min(g).min(b);
            if mx > 0.0 {
                sat += (mx - mn) / mx;
            }
            n += 1;
        }
        if n == 0 {
            return Stats::default();
        }
        let pct = |q: f64| {
            let target = (q * n as f64).ceil().max(1.0) as u64;
            let mut acc = 0;
            for (i, c) in hist.iter().enumerate() {
                acc += c;
                if acc >= target {
                    return i as f64 / 255.0;
                }
            }
            1.0
        };
        let share = |r: std::ops::Range<usize>| hist[r].iter().sum::<u64>() as f64 / n as f64;
        Stats { p01: pct(0.01), p05: pct(0.05), p50: pct(0.5), p95: pct(0.95), p99: pct(0.99), dark: share(0..39), bright: share(230..256), sat: sat / n as f64 }
    }
}

/// A face from `image.detectFaces`, in document pixels.
#[derive(Clone, Debug, Default)]
pub struct Face {
    pub rect: [f64; 4],
    /// Right eye (on the image's left), then left eye.
    pub eyes: Vec<[f64; 2]>,
    pub nose: Option<[f64; 2]>,
    /// Mouth corners: right (image left), then left.
    pub mouth: Vec<[f64; 2]>,
}

#[derive(Clone, Debug, Default)]
struct Scene {
    faces: Vec<Face>,
    /// Share of the canvas Select Sky finds (0 when none).
    sky: f64,
}

type Key = (String, u64);
static STATS: Mutex<Option<(Key, Stats)>> = Mutex::new(None);
static SCENE: Mutex<Option<(Key, Scene)>> = Mutex::new(None);

/// Forget the measured stats and scene (a document was opened, created or closed: history row
/// ids restart per open document, so a reopened sidecar could otherwise match a stale entry).
pub fn forget_caches() {
    *STATS.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    *SCENE.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

fn key(h: &Headless) -> ApiResult<Key> {
    let d = h.session.active().ok_or_else(|| ApiError::not_found("no open document; doc.open a photo first"))?;
    // The current history state's id: unlike the revision, it is the same again after a
    // preview is undone, so a slider that re-runs a look measures the photo once.
    let state = d.history.row_id(d.history.past_len()).unwrap_or(d.revision);
    Ok((format!("{:?}:{}x{}", d.doc.id, d.doc.size.width, d.doc.size.height), state))
}

fn cached<T: Clone>(cell: &Mutex<Option<(Key, T)>>, k: &Key) -> Option<T> {
    let g = cell.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    g.as_ref().filter(|(ck, _)| ck == k).map(|(_, v)| v.clone())
}

fn store<T>(cell: &Mutex<Option<(Key, T)>>, k: Key, v: T) {
    *cell.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some((k, v));
}

/// The photo's tones, from a ≤ 256 px render of the whole document.
fn stats(h: &Headless) -> ApiResult<Stats> {
    let k = key(h)?;
    if let Some(s) = cached(&STATS, &k) {
        return Ok(s);
    }
    let d = h.session.active().ok_or_else(|| ApiError::not_found("no open document"))?;
    let (w, hh) = (d.doc.size.width, d.doc.size.height);
    let long = w.max(hh).max(1);
    let scale = (256.0 / f64::from(long)).min(1.0);
    let (pw, ph) = (((f64::from(w) * scale).round() as u32).max(1), ((f64::from(hh) * scale).round() as u32).max(1));
    let px = photocraft_compose::render_reduced_rect(&d.doc, photocraft_geom::Rect::from_xywh(0, 0, w, hh), pw, ph).to_rgba8().pixels;
    let s = Stats::of_rgba(&px);
    store(&STATS, k, s);
    Ok(s)
}

/// Faces and the sky's share, measured without touching the document: the sky with PhotoCraft's
/// Select Sky model run on a small render (running `select.sky` and undoing it would clear the
/// redo states and could evict the oldest undo state).
fn scene(h: &mut Headless) -> ApiResult<Scene> {
    let k = key(h)?;
    if let Some(s) = cached(&SCENE, &k) {
        return Ok(s);
    }
    let mut out = Scene::default();
    if let Ok(v) = h.command_run("image.detectFaces", json!({"minScore": 0.6})) {
        for f in v.get("faces").and_then(Value::as_array).into_iter().flatten() {
            let Some(r) = f.get("rect").and_then(Value::as_array) else { continue };
            let r: Vec<f64> = r.iter().filter_map(Value::as_f64).collect();
            if r.len() != 4 || r[2] < 8.0 || r[3] < 8.0 {
                continue;
            }
            let lm = f.get("landmarks");
            let point = |name: &str| -> Option<[f64; 2]> {
                let a = lm?.get(name)?.as_array()?;
                Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?])
            };
            let eyes = ["rightEye", "leftEye"].iter().filter_map(|e| point(e)).collect();
            let mouth = ["mouthRight", "mouthLeft"].iter().filter_map(|e| point(e)).collect();
            out.faces.push(Face { rect: [r[0], r[1], r[2], r[3]], eyes, nose: point("nose"), mouth });
        }
    }
    out.sky = sky_coverage(h).unwrap_or(0.0);
    store(&SCENE, k, out.clone());
    Ok(out)
}

/// The sky's share of the canvas (0..1), from PhotoCraft's sky model on a ≤ 160 kpx render at
/// Select Sky's default threshold; None without a plausible sky.
fn sky_coverage(h: &Headless) -> Option<f64> {
    let d = h.session.active()?;
    let (w, hh) = (d.doc.size.width, d.doc.size.height);
    if w < 4 || hh < 4 {
        return None;
    }
    let scale = (photocraft_algo::segment::sky::WORK_PX as f64 / (f64::from(w) * f64::from(hh))).sqrt().min(1.0);
    let (pw, ph) = (((f64::from(w) * scale).round() as u32).max(2), ((f64::from(hh) * scale).round() as u32).max(2));
    let px = photocraft_compose::render_reduced_rect(&d.doc, photocraft_geom::Rect::from_xywh(0, 0, w, hh), pw, ph).to_rgba8().pixels;
    let rgba: Vec<[f32; 4]> = px.as_chunks::<4>().0.iter().map(|p| p.map(|c| f32::from(c) / 255.0)).collect();
    let img = photocraft_algo::segment::RgbImage::from_rgba(&rgba, pw as usize, ph as usize);
    let mask = photocraft_algo::segment::sky::sky_mask_low(&img, 50.0)?;
    Some(mask.iter().filter(|m| **m).count() as f64 / mask.len().max(1) as f64)
}

/// Faces in the active document (cached per revision).
pub fn faces(h: &mut Headless) -> ApiResult<Vec<Face>> {
    Ok(scene(h)?.faces)
}

// ── plans ────────────────────────────────────────────────────────────────────────────────────

/// Ranges of the `filter.cameraRaw` sliders a plan sets (the rest are ±100).
fn raw_range(k: &str) -> (f64, f64) {
    match k {
        "exposure" => (-5.0, 5.0),
        "sharpenAmount" => (0.0, 150.0),
        "noiseLuminance" | "noiseColor" | "grainAmount" | "grainSize" | "grainRoughness" | "sharpenMasking" | "vignetteFeather" | "vignetteMidpoint" => (0.0, 100.0),
        _ => (-100.0, 100.0),
    }
}

/// HSL bands of `hslHue` / `hslSat` / `hslLum`.
#[derive(Clone, Copy)]
enum Band {
    Orange = 1,
    Yellow = 2,
    Green = 3,
}

/// One cameraRaw parameter set, accumulated from several looks.
#[derive(Default, Clone)]
struct Raw(Map<String, Value>);

impl Raw {
    fn add(&mut self, k: &str, v: f64) -> &mut Self {
        if v.abs() < 1e-6 {
            return self;
        }
        let (lo, hi) = raw_range(k);
        let cur = self.0.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        let digits = if k == "exposure" { 100.0 } else { 10.0 };
        self.0.insert(k.into(), json!(((cur + v).clamp(lo, hi) * digits).round() / digits));
        self
    }
    fn set(&mut self, k: &str, v: Value) -> &mut Self {
        self.0.insert(k.into(), v);
        self
    }
    fn hsl(&mut self, k: &str, band: Band, v: f64) -> &mut Self {
        let mut arr: Vec<f64> = self.0.get(k).and_then(Value::as_array).map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_else(|| vec![0.0; 8]);
        arr.resize(8, 0.0);
        arr[band as usize] = (arr[band as usize] + v).clamp(-100.0, 100.0).round();
        self.0.insert(k.into(), json!(arr));
        self
    }
    /// Colour grading: the stronger of two grades of the same range wins its hue; sats add.
    fn grade(&mut self, range: &str, hue: f64, sat: f64) -> &mut Self {
        if sat <= 0.0 {
            return self;
        }
        let prev = self.0.get(range).cloned().unwrap_or(Value::Null);
        let prev_sat = prev.get("sat").and_then(Value::as_f64).unwrap_or(0.0);
        let hue = if prev_sat > sat { prev.get("hue").and_then(Value::as_f64).unwrap_or(hue) } else { hue };
        self.0.insert(range.into(), json!({"hue": hue.round(), "sat": (prev_sat + sat).min(100.0).round()}));
        self
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

type Step = (String, Value);

#[derive(Default)]
struct Plan {
    /// The whole photo (inside the user's selection, if there is one).
    global: Raw,
    /// Masked pixel work: each entry selects a region, then runs commands inside it.
    masked: Vec<Vec<Step>>,
    /// Layers added on top (they change the active layer, so they come last).
    layers: Vec<Vec<Step>>,
    /// Looks that found nothing to work on ("no faces", "no sky").
    skipped: Vec<String>,
}

impl Plan {
    fn steps(&self) -> Vec<Step> {
        let mut out = Vec::new();
        if !self.global.is_empty() {
            out.push(("filter.cameraRaw".to_string(), Value::Object(self.global.0.clone())));
        }
        for m in &self.masked {
            out.extend(m.iter().cloned());
        }
        if !self.masked.is_empty() {
            out.push(("select.deselect".into(), json!({})));
        }
        for l in &self.layers {
            out.extend(l.iter().cloned());
        }
        out
    }
}

struct Cx<'a> {
    h: &'a mut Headless,
    stats: Option<Stats>,
    scene: Option<Scene>,
}

impl Cx<'_> {
    fn stats(&mut self) -> ApiResult<Stats> {
        if self.stats.is_none() {
            self.stats = Some(stats(self.h)?);
        }
        Ok(self.stats.unwrap_or_default())
    }
    fn scene(&mut self) -> ApiResult<Scene> {
        if self.scene.is_none() {
            self.scene = Some(scene(self.h)?);
        }
        Ok(self.scene.clone().unwrap_or_default())
    }
    fn long_side(&self) -> f64 {
        self.h.session.active().map_or(1000.0, |d| f64::from(d.doc.size.width.max(d.doc.size.height)))
    }
}

/// Enhance's corrections at full amount, from the photo's tones. Each one is zero when the
/// photo does not need it, so a well-exposed photo gets only a little colour and clarity.
pub fn enhance_raw(s: &Stats) -> Vec<(&'static str, f64)> {
    let exposure = ((0.46 - s.p50) * 2.0).clamp(-0.8, 1.2);
    let shadows = 8.0 + (s.dark * 220.0).min(55.0);
    let highlights = -(6.0 + (s.bright * 260.0).min(55.0));
    let whites = if s.p99 < 0.94 { ((0.94 - s.p99) * 160.0).min(40.0) } else { 0.0 };
    let blacks = if s.p01 > 0.04 { -((s.p01 - 0.04) * 220.0).min(40.0) } else { 0.0 };
    let spread = s.p95 - s.p05;
    let contrast = if spread < 0.62 { ((0.62 - spread) * 70.0).min(25.0) } else { 0.0 };
    let vibrance = ((0.38 - s.sat) * 110.0).clamp(6.0, 35.0);
    let dehaze = if s.p01 > 0.10 { ((s.p01 - 0.10) * 90.0).min(20.0) } else { 0.0 };
    vec![
        ("exposure", exposure),
        ("shadows", shadows),
        ("highlights", highlights),
        ("whites", whites),
        ("blacks", blacks),
        ("contrast", contrast),
        ("vibrance", vibrance),
        ("dehaze", dehaze),
        ("clarity", 10.0),
        ("texture", 6.0),
    ]
}

fn sky_select() -> Step {
    ("select.sky".into(), json!({"softness": 55}))
}

fn face_ellipse(f: &Face, add: bool) -> Step {
    let [x, y, w, h] = f.rect;
    let (ew, eh) = (w * 1.15, h * 1.4);
    let (cx, cy) = (x + w / 2.0, y + h / 2.0 - h * 0.08);
    (
        "select.rect".into(),
        json!({"x": (cx - ew / 2.0).round() as i32, "y": (cy - eh / 2.0).round() as i32, "width": ew.round() as u32, "height": eh.round() as u32,
               "ellipse": true, "feather": (w * 0.12).round().max(1.0), "mode": if add { "add" } else { "replace" }}),
    )
}

fn faces_steps(faces: &[Face]) -> Vec<Step> {
    faces.iter().enumerate().map(|(i, f)| face_ellipse(f, i > 0)).collect()
}

/// Adds `look` at `amount` to the plan.
fn build(plan: &mut Plan, cx: &mut Cx, look: &str, amount: f64, sky: Option<f64>, style: Option<&str>) -> ApiResult<()> {
    let t = amount / 100.0;
    let g = &mut plan.global;
    match look {
        "enhance" => {
            let s = cx.stats()?;
            for (k, v) in enhance_raw(&s) {
                plan.global.add(k, v * t);
            }
            if let Some(sky) = sky.filter(|v| *v > 0.0) {
                build(plan, cx, "sky", sky, None, None)?;
            }
        }
        "sky" => {
            if cx.scene()?.sky < 0.02 {
                plan.skipped.push("no sky found".into());
                return Ok(());
            }
            let mut r = Raw::default();
            r.add("dehaze", 35.0 * t).add("highlights", -30.0 * t).add("vibrance", 35.0 * t).add("saturation", 8.0 * t).add("clarity", 10.0 * t).add("temperature", -6.0 * t).add("exposure", -0.15 * t);
            plan.masked.push(vec![sky_select(), ("filter.cameraRaw".into(), Value::Object(r.0))]);
        }
        "structure" => {
            g.add("texture", 45.0 * t).add("clarity", 40.0 * t);
            if t > 0.0 {
                g.add("vibrance", 5.0 * t);
            }
        }
        "details" => {
            g.add("sharpenAmount", 110.0 * t).add("sharpenMasking", 45.0 * t.signum()).add("texture", 15.0 * t);
        }
        "denoise" => {
            g.add("noiseLuminance", 75.0 * t).add("noiseColor", 90.0 * t);
        }
        "bw" => {
            g.add("saturation", -100.0 * t).add("contrast", 15.0 * t).add("clarity", 8.0 * t);
        }
        "vignette" => {
            g.add("vignetteAmount", -55.0 * t);
            if t != 0.0 {
                g.set("vignetteFeather", json!(65)).set("vignetteMidpoint", json!(40));
            }
        }
        "goldenHour" => {
            g.add("temperature", 28.0 * t).add("tint", 6.0 * t).add("shadows", 12.0 * t).add("highlights", -10.0 * t).add("vibrance", 12.0 * t);
            g.hsl("hslSat", Band::Orange, 18.0 * t).hsl("hslLum", Band::Orange, 8.0 * t).hsl("hslHue", Band::Yellow, -6.0 * t);
            g.grade("gradeHighlights", 42.0, 30.0 * t).grade("gradeShadows", 28.0, 10.0 * t);
        }
        "foliage" => {
            g.hsl("hslSat", Band::Green, 35.0 * t).hsl("hslSat", Band::Yellow, 25.0 * t).hsl("hslLum", Band::Green, 8.0 * t).hsl("hslHue", Band::Yellow, 10.0 * t).hsl("hslHue", Band::Green, -4.0 * t);
        }
        "atmosphere" => {
            g.add("dehaze", -45.0 * t);
            if t > 0.0 {
                g.add("contrast", -12.0 * t).add("exposure", 0.12 * t).add("blacks", 10.0 * t);
            } else {
                g.add("blacks", 6.0 * t);
            }
        }
        "skyReplace" => {
            if cx.scene()?.sky < 0.02 {
                plan.skipped.push("no sky found".into());
                return Ok(());
            }
            let (horizon, zenith, fg): (&str, &str, &[(&str, f64)]) = match style.unwrap_or("blue") {
                "golden" => ("#ffd38c", "#5f9ee0", &[("temperature", 16.0), ("vibrance", 8.0)]),
                "sunset" => ("#ffad66", "#55377f", &[("temperature", 28.0), ("tint", 8.0), ("exposure", -0.12)]),
                "dusk" => ("#f08ba3", "#1c2958", &[("temperature", -10.0), ("tint", 6.0), ("exposure", -0.3)]),
                "storm" => ("#a1a9b3", "#373f4a", &[("saturation", -15.0), ("exposure", -0.2), ("clarity", 12.0)]),
                _ => ("#c4e2ff", "#2f6ed4", &[("vibrance", 6.0)]),
            };
            let mut r = Raw::default();
            for (k, v) in fg {
                r.add(k, v * t);
            }
            if !r.is_empty() {
                plan.masked.push(vec![sky_select(), ("select.inverse".into(), json!({})), ("filter.cameraRaw".into(), Value::Object(r.0))]);
            }
            plan.layers.push(vec![
                ("select.sky".into(), json!({"softness": 35, "threshold": 55})),
                ("layer.newFillLayer.gradient".into(), json!({"from": horizon, "to": zenith, "angle": 90})),
                ("layer.setProps".into(), json!({"opacity": (t.clamp(0.0, 1.0) * 100.0).round() / 100.0, "name": "Sky", "blend": "Color"})),
                ("select.deselect".into(), json!({})),
            ]);
        }
        "dramatic" => {
            g.add("clarity", 55.0 * t).add("texture", 25.0 * t).add("contrast", 20.0 * t).add("saturation", -30.0 * t).add("shadows", 25.0 * t).add("highlights", -35.0 * t).add("vignetteAmount", -25.0 * t);
        }
        "glow" => {
            g.add("clarity", -45.0 * t).add("exposure", 0.15 * t).add("highlights", -15.0 * t).add("contrast", -8.0 * t).add("saturation", 8.0 * t);
        }
        "matte" => {
            if t > 0.0 {
                let lift = (45.0 * t).round();
                let cap = (255.0 - 22.0 * t).round();
                g.set("pointCurve", json!([[0, lift], [64, (64.0 + 0.5 * lift).round()], [192, (192.0 - 6.0 * t).round()], [255, cap]]));
                g.add("saturation", -12.0 * t);
            }
        }
        "toning" => {
            let (sh, hi) = match style.unwrap_or("tealOrange") {
                "warmFade" => ((40.0, 25.0), (45.0, 25.0)),
                "coolBlue" => ((215.0, 30.0), (200.0, 15.0)),
                "cinematic" => ((185.0, 35.0), (50.0, 25.0)),
                "sepia" => ((35.0, 35.0), (40.0, 30.0)),
                "rose" => ((330.0, 20.0), (15.0, 25.0)),
                _ => ((195.0, 40.0), (35.0, 35.0)),
            };
            g.grade("gradeShadows", sh.0, sh.1 * t).grade("gradeHighlights", hi.0, hi.1 * t);
            if style == Some("sepia") {
                g.add("saturation", -70.0 * t);
            }
        }
        "grain" => {
            g.add("grainAmount", 60.0 * t);
            if t > 0.0 {
                g.set("grainSize", json!(30)).set("grainRoughness", json!(55));
            }
        }
        "faceLight" | "skin" | "eyes" | "redEye" => {
            let faces = cx.scene()?.faces;
            if faces.is_empty() {
                plan.skipped.push("no faces found".into());
                return Ok(());
            }
            if t <= 0.0 {
                return Ok(());
            }
            match look {
                "faceLight" => {
                    let mut r = Raw::default();
                    r.add("exposure", 0.45 * t).add("shadows", 25.0 * t).add("highlights", -10.0 * t).add("clarity", -6.0 * t).add("temperature", 4.0 * t);
                    let mut steps = faces_steps(&faces);
                    steps.push(("filter.cameraRaw".into(), Value::Object(r.0)));
                    plan.masked.push(steps);
                }
                "skin" => {
                    let mut steps = faces_steps(&faces);
                    steps.push(("select.colorRange".into(), json!({"select": "skinTones", "mode": "intersect"})));
                    let face_w = faces.iter().map(|f| f.rect[2]).fold(0.0, f64::max);
                    let radius = (face_w * 0.014 * t).round().clamp(1.0, 100.0);
                    let threshold = (6.0 + 22.0 * t).round();
                    steps.push(("filter.blur.surfaceBlur".into(), json!({"radius": radius, "threshold": threshold})));
                    plan.masked.push(steps);
                }
                "eyes" => {
                    let mut steps = Vec::new();
                    for f in &faces {
                        let r = (f.rect[2] * 0.11).max(3.0);
                        for e in &f.eyes {
                            steps.push((
                                "select.rect".into(),
                                json!({"x": (e[0] - r).round() as i32, "y": (e[1] - r * 0.7).round() as i32, "width": (2.0 * r).round() as u32, "height": (1.4 * r).round() as u32,
                                       "ellipse": true, "feather": (r * 0.4).round().max(1.0), "mode": if steps.is_empty() { "replace" } else { "add" }}),
                            ));
                        }
                    }
                    if steps.is_empty() {
                        plan.skipped.push("no eyes found".into());
                        return Ok(());
                    }
                    let mut r = Raw::default();
                    r.add("clarity", 35.0 * t).add("exposure", 0.3 * t).add("vibrance", 18.0 * t).add("whites", 12.0 * t).add("texture", 15.0 * t);
                    steps.push(("filter.cameraRaw".into(), Value::Object(r.0)));
                    plan.masked.push(steps);
                }
                _ => {
                    let points: Vec<Value> = faces.iter().flat_map(|f| f.eyes.iter().map(|e| json!([e[0].round(), e[1].round()]))).take(64).collect();
                    if points.is_empty() {
                        plan.skipped.push("no eyes found".into());
                        return Ok(());
                    }
                    let pupil = (faces.iter().map(|f| f.rect[2]).fold(0.0, f64::max) * 0.04).clamp(1.0, 100.0).round();
                    plan.masked.push(vec![("paint.redEye".into(), json!({"points": points, "pupilSize": pupil, "darken": (100.0 * t).round().clamp(0.0, 100.0)}))]);
                }
            }
        }
        "bokeh" => {
            if t <= 0.0 {
                return Ok(());
            }
            let long = cx.long_side();
            let radius = (long * 0.012 * t).clamp(0.5, 1000.0);
            plan.masked.push(vec![
                ("select.subject".into(), json!({})),
                ("select.inverse".into(), json!({})),
                ("select.modify.feather".into(), json!({"radius": (long / 500.0).max(1.0).round()})),
                ("filter.blur.gaussianBlur".into(), json!({"radius": (radius * 10.0).round() / 10.0})),
            ]);
        }
        "template" => {
            let id = style.unwrap_or("auto");
            let (_, _, _, recipe) = TEMPLATES.iter().find(|(tid, ..)| *tid == id).ok_or_else(|| ApiError::bad_params(format!("unknown template `{id}`")))?;
            for (l, a, s) in recipe.iter() {
                build(plan, cx, l, a * t, None, *s)?;
            }
        }
        other => return Err(ApiError::bad_params(format!("unknown look `{other}`"))),
    }
    Ok(())
}

// ── running ──────────────────────────────────────────────────────────────────────────────────

/// Run `steps` as one history step named after `name` (a temporary action, deleted after).
fn run_grouped(h: &mut Headless, name: &str, steps: &[Step]) -> ApiResult<Value> {
    let existing: Vec<String> = h
        .command_run("actions.list", json!({}))
        .ok()
        .and_then(|v| v.get("actions").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.get("name").and_then(Value::as_str).map(String::from)).collect()))
        .unwrap_or_default();
    let mut action = name.to_string();
    let mut n = 2;
    while existing.contains(&action) {
        action = format!("{name} {n}");
        n += 1;
    }
    let before = ops::history_mark(h);
    h.command_run("actions.record", json!({"name": action}))?;
    let played = (|| -> ApiResult<Value> {
        h.command_run("actions.stop", json!({}))?;
        for (id, params) in steps {
            h.command_run("actions.insert", json!({"action": action, "id": id, "params": params}))?;
        }
        Ok(h.command_run("actions.play", json!({"action": action}))?)
    })();
    let _ = h.command_run("actions.delete", json!({"action": action}));
    let v = played?;
    if let Some(f) = v.get("failed") {
        // Playback keeps the steps before the failure (one grouped step, or several when the
        // engine does not group actions): take them all back.
        let n = ops::steps_since(h, before);
        if n > 0 {
            ops::undo_n(h, n)?;
        }
        let msg = f.get("error").and_then(Value::as_str).unwrap_or("a step failed");
        let id = f.get("id").and_then(Value::as_str).unwrap_or("?");
        return Err(ApiError::new("failed", format!("{name}: {id} failed ({msg}); the photo is as before")));
    }
    Ok(v)
}

fn needs_pixels(e: &ApiError) -> bool {
    let m = e.message.to_ascii_lowercase();
    m.contains("has no pixels") || m.contains("pixel layer")
}

/// `edit.look`: apply one look (or template) as one history step.
pub fn apply(h: &mut Headless, p: &LookParams) -> ApiResult<Value> {
    let def = find(&p.look)?;
    let amount = p.amount.unwrap_or(def.default).clamp(def.min, def.max);
    if let Some(s) = &p.style
        && !def.styles.is_empty()
        && !def.styles.iter().any(|(v, _)| v == s)
    {
        let names: Vec<&str> = def.styles.iter().map(|(v, _)| *v).collect();
        return Err(ApiError::bad_params(format!("style `{s}` is not one of {}", names.join(", "))));
    }
    let style = p.style.as_deref().or_else(|| def.styles.first().map(|(v, _)| *v));
    let mut plan = Plan::default();
    {
        let mut cx = Cx { h, stats: None, scene: None };
        build(&mut plan, &mut cx, def.id, amount, p.sky, style)?;
    }
    let steps = plan.steps();
    let label = if def.id == "template" { def.styles.iter().find(|(v, _)| Some(*v) == style).map_or(def.label, |(_, l)| *l) } else { def.label };
    if steps.is_empty() {
        return Ok(json!({"look": def.id, "amount": amount, "applied": false, "skipped": plan.skipped}));
    }
    let run = match run_grouped(h, label, &steps) {
        Err(e) if needs_pixels(&e) => {
            // An adjustment, fill or text layer is active: work on the topmost pixel layer.
            let layer = ops::top_pixel_layer(h).ok_or(e)?;
            h.command_run("layer.select", json!({"layer": layer}))?;
            run_grouped(h, label, &steps)?
        }
        r => r?,
    };
    let _ = run;
    Ok(json!({"look": def.id, "label": label, "amount": amount, "applied": true, "steps": steps.len(), "skipped": plan.skipped,
              "commands": steps.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>()}))
}

/// `edit.lookSuggest`: the scene and the templates that suit it.
pub fn suggest(h: &mut Headless) -> ApiResult<Value> {
    let sc = scene(h)?;
    let st = stats(h)?;
    let scene = if !sc.faces.is_empty() {
        "portrait"
    } else if sc.sky >= 0.05 {
        "landscape"
    } else {
        "general"
    };
    let mut ids: Vec<&str> = vec!["auto"];
    match scene {
        "portrait" => ids.extend(["portrait", "airy", "film", "noir"]),
        "landscape" => ids.extend(["crisp", "golden", "moody", "tealOrange"]),
        _ => ids.extend(["airy", "moody", "film", "noir"]),
    }
    if st.p50 < 0.3 && !ids.contains(&"airy") {
        ids.insert(1, "airy");
    }
    let templates: Vec<Value> = ids.iter().filter_map(|id| TEMPLATES.iter().find(|t| t.0 == *id)).map(|(id, label, ..)| json!({"id": id, "label": label})).collect();
    Ok(json!({"scene": scene, "faces": sc.faces.len(), "sky": (sc.sky * 1000.0).round() / 1000.0, "brightness": (st.p50 * 1000.0).round() / 1000.0, "templates": templates}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enhance_brightens_dark_and_calms_bright() {
        let dark = Stats { p01: 0.0, p05: 0.02, p50: 0.18, p95: 0.5, p99: 0.6, dark: 0.45, bright: 0.0, sat: 0.3 };
        let bright = Stats { p01: 0.2, p05: 0.3, p50: 0.75, p95: 0.98, p99: 1.0, dark: 0.0, bright: 0.3, sat: 0.2 };
        let get = |s: &Stats, k: &str| enhance_raw(s).into_iter().find(|(n, _)| *n == k).map_or(0.0, |(_, v)| v);
        assert!(get(&dark, "exposure") > 0.4);
        assert!(get(&dark, "shadows") > 40.0);
        assert!(get(&dark, "whites") > 20.0);
        assert!(get(&bright, "exposure") < -0.3);
        assert!(get(&bright, "highlights") < -40.0);
        assert!(get(&bright, "dehaze") > 5.0);
    }

    #[test]
    fn raw_accumulates_and_clamps() {
        let mut r = Raw::default();
        r.add("clarity", 80.0).add("clarity", 50.0).add("exposure", 0.333).hsl("hslSat", Band::Green, 30.0).grade("gradeShadows", 200.0, 20.0).grade("gradeShadows", 30.0, 10.0);
        assert_eq!(r.0["clarity"], json!(100.0));
        assert_eq!(r.0["exposure"], json!(0.33));
        assert_eq!(r.0["hslSat"][3], json!(30.0));
        assert_eq!(r.0["gradeShadows"], json!({"hue": 200.0, "sat": 30.0}));
    }

    #[test]
    fn stats_of_a_gradient() {
        let px: Vec<u8> = (0..=255u8).flat_map(|v| [v, v, v, 255]).collect();
        let s = Stats::of_rgba(&px);
        assert!((s.p50 - 0.5).abs() < 0.01, "{s:?}");
        assert!(s.sat < 1e-9);
    }

    #[test]
    fn every_template_names_real_looks() {
        let ids: Vec<&str> = looks().iter().map(|l| l.id).collect();
        for (t, _, _, recipe) in TEMPLATES {
            for (l, _, _) in recipe.iter() {
                assert!(ids.contains(l), "template {t} uses unknown look {l}");
            }
            assert!(template_styles().iter().any(|(v, _)| v == t));
        }
    }
}
