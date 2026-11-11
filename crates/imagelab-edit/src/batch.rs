//! Area `batch`: a recipe or steps over many photos, each in its own throwaway PhotoCraft
//! session (the UI's open document is untouched), with progress events, cancel and export
//! presets.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use imagelab_api::{ApiError, ApiResult, Ctx, Effects, Event, Headless, Registry};
use photocraft_automation::AuthorizedWorkspace;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::doc::{ensure_parent, resolve_open};
use crate::edit::load_recipe;
use crate::ops::{self, ExportSpec, Step, MAX_STEPS};
use crate::paths;

/// Most photos in one batch.
pub const MAX_PHOTOS: usize = 2000;
/// Most parallel sessions.
pub const MAX_CONCURRENCY: usize = 4;

/// An export preset: data, not code. `batch.export` and `batch.apply({export: {preset}})` read it.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Preset {
    pub name: &'static str,
    pub summary: &'static str,
    pub format: &'static str,
    pub max_side: Option<u32>,
    pub width: Option<u32>,
    /// Aspect applied only when the caller passes `crop: true`.
    pub optional_aspect: Option<&'static str>,
    pub quality: Option<u8>,
    pub suffix: &'static str,
}

pub const PRESETS: &[Preset] = &[
    Preset { name: "instagram", summary: "1080 px wide JPEG q90; crop: true crops to 4:5 first", format: "jpg", max_side: None, width: Some(1080), optional_aspect: Some("4:5"), quality: Some(90), suffix: "_ig" },
    Preset { name: "web2048", summary: "2048 px long side JPEG q85", format: "jpg", max_side: Some(2048), width: None, optional_aspect: None, quality: Some(85), suffix: "_web" },
    Preset { name: "fullJpeg", summary: "Full-size JPEG q95", format: "jpg", max_side: None, width: None, optional_aspect: None, quality: Some(95), suffix: "" },
    Preset { name: "png", summary: "Full-size PNG (lossless)", format: "png", max_side: None, width: None, optional_aspect: None, quality: None, suffix: "" },
];

pub fn preset(name: &str) -> ApiResult<&'static Preset> {
    PRESETS.iter().find(|p| p.name.eq_ignore_ascii_case(name)).ok_or_else(|| {
        ApiError::bad_params(format!("no preset `{name}`; presets: {}", PRESETS.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))
    })
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExportOpts {
    /// Workspace folder (default exports/).
    #[serde(default)]
    pub dir: Option<String>,
    /// jpg | png | webp | pcraft (also tif, psd).
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub max_side: Option<u32>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub aspect: Option<String>,
    #[serde(default)]
    pub quality: Option<u8>,
    /// Appended to the file stem, e.g. "_web".
    #[serde(default)]
    pub suffix: Option<String>,
    /// Start from a preset; explicit fields override it.
    #[serde(default)]
    pub preset: Option<String>,
    /// With a preset that has an optional aspect (instagram 4:5): apply it.
    #[serde(default)]
    pub crop: bool,
}

/// The export plan: folder, suffix and spec, after presets.
#[derive(Debug, Clone)]
struct Plan {
    dir: String,
    suffix: String,
    format: String,
    spec: ExportSpec,
}

fn plan(e: &ExportOpts) -> ApiResult<Plan> {
    let base = e.preset.as_deref().map(preset).transpose()?;
    let format = ops::normalize_format(e.format.as_deref().or(base.map(|b| b.format)).unwrap_or("jpg"))?;
    let aspect = e.aspect.clone().or_else(|| if e.crop { base.and_then(|b| b.optional_aspect).map(str::to_string) } else { None });
    let spec = ExportSpec {
        format: Some(format.clone()),
        max_side: e.max_side.or(base.and_then(|b| b.max_side)),
        width: e.width.or(base.and_then(|b| b.width)),
        aspect,
        quality: e.quality.or(base.and_then(|b| b.quality)),
    };
    if let Some(a) = &spec.aspect {
        ops::parse_aspect(a)?;
    }
    ops::export_options(&spec)?;
    let dir = paths::writable(e.dir.as_deref().unwrap_or(paths::EXPORTS_DIR))?;
    let suffix: String = e.suffix.clone().unwrap_or_else(|| base.map(|b| b.suffix.to_string()).unwrap_or_default());
    if suffix.chars().count() > 32 || suffix.chars().any(|c| matches!(c, '/' | '\\') || c.is_control()) {
        return Err(ApiError::bad_params("suffix must be at most 32 characters without slashes"));
    }
    Ok(Plan { dir, suffix, format, spec })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApplyParams {
    pub photos: Vec<String>,
    #[serde(default)]
    pub recipe: Option<String>,
    #[serde(default)]
    pub steps: Option<Vec<Step>>,
    #[serde(default)]
    pub export: Option<ExportOpts>,
    #[serde(default)]
    pub save_sidecar: bool,
    #[serde(default)]
    pub concurrency: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhotoResult {
    pub photo: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sidecar: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn throwaway(root: &Path) -> ApiResult<Headless> {
    let ws = AuthorizedWorkspace::new(Some(root), Some(root))?;
    Ok(Headless::with_workspace(ws))
}

/// Output names reserved by this batch, so two workers never pick the same file.
struct Names {
    root: PathBuf,
    taken: Mutex<HashSet<String>>,
}

impl Names {
    fn reserve(&self, dir: &str, stem: &str, ext: &str) -> ApiResult<String> {
        let mut taken = self.taken.lock().unwrap_or_else(PoisonError::into_inner);
        for n in 1..10_000 {
            let name = if n == 1 { stem.to_string() } else { format!("{stem} ({n})") };
            let candidate = paths::unique(&self.root, dir, &name, ext)?;
            if taken.insert(candidate.clone()) {
                return Ok(candidate);
            }
        }
        Err(ApiError::new("exists", format!("too many files named {stem}.{ext}")))
    }
}

fn process(root: &Path, photo: &str, steps: &[Step], save_sidecar: bool, plan: Option<&Plan>, names: &Names) -> ApiResult<(Option<String>, Option<String>)> {
    let (file, _) = resolve_open(root, photo, false)?;
    let photo = paths::relative(photo)?;
    let mut h = throwaway(root)?;
    h.open(Path::new(&file))?;
    ops::apply_steps(&mut h, steps, true)?;
    let mut sidecar = None;
    if save_sidecar {
        let target = paths::sidecar_for(&photo);
        ensure_parent(root, &target)?;
        ops::save_as(&mut h, &target, &ExportSpec { format: Some("pcraft".into()), ..ExportSpec::default() })?;
        sidecar = Some(target);
    }
    let mut output = None;
    if let Some(plan) = plan {
        ops::shape_for_export(&mut h, &plan.spec)?;
        let stem = format!("{}{}", paths::stem(&photo), plan.suffix);
        let target = names.reserve(&plan.dir, &stem, &plan.format)?;
        ensure_parent(root, &target)?;
        ops::save_as(&mut h, &target, &plan.spec)?;
        output = Some(target);
    }
    Ok((output, sidecar))
}

static JOBS: AtomicU64 = AtomicU64::new(1);

/// Run a batch (the work behind `batch.apply` and `batch.export`).
pub fn apply(ctx: &Ctx, p: ApplyParams) -> ApiResult<Value> {
    if p.photos.is_empty() {
        return Err(ApiError::bad_params("photos is empty"));
    }
    if p.photos.len() > MAX_PHOTOS {
        return Err(ApiError::too_large(format!("{} photos; at most {MAX_PHOTOS} per batch", p.photos.len())));
    }
    let steps: Vec<Step> = match (&p.recipe, &p.steps) {
        (Some(_), Some(_)) => return Err(ApiError::bad_params("pass `recipe` or `steps`, not both")),
        (Some(name), None) => load_recipe(&ctx.shared.root, name)?.steps,
        (None, Some(s)) => s.clone(),
        (None, None) => Vec::new(),
    };
    if steps.len() > MAX_STEPS {
        return Err(ApiError::too_large(format!("{} steps; at most {MAX_STEPS}", steps.len())));
    }
    for (i, s) in steps.iter().enumerate() {
        ops::validate_step(s).map_err(|e| ApiError::new(&e.code, format!("step {i}: {}", e.message)))?;
    }
    let plan = p.export.as_ref().map(plan).transpose()?;
    if steps.is_empty() && plan.is_none() && !p.save_sidecar {
        return Err(ApiError::bad_params("nothing to do: pass recipe/steps, export, or saveSidecar"));
    }
    if let Some(pl) = &plan {
        std::fs::create_dir_all(ctx.shared.root.join(&pl.dir)).map_err(|e| ApiError::internal(format!("create {}: {e}", pl.dir)))?;
    }
    let concurrency = p.concurrency.unwrap_or(2).clamp(1, MAX_CONCURRENCY).min(p.photos.len());
    let job = format!("batch-{}", JOBS.fetch_add(1, Ordering::Relaxed));
    let total = p.photos.len() as u64;
    let root = ctx.shared.root.clone();
    let names = Names { root: root.clone(), taken: Mutex::new(HashSet::new()) };
    let next = AtomicUsize::new(0);
    let done = AtomicU64::new(0);
    let results: Mutex<Vec<Option<PhotoResult>>> = Mutex::new(vec![None; p.photos.len()]);

    let worker = || {
        loop {
            if ctx.check_cancel().is_err() {
                break;
            }
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(photo) = p.photos.get(i) else { break };
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| process(&root, photo, &steps, p.save_sidecar, plan.as_ref(), &names)))
                .unwrap_or_else(|_| Err(ApiError::internal("processing panicked")));
            let r = match outcome {
                Ok((output, sidecar)) => PhotoResult { photo: photo.clone(), ok: true, output, sidecar, error: None },
                Err(e) => PhotoResult { photo: photo.clone(), ok: false, output: None, sidecar: None, error: Some(format!("{}: {}", e.code, e.message)) },
            };
            let message = if r.ok { format!("{} done", r.photo) } else { format!("{} failed", r.photo) };
            if let Some(slot) = results.lock().unwrap_or_else(PoisonError::into_inner).get_mut(i) {
                *slot = Some(r);
            }
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            ctx.emit(Event::JobProgress { job: job.clone(), done: n, total, message });
        }
    };
    if concurrency <= 1 {
        worker();
    } else {
        std::thread::scope(|s| {
            for _ in 0..concurrency {
                s.spawn(worker);
            }
        });
    }

    let results: Vec<PhotoResult> = results.into_inner().unwrap_or_else(PoisonError::into_inner).into_iter().flatten().collect();
    let cancelled = ctx.check_cancel().is_err() && (results.len() as u64) < total;
    let ok = results.iter().filter(|r| r.ok).count();
    let failed = results.len() - ok;
    let summary = if cancelled {
        format!("cancelled after {} of {total}: {ok} ok, {failed} failed", results.len())
    } else {
        format!("{ok} of {total} ok, {failed} failed")
    };
    ctx.emit(Event::JobFinished { job: job.clone(), ok: !cancelled && failed == 0, summary: summary.clone() });
    Ok(json!({"job": job, "total": total, "done": results.len(), "ok": ok, "failed": failed, "cancelled": cancelled, "summary": summary, "results": results}))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExportParams {
    photos: Vec<String>,
    preset: String,
    #[serde(default)]
    dir: Option<String>,
    #[serde(default)]
    crop: bool,
    #[serde(default)]
    recipe: Option<String>,
    #[serde(default)]
    concurrency: Option<usize>,
}

fn export(ctx: &Ctx, p: ExportParams) -> ApiResult<Value> {
    preset(&p.preset)?;
    apply(
        ctx,
        ApplyParams {
            photos: p.photos,
            recipe: p.recipe,
            steps: None,
            export: Some(ExportOpts { dir: p.dir, preset: Some(p.preset), crop: p.crop, ..ExportOpts::default() }),
            save_sidecar: false,
            concurrency: p.concurrency,
        },
    )
}

pub fn register(r: &mut Registry) {
    let preset_names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
    let preset_lines = PRESETS.iter().map(|p| format!("{}: {}", p.name, p.summary)).collect::<Vec<_>>().join("; ");
    r.function("batch", "apply", "Apply a recipe or steps to many photos, optionally exporting each; progress events, cancel")
        .description(&format!(
            "{{photos:[id] (≤{MAX_PHOTOS}), recipe? | steps?, export?: {{dir='exports', format: jpg|png|webp|pcraft, maxSide?, width?, aspect?, quality?, suffix?, preset?, crop?}}, saveSidecar?: false, concurrency?: 2 (max {MAX_CONCURRENCY})}}. Each photo (its sidecar if it has one) is edited in its own throwaway session — the open document is untouched — steps all-or-nothing per photo. Exports never overwrite: a taken name gets ' (2)'. Emits JobProgress after each photo and JobFinished at the end; on cancel returns the results so far with cancelled: true. Returns {{job, results:[{{photo, ok, output?, sidecar?, error?}}], ok, failed, cancelled}}."
        ))
        .params(json!({"type":"object","required":["photos"],"properties":{
            "photos":{"type":"array","items":{"type":"string"},"maxItems":MAX_PHOTOS},
            "recipe":{"type":"string"},
            "steps":{"type":"array","maxItems":MAX_STEPS,"items":{"type":"object"}},
            "export":{"type":"object","properties":{"dir":{"type":"string"},"format":{"type":"string","enum":["jpg","png","webp","pcraft","tif","psd"]},"maxSide":{"type":"integer"},"width":{"type":"integer"},"aspect":{"type":"string"},"quality":{"type":"integer","minimum":1,"maximum":100},"suffix":{"type":"string"},"preset":{"type":"string","enum":preset_names},"crop":{"type":"boolean"}}},
            "saveSidecar":{"type":"boolean","default":false},
            "concurrency":{"type":"integer","minimum":1,"maximum":MAX_CONCURRENCY}}}))
        .effects(Effects::Revertible)
        .typed(apply)
        .register();
    r.function("batch", "export", "Export many photos with a preset (instagram, web2048, fullJpeg, png)")
        .description(&format!("{{photos, preset, dir?='exports', crop?, recipe?, concurrency?}}. Presets — {preset_lines}. Same progress, cancel and no-overwrite rules as batch.apply."))
        .params(json!({"type":"object","required":["photos","preset"],"properties":{"photos":{"type":"array","items":{"type":"string"},"maxItems":MAX_PHOTOS},"preset":{"type":"string","enum":preset_names},"dir":{"type":"string"},"crop":{"type":"boolean"},"recipe":{"type":"string"},"concurrency":{"type":"integer","minimum":1,"maximum":MAX_CONCURRENCY}}}))
        .effects(Effects::Revertible)
        .typed(export)
        .register();
    r.function("batch", "presets", "Export presets")
        .typed(|_: &Ctx, _: Value| Ok(json!({"presets": PRESETS})))
        .register();
}
