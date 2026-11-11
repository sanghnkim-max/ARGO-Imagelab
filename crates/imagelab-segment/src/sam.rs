//! SAM 3 through `lightcraft-segment`: the model loaded lazily and dropped when idle, one
//! encoder pass per image (cached by content hash), prompts run only the decoder; and the model
//! files: status, download (only on `confirm: true`), install of a file the app downloaded.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use candle_core::IndexOp;
use imagelab_api::{ApiError, ApiResult, Ctx, Event, EventSink};
use lightcraft_segment::fetch::{self, FileSpec};
use lightcraft_segment::{Encoded, Sam3};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::cache::{EmbeddingCache, Encoder};
use crate::input::Input;
use crate::segment::{Click, Pred};

/// Model name in events and status.
pub const MODEL: &str = "sam3";
/// Side of the model's mask logits.
pub const MASK_SIDE: usize = lightcraft_segment::MASK_SIDE;
/// Unload after this long without a call (the model holds several GB).
const IDLE: Duration = Duration::from_secs(10 * 60);
/// Embeddings kept (each ~20–60 MB with its head features).
const CACHE_ENTRIES: usize = 3;
/// Memory the model needs free before it loads where that can be measured (Linux/Android):
/// 3.44 GB of F32 weights plus activations.
const MIN_AVAILABLE: u64 = 5 << 30;
/// Mirrors from the environment (base URLs; file `f` is `<base>/<f>`), then
/// `<root>/.imagelab/models/sam3-mirrors.txt`. Never from call params: a mirror's host receives
/// the token below.
pub const MIRRORS_ENV: &str = "IMAGELAB_SAM3_MIRRORS";
pub const TOKEN_ENV: &str = "IMAGELAB_SAM3_TOKEN";
pub const LICENCE: &str = "SAM License (Meta Platforms) — Meta's terms, not ImageLab's; downloading the weights means accepting them: https://huggingface.co/facebook/sam3";

pub fn model_dir(root: &Path) -> PathBuf {
    root.join(".imagelab").join("models").join(MODEL)
}

fn mirrors_file(root: &Path) -> PathBuf {
    root.join(".imagelab").join("models").join("sam3-mirrors.txt")
}

/// Normalized `[x0, y0, x1, y1]` of a text instance.
pub type NormBox = [f32; 4];
pub type Instance = (Pred, Option<NormBox>);

struct Engine {
    model: Option<Sam3>,
    cache: EmbeddingCache<Encoded>,
    last_used: Instant,
    reaper: bool,
}

struct ModelEncoder<'a>(&'a Sam3);
impl Encoder for ModelEncoder<'_> {
    type Embedding = Encoded;
    fn encode(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Encoded, String> {
        self.0.encode(rgb, w, h).map_err(|e| e.to_string())
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Download {
    Idle,
    Running { done: u64, total: u64 },
    Failed(String),
}

/// SAM state shared by every `segment.*` handler of one registry.
pub struct Sam {
    engine: Arc<Mutex<Engine>>,
    download: Arc<Mutex<Download>>,
    cancel: Arc<AtomicBool>,
}

impl Default for Sam {
    fn default() -> Self {
        Self {
            engine: Arc::new(Mutex::new(Engine { model: None, cache: EmbeddingCache::new(CACHE_ENTRIES), last_used: Instant::now(), reaper: false })),
            download: Arc::new(Mutex::new(Download::Idle)),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Free memory reported by the kernel (Linux/Android), when it can be read.
fn available_memory() -> Option<u64> {
    if !cfg!(any(target_os = "linux", target_os = "android")) {
        return None;
    }
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

impl Sam {
    pub fn installed(root: &Path) -> bool {
        lightcraft_segment::is_model_dir(&model_dir(root))
    }

    /// Run `f` with the model and the image's embedding (encoded once, then cached). Returns
    /// `f`'s result and whether the embedding was cached. Errors are strings: the caller falls
    /// back to PhotoCraft and reports them.
    pub fn run<R>(&self, root: &Path, input: &Input, f: impl FnOnce(&mut Sam3, &mut Encoded) -> lightcraft_segment::Result<R>) -> Result<(R, bool), String> {
        let dir = model_dir(root);
        if !lightcraft_segment::is_model_dir(&dir) {
            return Err(format!("the SAM 3 model is not installed ({}); see segment.models / segment.ensureModel", dir.display()));
        }
        let mut g = self.engine.lock().unwrap_or_else(PoisonError::into_inner);
        if g.model.is_none() {
            if let Some(avail) = available_memory()
                && avail < MIN_AVAILABLE
            {
                return Err(format!("not enough free memory to load SAM 3 ({} MB free, {} MB needed)", avail >> 20, MIN_AVAILABLE >> 20));
            }
            let model = match Sam3::load(&dir) {
                Ok(m) => m,
                // The GPU (Metal) build can fail where the CPU works.
                Err(_) if lightcraft_segment::GPU_BUILD => Sam3::load_cpu(&dir).map_err(|e| e.to_string())?,
                Err(e) => return Err(e.to_string()),
            };
            g.model = Some(model);
            g.cache.clear();
            if !g.reaper {
                g.reaper = spawn_reaper(Arc::downgrade(&self.engine));
            }
        }
        g.last_used = Instant::now();
        let Engine { model, cache, .. } = &mut *g;
        let model = model.as_mut().ok_or("model not loaded")?;
        let (enc, cached) = cache.get_or_encode(&mut ModelEncoder(&*model), &input.rgb, input.pw as usize, input.ph as usize)?;
        let r = f(model, enc).map_err(|e| e.to_string())?;
        Ok((r, cached))
    }

    pub fn clicks(&self, root: &Path, input: &Input, clicks: &[Click]) -> Result<(Pred, bool), String> {
        let clicks: Vec<lightcraft_segment::Click> = clicks.iter().map(|c| lightcraft_segment::Click { x: c.x, y: c.y, positive: c.positive }).collect();
        self.run(root, input, move |m, enc| {
            let p = m.segment_clicks(enc, &clicks)?;
            Ok(Pred { logits: p.logits, side: MASK_SIDE, score: p.score })
        })
    }

    /// Every instance of `prompt` above `threshold`, best first, at most `max`.
    pub fn text(&self, root: &Path, input: &Input, prompt: &str, threshold: f32, max: usize) -> Result<(Vec<Instance>, bool), String> {
        self.run(root, input, |m, enc| {
            let out = m.detect(enc, prompt)?;
            let s = |x: f32| 1.0 / (1.0 + (-x).exp());
            let presence = s(out.presence);
            let mut keep: Vec<(usize, f32)> = out.logits.iter().enumerate().map(|(i, l)| (i, s(*l) * presence)).filter(|(_, p)| *p > threshold).collect();
            keep.sort_by(|a, b| b.1.total_cmp(&a.1));
            keep.truncate(max);
            let mut preds = Vec::with_capacity(keep.len());
            for (i, score) in keep {
                let logits = out.masks.i(i)?.flatten_all()?.to_vec1::<f32>()?;
                preds.push((Pred { logits, side: MASK_SIDE, score }, out.boxes.get(i).copied()));
            }
            Ok(preds)
        })
    }

    /// People/animals the photo is about (LightCraft's subject heuristic); `None` = none found.
    pub fn subject(&self, root: &Path, input: &Input) -> Result<(Option<Pred>, bool), String> {
        self.run(root, input, |m, enc| Ok(m.segment_subject(enc, None, 0.5)?.map(|p| Pred { logits: p.logits, side: MASK_SIDE, score: p.score })))
    }

    pub fn unload(&self) -> bool {
        let mut g = self.engine.lock().unwrap_or_else(PoisonError::into_inner);
        g.cache.clear();
        g.model.take().is_some()
    }

    pub fn status(&self, root: &Path) -> Value {
        let dir = model_dir(root);
        let download = self.download.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let (loaded, cached) = {
            let g = self.engine.lock().unwrap_or_else(PoisonError::into_inner);
            (g.model.is_some(), g.cache.len())
        };
        let state = match (&download, lightcraft_segment::is_model_dir(&dir)) {
            (Download::Running { .. }, _) => "downloading",
            (_, true) => "ready",
            _ => "missing",
        };
        let mirrors = self.mirrors(root);
        let mut v = json!({
            "sam3": state,
            "sizeBytes": fetch_size(),
            "path": dir.display().to_string(),
            "loaded": loaded,
            "cachedEmbeddings": cached,
            "licence": LICENCE,
            "downloadConfigured": !mirrors.is_empty(),
            "files": lightcraft_segment::fetch::SAM3_FILES.iter().map(|f| f.name).collect::<Vec<_>>(),
        });
        match download {
            Download::Running { done, total } => v["progress"] = json!(if total > 0 { done as f64 / total as f64 } else { 0.0 }),
            Download::Failed(e) => v["lastError"] = json!(e),
            Download::Idle => {}
        }
        v
    }

    fn mirrors(&self, root: &Path) -> Vec<String> {
        let env = std::env::var(MIRRORS_ENV).ok();
        fetch::mirrors(env.as_deref(), Some(&mirrors_file(root)))
    }

    /// Start downloading the model in the background (progress as `ModelStatus` events).
    pub fn ensure(&self, ctx: &Ctx, confirm: bool) -> ApiResult<Value> {
        let root = ctx.shared.root.clone();
        if Self::installed(&root) {
            return Ok(json!({"sam3": "ready", "path": model_dir(&root).display().to_string()}));
        }
        let size = fetch_size();
        if !confirm {
            return Err(ApiError::new(
                "confirm_required",
                format!("downloading SAM 3 writes {:.2} GB to {}. Licence: {LICENCE}. Ask the person, then call again with confirm: true.", size as f64 / 1e9, model_dir(&root).display()),
            ));
        }
        let mirrors = self.mirrors(&root);
        if mirrors.is_empty() {
            return Err(ApiError::unavailable(format!(
                "no download location is configured for SAM 3 (set {MIRRORS_ENV} or write base URLs to {}; Hugging Face facebook/sam3 is gated and needs the person's own token in {TOKEN_ENV}). Or let the app download the files and call segment.installModel.",
                mirrors_file(&root).display()
            )));
        }
        {
            let mut d = self.download.lock().unwrap_or_else(PoisonError::into_inner);
            if matches!(*d, Download::Running { .. }) {
                return Ok(json!({"sam3": "downloading"}));
            }
            *d = Download::Running { done: 0, total: size };
        }
        self.cancel.store(false, Ordering::Relaxed);
        let (download, cancel, events) = (self.download.clone(), self.cancel.clone(), ctx.shared.events.clone());
        let dir = model_dir(&root);
        let spawned = std::thread::Builder::new().name("imagelab-sam3-download".into()).spawn(move || run_download(&dir, &mirrors, &download, &cancel, events.as_ref()));
        if let Err(e) = spawned {
            *self.download.lock().unwrap_or_else(PoisonError::into_inner) = Download::Failed(e.to_string());
            return Err(ApiError::internal(format!("could not start the download: {e}")));
        }
        ctx.emit(Event::ModelStatus { model: MODEL.into(), state: "downloading".into(), progress: 0.0 });
        Ok(json!({"sam3": "downloading", "sizeBytes": size, "path": model_dir(&root).display().to_string()}))
    }

    pub fn cancel_download(&self) -> bool {
        let running = matches!(*self.download.lock().unwrap_or_else(PoisonError::into_inner), Download::Running { .. });
        self.cancel.store(true, Ordering::Relaxed);
        running
    }

    /// Install model files the app downloaded: `path` is a workspace folder holding
    /// model.safetensors, vocab.json and merges.txt. The weights must match the official
    /// checkpoint's size and SHA-256. Files are hard-linked when possible, else copied.
    pub fn install(&self, ctx: &Ctx, src: &Path) -> ApiResult<Value> {
        let dst = model_dir(&ctx.shared.root);
        std::fs::create_dir_all(&dst).map_err(|e| ApiError::new("disk", format!("{}: {e}", dst.display())))?;
        for spec in fetch::SAM3_FILES {
            let from = src.join(spec.name);
            let meta = std::fs::metadata(&from).map_err(|e| ApiError::not_found(format!("{}: {e}", from.display())))?;
            check_spec(spec, &from, meta.len(), ctx)?;
        }
        for spec in fetch::SAM3_FILES {
            let (from, to) = (src.join(spec.name), dst.join(spec.name));
            let part = dst.join(format!("{}.part", spec.name));
            let _ = std::fs::remove_file(&part);
            if std::fs::hard_link(&from, &part).is_err() {
                std::fs::copy(&from, &part).map_err(|e| ApiError::new("disk", format!("copying {}: {e}", spec.name)))?;
            }
            std::fs::rename(&part, &to).map_err(|e| ApiError::new("disk", format!("{}: {e}", to.display())))?;
        }
        ctx.emit(Event::ModelStatus { model: MODEL.into(), state: "ready".into(), progress: 1.0 });
        Ok(json!({"sam3": "ready", "path": dst.display().to_string()}))
    }
}

fn fetch_size() -> u64 {
    fetch::SAM3_FILES.iter().filter_map(|f| f.size).sum()
}

fn check_spec(spec: &FileSpec<'_>, path: &Path, len: u64, ctx: &Ctx) -> ApiResult<()> {
    match spec.size {
        Some(s) if s != len => return Err(ApiError::bad_params(format!("{} is {len} bytes; the official file is {s}", spec.name))),
        None if len > spec.max => return Err(ApiError::bad_params(format!("{} is {len} bytes (limit {})", spec.name, spec.max))),
        _ => {}
    }
    if let Some(want) = spec.sha256 {
        let mut f = std::fs::File::open(path).map_err(|e| ApiError::not_found(format!("{}: {e}", path.display())))?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            ctx.check_cancel()?;
            let n = f.read(&mut buf).map_err(|e| ApiError::new("disk", e.to_string()))?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if got != want {
            return Err(ApiError::bad_params(format!("{} does not match the official checkpoint (SHA-256 {got})", spec.name)));
        }
    }
    Ok(())
}

fn run_download(dir: &Path, mirrors: &[String], state: &Mutex<Download>, cancel: &AtomicBool, events: &dyn EventSink) {
    let opts = fetch::Options { token_env: Some(TOKEN_ENV), ..fetch::Options::default() };
    let mut last = (Instant::now(), 0u64);
    let mut progress = |p: &fetch::Progress| {
        *state.lock().unwrap_or_else(PoisonError::into_inner) = Download::Running { done: p.done, total: p.total };
        // At most ~one event a second or per 1 %.
        let step = (p.total / 100).max(1);
        if last.0.elapsed() >= Duration::from_secs(1) || p.done >= last.1.saturating_add(step) {
            last = (Instant::now(), p.done);
            let progress = if p.total > 0 { (p.done as f64 / p.total as f64) as f32 } else { 0.0 };
            events.emit(Event::ModelStatus { model: MODEL.into(), state: "downloading".into(), progress });
        }
    };
    let result = fetch::download(fetch::SAM3_FILES, mirrors, dir, &opts, cancel, &mut progress);
    let (next, event) = match result {
        Ok(()) => (Download::Idle, Event::ModelStatus { model: MODEL.into(), state: "ready".into(), progress: 1.0 }),
        Err(e) => (Download::Failed(e.to_string()), Event::ModelStatus { model: MODEL.into(), state: "missing".into(), progress: 0.0 }),
    };
    *state.lock().unwrap_or_else(PoisonError::into_inner) = next;
    events.emit(event);
}

/// Drops the model (and embeddings) after [`IDLE`]; ends with the registry.
fn spawn_reaper(engine: Weak<Mutex<Engine>>) -> bool {
    std::thread::Builder::new()
        .name("imagelab-sam3-idle".into())
        .spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(30));
                let Some(e) = engine.upgrade() else { return };
                let mut g = e.lock().unwrap_or_else(PoisonError::into_inner);
                if g.model.is_some() && g.last_used.elapsed() > IDLE {
                    g.model = None;
                    g.cache.clear();
                }
            }
        })
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_is_reported_not_loaded() {
        let root = tempfile::tempdir().unwrap();
        let sam = Sam::default();
        let input = Input { width: 4, height: 4, rgb: vec![0; 48], pw: 4, ph: 4, kind: "image" };
        let err = sam.clicks(root.path(), &input, &[Click { x: 0.5, y: 0.5, positive: true }]).err().unwrap();
        assert!(err.contains("not installed"), "{err}");
        let st = sam.status(root.path());
        assert_eq!(st["sam3"], "missing");
        assert_eq!(st["sizeBytes"], 3_439_938_512u64);
        assert_eq!(st["loaded"], false);
    }
}
