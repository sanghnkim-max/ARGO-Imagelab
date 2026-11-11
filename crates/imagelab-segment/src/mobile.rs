//! MobileSAM: the phone-sized interactive segmenter (feature `mobile-sam`).
//!
//! SAM's prompt encoder and mask decoder with MobileSAM's TinyViT-5M image encoder, assembled
//! from candle's own segment-anything modules (candle-transformers, Apache-2.0). Weights:
//! `mobile_sam-tiny-vitt.safetensors`, 40.6 MB F32, the safetensors conversion of the MobileSAM
//! checkpoint (github.com/ChaoningZhang/MobileSAM, Apache-2.0) published at
//! huggingface.co/lmz/candle-sam. Public, ungated: the app downloads it (its own TLS stack) and
//! hands it to `segment.installModel {model: "mobilesam"}`, which checks size and SHA-256.
//!
//! Assembled here rather than through `candle_transformers::…::sam::Sam` because `Sam` only
//! takes points; the prompt encoder underneath takes real box prompts, which `segment.box` needs.
//!
//! "Used wisely" (docs/INTERFACES.md §6): the image is encoded once on the ≤1008 px proxy
//! (padded to SAM's 1024 square) and the embedding is cached by content; each prompt then runs
//! only the prompt encoder and the mask decoder (a few ms).

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::mask_decoder::MaskDecoder;
use candle_transformers::models::segment_anything::prompt_encoder::PromptEncoder;
use candle_transformers::models::segment_anything::tiny_vit::{TinyViT, tiny_vit_5m};
use imagelab_api::{ApiError, ApiResult, Ctx, Event};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::cache::{EmbeddingCache, Encoder};
use crate::input::Input;
use crate::segment::{Click, Pred};

/// Model id in paths, events and params.
pub const MODEL: &str = "mobilesam";
pub const FILE: &str = "mobile_sam-tiny-vitt.safetensors";
pub const SIZE: u64 = 40_613_024;
pub const SHA256: &str = "5ad43a841b57cc798e2837e251c15aa9f602868a1780d195459383d117784cf7";
pub const URL: &str = "https://huggingface.co/lmz/candle-sam/resolve/main/mobile_sam-tiny-vitt.safetensors";
pub const LICENCE: &str = "Apache-2.0 (MobileSAM checkpoint, github.com/ChaoningZhang/MobileSAM; safetensors conversion at huggingface.co/lmz/candle-sam)";

const IMAGE_SIZE: usize = 1024;
const PATCH: usize = 16;
const EMBED_DIM: usize = 256;
/// Low-resolution mask side the decoder outputs (over the padded 1024 square).
const LOW_RES: usize = 256;
/// Embeddings kept (each 256×64×64 f32 = 4 MB).
const CACHE_ENTRIES: usize = 4;

pub fn model_path(root: &Path) -> PathBuf {
    root.join(".imagelab").join("models").join(MODEL).join(FILE)
}

struct Model {
    encoder: TinyViT,
    prompt: PromptEncoder,
    decoder: MaskDecoder,
    mean: Tensor,
    std: Tensor,
    device: Device,
}

impl Model {
    fn load(path: &Path) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        let bytes = std::fs::read(path).map_err(|e| candle_core::Error::Msg(format!("{}: {e}", path.display())))?;
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &device)?;
        let grid = IMAGE_SIZE / PATCH;
        Ok(Self {
            encoder: tiny_vit_5m(vb.pp("image_encoder"))?,
            prompt: PromptEncoder::new(EMBED_DIM, (grid, grid), (IMAGE_SIZE, IMAGE_SIZE), 16, vb.pp("prompt_encoder"))?,
            decoder: MaskDecoder::new(EMBED_DIM, 3, 3, 256, vb.pp("mask_decoder"))?,
            mean: Tensor::new(&[123.675f32, 116.28, 103.53], &device)?.reshape((3, 1, 1))?,
            std: Tensor::new(&[58.395f32, 57.12, 57.375], &device)?.reshape((3, 1, 1))?,
            device,
        })
    }

    /// RGB proxy (`w×h`, both ≤ 1024) → image embedding (1×256×64×64).
    fn encode(&self, rgb: &[u8], w: usize, h: usize) -> candle_core::Result<Tensor> {
        if w > IMAGE_SIZE || h > IMAGE_SIZE || rgb.len() != w * h * 3 {
            candle_core::bail!("proxy {w}x{h} does not fit the {IMAGE_SIZE} px encoder");
        }
        let img = Tensor::from_slice(rgb, (h, w, 3), &self.device)?.permute((2, 0, 1))?.to_dtype(DType::F32)?;
        let img = img.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?;
        let img = img.pad_with_zeros(1, 0, IMAGE_SIZE - h)?.pad_with_zeros(2, 0, IMAGE_SIZE - w)?;
        self.encoder.forward(&img.unsqueeze(0)?)
    }

    /// Prompts in proxy pixels → the best mask's low-res logits (256², padded square) + IoU.
    fn decode(&self, emb: &Tensor, points: &[(f32, f32, bool)], rect: Option<[f32; 4]>) -> candle_core::Result<(Vec<f32>, f32)> {
        let pts = if points.is_empty() {
            None
        } else {
            let xy: Vec<f32> = points.iter().flat_map(|(x, y, _)| [*x, *y]).collect();
            let labels: Vec<f32> = points.iter().map(|(_, _, p)| if *p { 1.0 } else { 0.0 }).collect();
            Some((Tensor::from_vec(xy, (1, points.len(), 2), &self.device)?, Tensor::from_vec(labels, (1, points.len()), &self.device)?))
        };
        let boxes = match rect {
            Some(b) => Some(Tensor::from_vec(b.to_vec(), (1, 4), &self.device)?),
            None => None,
        };
        // candle-transformers 0.9.2's PromptEncoder::embed_boxes returns the two corner tokens
        // flattened to (1, 512) instead of (1, 2, 256) (it concatenates rank-2 corner embeddings
        // along dim 1), which the decoder then rejects. Reshape them back: same values, SAM's
        // layout. With both points and a box, the box tokens follow the (padded) point tokens.
        let (sparse, dense) = match (pts.as_ref(), boxes.as_ref()) {
            (None, Some(b)) => {
                let (sp, de) = self.prompt.forward(None, Some(b), None)?;
                (sp.reshape((1, 2, EMBED_DIM))?, de)
            }
            (Some((a, l)), Some(b)) => {
                let (sp, de) = self.prompt.forward(Some((a, l)), None, None)?;
                let (bx, _) = self.prompt.forward(None, Some(b), None)?;
                (Tensor::cat(&[&sp, &bx.reshape((1, 2, EMBED_DIM))?], 1)?, de)
            }
            (p, None) => self.prompt.forward(p.map(|(a, b)| (a, b)), None, None)?,
        };
        // One positive point is ambiguous (part vs whole): ask for three masks, keep the best.
        let multi = rect.is_none() && points.len() == 1;
        let pe = self.prompt.get_dense_pe()?;
        let (masks, iou) = self.decoder.forward(emb, &pe, &sparse, &dense, multi)?;
        let iou: Vec<f32> = iou.flatten_all()?.to_vec1()?;
        let best = iou.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        let logits: Vec<f32> = masks.i((0, best))?.flatten_all()?.to_vec1()?;
        Ok((logits, iou.get(best).copied().unwrap_or(0.0)))
    }
}

struct ModelEncoder<'a>(&'a Model);
impl Encoder for ModelEncoder<'_> {
    type Embedding = Tensor;
    fn encode(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Tensor, String> {
        self.0.encode(rgb, w, h).map_err(|e| e.to_string())
    }
}

struct Engine {
    model: Option<Model>,
    cache: EmbeddingCache<Tensor>,
}

/// MobileSAM state shared by every `segment.*` handler of one registry.
pub struct MobileSam {
    engine: Mutex<Engine>,
}

impl Default for MobileSam {
    fn default() -> Self {
        Self { engine: Mutex::new(Engine { model: None, cache: EmbeddingCache::new(CACHE_ENTRIES) }) }
    }
}

/// The decoder's low-res logits cover the padded 1024 square with the proxy at the top-left.
/// Crop the proxy's part and resample it to a `side²` grid over the stretched image (the shape
/// [`Pred`] uses, so [`crate::mask::logits_to_mask`] serves every engine).
fn to_stretched(low: &[f32], pw: usize, ph: usize) -> Vec<f32> {
    let side = LOW_RES;
    let (cw, ch) = ((pw * LOW_RES) as f32 / IMAGE_SIZE as f32, (ph * LOW_RES) as f32 / IMAGE_SIZE as f32);
    let sample = |x: f32, y: f32| {
        let (x, y) = (x.clamp(0.0, (LOW_RES - 1) as f32), y.clamp(0.0, (LOW_RES - 1) as f32));
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(LOW_RES - 1), (y0 + 1).min(LOW_RES - 1));
        let (tx, ty) = (x.fract(), y.fract());
        let at = |xx: usize, yy: usize| low.get(yy * LOW_RES + xx).copied().filter(|v| v.is_finite()).unwrap_or(-20.0);
        let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
        let bot = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
        top * (1.0 - ty) + bot * ty
    };
    let mut out = Vec::with_capacity(side * side);
    for y in 0..side {
        let fy = (y as f32 + 0.5) * ch / side as f32 - 0.5;
        for x in 0..side {
            let fx = (x as f32 + 0.5) * cw / side as f32 - 0.5;
            out.push(sample(fx, fy));
        }
    }
    out
}

impl MobileSam {
    pub fn installed(root: &Path) -> bool {
        std::fs::metadata(model_path(root)).is_ok_and(|m| m.len() == SIZE)
    }

    /// Clicks (normalized 0..1) and/or a box (normalized `[x0, y0, x1, y1]`) → a prediction.
    /// Errors are strings: the caller falls back and reports them.
    pub fn segment(&self, root: &Path, input: &Input, clicks: &[Click], rect: Option<[f32; 4]>) -> Result<(Pred, bool), String> {
        let path = model_path(root);
        if !Self::installed(root) {
            return Err(format!("MobileSAM is not installed ({} missing; see segment.models)", path.display()));
        }
        let mut g = self.engine.lock().unwrap_or_else(PoisonError::into_inner);
        if g.model.is_none() {
            g.model = Some(Model::load(&path).map_err(|e| format!("loading MobileSAM: {e}"))?);
        }
        let Engine { model, cache } = &mut *g;
        let model = model.as_ref().ok_or_else(|| "MobileSAM did not load".to_string())?;
        let (pw, ph) = (input.pw as usize, input.ph as usize);
        let (emb, cached) = cache.get_or_encode_similar(&mut ModelEncoder(model), &input.rgb, pw, ph)?;
        let points: Vec<(f32, f32, bool)> = clicks.iter().map(|c| (c.x * pw as f32, c.y * ph as f32, c.positive)).collect();
        let rect = rect.map(|[x0, y0, x1, y1]| [x0 * pw as f32, y0 * ph as f32, x1 * pw as f32, y1 * ph as f32]);
        let (low, score) = model.decode(emb, &points, rect).map_err(|e| format!("MobileSAM decode: {e}"))?;
        Ok((Pred { logits: to_stretched(&low, pw, ph), side: LOW_RES, score }, cached))
    }

    /// Encode `input` now and keep the embedding (no prompt): taps that follow only decode.
    /// Returns whether it was already cached.
    pub fn prepare(&self, root: &Path, input: &Input) -> Result<bool, String> {
        if !Self::installed(root) {
            return Err("MobileSAM is not installed".into());
        }
        let mut g = self.engine.lock().unwrap_or_else(PoisonError::into_inner);
        if g.model.is_none() {
            g.model = Some(Model::load(&model_path(root)).map_err(|e| format!("loading MobileSAM: {e}"))?);
        }
        let Engine { model, cache } = &mut *g;
        let model = model.as_ref().ok_or_else(|| "MobileSAM did not load".to_string())?;
        let (_, cached) = cache.get_or_encode_similar(&mut ModelEncoder(model), &input.rgb, input.pw as usize, input.ph as usize)?;
        Ok(cached)
    }

    pub fn loaded(&self) -> (bool, usize) {
        let g = self.engine.lock().unwrap_or_else(PoisonError::into_inner);
        (g.model.is_some(), g.cache.len())
    }

    pub fn unload(&self) -> bool {
        let mut g = self.engine.lock().unwrap_or_else(PoisonError::into_inner);
        let was = g.model.is_some();
        g.model = None;
        g.cache.clear();
        was
    }

    pub fn status(&self, root: &Path) -> Value {
        let (loaded, cached) = self.loaded();
        json!({
            "state": if Self::installed(root) { "ready" } else { "missing" },
            "sizeBytes": SIZE,
            "sha256": SHA256,
            "url": URL,
            "file": FILE,
            "path": model_path(root).display().to_string(),
            "licence": LICENCE,
            "loaded": loaded,
            "cachedEmbeddings": cached,
        })
    }

    /// Install a downloaded weights file (workspace path) after checking size and SHA-256.
    pub fn install(&self, ctx: &Ctx, src: &Path) -> ApiResult<Value> {
        let meta = std::fs::metadata(src).map_err(|e| ApiError::not_found(format!("{}: {e}", src.display())))?;
        if !meta.is_file() || meta.len() != SIZE {
            return Err(ApiError::bad_params(format!("{} is {} bytes; {FILE} is {SIZE}", src.display(), meta.len())));
        }
        let mut f = std::fs::File::open(src).map_err(|e| ApiError::not_found(format!("{}: {e}", src.display())))?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            ctx.check_cancel()?;
            let n = std::io::Read::read(&mut f, &mut buf).map_err(|e| ApiError::new("disk", e.to_string()))?;
            if n == 0 {
                break;
            }
            h.update(buf.get(..n).unwrap_or_default());
        }
        let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if got != SHA256 {
            return Err(ApiError::bad_params(format!("{FILE} checksum mismatch (got {got})")));
        }
        let dst = model_path(&ctx.shared.root);
        let dir = dst.parent().ok_or_else(|| ApiError::internal("model path has no parent"))?;
        std::fs::create_dir_all(dir).map_err(|e| ApiError::new("disk", format!("{}: {e}", dir.display())))?;
        let part = dst.with_extension("part");
        let _ = std::fs::remove_file(&part);
        if std::fs::hard_link(src, &part).is_err() {
            std::fs::copy(src, &part).map_err(|e| ApiError::new("disk", format!("copying {FILE}: {e}")))?;
        }
        std::fs::rename(&part, &dst).map_err(|e| ApiError::new("disk", format!("{}: {e}", dst.display())))?;
        self.unload();
        ctx.emit(Event::ModelStatus { model: MODEL.into(), state: "ready".into(), progress: 1.0 });
        Ok(json!({"mobilesam": "ready", "path": dst.display().to_string()}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretched_crop_maps_the_proxy_region() {
        // Logits positive only in the top-left 64×48 cells = a 256×192 px proxy region.
        let mut low = vec![-10.0f32; LOW_RES * LOW_RES];
        for y in 0..48 {
            for x in 0..64 {
                low[y * LOW_RES + x] = 10.0;
            }
        }
        let s = to_stretched(&low, 256, 192);
        assert_eq!(s.len(), LOW_RES * LOW_RES);
        // The whole stretched grid is inside the region (allow the bilinear edge row/col).
        let positive = s.iter().filter(|v| **v > 0.0).count();
        assert!(positive > LOW_RES * LOW_RES * 9 / 10, "{positive}");
    }
}
