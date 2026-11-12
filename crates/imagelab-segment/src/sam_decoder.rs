//! SAM's mask decoder (two-way transformer, upscaling, hypernetwork heads), laid out for a
//! multi-core phone CPU: what each MobileSAM tap runs.
//!
//! Same weights, tensor names and arithmetic as candle-transformers 0.9.2's
//! `segment_anything::{mask_decoder, transformer}` (adapted from them; Apache-2.0/MIT), with the
//! kernels of [`crate::tiny_vit`]:
//!
//! * everything stays pixel-major (`4096 × 256` for the image side), taking the encoder's tokens
//!   as they come;
//! * attention is one parallel kernel over query rows, all heads at once (candle splits heads
//!   into separate small matmuls);
//! * the stride-2 transposed convolutions of the upscaler are a gemm plus a parallel pixel
//!   shuffle with bias and GELU fused (candle's CPU `conv_transpose2d` is a scalar loop and was
//!   the largest share of a tap);
//! * residual adds are fused into the projections; only the best mask is upscaled into logits.

use candle_core::{D, Device, IndexOp, Result, Tensor, bail};
use candle_nn::VarBuilder;
use rayon::prelude::*;

use crate::tiny_vit::{Ep, Norm, Pointwise, epilogue, exp_nonpositive, gelu_all, linear, vec_f32, with_f32};

/// Image-side tokens: the 64×64 embedding grid.
const CELLS: usize = 64 * 64;
const DIM: usize = 256;
const HEADS: usize = 8;
pub(crate) const MASK_TOKENS: usize = 4;
/// Output logits side (4× the embedding grid).
pub const LOW_RES: usize = 256;

fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    epilogue(a, None, Ep { residual: Some(b), ..Ep::default() })
}

/// Multi-head softmax attention, `q (nq × c)` against `k, v (nk × c)` → `nq × c`.
fn mha(q: &[f32], k: &[f32], v: &[f32], nq: usize, nk: usize, c: usize) -> Result<Vec<f32>> {
    if q.len() != nq * c || k.len() != nk * c || v.len() != nk * c {
        bail!("attention shapes do not match");
    }
    let dh = c / HEADS;
    let scale = 1.0 / (dh as f32).sqrt();
    // Keys channel-major, so a score row accumulates as vector ops over keys.
    let mut kt = vec![0f32; c * nk];
    kt.par_chunks_mut(nk).enumerate().for_each(|(ch, row)| {
        for (j, r) in row.iter_mut().enumerate() {
            *r = k[j * c + ch];
        }
    });
    let rows = nq.div_ceil(rayon::current_num_threads() * 4).max(1);
    let mut out = vec![0f32; nq * c];
    out.par_chunks_mut(rows * c).enumerate().for_each(|(t, o)| {
        let mut s = vec![0f32; nk];
        for (r, orow) in o.chunks_exact_mut(c).enumerate() {
            let i = t * rows + r;
            let qi = &q[i * c..(i + 1) * c];
            for h in 0..HEADS {
                s.fill(0.0);
                for d in 0..dh {
                    let qd = qi[h * dh + d] * scale;
                    for (sj, kj) in s.iter_mut().zip(&kt[(h * dh + d) * nk..(h * dh + d + 1) * nk]) {
                        *sj = qd.mul_add(*kj, *sj);
                    }
                }
                let mx = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                s.iter_mut().for_each(|x| *x = exp_nonpositive(*x - mx));
                let inv = 1.0 / s.iter().sum::<f32>();
                let oh = &mut orow[h * dh..(h + 1) * dh];
                oh.fill(0.0);
                for (j, p) in s.iter().enumerate() {
                    let p = p * inv;
                    for (ov, vj) in oh.iter_mut().zip(&v[j * c + h * dh..j * c + (h + 1) * dh]) {
                        *ov = p.mul_add(*vj, *ov);
                    }
                }
            }
        }
    });
    Ok(out)
}

pub(crate) struct Attention {
    pub q: Pointwise,
    pub k: Pointwise,
    pub v: Pointwise,
    pub out: Pointwise,
}

impl Attention {
    fn new(vb: VarBuilder, downsample: usize) -> Result<Self> {
        let inner = DIM / downsample;
        Ok(Self {
            q: linear(vb.pp("q_proj"), DIM, inner)?,
            k: linear(vb.pp("k_proj"), DIM, inner)?,
            v: linear(vb.pp("v_proj"), DIM, inner)?,
            out: linear(vb.pp("out_proj"), inner, DIM)?,
        })
    }

    /// `out_proj(attention(q, k, v))`, plus `residual` when given.
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let (q, k, v) = (self.q.forward(q, Ep::default())?, self.k.forward(k, Ep::default())?, self.v.forward(v, Ep::default())?);
        let (nq, c) = q.dims2()?;
        let nk = k.dim(0)?;
        let o = with_f32(&q, |qv| with_f32(&k, |kv| with_f32(&v, |vv| mha(qv, kv, vv, nq, nk, c))))?;
        self.out.forward(&Tensor::from_vec(o, (nq, c), &Device::Cpu)?, Ep { residual, ..Ep::default() })
    }
}

/// Linear layers with ReLU between them.
pub(crate) struct Mlp(pub Vec<Pointwise>);

impl Mlp {
    fn new(vb: VarBuilder, dims: &[usize]) -> Result<Self> {
        let vb = vb.pp("layers");
        Ok(Self(dims.windows(2).enumerate().map(|(i, d)| linear(vb.pp(i), d[0], d[1])).collect::<Result<Vec<_>>>()?))
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, l) in self.0.iter().enumerate() {
            x = l.forward(&x, Ep { relu: i + 1 < self.0.len(), ..Ep::default() })?;
        }
        Ok(x)
    }
}

pub(crate) struct TwoWayBlock {
    pub self_attn: Attention,
    pub norm1: Norm,
    pub t2i: Attention,
    pub norm2: Norm,
    pub lin1: Pointwise,
    pub lin2: Pointwise,
    pub norm3: Norm,
    pub i2t: Attention,
    pub norm4: Norm,
    pub skip_first_pe: bool,
}

impl TwoWayBlock {
    fn new(vb: VarBuilder, skip_first_pe: bool) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(vb.pp("self_attn"), 1)?,
            norm1: Norm::new(vb.pp("norm1"), DIM, 1e-5)?,
            t2i: Attention::new(vb.pp("cross_attn_token_to_image"), 2)?,
            norm2: Norm::new(vb.pp("norm2"), DIM, 1e-5)?,
            lin1: linear(vb.pp("mlp").pp("lin1"), DIM, 2048)?,
            lin2: linear(vb.pp("mlp").pp("lin2"), 2048, DIM)?,
            norm3: Norm::new(vb.pp("norm3"), DIM, 1e-5)?,
            i2t: Attention::new(vb.pp("cross_attn_image_to_token"), 2)?,
            norm4: Norm::new(vb.pp("norm4"), DIM, 1e-5)?,
            skip_first_pe,
        })
    }

    fn forward(&self, queries: &Tensor, keys: &Tensor, query_pe: &Tensor, key_pe: &Tensor) -> Result<(Tensor, Tensor)> {
        let queries = if self.skip_first_pe {
            self.self_attn.forward(queries, queries, queries, None)?
        } else {
            let q = add(queries, query_pe)?;
            self.self_attn.forward(&q, &q, queries, Some(queries))?
        };
        let queries = self.norm1.forward(&queries)?;
        // Tokens attend to the image.
        let q = add(&queries, query_pe)?;
        let k = add(keys, key_pe)?;
        let queries = self.norm2.forward(&self.t2i.forward(&q, &k, keys, Some(&queries))?)?;
        let h = self.lin1.forward(&queries, Ep { relu: true, ..Ep::default() })?;
        let queries = self.norm3.forward(&self.lin2.forward(&h, Ep { residual: Some(&queries), ..Ep::default() })?)?;
        // The image attends to the tokens.
        let q = add(&queries, query_pe)?;
        let keys = self.norm4.forward(&self.i2t.forward(&k, &q, &queries, Some(keys))?)?;
        Ok((queries, keys))
    }
}

/// A stride-2, 2×2 transposed convolution: a gemm to the four sub-pixels, then a pixel shuffle
/// (with bias, and GELU when asked) into the doubled grid.
pub(crate) struct Up {
    /// cin × (4·cout), columns [dy][dx][co].
    pub w: Tensor,
    pub b: Vec<f32>,
    pub cout: usize,
}

impl Up {
    fn new(vb: VarBuilder, cin: usize, cout: usize) -> Result<Self> {
        let w = vb.get((cin, cout, 2, 2), "weight")?;
        Ok(Self { w: w.permute((0, 2, 3, 1))?.contiguous()?.reshape((cin, 4 * cout))?, b: vec_f32(&vb.get(cout, "bias")?)?, cout })
    }

    /// `x`: `h·w × cin` → `(2h·2w) × cout`.
    fn forward(&self, x: &Tensor, h: usize, w: usize, act: bool) -> Result<Tensor> {
        let co = self.cout;
        let p = x.matmul(&self.w)?;
        let out = with_f32(&p, |pv| {
            if pv.len() != h * w * 4 * co {
                bail!("upscaler input is {} values", pv.len());
            }
            let mut out = vec![0f32; 4 * h * w * co];
            out.par_chunks_mut(2 * w * co).enumerate().for_each(|(oy, orow)| {
                let (i, dy) = (oy / 2, oy % 2);
                for (ox, o) in orow.chunks_exact_mut(co).enumerate() {
                    let (j, dx) = (ox / 2, ox % 2);
                    let at = (i * w + j) * 4 * co + (dy * 2 + dx) * co;
                    o.iter_mut().zip(&pv[at..at + co]).zip(&self.b).for_each(|((o, s), b)| *o = s + b);
                    if act {
                        gelu_all(o);
                    }
                }
            });
            Ok(out)
        })?;
        Tensor::from_vec(out, (4 * h * w, co), &Device::Cpu)
    }
}

/// SAM's mask decoder (3 multimask outputs + 1, IoU head depth 3, hidden 256).
pub struct MaskDecoder {
    /// IoU token then the four mask tokens, `5 × 256`.
    pub(crate) output_tokens: Tensor,
    pub(crate) iou_head: Mlp,
    pub(crate) hyper: Vec<Mlp>,
    pub(crate) up1: Up,
    pub(crate) up_ln: Norm,
    pub(crate) up2: Up,
    pub(crate) layers: Vec<TwoWayBlock>,
    pub(crate) final_t2i: Attention,
    pub(crate) norm_final: Norm,
}

impl MaskDecoder {
    /// From `mask_decoder.*` weights.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let iou = vb.pp("iou_token").get((1, DIM), "weight")?;
        let masks = vb.pp("mask_tokens").get((MASK_TOKENS, DIM), "weight")?;
        let t = vb.pp("transformer");
        Ok(Self {
            output_tokens: Tensor::cat(&[&iou, &masks], 0)?,
            iou_head: Mlp::new(vb.pp("iou_prediction_head"), &[DIM, DIM, DIM, MASK_TOKENS])?,
            hyper: (0..MASK_TOKENS).map(|i| Mlp::new(vb.pp("output_hypernetworks_mlps").pp(i), &[DIM, DIM, DIM, DIM / 8])).collect::<Result<Vec<_>>>()?,
            up1: Up::new(vb.pp("output_upscaling.0"), DIM, DIM / 4)?,
            up_ln: Norm::new(vb.pp("output_upscaling.1"), DIM / 4, 1e-6)?,
            up2: Up::new(vb.pp("output_upscaling.3"), DIM / 4, DIM / 8)?,
            layers: (0..2).map(|i| TwoWayBlock::new(t.pp("layers").pp(i), i == 0)).collect::<Result<Vec<_>>>()?,
            final_t2i: Attention::new(t.pp("final_attn_token_to_image"), 2)?,
            norm_final: Norm::new(t.pp("norm_final_attn"), DIM, 1e-5)?,
        })
    }

    /// Image tokens `emb`, positional encoding `pe` and dense prompt `dense` (each
    /// `4096 × 256`, pixel-major) and sparse prompt tokens (`n × 256`) → the best mask's
    /// `256 × 256` logits (row-major over the padded square) and its predicted IoU. With
    /// `multimask`, the best of the three multimask outputs (SAM's answer to an ambiguous
    /// single point); otherwise the single-mask output.
    pub fn forward(&self, emb: &Tensor, pe: &Tensor, dense: &Tensor, sparse: &Tensor, multimask: bool) -> Result<(Vec<f32>, f32)> {
        if emb.dims2()? != (CELLS, DIM) {
            bail!("image embedding must be {CELLS}×{DIM} tokens");
        }
        let sparse = if sparse.rank() == 3 { sparse.squeeze(0)? } else { sparse.clone() };
        let tokens = Tensor::cat(&[&self.output_tokens, &sparse], 0)?;
        let mut queries = tokens.clone();
        let mut keys = add(emb, dense)?;
        for l in &self.layers {
            (queries, keys) = l.forward(&queries, &keys, &tokens, pe)?;
        }
        let q = add(&queries, &tokens)?;
        let k = add(&keys, pe)?;
        let queries = self.norm_final.forward(&self.final_t2i.forward(&q, &k, &keys, Some(&queries))?)?;

        let iou = vec_f32(&self.iou_head.forward(&queries.i(0..1)?)?)?;
        let range = if multimask { 1..MASK_TOKENS } else { 0..1 };
        let Some(best) = range.max_by(|a, b| iou[*a].total_cmp(&iou[*b])) else { bail!("no mask outputs") };
        let hyper = vec_f32(&self.hyper[best].forward(&queries.i(1 + best..2 + best)?)?)?;

        let up = self.up1.forward(&keys, 64, 64, false)?;
        let up = epilogue(&self.up_ln.forward(&up)?, None, Ep { gelu: true, ..Ep::default() })?;
        let up = self.up2.forward(&up, 128, 128, true)?;
        let c = up.dim(D::Minus1)?;
        if hyper.len() != c {
            bail!("hypernetwork gives {} weights for {c} channels", hyper.len());
        }
        let logits = with_f32(&up, |uv| Ok(uv.par_chunks(c).map(|px| px.iter().zip(&hyper).map(|(a, b)| a * b).sum::<f32>()).collect::<Vec<f32>>()))?;
        if logits.len() != LOW_RES * LOW_RES {
            bail!("decoder produced {} logits", logits.len());
        }
        Ok((logits, iou[best]))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::DType;
    use candle_transformers::models::segment_anything as cs;

    use super::*;

    /// Matches candle's decoder on the real checkpoint (set IMAGELAB_MOBILESAM_WEIGHTS), for a
    /// single point (multimask), two points, and a box.
    #[test]
    fn matches_candle_decoder() {
        let Ok(path) = std::env::var("IMAGELAB_MOBILESAM_WEIGHTS") else {
            eprintln!("IMAGELAB_MOBILESAM_WEIGHTS not set; skipped");
            return;
        };
        let dev = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(std::fs::read(path).unwrap(), DType::F32, &dev).unwrap();
        let enc = crate::tiny_vit::TinyViT::new(vb.pp("image_encoder")).unwrap();
        let prompt = cs::prompt_encoder::PromptEncoder::new(256, (64, 64), (1024, 1024), 16, vb.pp("prompt_encoder")).unwrap();
        let theirs = cs::mask_decoder::MaskDecoder::new(256, 3, 3, 256, vb.pp("mask_decoder")).unwrap();
        let ours = MaskDecoder::new(vb.pp("mask_decoder")).unwrap();
        let (w, h) = (800usize, 600usize);
        let rgb: Vec<u8> = (0..w * h * 3)
            .map(|i| {
                let (x, y) = ((i / 3) % w, (i / 3) / w);
                let inside = (x as i64 - 300).pow(2) + (y as i64 - 280).pow(2) < 120 * 120;
                if inside { [220u8, 60, 40][i % 3] } else { [40u8, 90, 160][i % 3] }
            })
            .collect();
        let emb_nchw = enc.encode(&rgb, w, h).unwrap();
        let emb = emb_nchw.squeeze(0).unwrap().flatten_from(1).unwrap().t().unwrap().contiguous().unwrap();
        let pe_nchw = prompt.get_dense_pe().unwrap();
        let pe = pe_nchw.squeeze(0).unwrap().flatten_from(1).unwrap().t().unwrap().contiguous().unwrap();
        let (_, dense_nchw) = prompt.forward(None, None, None).unwrap();
        let dense = dense_nchw.squeeze(0).unwrap().flatten_from(1).unwrap().t().unwrap().contiguous().unwrap();
        let pts = |v: &[(f32, f32, f32)]| {
            let xy: Vec<f32> = v.iter().flat_map(|p| [p.0, p.1]).collect();
            let l: Vec<f32> = v.iter().map(|p| p.2).collect();
            (Tensor::from_vec(xy, (1, v.len(), 2), &dev).unwrap(), Tensor::from_vec(l, (1, v.len()), &dev).unwrap())
        };
        let cases: Vec<(&str, Tensor, bool)> = vec![
            ("one point", { let (a, l) = pts(&[(300.0, 280.0, 1.0)]); prompt.forward(Some((&a, &l)), None, None).unwrap().0 }, true),
            ("two points", { let (a, l) = pts(&[(300.0, 280.0, 1.0), (650.0, 500.0, 0.0)]); prompt.forward(Some((&a, &l)), None, None).unwrap().0 }, false),
            ("box", prompt.forward(None, Some(&Tensor::new(&[[170f32, 150., 430., 410.]], &dev).unwrap()), None).unwrap().0.reshape((1, 2, 256)).unwrap(), false),
        ];
        for (name, sparse, multi) in cases {
            let t = std::time::Instant::now();
            let (logits, iou) = ours.forward(&emb, &pe, &dense, &sparse, multi).unwrap();
            let ours_t = t.elapsed();
            let t = std::time::Instant::now();
            let (masks, ious) = theirs.forward(&emb_nchw, &pe_nchw, &sparse, &dense_nchw, multi).unwrap();
            let theirs_t = t.elapsed();
            let ious: Vec<f32> = ious.flatten_all().unwrap().to_vec1().unwrap();
            let best = ious.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            let want: Vec<f32> = masks.i((0, best)).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            let diff = logits.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            eprintln!("{name}: ours {ours_t:?}, candle {theirs_t:?}; iou {iou:.4} vs {:.4}; max |diff| {diff:.2e} (max |logit| {scale:.1})", ious[best]);
            assert!((iou - ious[best]).abs() < 1e-3, "{name} iou");
            assert!(diff < 1e-3 * scale.max(1.0), "{name} diff {diff}");
        }
    }
}
