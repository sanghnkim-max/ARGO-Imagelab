//! TinyViT-5M, MobileSAM's image encoder, laid out for a multi-core phone CPU.
//!
//! Same weights, tensor names and arithmetic as candle-transformers 0.9.2's
//! `segment_anything::tiny_vit` (which this is adapted from; Apache-2.0/MIT), so the
//! `mobile_sam-tiny-vitt.safetensors` checkpoint loads unchanged and the embedding matches it to
//! float rounding. What changes is where the time goes; candle's version keeps most cores idle:
//!
//! * batch norms are folded into their convolutions at load, instead of four element-wise passes
//!   over every activation;
//! * a depthwise 3×3 convolution is one parallel kernel (candle runs a grouped convolution as one
//!   small convolution per channel, one after another);
//! * bias, GELU and residual adds run as one fused parallel pass after each matmul (candle's
//!   element-wise ops run on a single thread);
//! * windowed attention is one parallel kernel over windows, with the relative-position bias and
//!   the softmax fused in (candle issues a separate small matmul per window and head);
//! * activations stay in pixel-major (NHWC) layout from the RGB input to the neck, so the conv
//!   and transformer stages never transpose between layouts.
//!
//! Matmuls stay on candle's multithreaded gemm.

use candle_core::{CpuStorage, D, DType, Device, IndexOp, Result, Storage, Tensor, bail};
use candle_nn::VarBuilder;
use rayon::prelude::*;

/// Encoder input side (the proxy is padded to this square).
pub const IMG: usize = 1024;
/// Embedding grid side (`IMG / 16`).
pub const GRID: usize = 64;
/// Head width in every TinyViT-5M stage (128/4, 160/5, 320/10).
const HEAD: usize = 32;
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];

/// Contiguous f32 view of a CPU tensor (no copy when it is already contiguous).
pub(crate) fn with_f32<R>(t: &Tensor, f: impl FnOnce(&[f32]) -> Result<R>) -> Result<R> {
    let t = t.contiguous()?;
    let (s, l) = t.storage_and_layout();
    let Storage::Cpu(CpuStorage::F32(v)) = &*s else { bail!("TinyViT expects CPU f32 tensors") };
    let Some((a, b)) = l.contiguous_offsets() else { bail!("TinyViT expects contiguous tensors") };
    f(v.get(a..b).unwrap_or_default())
}

pub(crate) fn vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_dtype(DType::F32)?.to_vec1()
}

/// tanh as a clamped 13/6 rational polynomial (Eigen's `generic_fast_tanh_float`): within a few
/// ulp of `f32::tanh`, branch-free, so the loops around it vectorize (libm's `tanhf` does not,
/// and was a third of the encoder's time).
#[inline]
fn tanh(x: f32) -> f32 {
    let x = x.clamp(-7.905_311, 7.905_311);
    let x2 = x * x;
    let mut p = x2.mul_add(-2.760_768_5e-16, 2.000_188e-13);
    p = x2.mul_add(p, -8.604_672e-11);
    p = x2.mul_add(p, 5.122_297e-8);
    p = x2.mul_add(p, 1.485_722_4e-5);
    p = x2.mul_add(p, 6.372_619e-4);
    p = x2.mul_add(p, 4.893_524_6e-3);
    let mut q = x2.mul_add(1.198_258_4e-6, 1.185_347_1e-4);
    q = x2.mul_add(q, 2.268_434_6e-3);
    q = x2.mul_add(q, 4.893_525e-3);
    x * p / q
}

/// candle's `Tensor::gelu` (the tanh approximation), so the result matches candle's encoder.
#[inline]
pub(crate) fn gelu(v: f32) -> f32 {
    const SQRT_TWO_OVER_PI: f32 = 0.797_884_6;
    0.5 * v * (1.0 + tanh(SQRT_TWO_OVER_PI * v * (1.0 + 0.044_715 * v * v)))
}

/// e^x for x ≤ 0 (softmax after subtracting the row max): Cephes' range reduction and degree-6
/// polynomial, ~1 ulp, branch-free so the softmax loop vectorizes (libm's `expf` does not).
#[inline]
pub(crate) fn exp_nonpositive(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.0, 0.0);
    let n = (x * LOG2E).round();
    let r = n.mul_add(-LN2_LO, n.mul_add(-LN2_HI, x));
    let mut p = r.mul_add(1.987_569_1e-4, 1.398_199_9e-3);
    p = r.mul_add(p, 8.333_452e-3);
    p = r.mul_add(p, 4.166_579_6e-2);
    p = r.mul_add(p, 1.666_666_5e-1);
    p = r.mul_add(p, 0.5);
    let y = (p * r).mul_add(r, r) + 1.0;
    // 2^n by building the float's exponent (n ∈ [-126, 0] here).
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

pub(crate) fn gelu_all(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x = gelu(*x));
}

/// What runs after a matmul / convolution, fused into one pass over the output.
#[derive(Clone, Copy, Default)]
pub(crate) struct Ep<'a> {
    pub gelu: bool,
    pub relu: bool,
    pub residual: Option<&'a Tensor>,
    pub gelu_after_residual: bool,
}

/// `y` (rows × c) + per-column bias, then the [`Ep`] steps; parallel over row blocks.
pub(crate) fn epilogue(y: &Tensor, bias: Option<&[f32]>, ep: Ep) -> Result<Tensor> {
    let shape = y.shape().clone();
    let c = y.dim(D::Minus1)?;
    if bias.is_some_and(|b| b.len() != c) {
        bail!("bias has {} values for {c} channels", bias.map_or(0, <[f32]>::len));
    }
    let block = c * (16_384 / c).max(1);
    let run = |v: &[f32], r: Option<&[f32]>| {
        let mut out = vec![0f32; v.len()];
        out.par_chunks_mut(block).enumerate().for_each(|(i, o)| {
            let src = &v[i * block..i * block + o.len()];
            let res = r.map(|r| &r[i * block..i * block + o.len()]);
            // One step at a time over each row (a row stays in L1), so each loop vectorizes.
            for (row, (orow, srow)) in o.chunks_mut(c).zip(src.chunks(c)).enumerate() {
                match bias {
                    Some(b) => orow.iter_mut().zip(srow).zip(b).for_each(|((o, s), b)| *o = s + b),
                    None => orow.copy_from_slice(srow),
                }
                if ep.gelu {
                    gelu_all(orow);
                }
                if ep.relu {
                    orow.iter_mut().for_each(|v| *v = v.max(0.0));
                }
                if let Some(res) = res {
                    orow.iter_mut().zip(&res[row * c..(row + 1) * c]).for_each(|(o, r)| *o += r);
                }
                if ep.gelu_after_residual {
                    gelu_all(orow);
                }
            }
        });
        out
    };
    let out = with_f32(y, |v| match ep.residual {
        Some(r) => with_f32(r, |rv| {
            if rv.len() != v.len() {
                bail!("residual has {} values for {}", rv.len(), v.len());
            }
            Ok(run(v, Some(rv)))
        }),
        None => Ok(run(v, None)),
    })?;
    Tensor::from_vec(out, shape, &Device::Cpu)
}

/// A 1×1 convolution (or linear layer) over pixel-major rows: `x (rows × cin) · w + b`.
pub(crate) struct Pointwise {
    /// cin × cout.
    pub w: Tensor,
    pub b: Option<Vec<f32>>,
}

impl Pointwise {
    pub fn forward(&self, x: &Tensor, ep: Ep) -> Result<Tensor> {
        epilogue(&x.matmul(&self.w)?, self.b.as_deref(), ep)
    }
}

/// Depthwise 3×3, padding 1, over an NHWC image; weights as [tap][channel].
struct Depthwise {
    w: Vec<f32>,
    b: Vec<f32>,
    c: usize,
    stride: usize,
}

impl Depthwise {
    /// `x` is `h·w × c`; returns `(oh·ow × c, oh, ow)`.
    fn forward(&self, x: &Tensor, h: usize, w: usize, act: bool) -> Result<(Tensor, usize, usize)> {
        let (c, s) = (self.c, self.stride);
        let (oh, ow) = ((h - 1) / s + 1, (w - 1) / s + 1);
        let out = with_f32(x, |xv| {
            if xv.len() != h * w * c {
                bail!("depthwise input is {} values, not {h}×{w}×{c}", xv.len());
            }
            let mut out = vec![0f32; oh * ow * c];
            out.par_chunks_mut(ow * c).enumerate().for_each(|(oy, orow)| {
                for ox in 0..ow {
                    let o = &mut orow[ox * c..(ox + 1) * c];
                    o.copy_from_slice(&self.b);
                    for ky in 0..3 {
                        let Some(iy) = (oy * s + ky).checked_sub(1).filter(|&iy| iy < h) else { continue };
                        for kx in 0..3 {
                            let Some(ix) = (ox * s + kx).checked_sub(1).filter(|&ix| ix < w) else { continue };
                            let px = &xv[(iy * w + ix) * c..(iy * w + ix + 1) * c];
                            let k = &self.w[(ky * 3 + kx) * c..(ky * 3 + kx + 1) * c];
                            for ((ov, pv), kv) in o.iter_mut().zip(px).zip(k) {
                                *ov += pv * kv;
                            }
                        }
                    }
                    if act {
                        gelu_all(o);
                    }
                }
            });
            Ok(out)
        })?;
        Ok((Tensor::from_vec(out, (oh * ow, c), &Device::Cpu)?, oh, ow))
    }
}

/// Dense 3×3, padding 1, over an NHWC image: im2col in bands of output rows, then gemm.
struct Dense {
    /// (9·cin) × cout, rows ordered [ky][kx][ci].
    w: Tensor,
    b: Option<Vec<f32>>,
    cin: usize,
    stride: usize,
}

impl Dense {
    fn forward(&self, x: &[f32], h: usize, w: usize, act: bool) -> Result<(Tensor, usize, usize)> {
        let (cin, s) = (self.cin, self.stride);
        if x.len() != h * w * cin {
            bail!("conv input is {} values, not {h}×{w}×{cin}", x.len());
        }
        let (oh, ow) = ((h - 1) / s + 1, (w - 1) / s + 1);
        let k = 9 * cin;
        // Bands of ≤ ~4M im2col values: bounded memory, still large gemms.
        let band = ((4 << 20) / (ow * k)).clamp(1, oh);
        let mut parts = Vec::with_capacity(oh.div_ceil(band));
        for y0 in (0..oh).step_by(band) {
            let rows = band.min(oh - y0);
            let mut col = vec![0f32; rows * ow * k];
            col.par_chunks_mut(ow * k).enumerate().for_each(|(r, crow)| {
                let oy = y0 + r;
                for ox in 0..ow {
                    let cell = &mut crow[ox * k..(ox + 1) * k];
                    for ky in 0..3 {
                        let Some(iy) = (oy * s + ky).checked_sub(1).filter(|&iy| iy < h) else { continue };
                        for kx in 0..3 {
                            let Some(ix) = (ox * s + kx).checked_sub(1).filter(|&ix| ix < w) else { continue };
                            cell[(ky * 3 + kx) * cin..(ky * 3 + kx + 1) * cin].copy_from_slice(&x[(iy * w + ix) * cin..(iy * w + ix + 1) * cin]);
                        }
                    }
                }
            });
            let col = Tensor::from_vec(col, (rows * ow, k), &Device::Cpu)?;
            parts.push(epilogue(&col.matmul(&self.w)?, self.b.as_deref(), Ep { gelu: act, ..Ep::default() })?);
        }
        Ok((Tensor::cat(&parts, 0)?, oh, ow))
    }
}

/// Conv weight `(cout, cin/groups, k, k)` and its batch norm, folded: `w·s`, `β − μ·s` with
/// `s = γ / √(σ² + ε)`.
fn conv_bn(vb: &VarBuilder, cin: usize, cout: usize, k: usize, groups: usize) -> Result<(Tensor, Vec<f32>)> {
    let w = vb.pp("c").get((cout, cin / groups, k, k), "weight")?;
    let bn = vb.pp("bn");
    let mean = bn.get(cout, "running_mean")?;
    let var = bn.get(cout, "running_var")?;
    let gamma = bn.get(cout, "weight")?;
    let beta = bn.get(cout, "bias")?;
    let scale = gamma.div(&var.affine(1.0, 1e-5)?.sqrt()?)?;
    let bias = beta.sub(&mean.mul(&scale)?)?;
    Ok((w.broadcast_mul(&scale.reshape((cout, 1, 1, 1))?)?, vec_f32(&bias)?))
}

fn pointwise_bn(vb: VarBuilder, cin: usize, cout: usize) -> Result<Pointwise> {
    let (w, b) = conv_bn(&vb, cin, cout, 1, 1)?;
    Ok(Pointwise { w: w.reshape((cout, cin))?.t()?.contiguous()?, b: Some(b) })
}

fn depthwise_bn(vb: VarBuilder, c: usize, stride: usize) -> Result<Depthwise> {
    let (w, b) = conv_bn(&vb, c, c, 3, c)?;
    // (c, 1, 3, 3) → [tap][channel].
    let w = vec_f32(&w.reshape((c, 9))?.t()?.contiguous()?)?;
    Ok(Depthwise { w, b, c, stride })
}

/// `(cout, cin, 3, 3)` → `(9·cin) × cout`, rows [ky][kx][ci].
fn dense_weight(w: &Tensor) -> Result<Tensor> {
    let (cout, cin, _, _) = w.dims4()?;
    w.permute((2, 3, 1, 0))?.contiguous()?.reshape((9 * cin, cout))
}

fn dense_bn(vb: VarBuilder, cin: usize, cout: usize, stride: usize) -> Result<Dense> {
    let (w, b) = conv_bn(&vb, cin, cout, 3, 1)?;
    Ok(Dense { w: dense_weight(&w)?, b: Some(b), cin, stride })
}

pub(crate) fn linear(vb: VarBuilder, cin: usize, cout: usize) -> Result<Pointwise> {
    let w = vb.get((cout, cin), "weight")?;
    let b = vec_f32(&vb.get(cout, "bias")?)?;
    Ok(Pointwise { w: w.t()?.contiguous()?, b: Some(b) })
}

pub(crate) struct Norm {
    w: Tensor,
    b: Tensor,
    eps: f32,
}

impl Norm {
    pub fn new(vb: VarBuilder, c: usize, eps: f32) -> Result<Self> {
        Ok(Self { w: vb.get(c, "weight")?, b: vb.get(c, "bias")?, eps })
    }
    /// Normalise each row (pixel) over its channels.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm(&x.contiguous()?, &self.w, &self.b, self.eps)
    }
}

/// Inverted residual: expand 1×1 → GELU → depthwise 3×3 → GELU → project 1×1, + input → GELU.
struct MbConv {
    expand: Pointwise,
    dw: Depthwise,
    project: Pointwise,
}

impl MbConv {
    fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let y = self.expand.forward(x, Ep { gelu: true, ..Ep::default() })?;
        let (y, _, _) = self.dw.forward(&y, h, w, true)?;
        self.project.forward(&y, Ep { residual: Some(x), gelu_after_residual: true, ..Ep::default() })
    }
}

/// Downsampling between stages: 1×1 → GELU → depthwise 3×3 (stride 1 or 2) → GELU → 1×1.
struct PatchMerging {
    conv1: Pointwise,
    dw: Depthwise,
    conv3: Pointwise,
}

impl PatchMerging {
    fn new(vb: VarBuilder, dim: usize, out: usize) -> Result<Self> {
        let stride = if [320, 448, 576].contains(&out) { 1 } else { 2 };
        Ok(Self { conv1: pointwise_bn(vb.pp("conv1"), dim, out)?, dw: depthwise_bn(vb.pp("conv2"), out, stride)?, conv3: pointwise_bn(vb.pp("conv3"), out, out)? })
    }
    fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<(Tensor, usize, usize)> {
        let y = self.conv1.forward(x, Ep { gelu: true, ..Ep::default() })?;
        let (y, oh, ow) = self.dw.forward(&y, h, w, true)?;
        Ok((self.conv3.forward(&y, Ep::default())?, oh, ow))
    }
}

struct Attention {
    norm: Norm,
    qkv: Pointwise,
    proj: Pointwise,
    /// Relative-position bias, `heads × n × n`.
    ab: Vec<f32>,
    heads: usize,
    key_dim: usize,
    scale: f32,
}

impl Attention {
    fn new(vb: VarBuilder, dim: usize, heads: usize, window: usize) -> Result<Self> {
        let key_dim = dim / heads;
        let points: Vec<(i64, i64)> = (0..window as i64).flat_map(|x| (0..window as i64).map(move |y| (x, y))).collect();
        let mut offsets = std::collections::HashMap::new();
        let mut idxs = Vec::with_capacity(points.len() * points.len());
        for &(x1, y1) in &points {
            for &(x2, y2) in &points {
                let l = offsets.len();
                idxs.push(*offsets.entry(((x2 - x1).abs(), (y2 - y1).abs())).or_insert(l) as u32);
            }
        }
        let biases = vb.get((heads, offsets.len()), "attention_biases")?;
        let ab = vec_f32(&biases.index_select(&Tensor::new(idxs, &Device::Cpu)?, 1)?)?;
        Ok(Self {
            norm: Norm::new(vb.pp("norm"), dim, 1e-5)?,
            qkv: linear(vb.pp("qkv"), dim, 3 * dim)?,
            proj: linear(vb.pp("proj"), dim, dim)?,
            ab,
            heads,
            key_dim,
            scale: 1.0 / (key_dim as f32).sqrt(),
        })
    }

    /// `x`: `windows·n × dim` (pixels grouped by window) → same shape.
    fn forward(&self, x: &Tensor, windows: usize, n: usize) -> Result<Tensor> {
        let qkv = self.qkv.forward(&self.norm.forward(x)?, Ep::default())?;
        let (heads, kd) = (self.heads, self.key_dim);
        let d = kd;
        let stride = heads * 3 * kd;
        let out = with_f32(&qkv, |qv| {
            if qv.len() != windows * n * stride || self.ab.len() != heads * n * n || d != HEAD {
                bail!("attention shapes do not match ({} values, head {d})", qv.len());
            }
            // One task per (window, head): enough tasks for every core even when a stage has
            // few windows. Each writes its own `n × d` block; the blocks are interleaved back
            // into pixel rows below.
            let mut per_head = vec![0f32; windows * heads * n * d];
            per_head.par_chunks_mut(n * d).enumerate().for_each(|(task, o)| {
                let (wi, h) = (task / heads, task % heads);
                head_attention(&qv[wi * n * stride..(wi + 1) * n * stride], stride, h * 3 * kd, &self.ab[h * n * n..(h + 1) * n * n], self.scale, n, o);
            });
            let mut out = vec![0f32; windows * n * heads * d];
            out.par_chunks_mut(n * heads * d).enumerate().for_each(|(wi, o)| {
                for h in 0..heads {
                    let src = &per_head[(wi * heads + h) * n * d..(wi * heads + h + 1) * n * d];
                    for (i, row) in src.chunks_exact(d).enumerate() {
                        o[i * heads * d + h * d..i * heads * d + (h + 1) * d].copy_from_slice(row);
                    }
                }
            });
            Ok(out)
        })?;
        let out = Tensor::from_vec(out, (windows * n, heads * d), &Device::Cpu)?;
        self.proj.forward(&out, Ep::default())
    }
}

/// Softmax attention for one head of one window. `qkv` holds the window's `n` rows of `stride`
/// values; this head's `[q | k | v]` (each [`HEAD`] wide) starts at `base`. `ab` is the head's
/// `n × n` position bias; `out` receives `n × HEAD`.
fn head_attention(qkv: &[f32], stride: usize, base: usize, ab: &[f32], scale: f32, n: usize, out: &mut [f32]) {
    const D: usize = HEAD;
    // Keys transposed and padded to whole 8-wide tiles; values contiguous.
    let np = n.div_ceil(8) * 8;
    let mut kt = vec![0f32; D * np];
    let mut vv = vec![0f32; n * D];
    for j in 0..n {
        let row = &qkv[j * stride + base..j * stride + base + 3 * D];
        for k in 0..D {
            kt[k * np + j] = row[D + k];
        }
        vv[j * D..(j + 1) * D].copy_from_slice(&row[2 * D..3 * D]);
    }
    let q = |i: usize| -> &[f32] { &qkv[i * stride + base..i * stride + base + D] };
    // Scores for up to 4 queries at a time: a 4×8 tile accumulates in registers over the key
    // dimension, each key load serving four queries.
    let mut s = vec![0f32; 4 * np];
    let mut i0 = 0;
    while i0 < n {
        let m = 4.min(n - i0);
        let qs = [q(i0), q((i0 + 1).min(n - 1)), q((i0 + 2).min(n - 1)), q((i0 + 3).min(n - 1))];
        for t in 0..np / 8 {
            let mut acc = [[0f32; 8]; 4];
            for k in 0..D {
                let kr = &kt[k * np + t * 8..k * np + t * 8 + 8];
                for (a, qr) in acc.iter_mut().zip(&qs) {
                    let qk = qr[k];
                    for (av, kv) in a.iter_mut().zip(kr) {
                        *av = qk.mul_add(*kv, *av);
                    }
                }
            }
            for (r, a) in acc.iter().enumerate() {
                s[r * np + t * 8..r * np + t * 8 + 8].copy_from_slice(a);
            }
        }
        for r in 0..m {
            let i = i0 + r;
            let row = &mut s[r * np..r * np + n];
            for (sj, b) in row.iter_mut().zip(&ab[i * n..(i + 1) * n]) {
                *sj = sj.mul_add(scale, *b);
            }
            let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            row.iter_mut().for_each(|v| *v = exp_nonpositive(*v - mx));
            let inv = 1.0 / row.iter().sum::<f32>();
            let mut acc = [0f32; D];
            for (j, p) in row.iter().enumerate() {
                let p = p * inv;
                for (a, vj) in acc.iter_mut().zip(&vv[j * D..(j + 1) * D]) {
                    *a = p.mul_add(*vj, *a);
                }
            }
            out[i * D..(i + 1) * D].copy_from_slice(&acc);
        }
        i0 += 4;
    }
}

struct Block {
    attn: Attention,
    local: Depthwise,
    norm: Norm,
    fc1: Pointwise,
    fc2: Pointwise,
    window: usize,
}

impl Block {
    fn new(vb: VarBuilder, dim: usize, heads: usize, window: usize) -> Result<Self> {
        let mlp = vb.pp("mlp");
        Ok(Self {
            attn: Attention::new(vb.pp("attn"), dim, heads, window)?,
            local: depthwise_bn(vb.pp("local_conv"), dim, 1)?,
            norm: Norm::new(mlp.pp("norm"), dim, 1e-5)?,
            fc1: linear(mlp.pp("fc1"), dim, 4 * dim)?,
            fc2: linear(mlp.pp("fc2"), 4 * dim, dim)?,
            window,
        })
    }

    /// `x`: `h·w × c` pixels.
    fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (_, c) = x.dims2()?;
        let ws = self.window;
        let (pb, pr) = ((ws - h % ws) % ws, (ws - w % ws) % ws);
        let (ph, pw) = (h + pb, w + pr);
        let (nh, nw) = (ph / ws, pw / ws);
        // Pad to whole windows (zeros, before the attention's norm — as candle does), then
        // group pixels by window.
        let mut xs = x.reshape((h, w, c))?;
        if pb > 0 {
            xs = xs.pad_with_zeros(0, 0, pb)?;
        }
        if pr > 0 {
            xs = xs.pad_with_zeros(1, 0, pr)?;
        }
        let xs = xs.reshape((nh, ws, nw, ws, c))?.transpose(1, 2)?.contiguous()?.reshape((nh * nw * ws * ws, c))?;
        let a = self.attn.forward(&xs, nh * nw, ws * ws)?;
        let mut a = a.reshape((nh, nw, ws, ws, c))?.transpose(1, 2)?.reshape((ph, pw, c))?;
        if pb > 0 || pr > 0 {
            a = a.i((..h, ..w))?;
        }
        let x = (a.contiguous()?.reshape((h * w, c))? + x)?;
        let (x, _, _) = self.local.forward(&x, h, w, false)?;
        let y = self.fc1.forward(&self.norm.forward(&x)?, Ep { gelu: true, ..Ep::default() })?;
        self.fc2.forward(&y, Ep { residual: Some(&x), ..Ep::default() })
    }
}

struct Stage {
    blocks: Vec<Block>,
    down: Option<PatchMerging>,
}

/// MobileSAM's image encoder: an RGB proxy → the 1×256×64×64 image embedding.
pub struct TinyViT {
    embed1: Dense,
    embed2: Dense,
    conv: Vec<MbConv>,
    conv_down: PatchMerging,
    stages: Vec<Stage>,
    neck1: Pointwise,
    neck_ln1: Norm,
    neck2: Dense,
    neck_ln2: Norm,
}

impl TinyViT {
    /// TinyViT-5M from `image_encoder.*` weights (MobileSAM's checkpoint layout).
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let dims = [64, 128, 160, 320];
        let depths = [2, 2, 6, 2];
        let heads = [2, 4, 5, 10];
        let windows = [7, 7, 14, 7];
        let pe = vb.pp("patch_embed").pp("seq");
        let embed1 = dense_bn(pe.pp("0"), 3, dims[0] / 2, 2)?;
        let embed2 = dense_bn(pe.pp("2"), dims[0] / 2, dims[0], 2)?;
        let l0 = vb.pp("layers").pp("0");
        let conv = (0..depths[0])
            .map(|i| {
                let b = l0.pp("blocks").pp(i);
                let hidden = dims[0] * 4;
                Ok(MbConv { expand: pointwise_bn(b.pp("conv1"), dims[0], hidden)?, dw: depthwise_bn(b.pp("conv2"), hidden, 1)?, project: pointwise_bn(b.pp("conv3"), hidden, dims[0])? })
            })
            .collect::<Result<Vec<_>>>()?;
        let conv_down = PatchMerging::new(l0.pp("downsample"), dims[0], dims[1])?;
        let mut stages = Vec::new();
        for i in 1..dims.len() {
            let l = vb.pp("layers").pp(i);
            let blocks = (0..depths[i]).map(|j| Block::new(l.pp("blocks").pp(j), dims[i], heads[i], windows[i])).collect::<Result<Vec<_>>>()?;
            let down = if i + 1 < dims.len() { Some(PatchMerging::new(l.pp("downsample"), dims[i], dims[i + 1])?) } else { None };
            stages.push(Stage { blocks, down });
        }
        let last = dims[dims.len() - 1];
        let neck = vb.pp("neck");
        let n1 = neck.pp("0").get((256, last, 1, 1), "weight")?;
        let n2 = neck.pp("2").get((256, 256, 3, 3), "weight")?;
        Ok(Self {
            embed1,
            embed2,
            conv,
            conv_down,
            stages,
            neck1: Pointwise { w: n1.reshape((256, last))?.t()?.contiguous()?, b: None },
            neck_ln1: Norm::new(neck.pp("1"), 256, 1e-6)?,
            neck2: Dense { w: dense_weight(&n2)?, b: None, cin: 256, stride: 1 },
            neck_ln2: Norm::new(neck.pp("3"), 256, 1e-6)?,
        })
    }

    /// RGB proxy (`w×h` u8, both ≤ [`IMG`]) → embedding `1×256×64×64`. The proxy is normalised
    /// and zero-padded at the bottom/right to the encoder's square, as SAM does. (Tests compare
    /// it with candle's encoder; the app uses [`Self::encode_tokens`].)
    #[cfg(test)]
    pub fn encode(&self, rgb: &[u8], w: usize, h: usize) -> Result<Tensor> {
        // Pixel-major → channel-major, the layout candle's SAM takes.
        self.encode_tokens(rgb, w, h)?.t()?.contiguous()?.reshape((1, 256, GRID, GRID))
    }

    /// Like [`Self::encode`], pixel-major: `4096 × 256` (row = embedding cell), the layout
    /// [`crate::sam_decoder::MaskDecoder`] takes.
    pub fn encode_tokens(&self, rgb: &[u8], w: usize, h: usize) -> Result<Tensor> {
        if w > IMG || h > IMG || rgb.len() != w * h * 3 {
            bail!("proxy {w}x{h} does not fit the {IMG} px encoder");
        }
        let mut input = vec![0f32; IMG * IMG * 3];
        input.par_chunks_mut(IMG * 3).take(h).enumerate().for_each(|(y, row)| {
            for (x, px) in row.chunks_mut(3).take(w).enumerate() {
                for c in 0..3 {
                    px[c] = (f32::from(rgb[(y * w + x) * 3 + c]) - MEAN[c]) / STD[c];
                }
            }
        });
        let (x, h, w) = self.embed1.forward(&input, IMG, IMG, true)?;
        let (x, h, w) = with_f32(&x, |v| self.embed2.forward(v, h, w, false))?;
        let mut x = x;
        for b in &self.conv {
            x = b.forward(&x, h, w)?;
        }
        let (mut x, mut h, mut w) = self.conv_down.forward(&x, h, w)?;
        for s in &self.stages {
            for b in &s.blocks {
                x = b.forward(&x, h, w)?;
            }
            if let Some(d) = &s.down {
                (x, h, w) = d.forward(&x, h, w)?;
            }
        }
        if (h, w) != (GRID, GRID) {
            bail!("TinyViT produced a {h}×{w} grid, not {GRID}×{GRID}");
        }
        let x = self.neck_ln1.forward(&self.neck1.forward(&x, Ep::default())?)?;
        let (x, _, _) = with_f32(&x, |v| self.neck2.forward(v, h, w, false))?;
        self.neck_ln2.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matches candle's TinyViT on the real checkpoint (set IMAGELAB_MOBILESAM_WEIGHTS).
    #[test]
    fn matches_candle_encoder() {
        let Ok(path) = std::env::var("IMAGELAB_MOBILESAM_WEIGHTS") else {
            eprintln!("IMAGELAB_MOBILESAM_WEIGHTS not set; skipped");
            return;
        };
        let dev = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(std::fs::read(path).unwrap(), DType::F32, &dev).unwrap();
        let ours = TinyViT::new(vb.pp("image_encoder")).unwrap();
        let theirs = candle_transformers::models::segment_anything::tiny_vit::tiny_vit_5m(vb.pp("image_encoder")).unwrap();
        // A 900×700 proxy with structure (gradients and blocks), padded like a real one.
        let (w, h) = (900usize, 700usize);
        let rgb: Vec<u8> = (0..w * h * 3)
            .map(|i| {
                let (p, c) = (i / 3, i % 3);
                let (x, y) = (p % w, p / w);
                ((x * (c + 1) / 4 + y / 3 + if (x / 90 + y / 70) % 2 == 0 { 60 } else { 0 }) % 256) as u8
            })
            .collect();
        let t = std::time::Instant::now();
        let a = ours.encode(&rgb, w, h).unwrap();
        let ours_time = t.elapsed();
        let img = Tensor::from_slice(&rgb, (h, w, 3), &dev).unwrap().permute((2, 0, 1)).unwrap().to_dtype(DType::F32).unwrap();
        let mean = Tensor::new(&MEAN, &dev).unwrap().reshape((3, 1, 1)).unwrap();
        let std = Tensor::new(&STD, &dev).unwrap().reshape((3, 1, 1)).unwrap();
        let img = img.broadcast_sub(&mean).unwrap().broadcast_div(&std).unwrap();
        let img = img.pad_with_zeros(1, 0, IMG - h).unwrap().pad_with_zeros(2, 0, IMG - w).unwrap();
        let t = std::time::Instant::now();
        let b = candle_core::Module::forward(&theirs, &img.unsqueeze(0).unwrap()).unwrap();
        let theirs_time = t.elapsed();
        assert_eq!(a.dims(), b.dims());
        let diff = (&a - &b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        let scale = b.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        eprintln!("ours {ours_time:?}, candle {theirs_time:?}; max |diff| {diff:.2e} (max |value| {scale:.2})");
        assert!(diff < 1e-3 * scale.max(1.0), "max diff {diff}");
    }

    /// Repeated encodes for profiling (IMAGELAB_MOBILESAM_WEIGHTS and IMAGELAB_BENCH=<runs>).
    #[test]
    fn bench_encode() {
        let (Ok(path), Ok(runs)) = (std::env::var("IMAGELAB_MOBILESAM_WEIGHTS"), std::env::var("IMAGELAB_BENCH")) else { return };
        let vb = VarBuilder::from_buffered_safetensors(std::fs::read(path).unwrap(), DType::F32, &Device::Cpu).unwrap();
        let enc = TinyViT::new(vb.pp("image_encoder")).unwrap();
        let (w, h) = (1008usize, 756usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        for _ in 0..runs.parse::<usize>().unwrap_or(10) {
            let t = std::time::Instant::now();
            enc.encode(&rgb, w, h).unwrap();
            eprintln!("encode {:?}", t.elapsed());
        }
    }
}
