//! The TinyViT encoder on the GPU (wgpu compute: Vulkan on Android, Metal on macOS).
//!
//! The same network as the CPU encoder in the parent module, run from its already-loaded
//! weights (batch norms folded): each CPU step has a compute kernel here, and an encode is one
//! command submission and one read-back of the `4096 × 256` embedding.
//!
//! * gemm: no shared memory, vec4 loads straight from the buffers, a 4-row × 8-column tile per
//!   thread; bias, GELU/ReLU, residual and a second GELU fused in. ~1.1 TFLOP/s on an
//!   Adreno 840, ~5–15× the same phone's CPU gemm.
//! * im2col (dense 3×3 convs), depthwise 3×3 (vec4 over channels), layer norm, window
//!   partition / un-partition (+ residual) and windowed attention with an online softmax (one
//!   thread per query and head, no score buffer).
//!
//! Intermediate buffers come from a pool and return to it as soon as the recording no longer
//! needs them; wgpu orders the reuse after the dispatches that read them.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::mpsc;

use candle_core::{Device, Result, Tensor, bail};
use wgpu::util::DeviceExt;

use super::{Attention, Block, Dense, Depthwise, GRID, IMG, MbConv, Norm, PatchMerging, Pointwise, TinyViT, normalized_input, vec_f32};

const HEAD: usize = super::HEAD;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("GPU: {e}"))
}

const GELU: &str = r#"
fn gelu(x: vec4f) -> vec4f {
    let t = tanh(clamp(0.7978846 * x * (1.0 + 0.044715 * x * x), vec4f(-10.0), vec4f(10.0)));
    return 0.5 * x * (1.0 + t);
}
// Linear index of this invocation in a (possibly 2-D) grid of 64-wide workgroups.
fn flat(wg: vec3u, nwg: vec3u, l: u32) -> u32 { return (wg.y * nwg.x + wg.x) * 64u + l; }
"#;

const GEMM: &str = r#"
struct P { m: u32, n: u32, k: u32, flags: u32 }
@group(0) @binding(0) var<storage, read> a: array<vec4f>;
@group(0) @binding(1) var<storage, read> b: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> c: array<vec4f>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> bias: array<vec4f>;
@group(0) @binding(5) var<storage, read> res: array<vec4f>;

fn ep(v: vec4f, col4: u32, idx: u32) -> vec4f {
    var x = v;
    if ((p.flags & 1u) != 0u) { x += bias[col4]; }
    if ((p.flags & 2u) != 0u) { x = gelu(x); }
    if ((p.flags & 4u) != 0u) { x = max(x, vec4f(0.0)); }
    if ((p.flags & 8u) != 0u) { x += res[idx]; }
    if ((p.flags & 16u) != 0u) { x = gelu(x); }
    return x;
}

@compute @workgroup_size(16, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
    let n4 = p.n / 4u;
    let k4 = p.k / 4u;
    let col4 = g.x * 2u;
    let row = g.y * 4u;
    if (col4 >= n4 || row >= p.m) { return; }
    let r1 = min(row + 1u, p.m - 1u);
    let r2 = min(row + 2u, p.m - 1u);
    let r3 = min(row + 3u, p.m - 1u);
    let c1 = min(col4 + 1u, n4 - 1u);
    var x0 = vec4f(0.0); var y0 = vec4f(0.0);
    var x1 = vec4f(0.0); var y1 = vec4f(0.0);
    var x2 = vec4f(0.0); var y2 = vec4f(0.0);
    var x3 = vec4f(0.0); var y3 = vec4f(0.0);
    for (var kk = 0u; kk < k4; kk++) {
        let a0 = a[row * k4 + kk];
        let a1 = a[r1 * k4 + kk];
        let a2 = a[r2 * k4 + kk];
        let a3 = a[r3 * k4 + kk];
        for (var s = 0u; s < 4u; s++) {
            let bk = (kk * 4u + s) * n4;
            let bx = b[bk + col4];
            let by = b[bk + c1];
            x0 = fma(vec4f(a0[s]), bx, x0); y0 = fma(vec4f(a0[s]), by, y0);
            x1 = fma(vec4f(a1[s]), bx, x1); y1 = fma(vec4f(a1[s]), by, y1);
            x2 = fma(vec4f(a2[s]), bx, x2); y2 = fma(vec4f(a2[s]), by, y2);
            x3 = fma(vec4f(a3[s]), bx, x3); y3 = fma(vec4f(a3[s]), by, y3);
        }
    }
    let two = col4 + 1u < n4;
    var i = row * n4 + col4;
    c[i] = ep(x0, col4, i); if (two) { c[i + 1u] = ep(y0, col4 + 1u, i + 1u); }
    if (row + 1u < p.m) { i = (row + 1u) * n4 + col4; c[i] = ep(x1, col4, i); if (two) { c[i + 1u] = ep(y1, col4 + 1u, i + 1u); } }
    if (row + 2u < p.m) { i = (row + 2u) * n4 + col4; c[i] = ep(x2, col4, i); if (two) { c[i + 1u] = ep(y2, col4 + 1u, i + 1u); } }
    if (row + 3u < p.m) { i = (row + 3u) * n4 + col4; c[i] = ep(x3, col4, i); if (two) { c[i + 1u] = ep(y3, col4 + 1u, i + 1u); } }
}
"#;

/// Dense 3×3 conv input → rows of `kp` taps ([ky][kx][ci], zero-padded to `kp`).
const IM2COL: &str = r#"
struct P { h: u32, w: u32, cin: u32, stride: u32, oh: u32, ow: u32, kp: u32, total: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) l: u32) {
    let idx = flat(wg, nwg, l);
    if (idx >= p.total) { return; }
    let col = idx % p.kp;
    let row = idx / p.kp;
    var v = 0.0;
    if (col < 9u * p.cin) {
        let tap = col / p.cin;
        let ci = col % p.cin;
        let iy = i32((row / p.ow) * p.stride + tap / 3u) - 1;
        let ix = i32((row % p.ow) * p.stride + tap % 3u) - 1;
        if (iy >= 0 && ix >= 0 && iy < i32(p.h) && ix < i32(p.w)) {
            v = x[(u32(iy) * p.w + u32(ix)) * p.cin + ci];
        }
    }
    o[idx] = v;
}
"#;

const DEPTHWISE: &str = r#"
struct P { h: u32, w: u32, c4: u32, stride: u32, oh: u32, ow: u32, act: u32, total: u32 }
@group(0) @binding(0) var<storage, read> x: array<vec4f>;
@group(0) @binding(1) var<storage, read> wt: array<vec4f>;
@group(0) @binding(2) var<storage, read> b: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> o: array<vec4f>;
@group(0) @binding(4) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) l: u32) {
    let idx = flat(wg, nwg, l);
    if (idx >= p.total) { return; }
    let c = idx % p.c4;
    let px = idx / p.c4;
    let oy = px / p.ow;
    let ox = px % p.ow;
    var acc = b[c];
    for (var ky = 0u; ky < 3u; ky++) {
        let iy = i32(oy * p.stride + ky) - 1;
        if (iy < 0 || iy >= i32(p.h)) { continue; }
        for (var kx = 0u; kx < 3u; kx++) {
            let ix = i32(ox * p.stride + kx) - 1;
            if (ix < 0 || ix >= i32(p.w)) { continue; }
            acc = fma(x[(u32(iy) * p.w + u32(ix)) * p.c4 + c], wt[(ky * 3u + kx) * p.c4 + c], acc);
        }
    }
    if (p.act != 0u) { acc = gelu(acc); }
    o[idx] = acc;
}
"#;

const LAYERNORM: &str = r#"
struct P { rows: u32, c4: u32, eps: f32, pad: u32 }
@group(0) @binding(0) var<storage, read> x: array<vec4f>;
@group(0) @binding(1) var<storage, read> gamma: array<vec4f>;
@group(0) @binding(2) var<storage, read> beta: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> o: array<vec4f>;
@group(0) @binding(4) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) l: u32) {
    let r = flat(wg, nwg, l);
    if (r >= p.rows) { return; }
    let base = r * p.c4;
    var s = vec4f(0.0);
    for (var i = 0u; i < p.c4; i++) { s += x[base + i]; }
    let mean = (s.x + s.y + s.z + s.w) / f32(p.c4 * 4u);
    var q = vec4f(0.0);
    for (var i = 0u; i < p.c4; i++) { let d = x[base + i] - vec4f(mean); q = fma(d, d, q); }
    let inv = inverseSqrt((q.x + q.y + q.z + q.w) / f32(p.c4 * 4u) + p.eps);
    for (var i = 0u; i < p.c4; i++) { o[base + i] = (x[base + i] - vec4f(mean)) * inv * gamma[i] + beta[i]; }
}
"#;

/// Pixels (h × w × c) → window-grouped rows, zero-padded to whole windows.
const PARTITION: &str = r#"
struct P { h: u32, w: u32, c4: u32, ws: u32, nw: u32, total: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> x: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> o: array<vec4f>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) l: u32) {
    let idx = flat(wg, nwg, l);
    if (idx >= p.total) { return; }
    let c = idx % p.c4;
    let r = idx / p.c4;
    let win = r / (p.ws * p.ws);
    let t = r % (p.ws * p.ws);
    let y = (win / p.nw) * p.ws + t / p.ws;
    let xx = (win % p.nw) * p.ws + t % p.ws;
    var v = vec4f(0.0);
    if (y < p.h && xx < p.w) { v = x[(y * p.w + xx) * p.c4 + c]; }
    o[idx] = v;
}
"#;

/// Window-grouped rows back to pixels (cropping the padding), plus the residual.
const UNPARTITION: &str = r#"
struct P { h: u32, w: u32, c4: u32, ws: u32, nw: u32, total: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> a: array<vec4f>;
@group(0) @binding(1) var<storage, read> res: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> o: array<vec4f>;
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) l: u32) {
    let idx = flat(wg, nwg, l);
    if (idx >= p.total) { return; }
    let c = idx % p.c4;
    let px = idx / p.c4;
    let y = px / p.w;
    let xx = px % p.w;
    let win = (y / p.ws) * p.nw + xx / p.ws;
    let t = (y % p.ws) * p.ws + xx % p.ws;
    o[idx] = a[(win * p.ws * p.ws + t) * p.c4 + c] + res[idx];
}
"#;

/// Windowed attention, one invocation per (window, head, query): an online softmax over the
/// window's keys with the position bias, 32-wide heads in eight vec4 registers.
fn attention_shader() -> String {
    let regs = |f: &dyn Fn(usize) -> String| (0..8).map(f).collect::<Vec<_>>().join("\n    ");
    format!(
        r#"
struct P {{ windows: u32, n: u32, heads: u32, scale: f32, total: u32, pad0: u32, pad1: u32, pad2: u32 }}
@group(0) @binding(0) var<storage, read> qkv: array<vec4f>;
@group(0) @binding(1) var<storage, read> ab: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<vec4f>;
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) l: u32) {{
    let idx = flat(wg, nwg, l);
    if (idx >= p.total) {{ return; }}
    let i = idx % p.n;
    let rest = idx / p.n;
    let h = rest % p.heads;
    let win = rest / p.heads;
    let stride = p.heads * 24u;
    let qb = (win * p.n + i) * stride + h * 24u;
    {q}
    var m = -3.0e38;
    var sum = 0.0;
    {acc}
    let abrow = (h * p.n + i) * p.n;
    for (var j = 0u; j < p.n; j++) {{
        let kb = (win * p.n + j) * stride + h * 24u + 8u;
        let s = {dot} + ab[abrow + j];
        let mn = max(m, s);
        let corr = exp(m - mn);
        let e = exp(s - mn);
        sum = sum * corr + e;
        {update}
        m = mn;
    }}
    let ob = (win * p.n + i) * p.heads * 8u + h * 8u;
    let inv = 1.0 / sum;
    {write}
}}
"#,
        q = regs(&|k| format!("let q{k} = qkv[qb + {k}u] * p.scale;")),
        acc = regs(&|k| format!("var o{k} = vec4f(0.0);")),
        dot = (0..8).map(|k| format!("dot(q{k}, qkv[kb + {k}u])")).collect::<Vec<_>>().join(" + "),
        update = regs(&|k| format!("o{k} = o{k} * corr + e * qkv[kb + 8u + {k}u];")),
        write = regs(&|k| format!("o[ob + {k}u] = o{k} * inv;")),
    )
}

/// A storage buffer of f32 that goes back to the pool when dropped.
struct Buf {
    buf: Option<wgpu::Buffer>,
    /// f32 values.
    len: usize,
    pool: Rc<RefCell<HashMap<u64, Vec<wgpu::Buffer>>>>,
}

impl Buf {
    fn raw(&self) -> &wgpu::Buffer {
        // Always Some until drop.
        self.buf.as_ref().unwrap_or_else(|| unreachable_buffer())
    }
}

#[allow(clippy::panic)]
fn unreachable_buffer() -> &'static wgpu::Buffer {
    panic!("GPU buffer used after release")
}

impl Drop for Buf {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            self.pool.borrow_mut().entry(b.size()).or_default().push(b);
        }
    }
}

/// An uploaded weight buffer (never pooled).
struct W(wgpu::Buffer);

struct GLin {
    w: W,
    b: Option<W>,
    k: usize,
    n: usize,
}

struct GDw {
    w: W,
    b: W,
    c: usize,
    stride: usize,
}

struct GDense {
    w: W,
    b: Option<W>,
    cin: usize,
    kp: usize,
    n: usize,
    stride: usize,
}

struct GNorm {
    g: W,
    b: W,
    c: usize,
    eps: f32,
}

struct GAttn {
    norm: GNorm,
    qkv: GLin,
    proj: GLin,
    ab: W,
    heads: usize,
    scale: f32,
}

struct GBlock {
    attn: GAttn,
    local: GDw,
    norm: GNorm,
    fc1: GLin,
    fc2: GLin,
    window: usize,
}

struct GMerge {
    conv1: GLin,
    dw: GDw,
    conv3: GLin,
}

struct GMb {
    expand: GLin,
    dw: GDw,
    project: GLin,
}

struct Pipes {
    gemm: wgpu::ComputePipeline,
    im2col: wgpu::ComputePipeline,
    depthwise: wgpu::ComputePipeline,
    layernorm: wgpu::ComputePipeline,
    partition: wgpu::ComputePipeline,
    unpartition: wgpu::ComputePipeline,
    attention: wgpu::ComputePipeline,
}

/// The encoder's weights on the GPU, its kernels compiled, and a device to run them.
pub struct GpuTinyViT {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: Pipes,
    adapter: String,
    /// A 16-byte buffer bound where a kernel's optional input is absent.
    dummy: W,
    embed1: GDense,
    embed2: GDense,
    conv: Vec<GMb>,
    conv_down: GMerge,
    stages: Vec<(Vec<GBlock>, Option<GMerge>)>,
    neck1: GLin,
    neck_ln1: GNorm,
    neck2: GDense,
    neck_ln2: GNorm,
}

/// What follows a gemm, fused (bit flags in the kernel).
#[derive(Clone, Copy, Default)]
struct GEp<'a> {
    gelu: bool,
    relu: bool,
    residual: Option<&'a Buf>,
    gelu_after: bool,
}

impl GpuTinyViT {
    /// The GPU's name, for status and logs.
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// Upload `cpu`'s weights to the first high-performance GPU wgpu finds and compile the
    /// kernels; then check one small gemm against the CPU (a driver that miscompiles fails here,
    /// not in a user's mask).
    pub fn new(cpu: &TinyViT) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(err)?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            bail!("GPU: only a software adapter ({})", info.name);
        }
        let mut limits = wgpu::Limits::default();
        let have = adapter.limits();
        limits.max_storage_buffer_binding_size = have.max_storage_buffer_binding_size;
        limits.max_buffer_size = have.max_buffer_size;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("mobilesam"), required_limits: limits, ..Default::default() })).map_err(err)?;
        // Errors outside an error scope (e.g. a lost device) are reported, not panicked on: the
        // encode that hits one fails its read-back and MobileSAM falls back to the CPU.
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| eprintln!("mobilesam GPU: {e}")));
        let make = |name: &str, src: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(format!("{GELU}\n{src}").into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(name), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipes = Pipes {
            gemm: make("gemm", GEMM),
            im2col: make("im2col", IM2COL),
            depthwise: make("depthwise", DEPTHWISE),
            layernorm: make("layernorm", LAYERNORM),
            partition: make("partition", PARTITION),
            unpartition: make("unpartition", UNPARTITION),
            attention: make("attention", &attention_shader()),
        };
        if let Some(e) = pollster::block_on(scope.pop()) {
            bail!("GPU: kernels did not compile: {e}");
        }
        let up = |v: &[f32]| W(device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(v), usage: wgpu::BufferUsages::STORAGE }));
        let lin = |p: &Pointwise| -> Result<GLin> {
            let (k, n) = p.w.dims2()?;
            Ok(GLin { w: up(&vec_f32(&p.w)?), b: p.b.as_deref().map(up), k, n })
        };
        let dw = |d: &Depthwise| GDw { w: up(&d.w), b: up(&d.b), c: d.c, stride: d.stride };
        let dense = |d: &Dense| -> Result<GDense> {
            let (k, n) = d.w.dims2()?;
            let kp = k.div_ceil(4) * 4;
            let mut w = vec_f32(&d.w)?;
            w.resize(kp * n, 0.0);
            Ok(GDense { w: up(&w), b: d.b.as_deref().map(up), cin: d.cin, kp, n, stride: d.stride })
        };
        let norm = |nm: &Norm| -> Result<GNorm> { Ok(GNorm { g: up(&vec_f32(&nm.w)?), b: up(&vec_f32(&nm.b)?), c: nm.w.dim(0)?, eps: nm.eps }) };
        let attn = |a: &Attention| -> Result<GAttn> {
            if a.key_dim != HEAD {
                bail!("GPU attention takes {HEAD}-wide heads, not {}", a.key_dim);
            }
            Ok(GAttn { norm: norm(&a.norm)?, qkv: lin(&a.qkv)?, proj: lin(&a.proj)?, ab: up(&a.ab), heads: a.heads, scale: a.scale })
        };
        let block = |b: &Block| -> Result<GBlock> { Ok(GBlock { attn: attn(&b.attn)?, local: dw(&b.local), norm: norm(&b.norm)?, fc1: lin(&b.fc1)?, fc2: lin(&b.fc2)?, window: b.window }) };
        let merge = |m: &PatchMerging| -> Result<GMerge> { Ok(GMerge { conv1: lin(&m.conv1)?, dw: dw(&m.dw), conv3: lin(&m.conv3)? }) };
        let mb = |m: &MbConv| -> Result<GMb> { Ok(GMb { expand: lin(&m.expand)?, dw: dw(&m.dw), project: lin(&m.project)? }) };
        let me = Self {
            embed1: dense(&cpu.embed1)?,
            embed2: dense(&cpu.embed2)?,
            conv: cpu.conv.iter().map(mb).collect::<Result<Vec<_>>>()?,
            conv_down: merge(&cpu.conv_down)?,
            stages: cpu
                .stages
                .iter()
                .map(|s| Ok((s.blocks.iter().map(block).collect::<Result<Vec<_>>>()?, s.down.as_ref().map(merge).transpose()?)))
                .collect::<Result<Vec<_>>>()?,
            neck1: lin(&cpu.neck1)?,
            neck_ln1: norm(&cpu.neck_ln1)?,
            neck2: dense(&cpu.neck2)?,
            neck_ln2: norm(&cpu.neck_ln2)?,
            dummy: up(&[0.0; 4]),
            adapter: format!("{} ({:?})", info.name, info.backend),
            device,
            queue,
            pipes,
        };
        me.self_test()?;
        Ok(me)
    }

    /// One 37×20 · 20×12 gemm with bias and GELU, against the same arithmetic on the CPU.
    fn self_test(&self) -> Result<()> {
        let (m, k, n) = (37usize, 20usize, 12usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32 - 6.0) / 7.0).collect();
        let w: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 11) as f32 - 5.0) / 9.0).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 / 10.0 - 0.5).collect();
        let up = |v: &[f32]| W(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(v), usage: wgpu::BufferUsages::STORAGE }));
        let lin = GLin { w: up(&w), b: Some(up(&b)), k, n };
        let mut run = Run::new(self);
        let x = run.upload(&a);
        let y = run.gemm(&x, m, &lin, GEp { gelu: true, ..GEp::default() });
        let got = run.finish(&y)?;
        let mut worst = 0f32;
        for r in 0..m {
            for c in 0..n {
                let v: f32 = (0..k).map(|i| a[r * k + i] * w[i * n + c]).sum::<f32>() + b[c];
                worst = worst.max((got[r * n + c] - super::gelu(v)).abs());
            }
        }
        if worst > 1e-3 {
            bail!("GPU: self-test gemm is off by {worst}");
        }
        Ok(())
    }

    /// RGB proxy → the pixel-major `4096 × 256` embedding (as [`TinyViT::encode_tokens`]).
    pub fn encode_tokens(&self, rgb: &[u8], w: usize, h: usize) -> Result<Tensor> {
        if w > IMG || h > IMG || rgb.len() != w * h * 3 {
            bail!("proxy {w}x{h} does not fit the {IMG} px encoder");
        }
        let input = normalized_input(rgb, w, h);
        let mut run = Run::new(self);
        let x = run.upload(&input);
        let (x, h, w) = run.dense(&x, IMG, IMG, &self.embed1, true);
        let (mut x, h, w) = run.dense(&x, h, w, &self.embed2, false);
        for b in &self.conv {
            let y = run.gemm(&x, h * w, &b.expand, GEp { gelu: true, ..GEp::default() });
            let (y, _, _) = run.depthwise(&y, h, w, &b.dw, true);
            x = run.gemm(&y, h * w, &b.project, GEp { residual: Some(&x), gelu_after: true, ..GEp::default() });
        }
        let (mut x, mut h, mut w) = run.merge(&x, h, w, &self.conv_down);
        for (blocks, down) in &self.stages {
            for b in blocks {
                x = run.block(&x, h, w, b);
            }
            if let Some(d) = down {
                (x, h, w) = run.merge(&x, h, w, d);
            }
        }
        if (h, w) != (GRID, GRID) {
            bail!("GPU TinyViT produced a {h}×{w} grid");
        }
        let x = run.gemm(&x, h * w, &self.neck1, GEp::default());
        let x = run.norm(&x, h * w, &self.neck_ln1);
        let (x, _, _) = run.dense(&x, h, w, &self.neck2, false);
        let x = run.norm(&x, h * w, &self.neck_ln2);
        let out = run.finish(&x)?;
        Tensor::from_vec(out, (GRID * GRID, 256), &Device::Cpu)
    }
}

/// One encode being recorded: dispatches go into one command encoder, submitted by [`Run::finish`].
struct Run<'a> {
    g: &'a GpuTinyViT,
    enc: wgpu::CommandEncoder,
    pool: Rc<RefCell<HashMap<u64, Vec<wgpu::Buffer>>>>,
}

impl<'a> Run<'a> {
    fn new(g: &'a GpuTinyViT) -> Self {
        Self { g, enc: g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tinyvit") }), pool: Rc::default() }
    }

    fn alloc(&mut self, len: usize) -> Buf {
        let size = (len.max(4) * 4) as u64;
        // Size classes of 1/8 octave, so buffers of nearby sizes are shared.
        let class = {
            let p = 63 - size.leading_zeros() as u64;
            let step = (1u64 << p.saturating_sub(3)).max(256);
            size.div_ceil(step) * step
        };
        let buf = self.pool.borrow_mut().get_mut(&class).and_then(Vec::pop).unwrap_or_else(|| {
            self.g.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: class, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
        });
        Buf { buf: Some(buf), len, pool: self.pool.clone() }
    }

    fn upload(&mut self, v: &[f32]) -> Buf {
        let b = self.alloc(v.len());
        self.g.queue.write_buffer(b.raw(), 0, bytemuck::cast_slice(v));
        b
    }

    fn uniform(&self, words: &[u32]) -> wgpu::Buffer {
        self.g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(words), usage: wgpu::BufferUsages::UNIFORM })
    }

    /// Dispatch `pipe` with `bufs` bound in order; `groups` workgroups (x, y).
    fn dispatch(&mut self, pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer], groups: (u32, u32)) {
        let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let bind = self.g.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries });
        let mut pass = self.enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    /// Workgroups for `items` invocations at 64 per group, folded into 2-D under the 65535 cap.
    fn groups_1d(items: usize) -> (u32, u32) {
        let g = items.div_ceil(64) as u32;
        let x = g.min(65_535);
        (x, g.div_ceil(x.max(1)))
    }

    fn gemm(&mut self, a: &Buf, m: usize, l: &GLin, ep: GEp) -> Buf {
        let out = self.alloc(m * l.n);
        let flags = u32::from(l.b.is_some()) | u32::from(ep.gelu) << 1 | u32::from(ep.relu) << 2 | u32::from(ep.residual.is_some()) << 3 | u32::from(ep.gelu_after) << 4;
        let u = self.uniform(&[m as u32, l.n as u32, l.k as u32, flags]);
        let bias = l.b.as_ref().map_or(&self.g.dummy.0, |b| &b.0);
        let res = ep.residual.map_or(&self.g.dummy.0, Buf::raw);
        let pipe = &self.g.pipes.gemm;
        self.dispatch(pipe, &[a.raw(), &l.w.0, out.raw(), &u, bias, res], ((l.n / 4).div_ceil(32) as u32, m.div_ceil(32) as u32));
        out
    }

    fn dense(&mut self, x: &Buf, h: usize, w: usize, d: &GDense, act: bool) -> (Buf, usize, usize) {
        let (oh, ow) = ((h - 1) / d.stride + 1, (w - 1) / d.stride + 1);
        let total = oh * ow * d.kp;
        let col = self.alloc(total);
        let u = self.uniform(&[h as u32, w as u32, d.cin as u32, d.stride as u32, oh as u32, ow as u32, d.kp as u32, total as u32]);
        let pipe = &self.g.pipes.im2col;
        self.dispatch(pipe, &[x.raw(), col.raw(), &u], Self::groups_1d(total));
        let lin = GLin { w: W(d.w.0.clone()), b: d.b.as_ref().map(|b| W(b.0.clone())), k: d.kp, n: d.n };
        (self.gemm(&col, oh * ow, &lin, GEp { gelu: act, ..GEp::default() }), oh, ow)
    }

    fn depthwise(&mut self, x: &Buf, h: usize, w: usize, d: &GDw, act: bool) -> (Buf, usize, usize) {
        let (oh, ow) = ((h - 1) / d.stride + 1, (w - 1) / d.stride + 1);
        let c4 = d.c / 4;
        let total = oh * ow * c4;
        let out = self.alloc(oh * ow * d.c);
        let u = self.uniform(&[h as u32, w as u32, c4 as u32, d.stride as u32, oh as u32, ow as u32, u32::from(act), total as u32]);
        let pipe = &self.g.pipes.depthwise;
        self.dispatch(pipe, &[x.raw(), &d.w.0, &d.b.0, out.raw(), &u], Self::groups_1d(total));
        (out, oh, ow)
    }

    fn norm(&mut self, x: &Buf, rows: usize, n: &GNorm) -> Buf {
        let out = self.alloc(rows * n.c);
        let u = self.uniform(&[rows as u32, (n.c / 4) as u32, n.eps.to_bits(), 0]);
        let pipe = &self.g.pipes.layernorm;
        self.dispatch(pipe, &[x.raw(), &n.g.0, &n.b.0, out.raw(), &u], Self::groups_1d(rows));
        out
    }

    fn merge(&mut self, x: &Buf, h: usize, w: usize, m: &GMerge) -> (Buf, usize, usize) {
        let y = self.gemm(x, h * w, &m.conv1, GEp { gelu: true, ..GEp::default() });
        let (y, oh, ow) = self.depthwise(&y, h, w, &m.dw, true);
        (self.gemm(&y, oh * ow, &m.conv3, GEp::default()), oh, ow)
    }

    fn block(&mut self, x: &Buf, h: usize, w: usize, b: &GBlock) -> Buf {
        let c = b.local.c;
        let ws = b.window;
        let (nh, nw) = (h.div_ceil(ws), w.div_ceil(ws));
        let n = ws * ws;
        let windows = nh * nw;
        let c4 = c / 4;
        // Window partition (zero-padded), attention, back to pixels + residual.
        let total = windows * n * c4;
        let xs = self.alloc(windows * n * c);
        let u = self.uniform(&[h as u32, w as u32, c4 as u32, ws as u32, nw as u32, total as u32, 0, 0]);
        let pipe = &self.g.pipes.partition;
        self.dispatch(pipe, &[x.raw(), xs.raw(), &u], Self::groups_1d(total));
        let xn = self.norm(&xs, windows * n, &b.attn.norm);
        drop(xs);
        let qkv = self.gemm(&xn, windows * n, &b.attn.qkv, GEp::default());
        drop(xn);
        let heads = b.attn.heads;
        let items = windows * heads * n;
        let att = self.alloc(windows * n * heads * HEAD);
        let u = self.uniform(&[windows as u32, n as u32, heads as u32, b.attn.scale.to_bits(), items as u32, 0, 0, 0]);
        let pipe = &self.g.pipes.attention;
        self.dispatch(pipe, &[qkv.raw(), &b.attn.ab.0, att.raw(), &u], Self::groups_1d(items));
        drop(qkv);
        let a = self.gemm(&att, windows * n, &b.attn.proj, GEp::default());
        drop(att);
        let total = h * w * c4;
        let x2 = self.alloc(h * w * c);
        let u = self.uniform(&[h as u32, w as u32, c4 as u32, ws as u32, nw as u32, total as u32, 0, 0]);
        let pipe = &self.g.pipes.unpartition;
        self.dispatch(pipe, &[a.raw(), x.raw(), x2.raw(), &u], Self::groups_1d(total));
        drop(a);
        let (x3, _, _) = self.depthwise(&x2, h, w, &b.local, false);
        drop(x2);
        let xn = self.norm(&x3, h * w, &b.norm);
        let hid = self.gemm(&xn, h * w, &b.fc1, GEp { gelu: true, ..GEp::default() });
        drop(xn);
        self.gemm(&hid, h * w, &b.fc2, GEp { residual: Some(&x3), ..GEp::default() })
    }

    /// Submit everything recorded and read `out` back.
    fn finish(self, out: &Buf) -> Result<Vec<f32>> {
        let Run { g, mut enc, .. } = self;
        let size = (out.len * 4) as u64;
        let read = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        enc.copy_buffer_to_buffer(out.raw(), 0, &read, 0, size);
        let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
        g.queue.submit([enc.finish()]);
        let (tx, rx) = mpsc::channel();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        g.device.poll(wgpu::PollType::wait_indefinitely()).map_err(err)?;
        if let Some(e) = pollster::block_on(scope.pop()) {
            bail!("GPU: {e}");
        }
        rx.recv().map_err(err)?.map_err(err)?;
        let data = read.slice(..).get_mapped_range().map_err(|e| err(format!("{e:?}")))?;
        let v = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        read.unmap();
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::DType;
    use candle_nn::VarBuilder;

    use super::*;

    /// The GPU encoder matches the CPU one on the real checkpoint (IMAGELAB_MOBILESAM_WEIGHTS).
    #[test]
    fn gpu_matches_cpu_encoder() {
        let Ok(path) = std::env::var("IMAGELAB_MOBILESAM_WEIGHTS") else {
            eprintln!("IMAGELAB_MOBILESAM_WEIGHTS not set; skipped");
            return;
        };
        let vb = VarBuilder::from_buffered_safetensors(std::fs::read(path).unwrap(), DType::F32, &Device::Cpu).unwrap();
        let cpu = TinyViT::new(vb.pp("image_encoder")).unwrap();
        let gpu = match GpuTinyViT::new(&cpu) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no usable GPU ({e}); skipped");
                return;
            }
        };
        let (w, h) = (900usize, 700usize);
        let rgb: Vec<u8> = (0..w * h * 3)
            .map(|i| {
                let (p, c) = (i / 3, i % 3);
                let (x, y) = (p % w, p / w);
                ((x * (c + 1) / 4 + y / 3 + if (x / 90 + y / 70) % 2 == 0 { 60 } else { 0 }) % 256) as u8
            })
            .collect();
        let want = vec_f32(&cpu.encode_tokens(&rgb, w, h).unwrap()).unwrap();
        gpu.encode_tokens(&rgb, w, h).unwrap(); // warm-up (pipelines, buffers)
        let t = std::time::Instant::now();
        let got = vec_f32(&gpu.encode_tokens(&rgb, w, h).unwrap()).unwrap();
        let dt = t.elapsed();
        let diff = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        eprintln!("GPU {} encode {dt:?}; max |diff| {diff:.2e} (max |value| {scale:.2})", gpu.adapter());
        assert!(diff < 2e-3 * scale.max(1.0), "max diff {diff}");
    }
}
