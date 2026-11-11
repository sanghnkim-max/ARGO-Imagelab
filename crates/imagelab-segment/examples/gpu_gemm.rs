//! Feasibility probe: f32 gemm on the GPU through wgpu compute, at MobileSAM's shapes.
//! `cargo run --release -p imagelab-segment --example gpu_gemm` (or cargo-ndk + adb push).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Instant;

use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct Dims { m: u32, n: u32, k: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;
const TK = 16u;
// A tile as [k][row] and B tile as [k][col]. Scalar: threads write single values (writing one
// component of a shared vec4 is a read-modify-write of the whole vector on some backends).
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
    let row0 = wg.y * 64u;
    let col0 = wg.x * 64u;
    var c0 = vec4f(0.0);
    var c1 = vec4f(0.0);
    var c2 = vec4f(0.0);
    var c3 = vec4f(0.0);
    for (var t = 0u; t < d.k; t += TK) {
        for (var i = 0u; i < 4u; i++) {
            let idx = li + i * 256u;
            // A: 64 rows × 16 k.
            let r = idx / TK;
            let kk = idx % TK;
            let gr = row0 + r;
            let gk = t + kk;
            var av = 0.0;
            if (gr < d.m && gk < d.k) { av = a[gr * d.k + gk]; }
            sa[kk * 64u + r] = av;
            // B: 16 k × 64 cols.
            let kb = idx / 64u;
            let cn = idx % 64u;
            let gkb = t + kb;
            let gc = col0 + cn;
            var bv = 0.0;
            if (gkb < d.k && gc < d.n) { bv = b[gkb * d.n + gc]; }
            sb[kb * 64u + cn] = bv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TK; kk++) {
            let ai = kk * 64u + l.y * 4u;
            let bi = kk * 64u + l.x * 4u;
            let a4 = vec4f(sa[ai], sa[ai + 1u], sa[ai + 2u], sa[ai + 3u]);
            let b4 = vec4f(sb[bi], sb[bi + 1u], sb[bi + 2u], sb[bi + 3u]);
            c0 = fma(vec4f(a4.x), b4, c0);
            c1 = fma(vec4f(a4.y), b4, c1);
            c2 = fma(vec4f(a4.z), b4, c2);
            c3 = fma(vec4f(a4.w), b4, c3);
        }
        workgroupBarrier();
    }
    let rows = array<vec4f, 4>(c0, c1, c2, c3);
    for (var i = 0u; i < 4u; i++) {
        let r = row0 + l.y * 4u + i;
        for (var j = 0u; j < 4u; j++) {
            let cc = col0 + l.x * 4u + j;
            if (r < d.m && cc < d.n) { c[r * d.n + cc] = rows[i][j]; }
        }
    }
}
"#;

/// No shared memory: vec4 loads straight from the buffers (Adreno caches them well), each
/// thread a 4-row × 8-column tile in eight vec4 accumulators. Needs k and n multiples of 4.
const SHADER_V4: &str = r#"
struct Dims { m: u32, n: u32, k: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<vec4f>;
@group(0) @binding(1) var<storage, read> b: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> c: array<vec4f>;
@group(0) @binding(3) var<uniform> d: Dims;

@compute @workgroup_size(WGX, WGY)
fn main(@builtin(global_invocation_id) g: vec3u) {
    let n4 = d.n / 4u;
    let k4 = d.k / 4u;
    let col4 = g.x * 2u;
    let row = g.y * 4u;
    if (col4 >= n4 || row >= d.m) { return; }
    let r1 = min(row + 1u, d.m - 1u);
    let r2 = min(row + 2u, d.m - 1u);
    let r3 = min(row + 3u, d.m - 1u);
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
    c[row * n4 + col4] = x0; if (two) { c[row * n4 + col4 + 1u] = y0; }
    if (row + 1u < d.m) { c[(row + 1u) * n4 + col4] = x1; if (two) { c[(row + 1u) * n4 + col4 + 1u] = y1; } }
    if (row + 2u < d.m) { c[(row + 2u) * n4 + col4] = x2; if (two) { c[(row + 2u) * n4 + col4 + 1u] = y2; } }
    if (row + 3u < d.m) { c[(row + 3u) * n4 + col4] = x3; if (two) { c[(row + 3u) * n4 + col4 + 1u] = y3; } }
}
"#;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }).await.expect("no GPU adapter");
    let info = adapter.get_info();
    println!("adapter: {} ({:?}, {:?})", info.name, info.backend, info.device_type);
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.expect("no device");
    let make = |src: String| {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gemm"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
    };
    // (name, pipeline, threads per workgroup x/y, output columns/rows per workgroup)
    let kernels = vec![
        ("tiled-shared", make(SHADER.to_string()), (64u32, 64u32)),
        ("v4 16x8", make(SHADER_V4.replace("WGX", "16").replace("WGY", "8")), (16 * 8, 8 * 4)),
        ("v4 8x8", make(SHADER_V4.replace("WGX", "8").replace("WGY", "8")), (8 * 8, 8 * 4)),
        ("v4 32x4", make(SHADER_V4.replace("WGX", "32").replace("WGY", "4")), (32 * 8, 4 * 4)),
        ("v4 4x16", make(SHADER_V4.replace("WGX", "4").replace("WGY", "16")), (4 * 8, 16 * 4)),
    ];

    // MobileSAM shapes: layer-3 MLP fc1, layer-0 expand, decoder k_proj.
    for (m, k, n) in [(4096u32, 320u32, 1280u32), (65536, 64, 256), (4096, 256, 128), (96, 80, 72)] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32 - 6.0) / 7.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 11) as f32 - 5.0) / 9.0).collect();
        let ba = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck_cast(&a), usage: wgpu::BufferUsages::STORAGE });
        let bb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck_cast(&b), usage: wgpu::BufferUsages::STORAGE });
        let size = u64::from(m * n) * 4;
        let bc = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let dims = [m, n, k, 0u32];
        let bd = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck_cast_u32(&dims), usage: wgpu::BufferUsages::UNIFORM });
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        for (name, pipeline, per) in &kernels {
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ba.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bc.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: bd.as_entire_binding() },
                ],
            });
            let reps = 20;
            let submit = |reps: usize, copy: bool| {
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    for _ in 0..reps {
                        pass.dispatch_workgroups(n.div_ceil(per.0), m.div_ceil(per.1), 1);
                    }
                }
                if copy {
                    enc.copy_buffer_to_buffer(&bc, 0, &read, 0, size);
                }
                queue.submit([enc.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            };
            submit(1, false); // warm-up
            let t = Instant::now();
            submit(reps, true);
            let dt = t.elapsed().as_secs_f64() / reps as f64;
            let slice = read.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let got: Vec<f32> = slice.get_mapped_range().unwrap().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            read.unmap();
            // Spot-check against the CPU.
            let mut max = 0f32;
            for s in 0..64u32 {
                let (r, cc) = ((s * 2_654_435_761u32) % m, (s * 40_503) % n);
                let want: f32 = (0..k).map(|kk| a[(r * k + kk) as usize] * b[(kk * n + cc) as usize]).sum();
                max = max.max((got[(r * n + cc) as usize] - want).abs());
            }
            let gflops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k) / dt / 1e9;
        println!("  {name}: {:.2} ms, {gflops:.0} GFLOP/s, max err {max:.1e}", dt * 1e3);
        }
        // The same product on the CPU (candle's gemm, all cores), as the app does it today.
        let ta = candle_core::Tensor::from_vec(a.clone(), (m as usize, k as usize), &candle_core::Device::Cpu).unwrap();
        let tb = candle_core::Tensor::from_vec(b.clone(), (k as usize, n as usize), &candle_core::Device::Cpu).unwrap();
        ta.matmul(&tb).unwrap();
        let t = Instant::now();
        for _ in 0..5 {
            ta.matmul(&tb).unwrap();
        }
        let cpu = t.elapsed().as_secs_f64() / 5.0;
        let cpu_gflops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k) / cpu / 1e9;
        println!("{m}x{k} @ {k}x{n}: CPU {:.2} ms ({cpu_gflops:.0} GFLOP/s)", cpu * 1e3);
    }
}

fn bytemuck_cast(v: &[f32]) -> &[u8] {
    // No unsafe: build a byte copy and leak it for the probe's lifetime.
    Box::leak(v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>().into_boxed_slice())
}

fn bytemuck_cast_u32(v: &[u32]) -> &[u8] {
    Box::leak(v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>().into_boxed_slice())
}
