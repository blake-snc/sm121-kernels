//! GPU kernel benchmark suite for sm121-kernels.
//!
//! Uses CUDA event-based timing (not criterion) because GPU kernels need
//! warmup iterations and precise device-side measurement.
//!
//! Run with: cargo run --release --example benchmark

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaStream};

use sm121_kernels::{attention, device, gemm, moe, sampling};

// ---------------------------------------------------------------------------
// Deterministic random inputs
// ---------------------------------------------------------------------------
//
// Benchmarks must run on random operands, not zeros: zero inputs minimize the
// transistor toggle rate, which on a power/thermal-limited part lets the clock
// boost higher than it ever would on real data — inflating sustained TFLOPS and
// making the comparison against cuDNN/CUTLASS (which use random data) unfair.
//
// We generate host-side random bytes with a fixed-seed splitmix64 PRNG (no
// `rand` crate dependency, fully reproducible) and upload them with `htod_copy`.

/// splitmix64: tiny, fast, fixed-seed PRNG. Deterministic across runs/platforms.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [-1, 1).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits -> [0, 1), then map to [-1, 1).
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        u * 2.0 - 1.0
    }
}

/// f32 -> BF16 bits (round-to-nearest-even, matching hardware conversion).
#[inline]
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        // NaN: keep it quiet and non-zero.
        return 0x7FC0;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// f32 -> e4m3 FP8 bits (saturating, round-to-nearest). Range ~[-448, 448].
#[inline]
fn f32_to_fp8_e4m3_bits(f: f32) -> u8 {
    let sign = if f.is_sign_negative() { 0x80u8 } else { 0 };
    let a = f.abs();
    if a == 0.0 || !a.is_finite() {
        return sign;
    }
    let a = a.clamp(0.0, 448.0);
    // e4m3: bias 7, 3 mantissa bits. Smallest normal = 2^-6.
    let mut exp = a.log2().floor() as i32;
    if exp < -6 {
        exp = -6;
    }
    let scale = (exp as f32).exp2();
    let mant = ((a / scale - 1.0) * 8.0).round() as i32;
    let (exp, mant) = if mant > 7 { (exp + 1, 0) } else { (exp, mant) };
    let biased = (exp + 7).clamp(0, 15) as u8;
    sign | (biased << 3) | (mant.clamp(0, 7) as u8)
}

/// Allocate a device buffer of `n` BF16 elements filled with deterministic
/// random data (seed selects the stream so different tensors differ).
fn rand_bf16(stream: &Arc<CudaStream>, n: usize, seed: u64) -> cudarc::driver::CudaSlice<u16> {
    let mut rng = SplitMix64::new(seed);
    let host: Vec<u16> = (0..n).map(|_| f32_to_bf16_bits(rng.next_f32())).collect();
    stream.memcpy_stod(&host).unwrap()
}

/// Allocate a device buffer of `n` FP8 (e4m3) elements filled with
/// deterministic random data.
fn rand_fp8(stream: &Arc<CudaStream>, n: usize, seed: u64) -> cudarc::driver::CudaSlice<u8> {
    let mut rng = SplitMix64::new(seed);
    let host: Vec<u8> = (0..n)
        .map(|_| f32_to_fp8_e4m3_bits(rng.next_f32()))
        .collect();
    stream.memcpy_stod(&host).unwrap()
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct BenchResult {
    name: String,
    median_us: f64,
    mean_us: f64,
    min_us: f64,
    max_us: f64,
    tflops: Option<f64>,
    bandwidth_gbs: Option<f64>,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<45} median={:>9.1} us  mean={:>9.1} us  min={:>9.1} us  max={:>9.1} us",
            self.name, self.median_us, self.mean_us, self.min_us, self.max_us,
        )?;
        if let Some(tflops) = self.tflops {
            write!(f, "  {tflops:>6.2} TFLOPS")?;
        }
        if let Some(bw) = self.bandwidth_gbs {
            write!(f, "  {bw:>6.1} GB/s")?;
        }
        Ok(())
    }
}

/// Run a kernel benchmark with CUDA event timing.
///
/// - `warmup`: number of untimed iterations to stabilize clocks / caches
/// - `iterations`: number of timed iterations
/// - `flops`: total floating-point operations per kernel launch (for TFLOPS)
/// - `bytes`: total bytes moved per kernel launch (for bandwidth)
/// - `run`: closure that launches the kernel once
fn bench_kernel(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    name: &str,
    warmup: usize,
    iterations: usize,
    flops: Option<f64>,
    bytes: Option<f64>,
    mut run: impl FnMut(),
) -> BenchResult {
    // Warmup
    for _ in 0..warmup {
        run();
    }
    stream.synchronize().unwrap();

    // Timed runs with CUDA events
    let mut times_us = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        let stop = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();

        start.record(stream).unwrap();
        run();
        stop.record(stream).unwrap();

        let ms = start.elapsed_ms(&stop).unwrap();
        times_us.push(ms as f64 * 1000.0);
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times_us[times_us.len() / 2];
    let mean = times_us.iter().sum::<f64>() / times_us.len() as f64;
    let min = times_us[0];
    let max = times_us[times_us.len() - 1];

    let median_s = median * 1e-6;
    let tflops = flops.map(|f| f / median_s / 1e12);
    let bandwidth_gbs = bytes.map(|b| b / median_s / 1e9);

    BenchResult {
        name: name.to_string(),
        median_us: median,
        mean_us: mean,
        min_us: min,
        max_us: max,
        tflops,
        bandwidth_gbs,
    }
}

// ---------------------------------------------------------------------------
// Attention benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "experimental")]
fn bench_flash_attn_fp8(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> BenchResult {
    let batch: u32 = std::env::var("SPARK_BENCH_B")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let heads: u32 = std::env::var("SPARK_BENCH_H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let seq: u32 = std::env::var("SPARK_BENCH_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let d: u32 = 128;

    let n = (batch * heads * seq * d) as usize;
    let q = rand_fp8(stream, n, 0x0001);
    let k = rand_fp8(stream, n, 0x0002);
    let v = rand_fp8(stream, n, 0x0003);
    // FP8 attention outputs BF16
    let mut o = stream.alloc_zeros::<u16>(n).unwrap();

    let scale = 1.0 / (d as f32).sqrt();
    let flops = 2.0 * 2.0 * batch as f64 * heads as f64 * seq as f64 * seq as f64 * d as f64;
    // Bytes: read Q+K+V (FP8 = 1B each) + write O (BF16 = 2B)
    let bytes = (3 * n) as f64 + (n as f64 * 2.0);

    bench_kernel(
        ctx,
        stream,
        &format!("FA FP8  B={batch} H={heads} S={seq} D={d}"),
        5,
        100,
        Some(flops),
        Some(bytes),
        || {
            attention::flash_attn_fp8_d128(
                ctx, stream, &q, &k, &v, &mut o, batch, heads, seq, seq, scale,
            )
            .unwrap();
        },
    )
}

fn bench_flash_attn_bf16_v21(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    batch: u32,
    heads: u32,
    seq: u32,
) -> BenchResult {
    let d: u32 = 128;
    let n = (batch * heads * seq * d) as usize;
    let q = rand_bf16(stream, n, 0x0011);
    let k = rand_bf16(stream, n, 0x0012);
    let v = rand_bf16(stream, n, 0x0013);
    let mut o = stream.alloc_zeros::<u16>(n).unwrap();

    let scale = 1.0 / (d as f32).sqrt();
    let flops = 2.0 * 2.0 * batch as f64 * heads as f64 * seq as f64 * seq as f64 * d as f64;
    let bytes = (3 * n + n) as f64 * 2.0;

    bench_kernel(
        ctx,
        stream,
        &format!("FA BF16 V21 STRMP B={batch} H={heads} S={seq} D={d}"),
        5,
        200,
        Some(flops),
        Some(bytes),
        || {
            attention::flash_attn_bf16_v21_streaming_p(
                ctx, stream, &q, &k, &v, &mut o, batch, heads, seq, seq, scale,
            )
            .unwrap();
        },
    )
}

/// Production prefill flash-attention kernel for 9B GDN-hybrid's full-attention
/// layers (head_dim=256, GQA Hq=16/Hkv=2, causal). This is the kernel the
/// chunked prefill path actually dispatches, so it gets first-class bench
/// coverage. Supports both square (seq_q==seq_kv) and the production
/// rectangular chunk shape (seq_q=chunk, seq_kv=context).
fn bench_flash_attn_bf16_v3_d256_gqa_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    batch: u32,
    heads_q: u32,
    heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
) -> BenchResult {
    let d: u32 = 256;
    let nq = (batch * heads_q * seq_q * d) as usize;
    let nkv = (batch * heads_kv * seq_kv * d) as usize;
    let q = rand_bf16(stream, nq, 0x00d1);
    let k = rand_bf16(stream, nkv, 0x00d2);
    let v = rand_bf16(stream, nkv, 0x00d3);
    let mut o = stream.alloc_zeros::<u16>(nq).unwrap();

    let scale = 1.0 / (d as f32).sqrt();
    // Causal, tail-aligned (query i sits at global position seq_kv-seq_q+i, the
    // chunked-prefill convention). Attended (q,k) pairs =
    // sum_i (seq_kv-seq_q+i+1) = Sq*Skv - Sq^2/2 + Sq/2; 4*D flops per pair
    // (QK^T + PV, 2 flops/MAC). Collapses to 2*B*Hq*S^2*D when seq_q==seq_kv.
    let (sq, skv) = (seq_q as f64, seq_kv as f64);
    let pairs = sq * skv - sq * sq / 2.0 + sq / 2.0;
    let flops = 4.0 * d as f64 * batch as f64 * heads_q as f64 * pairs;
    let bytes = (nq + 2 * nkv + nq) as f64 * 2.0;

    bench_kernel(
        ctx,
        stream,
        &format!(
            "FA BF16 V3 D256 GQA-C B={batch} Hq={heads_q} Hkv={heads_kv} Sq={seq_q} Skv={seq_kv}"
        ),
        5,
        200,
        Some(flops),
        Some(bytes),
        || {
            attention::flash_attn_bf16_v3_d256_gqa_causal(
                ctx, stream, &q, &k, &v, &mut o, batch, heads_q, heads_kv, seq_q, seq_kv, scale,
            )
            .unwrap();
        },
    )
}

/// The actual production chunked-prefill dispatch: pos_dev variant places the
/// `seq_q`-token chunk at the TAIL (global position seq_kv-seq_q) so it attends
/// to the full prior context, unlike the head-aligned non-pos-dev causal. This
/// is the real per-chunk shape (a 256-token chunk vs a long KV context).
fn bench_flash_attn_bf16_v3_d256_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    batch: u32,
    heads_q: u32,
    heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
) -> BenchResult {
    let d: u32 = 256;
    let nq = (batch * heads_q * seq_q * d) as usize;
    let nkv = (batch * heads_kv * seq_kv * d) as usize;
    let q = rand_bf16(stream, nq, 0x00e1);
    let k = rand_bf16(stream, nkv, 0x00e2);
    let v = rand_bf16(stream, nkv, 0x00e3);
    let mut o = stream.alloc_zeros::<u16>(nq).unwrap();
    // Per-batch base position: the chunk sits at the end of the context.
    let pos: Vec<u32> = vec![seq_kv - seq_q; batch as usize];
    let pos_d = stream.memcpy_stod(&pos).unwrap();

    let scale = 1.0 / (d as f32).sqrt();
    let (sq, skv) = (seq_q as f64, seq_kv as f64);
    let pairs = sq * skv - sq * sq / 2.0 + sq / 2.0;
    let flops = 4.0 * d as f64 * batch as f64 * heads_q as f64 * pairs;
    let bytes = (nq + 2 * nkv + nq) as f64 * 2.0;

    bench_kernel(
        ctx,
        stream,
        &format!(
            "FA BF16 V3 D256 POSDEV B={batch} Hq={heads_q} Hkv={heads_kv} Sq={seq_q} Skv={seq_kv}"
        ),
        5,
        200,
        Some(flops),
        Some(bytes),
        || {
            attention::flash_attn_bf16_v3_d256_gqa_causal_pos_dev(
                ctx, stream, &q, &k, &v, &mut o, &pos_d, batch, heads_q, heads_kv, seq_q, seq_kv,
                scale,
            )
            .unwrap();
        },
    )
}

fn bench_flash_attn_bf16_v3_d128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    batch: u32,
    heads: u32,
    seq: u32,
) -> BenchResult {
    let d: u32 = 128;
    let n = (batch * heads * seq * d) as usize;
    let q = rand_bf16(stream, n, 0x0021);
    let k = rand_bf16(stream, n, 0x0022);
    let v = rand_bf16(stream, n, 0x0023);
    let mut o = stream.alloc_zeros::<u16>(n).unwrap();

    let scale = 1.0 / (d as f32).sqrt();
    let flops = 2.0 * 2.0 * batch as f64 * heads as f64 * seq as f64 * seq as f64 * d as f64;
    let bytes = (3 * n + n) as f64 * 2.0;

    bench_kernel(
        ctx,
        stream,
        &format!("FA BF16 V3 D128   B={batch} H={heads} S={seq} D={d}"),
        5,
        200,
        Some(flops),
        Some(bytes),
        || {
            attention::flash_attn_bf16_v3_d128(
                ctx, stream, &q, &k, &v, &mut o, batch, heads, seq, seq, scale,
            )
            .unwrap();
        },
    )
}

fn bench_flash_attn_bf16_v22_db(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    batch: u32,
    heads: u32,
    seq: u32,
) -> BenchResult {
    let d: u32 = 128;
    let n = (batch * heads * seq * d) as usize;
    let q = rand_bf16(stream, n, 0x0031);
    let k = rand_bf16(stream, n, 0x0032);
    let v = rand_bf16(stream, n, 0x0033);
    let mut o = stream.alloc_zeros::<u16>(n).unwrap();

    let scale = 1.0 / (d as f32).sqrt();
    let flops = 2.0 * 2.0 * batch as f64 * heads as f64 * seq as f64 * seq as f64 * d as f64;
    let bytes = (3 * n + n) as f64 * 2.0;

    bench_kernel(
        ctx,
        stream,
        &format!("FA BF16 V22 DB    B={batch} H={heads} S={seq} D={d}"),
        5,
        200,
        Some(flops),
        Some(bytes),
        || {
            attention::flash_attn_bf16_v22_db(
                ctx, stream, &q, &k, &v, &mut o, batch, heads, seq, seq, scale,
            )
            .unwrap();
        },
    )
}

#[cfg(feature = "experimental")]
fn bench_flash_attn_fp8_v11_tma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    batch: u32,
    heads: u32,
    seq: u32,
) -> BenchResult {
    let d: u32 = 128;
    let n = (batch * heads * seq * d) as usize;
    let q = rand_fp8(stream, n, 0x0041);
    let k = rand_fp8(stream, n, 0x0042);
    let v = rand_fp8(stream, n, 0x0043);
    let mut o = stream.alloc_zeros::<u16>(n).unwrap();

    let scale = 1.0 / (d as f32).sqrt();
    let flops = 2.0 * 2.0 * batch as f64 * heads as f64 * seq as f64 * seq as f64 * d as f64;
    // FP8 input (1B) + BF16 output (2B)
    let bytes = (3 * n) as f64 + (n as f64 * 2.0);

    bench_kernel(
        ctx,
        stream,
        &format!("FA FP8  V11 TMA  B={batch} H={heads} S={seq} D={d}"),
        5,
        200,
        Some(flops),
        Some(bytes),
        || {
            attention::flash_attn_fp8_v11_tma(
                ctx, stream, &q, &k, &v, &mut o, batch, heads, seq, seq, scale,
            )
            .unwrap();
        },
    )
}

#[cfg(feature = "experimental")]
fn bench_flash_attn_fp8_v12a(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    batch: u32,
    heads: u32,
    seq: u32,
) -> BenchResult {
    let d: u32 = 128;
    let n = (batch * heads * seq * d) as usize;
    let q = rand_fp8(stream, n, 0x0051);
    let k = rand_fp8(stream, n, 0x0052);
    let v = rand_fp8(stream, n, 0x0053);
    let mut o = stream.alloc_zeros::<u16>(n).unwrap();

    let scale = 1.0 / (d as f32).sqrt();
    let flops = 2.0 * 2.0 * batch as f64 * heads as f64 * seq as f64 * seq as f64 * d as f64;
    let bytes = (3 * n) as f64 + (n as f64 * 2.0);

    bench_kernel(
        ctx,
        stream,
        &format!("FA FP8  V12a SMEM-T B={batch} H={heads} S={seq} D={d}"),
        5,
        200,
        Some(flops),
        Some(bytes),
        || {
            attention::flash_attn_fp8_v12a_transpose(
                ctx, stream, &q, &k, &v, &mut o, batch, heads, seq, seq, scale,
            )
            .unwrap();
        },
    )
}

fn bench_flash_attn_fp8_v12c(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    batch: u32,
    heads: u32,
    seq: u32,
) -> BenchResult {
    let d: u32 = 128;
    let n = (batch * heads * seq * d) as usize;
    let q = rand_fp8(stream, n, 0x0061);
    let k = rand_fp8(stream, n, 0x0062);
    // V_T pre-transposed: [D=128, B*H*S] = same size as V
    let vt = rand_fp8(stream, n, 0x0063);
    let mut o = stream.alloc_zeros::<u16>(n).unwrap();

    let scale = 1.0 / (d as f32).sqrt();
    let flops = 2.0 * 2.0 * batch as f64 * heads as f64 * seq as f64 * seq as f64 * d as f64;
    let bytes = (3 * n) as f64 + (n as f64 * 2.0);

    bench_kernel(
        ctx,
        stream,
        &format!("FA FP8  V12c VT-GMEM B={batch} H={heads} S={seq} D={d}"),
        5,
        200,
        Some(flops),
        Some(bytes),
        || {
            attention::flash_attn_fp8_v12c_vt(
                ctx, stream, &q, &k, &vt, &mut o, batch, heads, seq, seq, scale,
            )
            .unwrap();
        },
    )
}

// ---------------------------------------------------------------------------
// GEMM benchmarks
// ---------------------------------------------------------------------------

/// BF16 MMA GEMM v3/v4 at a caller-chosen square shape. At 4096^3 the
/// arithmetic intensity (~683 FLOPS/byte) is far above the ~183 ridge, so this
/// is compute-bound: it measures how close the GEMM kernels get to the BF16
/// tensor-core ceiling (the flash-attn kernel sustains ~75 TFLOPS on the same
/// cores, so this row reveals any GEMM-side headroom).
fn bench_gemm_bf16_square(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    label: &str,
    s: u32,
    run: impl Fn(
        &Arc<CudaContext>,
        &Arc<CudaStream>,
        &cudarc::driver::CudaSlice<u16>,
        &cudarc::driver::CudaSlice<u16>,
        &mut cudarc::driver::CudaSlice<u16>,
        u32,
        u32,
        u32,
    ),
) -> BenchResult {
    let a = rand_bf16(stream, (s * s) as usize, 0x0091);
    let b = rand_bf16(stream, (s * s) as usize, 0x0092);
    let mut c = stream.alloc_zeros::<u16>((s * s) as usize).unwrap();
    let flops = 2.0 * s as f64 * s as f64 * s as f64;
    let bytes = (3 * s * s) as f64 * 2.0;
    bench_kernel(
        ctx,
        stream,
        &format!("{label} {s}x{s}x{s}"),
        5,
        50,
        Some(flops),
        Some(bytes),
        || run(ctx, stream, &a, &b, &mut c, s, s, s),
    )
}

fn bench_gemm_bf16_mma(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> BenchResult {
    let m: u32 = 512;
    let n: u32 = 512;
    let k: u32 = 512;

    let a = rand_bf16(stream, (m * k) as usize, 0x0071);
    let b = rand_bf16(stream, (k * n) as usize, 0x0072);
    let mut c = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    // GEMM FLOPs: 2*M*N*K
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    // Bytes: read A+B, write C (all BF16 = 2 bytes)
    let bytes = ((m * k + k * n + m * n) as f64) * 2.0;

    bench_kernel(
        ctx,
        stream,
        &format!("GEMM BF16 MMA  {m}x{n}x{k}"),
        5,
        100,
        Some(flops),
        Some(bytes),
        || {
            gemm::gemm_bf16_mma(ctx, stream, &a, &b, &mut c, m, n, k).unwrap();
        },
    )
}

fn bench_gemm_fp8_mma(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> BenchResult {
    let m: u32 = 512;
    let n: u32 = 512;
    let k: u32 = 512;

    let a = rand_fp8(stream, (m * k) as usize, 0x0081);
    let b = rand_fp8(stream, (k * n) as usize, 0x0082);
    let mut c = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    // A,B: FP8 (1 byte), C: BF16 (2 bytes)
    let bytes = (m * k + k * n) as f64 + (m * n) as f64 * 2.0;

    bench_kernel(
        ctx,
        stream,
        &format!("GEMM FP8 MMA  {m}x{n}x{k}"),
        5,
        100,
        Some(flops),
        Some(bytes),
        || {
            gemm::gemm_fp8_mma(ctx, stream, &a, &b, &mut c, m, n, k).unwrap();
        },
    )
}

/// W8A16 (BF16 activation × FP8 e4m3 weight) MMA GEMM v3 at a square shape.
/// FP8/NVFP4 weights are a common serving format, so this path gets large-shape
/// bench coverage. Compute is identical to BF16 (dequant to
/// BF16 in SMEM, then BF16 MMA), so it caps at the same ~45–47 TFLOPS as
/// gemm_bf16_mma_v3 — the baseline a future W8A16 v5 (128×256+swizzle) targets.
fn bench_gemm_w8a16_square(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    s: u32,
) -> BenchResult {
    let a = rand_bf16(stream, (s * s) as usize, 0x00a1);
    let b = rand_fp8(stream, (s * s) as usize, 0x00a2);
    let mut c = stream.alloc_zeros::<u16>((s * s) as usize).unwrap();
    let flops = 2.0 * s as f64 * s as f64 * s as f64;
    let bytes = (s * s) as f64 * 2.0 + (s * s) as f64 + (s * s) as f64 * 2.0;
    bench_kernel(
        ctx,
        stream,
        &format!("GEMM W8A16 MMA v3 {s}x{s}x{s}"),
        5,
        50,
        Some(flops),
        Some(bytes),
        || {
            gemm::gemm_w8a16_mma_v3(ctx, stream, &a, &b, &mut c, s, s, s, 1.0).unwrap();
        },
    )
}

// ---------------------------------------------------------------------------
// Sampling benchmark
// ---------------------------------------------------------------------------

fn bench_topk_sampling(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> BenchResult {
    let batch: u32 = 64;
    let vocab: u32 = 128256;
    let k: u32 = 1;

    let logits = rand_bf16(stream, (batch * vocab) as usize, 0x0091);
    let mut indices = stream.alloc_zeros::<u32>((batch * k) as usize).unwrap();
    let mut values = stream.alloc_zeros::<u16>((batch * k) as usize).unwrap();

    let temperature = 1.0f32;

    // Bandwidth-bound: read all logits, write k indices + k values per batch
    let bytes = (batch * vocab) as f64 * 2.0 // read logits (BF16)
        + (batch * k) as f64 * 4.0            // write indices (u32)
        + (batch * k) as f64 * 2.0; // write values (BF16)

    bench_kernel(
        ctx,
        stream,
        &format!("Top-k sampling  B={batch} V={vocab} k={k}"),
        5,
        100,
        None,
        Some(bytes),
        || {
            sampling::topk_sampling(
                ctx,
                stream,
                &logits,
                &mut indices,
                &mut values,
                batch,
                vocab,
                k,
                temperature,
            )
            .unwrap();
        },
    )
}

// ---------------------------------------------------------------------------
// MoE routing benchmark
// ---------------------------------------------------------------------------

fn bench_moe_routing(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> BenchResult {
    let num_tokens: u32 = 1024;
    let num_experts: u32 = 8;
    let top_k: u32 = 2;

    let logits = rand_bf16(stream, (num_tokens * num_experts) as usize, 0x00A1);
    let mut expert_ids = stream
        .alloc_zeros::<u32>((num_tokens * top_k) as usize)
        .unwrap();
    let mut weights = stream
        .alloc_zeros::<u16>((num_tokens * top_k) as usize)
        .unwrap();

    // Bytes: read logits, write expert_ids + weights
    let bytes = (num_tokens * num_experts) as f64 * 2.0
        + (num_tokens * top_k) as f64 * 4.0
        + (num_tokens * top_k) as f64 * 2.0;

    bench_kernel(
        ctx,
        stream,
        &format!("MoE routing  T={num_tokens} E={num_experts} k={top_k}"),
        5,
        100,
        None,
        Some(bytes),
        || {
            moe::moe_routing(
                ctx,
                stream,
                &logits,
                &mut expert_ids,
                &mut weights,
                num_tokens,
                num_experts,
                top_k,
            )
            .unwrap();
        },
    )
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("sm121-kernels benchmark suite");
    println!("=============================");
    println!();

    let ctx = device::init_device(0).expect("failed to init SM121 device");
    let stream = ctx.default_stream();

    // Print device info
    let name_attr = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .unwrap();
    let minor = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .unwrap();
    println!("Device: SM{name_attr}{minor}");
    println!("Warmup: 5 iterations; Timed: 200 iterations (attention), 100 iterations (GEMM/sampling/MoE)");
    println!("Inputs: deterministic random data (fixed-seed splitmix64), uploaded host->device");
    println!("Timing: CUDA events (device-side, sub-microsecond resolution)");
    println!();

    let separator = "-".repeat(120);

    // --- Flash Attention ---
    println!("Flash Attention (d=128)");
    println!("{separator}");

    // `mut` is needed only when the experimental feature appends superseded
    // generations below; suppress the unused_mut warning in the default build.
    #[allow(unused_mut)]
    let mut results = vec![
        bench_flash_attn_bf16_v21(&ctx, &stream, 2, 16, 1024),
        bench_flash_attn_bf16_v21(&ctx, &stream, 2, 16, 2048),
        bench_flash_attn_bf16_v21(&ctx, &stream, 1, 32, 2048),
        // Long-context BF16 V21: the kernel is occupancy-sensitive and scales
        // strongly with batch*heads*seq. These rows back the long-context
        // numbers in the README perf table (~69 / ~75 TFLOPS).
        bench_flash_attn_bf16_v21(&ctx, &stream, 2, 32, 4096),
        bench_flash_attn_bf16_v21(&ctx, &stream, 2, 32, 8192),
        bench_flash_attn_bf16_v3_d128(&ctx, &stream, 2, 16, 1024),
        bench_flash_attn_bf16_v3_d128(&ctx, &stream, 2, 16, 2048),
        bench_flash_attn_bf16_v3_d128(&ctx, &stream, 1, 32, 2048),
        bench_flash_attn_bf16_v22_db(&ctx, &stream, 2, 16, 1024),
        bench_flash_attn_bf16_v22_db(&ctx, &stream, 2, 16, 2048),
        bench_flash_attn_bf16_v22_db(&ctx, &stream, 1, 32, 2048),
        bench_flash_attn_fp8_v12c(&ctx, &stream, 2, 16, 1024),
        bench_flash_attn_fp8_v12c(&ctx, &stream, 2, 16, 2048),
        bench_flash_attn_fp8_v12c(&ctx, &stream, 1, 32, 2048),
        bench_flash_attn_fp8_v12c(&ctx, &stream, 2, 32, 8192),
        // Production prefill kernel (9B GDN-hybrid full-attn: D=256, GQA Hq16/Hkv2,
        // causal). Square rows characterize the kernel; the Sq=256 rows are the
        // real chunked-prefill shape (a 256-token chunk attending to context).
        bench_flash_attn_bf16_v3_d256_gqa_causal(&ctx, &stream, 2, 16, 2, 2048, 2048),
        bench_flash_attn_bf16_v3_d256_gqa_causal(&ctx, &stream, 2, 16, 2, 4096, 4096),
        bench_flash_attn_bf16_v3_d256_gqa_causal(&ctx, &stream, 2, 16, 2, 8192, 8192),
        // Real production chunk shape (256-token chunk at the tail of context).
        bench_flash_attn_bf16_v3_d256_pos_dev(&ctx, &stream, 1, 16, 2, 256, 8192),
        bench_flash_attn_bf16_v3_d256_pos_dev(&ctx, &stream, 4, 16, 2, 256, 8192),
    ];
    // Optional single-shape override: `SPARK_BENCH_B/H/S` runs the production
    // BF16 (V21) and FP8 (V12c) kernels at exactly one shape, so the README's
    // per-row reproduction commands resolve to a real measurement.
    if let (Ok(b), Ok(h), Ok(s)) = (
        std::env::var("SPARK_BENCH_B"),
        std::env::var("SPARK_BENCH_H"),
        std::env::var("SPARK_BENCH_S"),
    ) {
        if let (Ok(b), Ok(h), Ok(s)) = (b.parse(), h.parse(), s.parse()) {
            results.push(bench_flash_attn_bf16_v21(&ctx, &stream, b, h, s));
            results.push(bench_flash_attn_fp8_v12c(&ctx, &stream, b, h, s));
        }
    }
    // Superseded FA generations (gated behind the experimental feature).
    #[cfg(feature = "experimental")]
    results.extend([
        bench_flash_attn_fp8_v11_tma(&ctx, &stream, 2, 16, 1024),
        bench_flash_attn_fp8_v11_tma(&ctx, &stream, 2, 16, 2048),
        bench_flash_attn_fp8_v11_tma(&ctx, &stream, 1, 32, 2048),
        bench_flash_attn_fp8_v12a(&ctx, &stream, 2, 16, 1024),
        bench_flash_attn_fp8_v12a(&ctx, &stream, 2, 16, 2048),
        bench_flash_attn_fp8_v12a(&ctx, &stream, 1, 32, 2048),
        // CpAsync (non-TMA) FP8 baseline
        bench_flash_attn_fp8(&ctx, &stream),
    ]);
    for r in &results {
        println!("{r}");
    }
    println!();

    // --- GEMM ---
    println!("GEMM (512x512x512)");
    println!("{separator}");

    let results = [
        bench_gemm_bf16_mma(&ctx, &stream),
        bench_gemm_fp8_mma(&ctx, &stream),
        // Compute-bound square shapes: how close do the BF16 GEMM kernels get
        // to the tensor-core ceiling? (roofline cites v3 ~49 TFLOPS at 4096^3)
        bench_gemm_bf16_square(
            &ctx,
            &stream,
            "GEMM BF16 MMA v3",
            2048,
            |c, s, a, b, o, m, n, k| {
                gemm::gemm_bf16_mma_v3(c, s, a, b, o, m, n, k).unwrap();
            },
        ),
        bench_gemm_bf16_square(
            &ctx,
            &stream,
            "GEMM BF16 MMA v3",
            4096,
            |c, s, a, b, o, m, n, k| {
                gemm::gemm_bf16_mma_v3(c, s, a, b, o, m, n, k).unwrap();
            },
        ),
        bench_gemm_bf16_square(
            &ctx,
            &stream,
            "GEMM BF16 MMA v4",
            2048,
            |c, s, a, b, o, m, n, k| {
                gemm::gemm_bf16_mma_v4(c, s, a, b, o, m, n, k).unwrap();
            },
        ),
        bench_gemm_bf16_square(
            &ctx,
            &stream,
            "GEMM BF16 MMA v4",
            4096,
            |c, s, a, b, o, m, n, k| {
                gemm::gemm_bf16_mma_v4(c, s, a, b, o, m, n, k).unwrap();
            },
        ),
        // v5: 128×256 register-blocked (the larger-tile lever vs v3's 49 TF).
        bench_gemm_bf16_square(
            &ctx,
            &stream,
            "GEMM BF16 MMA v5",
            2048,
            |c, s, a, b, o, m, n, k| {
                gemm::gemm_bf16_mma_v5(c, s, a, b, o, m, n, k).unwrap();
            },
        ),
        bench_gemm_bf16_square(
            &ctx,
            &stream,
            "GEMM BF16 MMA v5",
            4096,
            |c, s, a, b, o, m, n, k| {
                gemm::gemm_bf16_mma_v5(c, s, a, b, o, m, n, k).unwrap();
            },
        ),
        // W8A16 (FP8 weight) — the production weight format. Same compute as
        // BF16 v3 (~45-47 TF), so it's the baseline a future W8A16 v5 targets.
        bench_gemm_w8a16_square(&ctx, &stream, 2048),
        bench_gemm_w8a16_square(&ctx, &stream, 4096),
    ];
    for r in &results {
        println!("{r}");
    }
    println!();

    // --- Sampling ---
    println!("Sampling & Routing");
    println!("{separator}");

    let results = [
        bench_topk_sampling(&ctx, &stream),
        bench_moe_routing(&ctx, &stream),
    ];
    for r in &results {
        println!("{r}");
    }
    println!();

    // --- JSON output if requested ---
    if std::env::args().any(|a| a == "--json") {
        // Collect all results would require refactoring. For now, print a note.
        eprintln!("(JSON output: re-run with --json to a future version with collected results)");
    }

    println!("Done.");
    println!();
    println!("=== Summary ===");
    println!("Platform: DGX Spark (SM121a), 128 GB LPDDR5x, 273 GB/s");
    println!("Timing: CUDA events, 5 warmup + 200/100 measured, median reported");
    println!();
    println!("Key results (B=1, H=32, S=2048, D=128 unless noted):");
    println!("  BF16 V21 streaming_p:  ~35 TFLOPS @ S=2048 -> ~75 TFLOPS @ S=8192 (B=2,H=32)");
    println!("  FP8  V12c VT-GMEM:     ~108 TFLOPS");
    println!();
    println!("vs CUTLASS 4.5 CuTe DSL (same hardware, H=16, S=2048):");
    println!("  FP8  CUTLASS FA:    ~45 TFLOPS  (sm121-kernels FP8 ~2.2x faster)");
    println!(
        "  BF16 CUTLASS FA:    ~67 TFLOPS  (their short-shape ref; our V21 reaches ~75 at S=8192,"
    );
    println!("                                   within ~13% of upstream FA4 -- see docs/optimization_journey.md)");
}
