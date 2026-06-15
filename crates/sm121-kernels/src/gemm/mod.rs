use std::sync::Arc;
use std::sync::OnceLock;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};

use crate::error::{Result, SparkError};
use crate::module;

/// When `SPARK_DETERMINISTIC=1` is set in the environment, the
/// `*_split_k_managed` GEMV wrappers fall back to atomic-free non-split-K
/// variants so identical inputs produce bytewise-identical outputs. The
/// split-K kernels use `atom.global.add.f32` which is non-deterministic
/// (f32 addition is non-associative; hardware schedules atomics in arbitrary
/// order). Cost: roughly 1.5-3x slower decode latency, deemed acceptable for
/// eval/adapter-comparison workloads where reproducibility matters.
pub fn is_deterministic() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("SPARK_DETERMINISTIC")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn validate_gemm_dims(m: u32, n: u32, k: u32) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument(format!(
            "GEMM dimensions must be > 0: M={m}, N={n}, K={k}"
        )));
    }
    Ok(())
}

fn validate_mma_alignment(m: u32, n: u32, k: u32, k_align: u32) -> Result<()> {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) {
        return Err(SparkError::InvalidArgument(format!(
            "MMA GEMM requires M, N divisible by 32: M={m}, N={n}"
        )));
    }
    if !k.is_multiple_of(k_align) {
        return Err(SparkError::InvalidArgument(format!(
            "MMA GEMM requires K divisible by {k_align}: K={k}"
        )));
    }
    Ok(())
}

/// Launch BF16 GEMM kernel: C[M,N] = A[M,K] x B[K,N]
///
/// All matrices are row-major BF16 (represented as u16 raw bits).
/// Accumulation in FP32, output in BF16.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    if a.len() < m as usize * k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "A buffer too small: {} < {}",
            a.len(),
            m as usize * k as usize
        )));
    }
    if b.len() < k as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "B buffer too small: {} < {}",
            b.len(),
            k as usize * n as usize
        )));
    }
    if c.len() < m as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "C buffer too small: {} < {}",
            c.len(),
            m as usize * n as usize
        )));
    }
    let func = module::load_kernel(ctx, "gemm_bf16", "gemm_bf16")?;

    // 16x16 tile, 256 threads per block (1D — PTX maps tx=tid%16, ty=tid/16)
    let grid_x = n.div_ceil(16); // N dimension
    let grid_y = m.div_ceil(16); // M dimension
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 1024,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

/// View-accepting variant of `gemm_bf16` (forward C = A @ B).
/// Used by orchestrators that hand out slices of larger batched buffers.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &cudarc::driver::CudaView<u16>,
    b: &cudarc::driver::CudaView<u16>,
    c: &mut cudarc::driver::CudaViewMut<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    let func = module::load_kernel(ctx, "gemm_bf16", "gemm_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(16), m.div_ceil(16), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 1024,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// View-accepting variant of `gemm_bf16_backward_dA`. (`dA` matches the
/// standard math notation for ∂L/∂A; the lowercase-after-d is intentional.)
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn gemm_bf16_backward_dA_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dc: &cudarc::driver::CudaView<u16>,
    b: &cudarc::driver::CudaView<u16>,
    da: &mut cudarc::driver::CudaViewMut<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    let func = module::load_kernel(ctx, "gemm_bf16_backward_dA", "gemm_bf16_backward_dA")?;
    let cfg = LaunchConfig {
        grid_dim: (m.div_ceil(8), k.div_ceil(32), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(dc)
            .arg(b)
            .arg(da)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// View-accepting variant of `gemm_bf16_backward_dB`.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn gemm_bf16_backward_dB_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &cudarc::driver::CudaView<u16>,
    dc: &cudarc::driver::CudaView<u16>,
    db: &mut cudarc::driver::CudaViewMut<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    let func = module::load_kernel(ctx, "gemm_bf16_backward_dB", "gemm_bf16_backward_dB")?;
    let cfg = LaunchConfig {
        grid_dim: (k.div_ceil(8), n.div_ceil(32), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(dc)
            .arg(db)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// FP8 W8A16 GEMM backward — `dA` gradient via dequant-then-BF16.
///
/// For C = A_bf16 @ B_fp8_dequant where B_fp8_dequant[i] = B_fp8[i] * scale_B:
///   dA[m, k] = sum_n(dC[m, n] * B_fp8_dequant[k, n])
///
/// Implementation: dequantize B_fp8 → B_bf16 (scratch), then use the
/// MMA-backed BF16 dA backward (which itself falls back to scalar at
/// non-aligned shapes). Gradients flow in BF16 — sufficient for FP8
/// training per DeepSeek V3 / Tensor Engine convention.
///
/// FP8 backward inherits the same 21× MMA speedup over scalar at aligned
/// shapes (M%128, K%64, N%16).
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn gemm_fp8_w8a16_backward_dA(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dc: &CudaSlice<u16>,
    b_fp8: &CudaSlice<u8>,
    b_scale: f32,
    da: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    let kn = (k * n) as usize;
    let mut b_bf16 = stream
        .alloc_zeros::<u16>(kn)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc b_bf16 scratch: {e:?}")))?;
    crate::quantization::dequant_fp8_bf16_pertensor(
        ctx,
        stream,
        b_fp8,
        &mut b_bf16,
        k * n,
        b_scale,
    )?;
    gemm_bf16_backward_dA_mma(ctx, stream, dc, &b_bf16, da, m, n, k)
}

/// FP8 W8A16 GEMM backward — `dB` gradient.
///
/// dB stays in BF16 (gradient on FP8 weights). Quantization to FP8 for
/// the weight update is the optimizer's job. So this is just a BF16
/// backward dB — wrapper exists for API symmetry. Routes to the MMA
/// dispatcher (which falls back to scalar at non-aligned shapes).
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn gemm_fp8_w8a16_backward_dB(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    dc: &CudaSlice<u16>,
    db: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    gemm_bf16_backward_dB_mma(ctx, stream, a, dc, db, m, n, k)
}

/// Backward pass for `gemm_bf16` — `dA` gradient.
///
/// For `C = A @ B` with `A:[M,K]`, `B:[K,N]`, `C:[M,N]`:
///   `dA[m, k] = sum_n(dC[m, n] * B[k, n])`
///
/// Equivalent to `dA = dC @ B^T` but avoids materializing the transpose.
///
/// `dC`: [M, N] BF16 (input upstream gradient)
/// `b`: [K, N] BF16 (saved from forward)
/// `da`: [M, K] BF16 (output)
///
/// Note: scalar correctness-first impl (one thread per output element,
/// scalar inner loop over N). An MMA-tiled variant is future work.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn gemm_bf16_backward_dA(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dc: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    da: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    if dc.len() < m as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "dC too small: {} < {}",
            dc.len(),
            m as usize * n as usize
        )));
    }
    if b.len() < k as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "B too small: {} < {}",
            b.len(),
            k as usize * n as usize
        )));
    }
    if da.len() < m as usize * k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "dA too small: {} < {}",
            da.len(),
            m as usize * k as usize
        )));
    }
    let func = module::load_kernel(ctx, "gemm_bf16_backward_dA", "gemm_bf16_backward_dA")?;
    // Block layout matches kernel: BM=8, BK=32, 256 threads/block.
    let grid_x = m.div_ceil(8);
    let grid_y = k.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(dc)
            .arg(b)
            .arg(da)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Backward pass for `gemm_bf16` — `dB` gradient.
///
/// For `C = A @ B`:
///   `dB[k, n] = sum_m(A[m, k] * dC[m, n])`
///
/// Equivalent to `dB = A^T @ dC`.
///
/// `a`: [M, K] BF16 (saved from forward)
/// `dC`: [M, N] BF16
/// `db`: [K, N] BF16 (output)
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn gemm_bf16_backward_dB(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    dc: &CudaSlice<u16>,
    db: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    if a.len() < m as usize * k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "A too small: {} < {}",
            a.len(),
            m as usize * k as usize
        )));
    }
    if dc.len() < m as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "dC too small: {} < {}",
            dc.len(),
            m as usize * n as usize
        )));
    }
    if db.len() < k as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "dB too small: {} < {}",
            db.len(),
            k as usize * n as usize
        )));
    }
    let func = module::load_kernel(ctx, "gemm_bf16_backward_dB", "gemm_bf16_backward_dB")?;
    // Block layout matches kernel: BK=8, BN=32.
    let grid_x = k.div_ceil(8);
    let grid_y = n.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(dc)
            .arg(db)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Out-of-place 2-D BF16 transpose: dst[N, M] = src[M, N].
///
/// Used by the MMA backward GEMM dispatchers to materialize the transposes
/// without writing dedicated transposed-MMA kernels.
pub fn transpose_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u16>,
    dst: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
) -> Result<()> {
    if m == 0 || n == 0 {
        return Err(SparkError::InvalidArgument(format!(
            "transpose_bf16: M and N must be > 0 (got {m}x{n})"
        )));
    }
    if src.len() < (m * n) as usize || dst.len() < (m * n) as usize {
        return Err(SparkError::InvalidArgument(format!(
            "transpose_bf16: buffers too small src={} dst={} need={}",
            src.len(),
            dst.len(),
            m * n
        )));
    }
    let func = module::load_kernel(ctx, "transpose_bf16", "transpose_bf16")?;
    let grid_x = n.div_ceil(32);
    let grid_y = m.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src)
            .arg(dst)
            .arg(&m)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// MMA-backed BF16 GEMM backward — `dA = dC @ B^T`. Materializes B^T into
/// scratch then dispatches to the optimized forward MMA. ~10-50× faster
/// than the scalar `gemm_bf16_backward_dA` for large shapes; falls back to
/// the scalar path when MMA alignment requirements aren't met.
///
/// Forward C = A @ B with `A:[M,K]`, `B:[K,N]`, `C:[M,N]`. Backward needs
/// `dA:[M,K] = dC:[M,N] @ B^T:[N,K]`. We transpose B (K×N → N×K) into
/// scratch, then call `gemm_bf16_mma(dC, B^T, dA, M, K, N)` (treating M as
/// the new row-count, K as the new col-count, N as the new K-reduction).
///
/// Alignment for the MMA path: M%128 == 0, K%64 == 0, N%16 == 0. Outside
/// those, the scalar fallback runs.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn gemm_bf16_backward_dA_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dc: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    da: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    // MMA path requires the FORWARD-call's M%128, N%64, K%16. For the
    // remapped call gemm_mma(dC, B^T, dA, M_new=M, N_new=K, K_new=N), that's
    // M%128, K%64, N%16.
    let mma_eligible = m.is_multiple_of(128) && k.is_multiple_of(64) && n.is_multiple_of(16);
    if !mma_eligible {
        return gemm_bf16_backward_dA(ctx, stream, dc, b, da, m, n, k);
    }
    // Transpose B (K x N) -> Bt (N x K) into scratch.
    let n_b_t = (k * n) as usize;
    let mut b_t = stream
        .alloc_zeros::<u16>(n_b_t)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc Bt scratch: {e:?}")))?;
    transpose_bf16(ctx, stream, b, &mut b_t, k, n)?;
    // gemm_mma(dC[M,N], B^T[N,K], dA[M,K], M=M, N=K, K=N)
    gemm_bf16_mma(ctx, stream, dc, &b_t, da, m, k, n)
}

/// MMA-backed BF16 GEMM backward — `dB = A^T @ dC`. Same pattern as
/// `gemm_bf16_backward_dA_mma`: transpose A (M×K → K×M), then forward-MMA.
///
/// Backward maps to gemm_mma(A^T, dC, dB, K, N, M). Alignment: K%128 == 0,
/// N%64 == 0, M%16 == 0. Outside those, falls back to scalar.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn gemm_bf16_backward_dB_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    dc: &CudaSlice<u16>,
    db: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    let mma_eligible = k.is_multiple_of(128) && n.is_multiple_of(64) && m.is_multiple_of(16);
    if !mma_eligible {
        return gemm_bf16_backward_dB(ctx, stream, a, dc, db, m, n, k);
    }
    let n_a_t = (m * k) as usize;
    let mut a_t = stream
        .alloc_zeros::<u16>(n_a_t)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc At scratch: {e:?}")))?;
    transpose_bf16(ctx, stream, a, &mut a_t, m, k)?;
    // gemm_mma(A^T[K,M], dC[M,N], dB[K,N], M=K, N=N, K=M)
    gemm_bf16_mma(ctx, stream, &a_t, dc, db, k, n, m)
}

/// Launch BF16 GEMV kernel: out[N] = x[K] @ B[K, N].
///
/// Specialised for the M=1 decode shape that the scalar `gemm_bf16` (16x16 tile)
/// wastes most of its threads on. One thread per output column; X is staged into
/// dynamic SMEM once per block.
///
/// SMEM requirement: `K * 2` bytes. For K up to ~49152 fits in 99 KB on SM121a.
pub fn gemv_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,       // [K] bf16
    b: &CudaSlice<u16>,       // [K, N] bf16
    out: &mut CudaSlice<u16>, // [N] bf16
    n: u32,
    k: u32,
) -> Result<()> {
    if n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("n and k must be > 0".into()));
    }
    if x.len() < k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "gemv: x too small {} < {k}",
            x.len()
        )));
    }
    if b.len() < (k as usize) * (n as usize) {
        return Err(SparkError::InvalidArgument(format!(
            "gemv: b too small {} < {}",
            b.len(),
            (k as usize) * (n as usize)
        )));
    }
    if out.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "gemv: out too small {} < {n}",
            out.len()
        )));
    }
    let smem_bytes = (k as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv: K={k} exceeds SMEM budget (96 KB / 2 = 49152 elements)"
        )));
    }
    let func = module::load_kernel(ctx, "gemv_bf16", "gemv_bf16")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b)
            .arg(out)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch BF16 GEMV v2: each thread holds 8 column accumulators in registers
/// and does 8 independent FMAs per K-iteration, raising ILP from v1's 1 to 8.
///
/// Constraints: N must be a multiple of 8 (for packed b32 stores). K * 2 bytes
/// must fit in 96 KB SMEM (i.e., K <= 49152).
pub fn gemv_bf16_v2(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    n: u32,
    k: u32,
) -> Result<()> {
    if n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("n and k must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_v2 requires N divisible by 8, got {n}"
        )));
    }
    let smem_bytes = (k as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_v2: K={k} exceeds SMEM budget"
        )));
    }
    if x.len() < k as usize || b.len() < (k as usize) * (n as usize) || out.len() < n as usize {
        return Err(SparkError::InvalidArgument("gemv_v2: buffer sizes".into()));
    }
    let func = module::load_kernel(ctx, "gemv_bf16_v2", "gemv_bf16_v2")?;
    let threads = 128u32;
    let cols_per_block = threads * 8; // 1024
    let blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b)
            .arg(out)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch the BF16 GEMV split-K kernel.
///
/// Each block handles a (col_block, k_shard) pair and atomically accumulates
/// into `out_f32[N]`. Caller must zero `out_f32` before calling.
/// Caller is responsible for casting `out_f32` → BF16 afterwards
/// (via `activation::f32_to_bf16`).
///
/// Constraints: N must be a multiple of 8, k_shard*2 must fit in 96 KB SMEM.
/// W4A16 NVFP4 GEMV with split-K: BF16 activations × NVFP4 weights →
/// F32 accumulator. NVFP4 = FP4 e2m1 with 16-elem blocks along N and FP8
/// e4m3 per-block scales. Halves weight HBM bandwidth vs W8A16.
///
/// `b_packed`: `[K, N/2]` u8 (packed FP4)
/// `b_scales`: `[K, N/16]` u8 (FP8 e4m3 per-block scales)
/// Caller must zero `out_f32` before calling.
#[allow(clippy::too_many_arguments)]
pub fn gemv_w4a16_nvfp4_split_k(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_packed: &CudaSlice<u8>,
    b_scales: &CudaSlice<u8>,
    out_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 15 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w4a16_nvfp4_split_k requires N divisible by 16, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w4a16: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    let need_packed = (k as usize) * (n as usize / 2);
    let need_scales = (k as usize) * (n as usize / 16);
    if x.len() < k as usize
        || b_packed.len() < need_packed
        || b_scales.len() < need_scales
        || out_f32.len() < n as usize
    {
        return Err(SparkError::InvalidArgument(format!(
            "buffer sizes (need packed={need_packed} scales={need_scales})"
        )));
    }
    let func = module::load_kernel(ctx, "gemv_w4a16_nvfp4_split_k", "gemv_w4a16_nvfp4_split_k")?;
    let threads = 128u32;
    let cols_per_block = threads * 8;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b_packed)
            .arg(b_scales)
            .arg(out_f32)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// Deterministic NVFP4 GEMV (writes per-shard partials,
/// caller pairs with `f32_shard_reduce_to_bf16`). Same `n_blocks`
/// dimensioning as `gemv_w4a16_nvfp4_split_k`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_w4a16_nvfp4_split_k_det(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_packed: &CudaSlice<u8>,
    b_scales: &CudaSlice<u8>,
    stage_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n & 15 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w4a16_nvfp4_split_k_det requires N divisible by 16, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w4a16_det: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    let need = (num_shards as usize) * (n as usize);
    if stage_f32.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w4a16_det stage too small: {} < {need}",
            stage_f32.len()
        )));
    }
    let func = module::load_kernel(
        ctx,
        "gemv_w4a16_nvfp4_split_k_det",
        "gemv_w4a16_nvfp4_split_k_det",
    )?;
    let threads = 128u32;
    let n_blocks = n.div_ceil(threads * 8);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b_packed)
            .arg(b_scales)
            .arg(stage_f32)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// W8A16 GEMV with split-K: BF16 activations × FP8 e4m3 weights → F32
/// accumulator (cast to BF16 separately). Halves weight HBM bandwidth vs
/// `gemv_bf16_split_k`.
///
/// `b_fp8`: `[K, N]` FP8 e4m3 (1 byte per element).
/// `b_scale`: per-tensor dequant scale (such that `dequant(b) = b_scale * f32(b)`).
/// Caller must zero `out_f32` before calling.
#[allow(clippy::too_many_arguments)]
pub fn gemv_w8a16_split_k(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_fp8: &CudaSlice<u8>,
    b_scale: f32,
    out_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w8a16_split_k requires N divisible by 8, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w8a16_split_k: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    if x.len() < k as usize
        || b_fp8.len() < (k as usize) * (n as usize)
        || out_f32.len() < n as usize
    {
        return Err(SparkError::InvalidArgument("buffer sizes".into()));
    }
    let func = module::load_kernel(ctx, "gemv_w8a16_split_k", "gemv_w8a16_split_k")?;
    let threads = 128u32;
    let cols_per_block = threads * 8;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b_fp8)
            .arg(out_f32)
            .arg(&b_scale)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// Combined wrapper: zero F32 scratch → split-K W8A16 GEMV → cast to BF16.
pub fn gemv_w8a16_split_k_managed(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_fp8: &CudaSlice<u8>,
    b_scale: f32,
    out_f32_scratch: &mut CudaSlice<f32>,
    out_bf16: &mut CudaSlice<u16>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    // Deterministic mode uses the W8A16 det kernel +
    // shard-reducer (same pattern as gemv_bf16_split_k_managed in 92g).
    if is_deterministic() {
        let need = (num_shards as usize) * (n as usize);
        if out_f32_scratch.len() >= need {
            gemv_w8a16_split_k_det(
                ctx,
                stream,
                x,
                b_fp8,
                b_scale,
                out_f32_scratch,
                n,
                k,
                num_shards,
            )?;
            f32_shard_reduce_to_bf16(ctx, stream, out_f32_scratch, out_bf16, n, num_shards)?;
            return Ok(());
        }
        // Scratch too small for staging → fall through to atomic path
        // (still produces correct numerical output; just not bytewise stable).
    }
    stream
        .memset_zeros(out_f32_scratch)
        .map_err(SparkError::Driver)?;
    gemv_w8a16_split_k(
        ctx,
        stream,
        x,
        b_fp8,
        b_scale,
        out_f32_scratch,
        n,
        k,
        num_shards,
    )?;
    crate::activation::f32_to_bf16(ctx, stream, out_f32_scratch, out_bf16, n)?;
    Ok(())
}

/// Deterministic W8A16 GEMV — writes per-shard partials to
/// staging instead of atomic-adding. Pair with `f32_shard_reduce_to_bf16`.
pub fn gemv_w8a16_split_k_det(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_fp8: &CudaSlice<u8>,
    b_scale: f32,
    stage_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w8a16_split_k_det requires N divisible by 8, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w8a16_split_k_det: k_shard={k_shard} exceeds SMEM"
        )));
    }
    let need = (num_shards as usize) * (n as usize);
    if x.len() < k as usize || b_fp8.len() < (k as usize) * (n as usize) || stage_f32.len() < need {
        return Err(SparkError::InvalidArgument("buffer sizes".into()));
    }
    let func = module::load_kernel(ctx, "gemv_w8a16_split_k_det", "gemv_w8a16_split_k_det")?;
    let threads = 128u32;
    let n_blocks = n.div_ceil(threads * 8);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b_fp8)
            .arg(stage_f32)
            .arg(&b_scale)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// Fused W8A16 GEMV with direct BF16 output. Replaces the 3-launch
/// `gemv_w8a16_split_k_managed` (memset + split-K GEMV + cast) with a
/// single launch by trading split-K parallelism for launch-overhead
/// savings. Use when N is large enough to saturate SMs without splitting
/// (typically N ≥ ~4096 for SM121's 48 SMs).
///
/// Layout:
///  - `x`: [K] BF16
///  - `b_fp8`: [K, N] FP8 e4m3 (1 byte/elem)
///  - `out_bf16`: [N] BF16
///  - `b_scale`: per-tensor dequant scale
///
/// Constraints:
///  - N must be a multiple of 8.
///  - K * 2 bytes must fit in the per-CTA SMEM budget (≤ ~32K K).
///  - Caller does NOT need to pre-zero `out_bf16` (kernel writes once).
pub fn gemv_w8a16_fused_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_fp8: &CudaSlice<u8>,
    b_scale: f32,
    out_bf16: &mut CudaSlice<u16>,
    n: u32,
    k: u32,
) -> Result<()> {
    if n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "n ({n}) must be a multiple of 8"
        )));
    }
    if x.len() < k as usize {
        return Err(SparkError::InvalidArgument("x buffer too small".into()));
    }
    if b_fp8.len() < (k * n) as usize {
        return Err(SparkError::InvalidArgument("b buffer too small".into()));
    }
    if out_bf16.len() < n as usize {
        return Err(SparkError::InvalidArgument("out buffer too small".into()));
    }
    let smem_bytes = (k as usize) * 2;
    if smem_bytes > 99 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "K={k} too large for fused W8A16 SMEM (need {smem_bytes} > 99 KB)"
        )));
    }
    let func = module::load_kernel(ctx, "gemv_w8a16_fused_bf16", "gemv_w8a16_fused_bf16")?;
    let grid_x = n.div_ceil(1024);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b_fp8)
            .arg(out_bf16)
            .arg(&b_scale)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// W8A16 GEMV with 16 cols/thread (vs split_k's 8). Uses ld.global.v4.b32 for a
/// fully vectorized 16-byte weight load per K-iter and emits 2 independent groups
/// of 8 FMAs to keep the warp scheduler fed. Block: 128 threads × 16 = 2048
/// cols/block, halving CTA count vs the standard kernel — wins on weights with
/// large N (e.g. lm_head), loses on small N (would idle threads). Dispatch
/// selectively. Requires N divisible by 16.
#[allow(clippy::too_many_arguments)]
pub fn gemv_w8a16_split_k_w16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_fp8: &CudaSlice<u8>,
    b_scale: f32,
    out_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 15 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w8a16_split_k_w16 requires N divisible by 16, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w8a16_split_k_w16: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    if x.len() < k as usize
        || b_fp8.len() < (k as usize) * (n as usize)
        || out_f32.len() < n as usize
    {
        return Err(SparkError::InvalidArgument("buffer sizes".into()));
    }
    let func = module::load_kernel(ctx, "gemv_w8a16_split_k_w16", "gemv_w8a16_split_k_w16")?;
    let threads = 128u32;
    let cols_per_block = threads * 16;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b_fp8)
            .arg(out_f32)
            .arg(&b_scale)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// Combined wrapper for the wider-block W8A16 GEMV.
#[allow(clippy::too_many_arguments)]
pub fn gemv_w8a16_split_k_w16_managed(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_fp8: &CudaSlice<u8>,
    b_scale: f32,
    out_f32_scratch: &mut CudaSlice<f32>,
    out_bf16: &mut CudaSlice<u16>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    // Det route mirrors gemv_w8a16_split_k_managed.
    if is_deterministic() {
        let need = (num_shards as usize) * (n as usize);
        if out_f32_scratch.len() >= need {
            gemv_w8a16_split_k_w16_det(
                ctx,
                stream,
                x,
                b_fp8,
                b_scale,
                out_f32_scratch,
                n,
                k,
                num_shards,
            )?;
            f32_shard_reduce_to_bf16(ctx, stream, out_f32_scratch, out_bf16, n, num_shards)?;
            return Ok(());
        }
    }
    stream
        .memset_zeros(out_f32_scratch)
        .map_err(SparkError::Driver)?;
    gemv_w8a16_split_k_w16(
        ctx,
        stream,
        x,
        b_fp8,
        b_scale,
        out_f32_scratch,
        n,
        k,
        num_shards,
    )?;
    crate::activation::f32_to_bf16(ctx, stream, out_f32_scratch, out_bf16, n)?;
    Ok(())
}

/// Deterministic wide-block W8A16 GEMV (16 cols/thread).
#[allow(clippy::too_many_arguments)]
pub fn gemv_w8a16_split_k_w16_det(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_fp8: &CudaSlice<u8>,
    b_scale: f32,
    stage_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n & 15 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w8a16_split_k_w16_det requires N divisible by 16, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w8a16_split_k_w16_det: k_shard={k_shard} exceeds SMEM"
        )));
    }
    let need = (num_shards as usize) * (n as usize);
    if x.len() < k as usize || b_fp8.len() < (k as usize) * (n as usize) || stage_f32.len() < need {
        return Err(SparkError::InvalidArgument("buffer sizes".into()));
    }
    let func = module::load_kernel(
        ctx,
        "gemv_w8a16_split_k_w16_det",
        "gemv_w8a16_split_k_w16_det",
    )?;
    let threads = 128u32;
    let n_blocks = n.div_ceil(threads * 16);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b_fp8)
            .arg(stage_f32)
            .arg(&b_scale)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// Wide-block BF16 split-K GEMV variant: 16 cols/thread, 2048 cols/block.
/// Sibling of `gemv_w8a16_split_k_w16` for BF16-only mode (no FP8 env var)
/// at very large N (lm_head at vocab=262144 etc.). Loses on small N
/// (gate/up/down at inter≤10240) — caller picks. N must be a multiple of 16.
pub fn gemv_bf16_split_k_w16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    out_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 15 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_bf16_split_k_w16 requires N divisible by 16, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_bf16_split_k_w16: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    if x.len() < k as usize || b.len() < (k as usize) * (n as usize) || out_f32.len() < n as usize {
        return Err(SparkError::InvalidArgument("buffer sizes".into()));
    }
    let func = module::load_kernel(ctx, "gemv_bf16_split_k_w16", "gemv_bf16_split_k_w16")?;
    let threads = 128u32;
    let cols_per_block = threads * 16;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b)
            .arg(out_f32)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// Combined wrapper for the wider-block BF16 split-K GEMV.
pub fn gemv_bf16_split_k_w16_managed(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    out_f32_scratch: &mut CudaSlice<f32>,
    out_bf16: &mut CudaSlice<u16>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    stream
        .memset_zeros(out_f32_scratch)
        .map_err(SparkError::Driver)?;
    gemv_bf16_split_k_w16(ctx, stream, x, b, out_f32_scratch, n, k, num_shards)?;
    crate::activation::f32_to_bf16(ctx, stream, out_f32_scratch, out_bf16, n)?;
    Ok(())
}

/// Split-K BF16 GEMV: `out_f32 = x[1, K] * b[K, N]` accumulated in FP32, with the
/// K dimension split across `num_shards` partial sums. N must be divisible by 8.
/// This is the M=1 decode matvec; pair with `f32_to_bf16` to cast the result.
pub fn gemv_bf16_split_k(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    out_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_split_k requires N divisible by 8, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_split_k: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    if x.len() < k as usize || b.len() < (k as usize) * (n as usize) || out_f32.len() < n as usize {
        return Err(SparkError::InvalidArgument("buffer sizes".into()));
    }
    let func = module::load_kernel(ctx, "gemv_bf16_split_k", "gemv_bf16_split_k")?;
    let threads = 128u32;
    let cols_per_block = threads * 8;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b)
            .arg(out_f32)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// Deterministic-by-construction split-K GEMV. Same K-shard
/// parallelism as `gemv_bf16_split_k` but writes per-shard partial sums to
/// `stage_f32[num_shards × N]` instead of atomic-adding into `out_f32[N]`.
/// Pair with `f32_shard_reduce_to_bf16` to produce the final BF16 output.
pub fn gemv_bf16_split_k_det(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    stage_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_split_k_det requires N divisible by 8, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_split_k_det: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    let need = (num_shards as usize) * (n as usize);
    if x.len() < k as usize || b.len() < (k as usize) * (n as usize) || stage_f32.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_split_k_det: stage_f32 too small {} < {need}",
            stage_f32.len()
        )));
    }
    let func = module::load_kernel(ctx, "gemv_bf16_split_k_det", "gemv_bf16_split_k_det")?;
    let threads = 128u32;
    let cols_per_block = threads * 8;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b)
            .arg(stage_f32)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// Max rows the batched split-K verify kernel supports (compile-time unroll in
/// `gemm_bf16_split_k_det.ptx`). MTP self-spec uses M = k+1 (typically 4).
pub const SPLIT_K_DET_MAX_M: u32 = 8;

/// Batched (M-row) deterministic split-K GEMM — bit-identical PER ROW to
/// `gemv_bf16_split_k_det`. This is the MTP self-spec VERIFY kernel: it computes
/// `out[m, :] = x[m, :] · B` for M rows while loading each weight tile ONCE
/// (amortizing weight bandwidth across the k+1 verified tokens). Each output
/// `(m, col)` reduces over K in the SAME order as the M=1 kernel, so feeding the
/// same `num_shards` the decode path uses yields argmax-identical logits.
///
/// `stage_f32` layout is `[num_shards, M, N]` (per-shard stride `M*N`); pair with
/// `f32_shard_reduce_to_bf16` called with `n' = M*N` (see
/// `gemm_bf16_split_k_det_managed`).
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_split_k_det(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    stage_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    m: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || k == 0 || num_shards == 0 || m == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemm_bf16_split_k_det requires N divisible by 8, got {n}"
        )));
    }
    if m > SPLIT_K_DET_MAX_M {
        return Err(SparkError::InvalidArgument(format!(
            "gemm_bf16_split_k_det: m={m} exceeds MAX_M={SPLIT_K_DET_MAX_M}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    // Kernel stages exactly M rows of the X shard in SMEM (k-loop skips rows>=M).
    let smem_bytes = (m as usize) * (k_shard as usize) * 2;
    if smem_bytes > 48 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemm_bf16_split_k_det: smem {smem_bytes}B (m={m} k_shard={k_shard}) exceeds 48KB; raise num_shards"
        )));
    }
    let need = (num_shards as usize) * (m as usize) * (n as usize);
    if x.len() < (m as usize) * (k as usize)
        || b.len() < (k as usize) * (n as usize)
        || stage_f32.len() < need
    {
        return Err(SparkError::InvalidArgument(format!(
            "gemm_bf16_split_k_det: buffer too small (stage need {need}, got {})",
            stage_f32.len()
        )));
    }
    let func = module::load_kernel(ctx, "gemm_bf16_split_k_det", "gemm_bf16_split_k_det")?;
    let threads = 128u32;
    let cols_per_block = threads * 8;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b)
            .arg(stage_f32)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .arg(&m)
            .launch(cfg)?;
    }
    Ok(())
}

/// Managed batched split-K: runs `gemm_bf16_split_k_det` then the shard reducer
/// over the flattened `M*N` output, writing `out_bf16[m*N + col]`. `out_bf16`
/// must hold `m*n` elements; `stage_f32` must hold `num_shards*m*n`. Row `m` of
/// the result is byte-identical to `gemv_bf16_split_k_managed` on row `m` alone
/// (same `num_shards`), under SPARK_DETERMINISTIC or not (no atomics either way).
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_split_k_det_managed(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    stage_f32: &mut CudaSlice<f32>,
    out_bf16: &mut CudaSlice<u16>,
    n: u32,
    k: u32,
    m: u32,
    num_shards: u32,
) -> Result<()> {
    gemm_bf16_split_k_det(ctx, stream, x, b, stage_f32, n, k, m, num_shards)?;
    // Reduce over shards for each of the M*N flattened outputs in one shot.
    f32_shard_reduce_to_bf16(ctx, stream, stage_f32, out_bf16, m * n, num_shards)?;
    Ok(())
}

/// Deterministic shard-reducer + bf16 cast. Reads
/// `stage_f32[num_shards × N]`, sums per output across shards, writes BF16.
pub fn f32_shard_reduce_to_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    stage_f32: &CudaSlice<f32>,
    out_bf16: &mut CudaSlice<u16>,
    n: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let need = (num_shards as usize) * (n as usize);
    if stage_f32.len() < need || out_bf16.len() < n as usize {
        return Err(SparkError::InvalidArgument("shard_reduce buffers".into()));
    }
    let func = module::load_kernel(ctx, "f32_shard_reduce_to_bf16", "f32_shard_reduce_to_bf16")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(stage_f32)
            .arg(out_bf16)
            .arg(&n)
            .arg(&num_shards)
            .launch(cfg)?;
    }
    Ok(())
}

/// Wrapper: GEMV split-K with managed F32 scratch + final cast to BF16.
/// Caller provides a pre-allocated F32 scratch of size N (zeroed by this fn).
pub fn gemv_bf16_split_k_managed(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    out_f32_scratch: &mut CudaSlice<f32>,
    out_bf16: &mut CudaSlice<u16>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    // Deterministic mode keeps the K-shard parallelism
    // but writes partials to a staging buffer (no atomicAdd) and then
    // runs a tiny shard-reducer + cast. Same throughput as split-K, just
    // bytewise-stable. Requires the scratch to be ≥ num_shards × N.
    if is_deterministic() {
        let need = (num_shards as usize) * (n as usize);
        if out_f32_scratch.len() < need {
            // Fallback: scratch too small for det layout; use atomic-free
            // gemv_bf16 instead (still deterministic, just slower).
            return gemv_bf16(ctx, stream, x, b, out_bf16, n, k);
        }
        gemv_bf16_split_k_det(ctx, stream, x, b, out_f32_scratch, n, k, num_shards)?;
        f32_shard_reduce_to_bf16(ctx, stream, out_f32_scratch, out_bf16, n, num_shards)?;
        return Ok(());
    }
    stream
        .memset_zeros(out_f32_scratch)
        .map_err(SparkError::Driver)?;
    gemv_bf16_split_k(ctx, stream, x, b, out_f32_scratch, n, k, num_shards)?;
    crate::activation::f32_to_bf16(ctx, stream, out_f32_scratch, out_bf16, n)?;
    Ok(())
}

/// View variant of `gemv_bf16_split_k`: weight matrix B can be a sub-view (e.g.,
/// a single expert's slice of a stacked-experts buffer).
///
/// Signature mirrors `gemv_bf16_split_k` except `b` is a `&CudaView<u16>`.
pub fn gemv_bf16_split_k_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaView<u16>,
    out_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_split_k_view requires N divisible by 8, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_split_k_view: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    if x.len() < k as usize || b.len() < (k as usize) * (n as usize) || out_f32.len() < n as usize {
        return Err(SparkError::InvalidArgument("buffer sizes".into()));
    }
    let func = module::load_kernel(ctx, "gemv_bf16_split_k", "gemv_bf16_split_k")?;
    let threads = 128u32;
    let cols_per_block = threads * 8;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b)
            .arg(out_f32)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// View variant of `gemv_bf16_split_k_managed`: zeros F32 scratch, runs the
/// view-based split-K gemv, casts to BF16.
pub fn gemv_bf16_split_k_managed_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaView<u16>,
    out_f32_scratch: &mut CudaSlice<f32>,
    out_bf16: &mut CudaSlice<u16>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    stream
        .memset_zeros(out_f32_scratch)
        .map_err(SparkError::Driver)?;
    gemv_bf16_split_k_view(ctx, stream, x, b, out_f32_scratch, n, k, num_shards)?;
    crate::activation::f32_to_bf16(ctx, stream, out_f32_scratch, out_bf16, n)?;
    Ok(())
}

/// v4 of split-K GEMV: K loop unrolled by 2 → 16 FMAs per outer iter (vs v3's 8).
pub fn gemv_bf16_split_k_v4(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    out_f32: &mut CudaSlice<f32>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_split_k_v4 requires N divisible by 8, got {n}"
        )));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_split_k_v4: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    let func = module::load_kernel(ctx, "gemv_bf16_split_k_v4", "gemv_bf16_split_k_v4")?;
    let threads = 128u32;
    let cols_per_block = threads * 8;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b)
            .arg(out_f32)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}

/// v4 managed wrapper: zero + gemv_split_k_v4 + cast.
pub fn gemv_bf16_split_k_v4_managed(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    out_f32_scratch: &mut CudaSlice<f32>,
    out_bf16: &mut CudaSlice<u16>,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    stream
        .memset_zeros(out_f32_scratch)
        .map_err(SparkError::Driver)?;
    gemv_bf16_split_k_v4(ctx, stream, x, b, out_f32_scratch, n, k, num_shards)?;
    crate::activation::f32_to_bf16(ctx, stream, out_f32_scratch, out_bf16, n)?;
    Ok(())
}

/// Launch BF16 GEMM kernel using MMA (mma.sync.aligned.m16n8k16).
///
/// Uses 128x64 tile with 4 warps and cp.async double-buffered pipeline.
/// Requires M divisible by 128, N divisible by 64, K divisible by 16.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    if !m.is_multiple_of(128) || !n.is_multiple_of(64) {
        return Err(SparkError::InvalidArgument(format!(
            "BF16 MMA GEMM requires M divisible by 128 and N by 64: M={m}, N={n}"
        )));
    }
    if !k.is_multiple_of(16) {
        return Err(SparkError::InvalidArgument(format!(
            "BF16 MMA GEMM requires K divisible by 16: K={k}"
        )));
    }
    if a.len() < m as usize * k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "A buffer too small: {} < {}",
            a.len(),
            m as usize * k as usize
        )));
    }
    if b.len() < k as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "B buffer too small: {} < {}",
            b.len(),
            k as usize * n as usize
        )));
    }
    if c.len() < m as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "C buffer too small: {} < {}",
            c.len(),
            m as usize * n as usize
        )));
    }
    let func = module::load_kernel(ctx, "gemm_bf16_mma", "gemm_bf16_mma")?;

    // 128x64 tile, 4 warps (128 threads) per block, 12KB SMEM
    let grid_x = n / 64;
    let grid_y = m / 128;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 12288,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch BF16 GEMM v2 — K-block = 32 (vs v1's 16) and 3-stage cp.async pipeline
/// (vs 2). Same 128×64 tile, same 4 warps. Halves loop overhead, doubles MMA per
/// K-iter. Requires M%128, N%64, K%32.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_mma_v2(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    if !m.is_multiple_of(128) || !n.is_multiple_of(64) {
        return Err(SparkError::InvalidArgument(format!(
            "BF16 MMA GEMM v2 requires M divisible by 128 and N by 64: M={m}, N={n}"
        )));
    }
    if !k.is_multiple_of(32) {
        return Err(SparkError::InvalidArgument(format!(
            "BF16 MMA GEMM v2 requires K divisible by 32: K={k}"
        )));
    }
    if a.len() < m as usize * k as usize
        || b.len() < k as usize * n as usize
        || c.len() < m as usize * n as usize
    {
        return Err(SparkError::InvalidArgument(
            "GEMM v2 buffer size mismatch".into(),
        ));
    }
    let func = module::load_kernel(ctx, "gemm_bf16_mma_v2", "gemm_bf16_mma_v2")?;

    let grid_x = n / 64;
    let grid_y = m / 128;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0, // SMEM declared statically in kernel (.shared)
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

/// Auto-dispatch BF16 GEMM. Picks v3 (128×128 tile, 8 warps) for large shapes
/// where SM saturation amortizes the bigger tile — measured 1.30×–1.69× over v1
/// at ≥2048×2048. Falls back to v1 for smaller shapes (v1 wins below 1024² due
/// to more CTAs filling more SMs) or shapes that don't satisfy v3's M%128/N%128
/// constraints.
///
/// Bench (median of 50, BF16, K=K_dim, M=N=K):
///   shape          v1 TFLOPS  v3 TFLOPS  v3/v1
///   1024³                30.7       31.1   1.01
///   2048³                28.4       37.1   1.30
///   4096³                24.8       42.0   1.69  ← beats 40 TFLOPS target
///   1024×4096×4096       24.2       40.6   1.68
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_mma_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    // v3 wins when grid saturates SMs (≈48 CTAs ≥) AND meets shape constraints.
    let v3_eligible = m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(32);
    // Prefill runs M=256 dense GEMMs (chunk=128 → M next_multiple_of(128),
    // 256-token chunk → M=256) at N∈{4096,12288}. Those are compute-bound and v3's
    // 128×128 tile beats v1's 128×64 by ~1.7× at these shapes. The original gate
    // (m>=1024 && n>=1024) kept them on v1. Lower the M floor to 256 with a wide-N
    // requirement so prefill catches v3, while M<256 (e.g. M=128) stays on v1 —
    // Measurement showed v3 UNDER-saturates SMs at M=128 (97 blocks vs v1's many),
    // so we must NOT route M=128 here. Decode (M=128 batched, M=1 GEMV) uses gemv_*
    // not this GEMM, so it is untouched regardless. n>=4096 ensures the v3 win is real.
    let large_enough = (m >= 1024 && n >= 1024) || (m >= 256 && n >= 4096);
    // v4 (warp-spec + TMA + 3-stage) beats v3 in the huge-N regime
    // (lm_head shape, N=248K) by ~16% at M=512+. For more typical N
    // (up to MLP/QKV widths) v3 still wins. See docs/gemm_v4_design.md.
    let v4_huge_n = v3_eligible && m >= 128 && n >= 65536 && k >= 1024;
    // v5 (128×256 register-blocked + threadblock swizzle) beats v3 in the large
    // compute-bound square regime: measured +12% at 2048³ and +12–20% at 4096³.
    // Requires N%256. Gated to M,N ≥ 2048 where the win is measured; smaller/
    // rectangular shapes stay on v3 — v5 under-saturates the 48 SMs at low M
    // (1 block/SM). See the gemm_bf16_mma_v5 header below for the design.
    let v5_win = m.is_multiple_of(128)
        && n.is_multiple_of(256)
        && k.is_multiple_of(32)
        && m >= 2048
        && n >= 2048;
    // Small-M / unaligned fallback: the MMA kernels (v1/v2/v3/v4) all require
    // M divisible by 128 (or at least 32) and N by 64. Speculative-decode verify
    // passes run forward_prefill_chunk at tiny M (k+1, e.g. 4), which those
    // kernels reject. Route any M not aligned to 128 (or N not aligned to 64)
    // through the scalar 16x16-tile gemm_bf16, which handles arbitrary dims via
    // div_ceil. This does NOT change behavior for the M%128==0 callers (chunked
    // prefill at chunk=128/256, batched M=128) — they still take the MMA path.
    let mma_aligned = m.is_multiple_of(128) && n.is_multiple_of(64);
    if !mma_aligned {
        gemm_bf16(ctx, stream, a, b, c, m, n, k)
    } else if v4_huge_n {
        gemm_bf16_mma_v4(ctx, stream, a, b, c, m, n, k)
    } else if v5_win {
        gemm_bf16_mma_v5(ctx, stream, a, b, c, m, n, k)
    } else if v3_eligible && large_enough {
        gemm_bf16_mma_v3(ctx, stream, a, b, c, m, n, k)
    } else {
        gemm_bf16_mma(ctx, stream, a, b, c, m, n, k)
    }
}

/// Launch BF16 GEMM v3 — 128×128 tile (vs v1/v2's 128×64) with 8 warps in 2×4
/// layout. K=32 inner block, 2-stage cp.async pipeline. 32 KB static SMEM.
/// Targets large-shape regimes where v1's 4-warp 128×64 leaves tensor cores
/// unsaturated. Requires M%128, N%128, K%32.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_mma_v3(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    if !m.is_multiple_of(128) || !n.is_multiple_of(128) {
        return Err(SparkError::InvalidArgument(format!(
            "BF16 MMA GEMM v3 requires M and N divisible by 128: M={m}, N={n}"
        )));
    }
    if !k.is_multiple_of(32) {
        return Err(SparkError::InvalidArgument(format!(
            "BF16 MMA GEMM v3 requires K divisible by 32: K={k}"
        )));
    }
    if a.len() < m as usize * k as usize
        || b.len() < k as usize * n as usize
        || c.len() < m as usize * n as usize
    {
        return Err(SparkError::InvalidArgument(
            "GEMM v3 buffer size mismatch".into(),
        ));
    }
    let func = module::load_kernel(ctx, "gemm_bf16_mma_v3", "gemm_bf16_mma_v3")?;

    let grid_x = n / 128;
    let grid_y = m / 128;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (256, 1, 1), // 8 warps
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

/// BF16 MMA GEMM v5 — 128×256 register-blocked tile (vs v3's 128×128). Each
/// thread holds 128 FP32 accumulators (vs 64), trading occupancy (≈1 block/SM)
/// for compute-per-thread. NCU showed v3 is occupancy-bound but the register
/// knob regressed; raising compute-per-thread is the CUTLASS-style alternative.
/// Requires M%128, N%256, K%32.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_mma_v5(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    if !m.is_multiple_of(128) || !n.is_multiple_of(256) {
        return Err(SparkError::InvalidArgument(format!(
            "BF16 MMA GEMM v5 requires M%128==0 and N%256==0: M={m}, N={n}"
        )));
    }
    if !k.is_multiple_of(32) {
        return Err(SparkError::InvalidArgument(format!(
            "BF16 MMA GEMM v5 requires K divisible by 32: K={k}"
        )));
    }
    if a.len() < m as usize * k as usize
        || b.len() < k as usize * n as usize
        || c.len() < m as usize * n as usize
    {
        return Err(SparkError::InvalidArgument(
            "GEMM v5 buffer size mismatch".into(),
        ));
    }
    let func = module::load_kernel(ctx, "gemm_bf16_mma_v5", "gemm_bf16_mma_v5")?;

    let grid_x = n / 256;
    let grid_y = m / 128;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (256, 1, 1), // 8 warps
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

/// Dense W8A16 GEMM v3 — BF16 A × FP8 (E4M3) B × per-tensor scale → BF16 C.
/// Modeled on `gemm_bf16_mma_v3`: 128×128 tile, 8 warps, K-block 32. Only B
/// differs: cp.async loads FP8 (half bandwidth), inline dequant pass
/// converts to BF16 in SMEM (applies per-tensor scale), then standard BF16
/// MMA. Replaces the earlier 3-kernel composite (quant + GEMM + scale)
/// with ONE kernel — saves 2 launches per GEMM site and avoids the activation
/// quant + post-scale memory traffic.
///
/// Constraints: M%128, N%128, K%32 (same as BF16/FP8 v3).
#[allow(clippy::too_many_arguments)]
pub fn gemm_w8a16_mma_v3(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
    b_scale: f32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    if !m.is_multiple_of(128) || !n.is_multiple_of(128) {
        return Err(SparkError::InvalidArgument(format!(
            "W8A16 MMA GEMM v3 requires M and N divisible by 128: M={m}, N={n}"
        )));
    }
    if !k.is_multiple_of(32) {
        return Err(SparkError::InvalidArgument(format!(
            "W8A16 MMA GEMM v3 requires K divisible by 32: K={k}"
        )));
    }
    if a.len() < m as usize * k as usize
        || b.len() < k as usize * n as usize
        || c.len() < m as usize * n as usize
    {
        return Err(SparkError::InvalidArgument(
            "W8A16 v3 buffer size mismatch".into(),
        ));
    }
    if !(b_scale > 0.0 && b_scale.is_finite()) {
        return Err(SparkError::InvalidArgument(format!(
            "b_scale must be > 0 and finite; got {b_scale}"
        )));
    }
    let func = module::load_kernel(ctx, "gemm_w8a16_mma_v3", "gemm_w8a16_mma_v3")?;

    let grid_x = n / 128;
    let grid_y = m / 128;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (256, 1, 1), // 8 warps
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .arg(&b_scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch FP8 GEMM kernel using MMA (mma.sync.aligned.m16n8k32).
///
/// A, B: row-major FP8 e4m3 (represented as u8).
/// C: row-major BF16 (represented as u16).
/// Requires M, N divisible by 32 and K divisible by 32.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    validate_mma_alignment(m, n, k, 32)?;
    if a.len() < m as usize * k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "A buffer too small: {} < {}",
            a.len(),
            m as usize * k as usize
        )));
    }
    if b.len() < k as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "B buffer too small: {} < {}",
            b.len(),
            k as usize * n as usize
        )));
    }
    if c.len() < m as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "C buffer too small: {} < {}",
            c.len(),
            m as usize * n as usize
        )));
    }
    let func = module::load_kernel(ctx, "gemm_fp8_mma", "gemm_fp8_mma")?;

    // 32x32 tile, 1 warp (32 threads) per block
    let grid_x = n.div_ceil(32);
    let grid_y = m.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 2048,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch FP8 GEMM v3 — 128×128 tile (vs v1's 32×32) with 8 warps in 2×4 layout.
/// Same v3 strategy that gave 1.69× win in BF16 GEMM. K=32 inner block, 2-stage
/// cp.async pipeline. Requires M%128, N%128, K%32.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_mma_v3(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    if !m.is_multiple_of(128) || !n.is_multiple_of(128) {
        return Err(SparkError::InvalidArgument(format!(
            "FP8 MMA GEMM v3 requires M and N divisible by 128: M={m}, N={n}"
        )));
    }
    if !k.is_multiple_of(32) {
        return Err(SparkError::InvalidArgument(format!(
            "FP8 MMA GEMM v3 requires K divisible by 32: K={k}"
        )));
    }
    if a.len() < m as usize * k as usize
        || b.len() < k as usize * n as usize
        || c.len() < m as usize * n as usize
    {
        return Err(SparkError::InvalidArgument(
            "GEMM FP8 v3 buffer size mismatch".into(),
        ));
    }
    let func = module::load_kernel(ctx, "gemm_fp8_mma_v3", "gemm_fp8_mma_v3")?;

    let grid_x = n / 128;
    let grid_y = m / 128;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch FP8 GEMM v3.5 — same 128×128 tile + 8 warps as v3, 4-stage cp.async
/// pipeline (vs v3's 2). 32 KB SMEM. Up to 2 transfers in flight while computing
/// → more LSU bandwidth utilization. Requires K%96 (3-stage prologue covers K=0..95).
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_mma_v3_5(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    if !m.is_multiple_of(128) || !n.is_multiple_of(128) {
        return Err(SparkError::InvalidArgument(format!(
            "FP8 MMA GEMM v3.5 requires M and N divisible by 128: M={m}, N={n}"
        )));
    }
    if !k.is_multiple_of(32) {
        return Err(SparkError::InvalidArgument(format!(
            "FP8 MMA GEMM v3.5 requires K divisible by 32: K={k}"
        )));
    }
    if a.len() < m as usize * k as usize
        || b.len() < k as usize * n as usize
        || c.len() < m as usize * n as usize
    {
        return Err(SparkError::InvalidArgument(
            "GEMM FP8 v3.5 buffer size mismatch".into(),
        ));
    }
    let func = module::load_kernel(ctx, "gemm_fp8_mma_v3_5", "gemm_fp8_mma_v3_5")?;

    let grid_x = n / 128;
    let grid_y = m / 128;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

/// Auto-dispatch FP8 GEMM. Routes to v3 (128×128 tile, 8 warps) for shapes that
/// satisfy v3 constraints (M%128, N%128, K%32) and are large enough to saturate
/// SMs (M ≥ 1024 AND N ≥ 1024). Falls back to v1 (32×32 tile, broader shape
/// support) otherwise.
///
/// Note: v3.5 (4-stage pipeline) is also available but measured neutral vs v3
/// at large shapes — confirms cp.async latency is not the bottleneck. Kept as
/// a documented exploration; not in the auto path.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_mma_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    let v3_eligible = m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(32);
    // Mirror the BF16 auto gate — route M>=256 wide-N GEMMs (prefill shape)
    // to v3, keep M<256 on v1 (v3 under-saturates SMs at low M, measured).
    let large_enough = (m >= 1024 && n >= 1024) || (m >= 256 && n >= 4096);
    if v3_eligible && large_enough {
        gemm_fp8_mma_v3(ctx, stream, a, b, c, m, n, k)
    } else {
        gemm_fp8_mma(ctx, stream, a, b, c, m, n, k)
    }
}

/// Launch NVFP4 block-scaled GEMM kernel using MMA.
///
/// A: [M, K/2] u8 — e2m1 nibble-packed, row-major (K elements packed as K/2 bytes).
/// B: [K/2, N] u8 — e2m1 K-packed (each byte = 2 consecutive K-elements for 1 N-column).
/// C: [M, N] u16 (bf16) — output, row-major.
/// scale_a: [ceil(M/16) * ceil(K/64)] u8 — one UE8M0 per (16-row, 64-element) A block.
/// scale_b: [ceil(K/64) * ceil(N/8)] u8 — one UE8M0 per (64-element, 8-col) B block.
///
/// Requires M, N divisible by 32 and K divisible by 64.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nvfp4_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<u16>,
    scale_a: &CudaSlice<u8>,
    scale_b: &CudaSlice<u8>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    validate_mma_alignment(m, n, k, 64)?;
    // Scale factors: one UE8M0 byte per MMA fragment.
    // scale_a: ceil(M/16) * ceil(K/64) entries (one per A-fragment)
    // scale_b: ceil(K/64) * ceil(N/8) entries (one per B-fragment)
    let sa_need = m.div_ceil(16) as usize * k.div_ceil(64) as usize;
    let sb_need = k.div_ceil(64) as usize * n.div_ceil(8) as usize;
    if scale_a.len() < sa_need {
        return Err(SparkError::InvalidArgument(format!(
            "scale_a buffer too small: {} < {} (ceil(M/16)*ceil(K/64))",
            scale_a.len(),
            sa_need
        )));
    }
    if scale_b.len() < sb_need {
        return Err(SparkError::InvalidArgument(format!(
            "scale_b buffer too small: {} < {} (ceil(K/64)*ceil(N/8))",
            scale_b.len(),
            sb_need
        )));
    }
    // A and B are nibble-packed: K/2 bytes per row/col
    if a.len() < m as usize * k as usize / 2 {
        return Err(SparkError::InvalidArgument(format!(
            "A buffer too small: {} < {}",
            a.len(),
            m as usize * k as usize / 2
        )));
    }
    if b.len() < k as usize / 2 * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "B buffer too small: {} < {}",
            b.len(),
            k as usize / 2 * n as usize
        )));
    }
    if c.len() < m as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "C buffer too small: {} < {}",
            c.len(),
            m as usize * n as usize
        )));
    }
    let func = module::load_kernel(ctx, "gemm_nvfp4_mma", "gemm_nvfp4_mma")?;

    // 32x32 tile, 1 warp (32 threads) per block
    let grid_x = n.div_ceil(32);
    let grid_y = m.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 2048,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(scale_a)
            .arg(scale_b)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch W4A16 dequant GEMM kernel: C[M,N] = A[M,K] x dequant(W[K,N], scales, zeros)
///
/// A: [M, K] bf16 activations, row-major.
/// W: [K, N/2] u8 INT4-packed weights (2 per byte, lo nibble = even col).
/// C: [M, N] bf16 output, row-major.
/// scales: [N] bf16 per-column dequant scale.
/// zeros: [N] bf16 per-column zero point.
///
/// Dequant: w_float = (int4_val + zero) * scale
/// Requires M, N divisible by 32 and K divisible by 16.
#[allow(clippy::too_many_arguments)]
pub fn gemm_w4a16_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    w: &CudaSlice<u8>,
    c: &mut CudaSlice<u16>,
    scales: &CudaSlice<u16>,
    zeros: &CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_gemm_dims(m, n, k)?;
    validate_mma_alignment(m, n, k, 16)?;
    if a.len() < m as usize * k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "A buffer too small: {} < {}",
            a.len(),
            m as usize * k as usize
        )));
    }
    // W is INT4-packed: N/2 bytes per row
    if w.len() < k as usize * n as usize / 2 {
        return Err(SparkError::InvalidArgument(format!(
            "W buffer too small: {} < {}",
            w.len(),
            k as usize * n as usize / 2
        )));
    }
    if c.len() < m as usize * n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "C buffer too small: {} < {}",
            c.len(),
            m as usize * n as usize
        )));
    }
    if scales.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "scales buffer too small: {} < {n}",
            scales.len()
        )));
    }
    if zeros.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "zeros buffer too small: {} < {n}",
            zeros.len()
        )));
    }
    let func = module::load_kernel(ctx, "gemm_w4a16_mma", "gemm_w4a16_mma")?;

    // 32x32 tile, 1 warp (32 threads) per block
    let grid_x = n.div_ceil(32);
    let grid_y = m.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 2176,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(w)
            .arg(c)
            .arg(scales)
            .arg(zeros)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

/// Dense GEMM with DSv3 block-scaled FP8 weights (BF16 activations).
///
/// Computes `C[m, n] = Σ_k A[m, k] * B[n, k] * scales[n, k / 128]` where
/// `scales` carries one FP32 scale per 1×128 K-block per output row. Used by
/// DeepSeek V3 / V3.1 / V3.2 dense MLP layers and FP8 attention paths.
///
/// `A`: [M, K] BF16 row-major
/// `B`: [N, K] FP8 e4m3 row-major (1 byte per element)
/// `scales`: [N, K/128] FP32
/// `C`: [M, N] BF16
///
/// `K` must be divisible by 128.
///
/// Scalar reference. An MMA-optimized variant is tracked as follow-up.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_block128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u8>,
    scales: &CudaSlice<f32>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !k.is_multiple_of(128) {
        return Err(SparkError::InvalidArgument(
            "k must be divisible by 128 (DSv3 block size)".into(),
        ));
    }
    if a.len() < (m * k) as usize {
        return Err(SparkError::InvalidArgument(format!(
            "a buffer too small: {} < {}",
            a.len(),
            m * k
        )));
    }
    if b.len() < (n * k) as usize {
        return Err(SparkError::InvalidArgument(format!(
            "b buffer too small: {} < {}",
            b.len(),
            n * k
        )));
    }
    let scales_need = (n * (k / 128)) as usize;
    if scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "scales buffer too small: {} < {scales_need}",
            scales.len()
        )));
    }
    if c.len() < (m * n) as usize {
        return Err(SparkError::InvalidArgument(format!(
            "c buffer too small: {} < {}",
            c.len(),
            m * n
        )));
    }

    let func = module::load_kernel(ctx, "gemm_fp8_block128", "gemm_fp8_block128")?;
    let grid_n = n.div_ceil(32);
    let grid_m = m.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(scales)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// MMA-optimized variant of `gemm_fp8_block128`. Same math and layout, BF16
/// MMA m16n8k16 inner loop with vectorized B loads (1 col × 16 K-rows
/// contiguous per inner iter — same pattern as `gemm_fp8_block128_grouped_mma_v2`).
///
/// Use this in production when M, N, K are large enough that scalar accum
/// is the bottleneck. Target ≥40 TFLOPS (similar regime as the BF16 MMA v2
/// kernel) at DSv3 dense-MLP shapes.
///
/// `K` must be divisible by 128.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_block128_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u8>,
    scales: &CudaSlice<f32>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !k.is_multiple_of(128) {
        return Err(SparkError::InvalidArgument(
            "k must be divisible by 128 (DSv3 block size)".into(),
        ));
    }
    if !n.is_multiple_of(2) {
        // The kernel packs two BF16 outputs into a single b32 store. When N
        // is odd, consecutive rows are at non-4-aligned addresses. Always
        // true in practice for DeepSeek V3 weights (4096, 7168, 18432, ...).
        return Err(SparkError::InvalidArgument(
            "n must be even (kernel packs 2 BF16 per b32 store)".into(),
        ));
    }
    if a.len() < (m * k) as usize {
        return Err(SparkError::InvalidArgument(format!(
            "a buffer too small: {} < {}",
            a.len(),
            m * k
        )));
    }
    if b.len() < (n * k) as usize {
        return Err(SparkError::InvalidArgument(format!(
            "b buffer too small: {} < {}",
            b.len(),
            n * k
        )));
    }
    let scales_need = (n * (k / 128)) as usize;
    if scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "scales buffer too small: {} < {scales_need}",
            scales.len()
        )));
    }
    if c.len() < (m * n) as usize {
        return Err(SparkError::InvalidArgument(format!(
            "c buffer too small: {} < {}",
            c.len(),
            m * n
        )));
    }

    let func = module::load_kernel(ctx, "gemm_fp8_block128_mma", "gemm_fp8_block128_mma")?;
    let grid_n = n.div_ceil(32);
    let grid_m = m.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(scales)
            .arg(c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

// ============================================================================
// Warp-specialized + TMA GEMM v4
// ============================================================================

/// Create a 2D TMA tensor map for a BF16 matrix.
///
/// `rows × cols` is the logical shape (row-major), `tile_rows × tile_cols`
/// is the per-launch box that the kernel will pull via cp.async.bulk.tensor.
/// TMA convention: dim[0] is the innermost (cols), dim[1] is the outermost
/// (rows). The kernel passes coords as {coord_x=col_off, coord_y=row_off}.
fn create_tma_desc_bf16_2d(
    global_ptr: *mut core::ffi::c_void,
    rows: u32,
    cols: u32,
    tile_rows: u32,
    tile_cols: u32,
) -> Result<cudarc::driver::sys::CUtensorMap> {
    use cudarc::driver::sys::*;
    let mut tma = CUtensorMap_st::default();
    let global_dim: [u64; 2] = [cols as u64, rows as u64];
    let global_strides: [u64; 1] = [(cols as u64) * 2]; // bytes per row
    let box_dim: [u32; 2] = [tile_cols, tile_rows];
    let elem_strides: [u32; 2] = [1, 1];
    let r = unsafe {
        cuTensorMapEncodeTiled(
            &mut tma,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            global_ptr,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            elem_strides.as_ptr(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if r != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuTensorMapEncodeTiled failed: {r:?}"
        )));
    }
    Ok(tma)
}

/// Warp-specialized + TMA BF16 GEMM (v4).
///
/// Step 4 status: single-K-block kernel only. Requires K == 32 exactly
/// (single K-iteration), M%128==0, N%128==0. The K-loop and multi-stage
/// pipeline will be added in subsequent steps.
pub fn gemm_bf16_mma_v4(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    use cudarc::driver::sys::*;
    use cudarc::driver::DevicePtr;
    validate_gemm_dims(m, n, k)?;
    if !m.is_multiple_of(128) || !n.is_multiple_of(128) || !k.is_multiple_of(32) {
        return Err(SparkError::InvalidArgument(format!(
            "v4 requires M%128==0, N%128==0, K%32==0; got M={m}, N={n}, K={k}"
        )));
    }
    if a.len() < (m as usize) * (k as usize)
        || b.len() < (k as usize) * (n as usize)
        || c.len() < (m as usize) * (n as usize)
    {
        return Err(SparkError::InvalidArgument("buffer sizes too small".into()));
    }

    let (a_dptr, _ag) = a.device_ptr(stream);
    let (b_dptr, _bg) = b.device_ptr(stream);
    let (c_dptr, _cg) = c.device_ptr(stream);

    let a_tma = create_tma_desc_bf16_2d(a_dptr as *mut _, m, k, 128, 32)?;
    let b_tma = create_tma_desc_bf16_2d(b_dptr as *mut _, k, n, 32, 128)?;

    // Copy TMA descriptors to device (each CUtensorMap is 128 B = 32 u32).
    let a_tma_u32: [u32; 32] = unsafe { core::mem::transmute(a_tma) };
    let b_tma_u32: [u32; 32] = unsafe { core::mem::transmute(b_tma) };
    let a_tma_dev = stream.memcpy_stod(&a_tma_u32).map_err(SparkError::Driver)?;
    let b_tma_dev = stream.memcpy_stod(&b_tma_u32).map_err(SparkError::Driver)?;
    let (a_tma_dptr, _) = a_tma_dev.device_ptr(stream);
    let (b_tma_dptr, _) = b_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "gemm_bf16_mma_v4", "gemm_bf16_mma_v4")?;
    let cu_stream = stream.cu_stream();

    const SMEM_BYTES: i32 = 49232; // 48 KB stages + 72 B mbarriers (A/B/DONE per stage)
    let attr = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_BYTES,
        )
    };
    if attr != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES={SMEM_BYTES} failed: {attr:?}"
        )));
    }

    let grid_n = n / 128;
    let grid_m = m / 128;
    let params: [*mut core::ffi::c_void; 6] = [
        &a_tma_dptr as *const u64 as *mut _,
        &b_tma_dptr as *const u64 as *mut _,
        &c_dptr as *const u64 as *mut _,
        &m as *const u32 as *mut _,
        &n as *const u32 as *mut _,
        &k as *const u32 as *mut _,
    ];
    let r = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_n,
            grid_m,
            1,
            288,
            1,
            1, // 9 warps (1 producer + 8 consumer)
            SMEM_BYTES as u32,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };
    if r != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "v4 cuLaunchKernel failed: {r:?}"
        )));
    }
    Ok(())
}

/// v4 K-block=64 variant.
///
/// Same warp-spec + TMA architecture as v4 but with 2-stage pipeline
/// and K-block=64. Each producer TMA pair loads 16 KB A + 16 KB B
/// (vs v4's 8 + 8), and each consumer K-iter does 64 MMAs/warp (vs 32).
/// Trades 3-stage pipeline depth for larger per-stage compute.
///
/// SMEM: 64 KB stages + 48 B mbarriers = 65,584 B.
/// Requires M%128==0, N%128==0, K%64==0.
pub fn gemm_bf16_mma_v4_k64(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    c: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    use cudarc::driver::sys::*;
    use cudarc::driver::DevicePtr;
    validate_gemm_dims(m, n, k)?;
    if !m.is_multiple_of(128) || !n.is_multiple_of(128) || !k.is_multiple_of(64) {
        return Err(SparkError::InvalidArgument(format!(
            "v4_k64 requires M%128==0, N%128==0, K%64==0; got M={m}, N={n}, K={k}"
        )));
    }
    if a.len() < (m as usize) * (k as usize)
        || b.len() < (k as usize) * (n as usize)
        || c.len() < (m as usize) * (n as usize)
    {
        return Err(SparkError::InvalidArgument("buffer sizes too small".into()));
    }

    let (a_dptr, _ag) = a.device_ptr(stream);
    let (b_dptr, _bg) = b.device_ptr(stream);
    let (c_dptr, _cg) = c.device_ptr(stream);

    // TMA box is 128×64 for A and 64×128 for B (K=64 per stage)
    let a_tma = create_tma_desc_bf16_2d(a_dptr as *mut _, m, k, 128, 64)?;
    let b_tma = create_tma_desc_bf16_2d(b_dptr as *mut _, k, n, 64, 128)?;

    let a_tma_u32: [u32; 32] = unsafe { core::mem::transmute(a_tma) };
    let b_tma_u32: [u32; 32] = unsafe { core::mem::transmute(b_tma) };
    let a_tma_dev = stream.memcpy_stod(&a_tma_u32).map_err(SparkError::Driver)?;
    let b_tma_dev = stream.memcpy_stod(&b_tma_u32).map_err(SparkError::Driver)?;
    let (a_tma_dptr, _) = a_tma_dev.device_ptr(stream);
    let (b_tma_dptr, _) = b_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "gemm_bf16_mma_v4_k64", "gemm_bf16_mma_v4_k64")?;
    let cu_stream = stream.cu_stream();

    const SMEM_BYTES: i32 = 98376; // 96 KB stages + 72 B mbarriers (3-stage)
    let attr = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_BYTES,
        )
    };
    if attr != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES={SMEM_BYTES} failed: {attr:?}"
        )));
    }

    let grid_n = n / 128;
    let grid_m = m / 128;
    let params: [*mut core::ffi::c_void; 6] = [
        &a_tma_dptr as *const u64 as *mut _,
        &b_tma_dptr as *const u64 as *mut _,
        &c_dptr as *const u64 as *mut _,
        &m as *const u32 as *mut _,
        &n as *const u32 as *mut _,
        &k as *const u32 as *mut _,
    ];
    let r = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_n,
            grid_m,
            1,
            288,
            1,
            1,
            SMEM_BYTES as u32,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };
    if r != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "v4_k64 cuLaunchKernel failed: {r:?}"
        )));
    }
    Ok(())
}
