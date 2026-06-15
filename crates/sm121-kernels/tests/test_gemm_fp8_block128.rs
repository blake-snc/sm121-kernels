//! Dense FP8 1×128 block-scaled GEMM (DSv3 / DeepSeek V3 weight format).
//!
//! Uses pre-baked FP8 byte values (0x38 = 1.0 in e4m3) and known scales so
//! the expected math reduces to scalar multiplies on f32 sums — same pattern
//! as the existing grouped test in test_moe_fp8_block128.rs.

mod common;

use sm121_kernels::{device, gemm};

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

/// Drives both kernel variants through the same correctness suite. Reuses one
/// device + a single set of host inputs per case, so the kernels are A/B'd
/// against identical fixtures.
fn run_dense_correctness<F>(launch: F, label: &str)
where
    F: Fn(
        &std::sync::Arc<cudarc::driver::CudaContext>,
        &std::sync::Arc<cudarc::driver::CudaStream>,
        &cudarc::driver::CudaSlice<u16>,
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<f32>,
        &mut cudarc::driver::CudaSlice<u16>,
        u32,
        u32,
        u32,
    ) -> Result<(), sm121_kernels::SparkError>,
{
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    // Case 1: uniform — single K-block, all FP8 = 1.0, scale = 0.5
    {
        let m = 64u32;
        let n = 32u32;
        let k = 128u32;
        let mut a_bf16 = vec![0u16; (m * k) as usize];
        for i in 0..a_bf16.len() {
            let f = 0.1 * ((i % 7) as f32 - 3.0);
            a_bf16[i] = f32_to_bf16_bits(f);
        }
        let b_fp8 = vec![0x38u8; (n * k) as usize];
        let b_scales = vec![0.5f32; n as usize];
        let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
        let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
        let s_dev = stream.memcpy_stod(&b_scales).unwrap();
        let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
        launch(&ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, m, n, k)
            .unwrap_or_else(|e| panic!("{label} uniform: {e:?}"));
        let c_host = stream.memcpy_dtov(&c_dev).unwrap();
        let a_f32: Vec<f32> = a_bf16.iter().map(|b| bf16_to_f32(*b)).collect();
        let mut max_diff = 0f32;
        for mi in 0..m as usize {
            let mut row_sum = 0f32;
            for kk in 0..k as usize {
                row_sum += a_f32[mi * k as usize + kk];
            }
            let expect = row_sum * 0.5;
            for ni in 0..n as usize {
                let got = bf16_to_f32(c_host[mi * n as usize + ni]);
                let d = (got - expect).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        eprintln!("{label} uniform: max_diff={max_diff:.4}");
        assert!(max_diff <= 0.5, "{label} uniform diverged: {max_diff}");
    }

    // Case 2: multi-block, ramping per-block scales
    {
        let m = 32u32;
        let n = 32u32;
        let k = 512u32;
        let mut a_bf16 = vec![0u16; (m * k) as usize];
        for i in 0..a_bf16.len() {
            let f = 0.05 * (((i * 13 + 5) % 17) as f32 - 8.0);
            a_bf16[i] = f32_to_bf16_bits(f);
        }
        let b_fp8 = vec![0x38u8; (n * k) as usize];
        let block_scales = [0.5f32, 0.25, 0.125, 0.0625];
        let num_blocks = block_scales.len();
        let mut b_scales = Vec::with_capacity(n as usize * num_blocks);
        for _ in 0..n {
            for &s in &block_scales {
                b_scales.push(s);
            }
        }
        let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
        let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
        let s_dev = stream.memcpy_stod(&b_scales).unwrap();
        let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
        launch(&ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, m, n, k)
            .unwrap_or_else(|e| panic!("{label} multi: {e:?}"));
        let c_host = stream.memcpy_dtov(&c_dev).unwrap();
        let a_f32: Vec<f32> = a_bf16.iter().map(|b| bf16_to_f32(*b)).collect();
        let mut max_diff = 0f32;
        for mi in 0..m as usize {
            let mut expect = 0f32;
            for blk in 0..num_blocks {
                let mut bs = 0f32;
                for kk in 0..128 {
                    let k_idx = blk * 128 + kk;
                    bs += a_f32[mi * k as usize + k_idx];
                }
                expect += bs * block_scales[blk];
            }
            for ni in 0..n as usize {
                let got = bf16_to_f32(c_host[mi * n as usize + ni]);
                let d = (got - expect).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        eprintln!("{label} multi-block: max_diff={max_diff:.4}");
        assert!(max_diff <= 0.5, "{label} multi-block diverged: {max_diff}");
    }

    // Case 3: ragged M and N (tail masking on both axes; N kept even because
    // the MMA kernel packs 2 BF16 per b32 store).
    {
        let m = 17u32;
        let n = 34u32;
        let k = 128u32;
        let mut a_bf16 = vec![0u16; (m * k) as usize];
        for i in 0..a_bf16.len() {
            let f = 0.1 * ((i % 5) as f32 - 2.0);
            a_bf16[i] = f32_to_bf16_bits(f);
        }
        let b_fp8 = vec![0x38u8; (n * k) as usize];
        let b_scales = vec![0.5f32; n as usize];
        let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
        let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
        let s_dev = stream.memcpy_stod(&b_scales).unwrap();
        let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
        launch(&ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, m, n, k)
            .unwrap_or_else(|e| panic!("{label} ragged: {e:?}"));
        let c_host = stream.memcpy_dtov(&c_dev).unwrap();
        let a_f32: Vec<f32> = a_bf16.iter().map(|b| bf16_to_f32(*b)).collect();
        let mut max_diff = 0f32;
        for mi in 0..m as usize {
            let mut row_sum = 0f32;
            for kk in 0..k as usize {
                row_sum += a_f32[mi * k as usize + kk];
            }
            let expect = row_sum * 0.5;
            for ni in 0..n as usize {
                let got = bf16_to_f32(c_host[mi * n as usize + ni]);
                let d = (got - expect).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        eprintln!("{label} ragged: max_diff={max_diff:.4}");
        assert!(max_diff <= 0.5, "{label} ragged diverged: {max_diff}");
    }
}

#[test]
fn test_gemm_fp8_block128_mma_correctness_suite() {
    run_dense_correctness(gemm::gemm_fp8_block128_mma, "FP8 block128 MMA");
}

#[test]
fn test_gemm_fp8_block128_dense_uniform() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let m = 64u32;
    let n = 32u32;
    let k = 128u32; // one block

    // A: [M, K] BF16, deterministic values.
    let mut a_bf16 = vec![0u16; (m * k) as usize];
    for i in 0..a_bf16.len() {
        let f = 0.1 * ((i % 7) as f32 - 3.0);
        a_bf16[i] = f32_to_bf16_bits(f);
    }

    // B: [N, K] FP8 e4m3 — all 0x38 = 1.0
    let b_fp8 = vec![0x38u8; (n * k) as usize];
    // Scales: [N, K/128] = [N, 1] = 0.5 → dequantized B value = 0.5
    let b_scales = vec![0.5f32; n as usize];

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let s_dev = stream.memcpy_stod(&b_scales).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_fp8_block128(&ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, m, n, k)
        .expect("dense FP8 block128 GEMM failed");

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();

    // Expected: C[mi, ni] = Σ_k A[mi, k] * 0.5
    let a_f32: Vec<f32> = a_bf16.iter().map(|b| bf16_to_f32(*b)).collect();
    let mut max_diff = 0f32;
    for mi in 0..m as usize {
        let mut row_sum = 0f32;
        for kk in 0..k as usize {
            row_sum += a_f32[mi * k as usize + kk];
        }
        let expect = row_sum * 0.5;
        for ni in 0..n as usize {
            let got = bf16_to_f32(c_host[mi * n as usize + ni]);
            let d = (got - expect).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    eprintln!("FP8 block128 dense uniform: max_diff={max_diff:.4}");
    assert!(
        max_diff <= 0.5,
        "dense FP8 block128 GEMM diverged: {max_diff}"
    );
}

#[test]
fn test_gemm_fp8_block128_dense_multi_block() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let m = 32u32;
    let n = 32u32;
    let k = 512u32; // 4 blocks of 128

    // A: random-ish
    let mut a_bf16 = vec![0u16; (m * k) as usize];
    for i in 0..a_bf16.len() {
        let f = 0.05 * (((i * 13 + 5) % 17) as f32 - 8.0);
        a_bf16[i] = f32_to_bf16_bits(f);
    }

    // B: all 0x38 = 1.0
    let b_fp8 = vec![0x38u8; (n * k) as usize];
    // Scales: per-block ramp [0.5, 0.25, 0.125, 0.0625] for each n
    let num_blocks = (k / 128) as usize;
    let block_scales = [0.5f32, 0.25, 0.125, 0.0625];
    let mut b_scales = Vec::with_capacity(n as usize * num_blocks);
    for _ in 0..n {
        for &s in &block_scales {
            b_scales.push(s);
        }
    }

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let s_dev = stream.memcpy_stod(&b_scales).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_fp8_block128(&ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, m, n, k)
        .expect("dense FP8 block128 GEMM failed");

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();

    let a_f32: Vec<f32> = a_bf16.iter().map(|b| bf16_to_f32(*b)).collect();
    let mut max_diff = 0f32;
    for mi in 0..m as usize {
        // Expected per (m, n): Σ_block (scale_block * Σ_{k in block} A[m, k] * 1.0)
        let mut expect = 0f32;
        for blk in 0..num_blocks {
            let mut block_sum = 0f32;
            for kk in 0..128 {
                let k_idx = blk * 128 + kk;
                block_sum += a_f32[mi * k as usize + k_idx];
            }
            expect += block_sum * block_scales[blk];
        }
        for ni in 0..n as usize {
            let got = bf16_to_f32(c_host[mi * n as usize + ni]);
            let d = (got - expect).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    eprintln!("FP8 block128 dense multi-block: max_diff={max_diff:.4}");
    assert!(
        max_diff <= 0.5,
        "dense FP8 block128 multi-block diverged: {max_diff}"
    );
}

#[test]
fn test_gemm_fp8_block128_dense_ragged_m() {
    // M not a multiple of BM=32, exercises the row tail-mask path.
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let m = 17u32; // ragged — last m_tile only has 17 - 0 = 17 < 32 rows
    let n = 33u32; // ragged on N too — exercises the n bounds check
    let k = 128u32;

    let mut a_bf16 = vec![0u16; (m * k) as usize];
    for i in 0..a_bf16.len() {
        let f = 0.1 * ((i % 5) as f32 - 2.0);
        a_bf16[i] = f32_to_bf16_bits(f);
    }
    let b_fp8 = vec![0x38u8; (n * k) as usize];
    let b_scales = vec![0.5f32; n as usize];

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let s_dev = stream.memcpy_stod(&b_scales).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_fp8_block128(&ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, m, n, k)
        .expect("dense FP8 block128 GEMM (ragged) failed");

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let a_f32: Vec<f32> = a_bf16.iter().map(|b| bf16_to_f32(*b)).collect();
    let mut max_diff = 0f32;
    for mi in 0..m as usize {
        let mut row_sum = 0f32;
        for kk in 0..k as usize {
            row_sum += a_f32[mi * k as usize + kk];
        }
        let expect = row_sum * 0.5;
        for ni in 0..n as usize {
            let got = bf16_to_f32(c_host[mi * n as usize + ni]);
            let d = (got - expect).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    eprintln!("FP8 block128 dense ragged: max_diff={max_diff:.4}");
    assert!(
        max_diff <= 0.5,
        "dense FP8 block128 ragged diverged: {max_diff}"
    );
}
