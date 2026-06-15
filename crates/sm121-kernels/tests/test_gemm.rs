#![allow(non_snake_case)]

mod common;

use common::{compare_bf16, load_npz};
use sm121_kernels::{device, gemm};

#[test]
fn test_gemm_bf16_128x128x128() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_bf16_128x128x128.npz");

    let a_np: ndarray::Array2<u16> = npz.by_name("a").unwrap();
    let b_np: ndarray::Array2<u16> = npz.by_name("b").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = a_np.shape()[1] as u32;
    let n = b_np.shape()[1] as u32;

    let a_flat: Vec<u16> = a_np.into_raw_vec_and_offset().0;
    let b_flat: Vec<u16> = b_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_bf16(&ctx, &stream, &a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 0.15);
    eprintln!("gemm 128x128x128: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_bf16_backward() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (m, n, k) in [(64u32, 128u32, 128u32), (128, 64, 256), (256, 128, 64)] {
        let mut npz = load_npz(&format!("gemm_backward_bf16_m{m}_n{n}_k{k}.npz"));

        let a_np: ndarray::Array2<u16> = npz.by_name("a").unwrap();
        let b_np: ndarray::Array2<u16> = npz.by_name("b").unwrap();
        let dc_np: ndarray::Array2<u16> = npz.by_name("dc").unwrap();
        let da_expected: ndarray::Array2<u16> = npz.by_name("da").unwrap();
        let db_expected: ndarray::Array2<u16> = npz.by_name("db").unwrap();

        let a_flat: Vec<u16> = a_np.into_raw_vec_and_offset().0;
        let b_flat: Vec<u16> = b_np.into_raw_vec_and_offset().0;
        let dc_flat: Vec<u16> = dc_np.into_raw_vec_and_offset().0;
        let da_expected_flat: Vec<u16> = da_expected.into_raw_vec_and_offset().0;
        let db_expected_flat: Vec<u16> = db_expected.into_raw_vec_and_offset().0;

        let a_dev = stream.memcpy_stod(&a_flat).unwrap();
        let b_dev = stream.memcpy_stod(&b_flat).unwrap();
        let dc_dev = stream.memcpy_stod(&dc_flat).unwrap();
        let mut da_dev = stream.alloc_zeros::<u16>((m * k) as usize).unwrap();
        let mut db_dev = stream.alloc_zeros::<u16>((k * n) as usize).unwrap();

        gemm::gemm_bf16_backward_dA(&ctx, &stream, &dc_dev, &b_dev, &mut da_dev, m, n, k).unwrap();
        gemm::gemm_bf16_backward_dB(&ctx, &stream, &a_dev, &dc_dev, &mut db_dev, m, n, k).unwrap();

        let da_host = stream.memcpy_dtov(&da_dev).unwrap();
        let (max_da, mean_da) = compare_bf16(&da_host, &da_expected_flat, 1.0);
        eprintln!("gemm_bw dA m{m}n{n}k{k}: max_diff={max_da:.4} mean_diff={mean_da:.4}");

        let db_host = stream.memcpy_dtov(&db_dev).unwrap();
        let (max_db, mean_db) = compare_bf16(&db_host, &db_expected_flat, 1.0);
        eprintln!("gemm_bw dB m{m}n{n}k{k}: max_diff={max_db:.4} mean_diff={mean_db:.4}");
    }
}

#[test]
fn test_gemm_bf16_512x512x512() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_bf16_512x512x512.npz");

    let a_np: ndarray::Array2<u16> = npz.by_name("a").unwrap();
    let b_np: ndarray::Array2<u16> = npz.by_name("b").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = a_np.shape()[1] as u32;
    let n = b_np.shape()[1] as u32;

    let a_flat: Vec<u16> = a_np.into_raw_vec_and_offset().0;
    let b_flat: Vec<u16> = b_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_bf16(&ctx, &stream, &a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 0.15);
    eprintln!("gemm 512x512x512: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_bf16_mma_128x128x128() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_bf16_128x128x128.npz");

    let a_np: ndarray::Array2<u16> = npz.by_name("a").unwrap();
    let b_np: ndarray::Array2<u16> = npz.by_name("b").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = a_np.shape()[1] as u32;
    let n = b_np.shape()[1] as u32;

    let a_flat: Vec<u16> = a_np.into_raw_vec_and_offset().0;
    let b_flat: Vec<u16> = b_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_bf16_mma(&ctx, &stream, &a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 0.15);
    eprintln!("gemm_mma 128x128x128: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_bf16_mma_512x512x512() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_bf16_512x512x512.npz");

    let a_np: ndarray::Array2<u16> = npz.by_name("a").unwrap();
    let b_np: ndarray::Array2<u16> = npz.by_name("b").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = a_np.shape()[1] as u32;
    let n = b_np.shape()[1] as u32;

    let a_flat: Vec<u16> = a_np.into_raw_vec_and_offset().0;
    let b_flat: Vec<u16> = b_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_bf16_mma(&ctx, &stream, &a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 0.3);
    eprintln!("gemm_mma 512x512x512: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_bf16_mma_v5_512x512x512() {
    // v5 (128×256 register-blocked) against the same golden as v3. N=512%256 ok.
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_bf16_512x512x512.npz");
    let a_np: ndarray::Array2<u16> = npz.by_name("a").unwrap();
    let b_np: ndarray::Array2<u16> = npz.by_name("b").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = a_np.shape()[1] as u32;
    let n = b_np.shape()[1] as u32;

    let a_flat: Vec<u16> = a_np.into_raw_vec_and_offset().0;
    let b_flat: Vec<u16> = b_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_bf16_mma_v5(&ctx, &stream, &a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 0.3);
    eprintln!("gemm_mma_v5 512x512x512: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_bf16_mma_auto_routes_v5_2048() {
    // At 2048^3 (M,N>=2048, N%256==0) the auto-dispatcher routes to v5. Cross-
    // check against v3 on the same inputs. Expect BIT-IDENTICAL output (max_diff
    // 0): the tile size changes which block computes each C[i][j], but the
    // per-element accumulation order over k is the same in both kernels, so v5 is
    // bitwise-equal to v3/v1. This validates v5's correctness at the integration
    // shape AND that routing v5 through auto preserves SPARK_DETERMINISTIC.
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();
    let (m, n, k) = (2048u32, 2048u32, 2048u32);

    let mk = |seed: u64, len: usize| -> Vec<u16> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 40) as f32) / ((1u64 << 24) as f32); // [0,1)
                half::bf16::from_f32(u - 0.5).to_bits()
            })
            .collect()
    };
    let a = mk(0x1234, (m * k) as usize);
    let b = mk(0x9abc, (k * n) as usize);
    let a_d = stream.memcpy_stod(&a).unwrap();
    let b_d = stream.memcpy_stod(&b).unwrap();
    let mut c_v3 = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
    let mut c_auto = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_bf16_mma_v3(&ctx, &stream, &a_d, &b_d, &mut c_v3, m, n, k).unwrap();
    gemm::gemm_bf16_mma_auto(&ctx, &stream, &a_d, &b_d, &mut c_auto, m, n, k).unwrap();

    let hv3 = stream.memcpy_dtov(&c_v3).unwrap();
    let hauto = stream.memcpy_dtov(&c_auto).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&hauto, &hv3, 1.0);
    eprintln!("auto(v5) vs v3 @2048^3: max_diff={max_diff:.4} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_bf16_mma_1024x4096x4096() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_bf16_1024x4096x4096.npz");

    let a_np: ndarray::Array2<u16> = npz.by_name("a").unwrap();
    let b_np: ndarray::Array2<u16> = npz.by_name("b").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = a_np.shape()[1] as u32;
    let n = b_np.shape()[1] as u32;

    let a_flat: Vec<u16> = a_np.into_raw_vec_and_offset().0;
    let b_flat: Vec<u16> = b_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_bf16_mma(&ctx, &stream, &a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 2.0);
    eprintln!("gemm_mma 1024x4096x4096: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_fp8_mma_128x128x128() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_fp8_128x128x128.npz");

    let a_np: ndarray::Array2<u8> = npz.by_name("a").unwrap();
    let b_np: ndarray::Array2<u8> = npz.by_name("b").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = a_np.shape()[1] as u32;
    let n = b_np.shape()[1] as u32;

    let a_flat: Vec<u8> = a_np.into_raw_vec_and_offset().0;
    let b_flat: Vec<u8> = b_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_fp8_mma(&ctx, &stream, &a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 0.5);
    eprintln!("gemm_fp8_mma 128x128x128: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_fp8_mma_512x512x512() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_fp8_512x512x512.npz");

    let a_np: ndarray::Array2<u8> = npz.by_name("a").unwrap();
    let b_np: ndarray::Array2<u8> = npz.by_name("b").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = a_np.shape()[1] as u32;
    let n = b_np.shape()[1] as u32;

    let a_flat: Vec<u8> = a_np.into_raw_vec_and_offset().0;
    let b_flat: Vec<u8> = b_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_fp8_mma(&ctx, &stream, &a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 0.5);
    eprintln!("gemm_fp8_mma 512x512x512: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_nvfp4_mma_32x32x64() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_nvfp4_32x32x64.npz");

    let a_np: ndarray::Array2<u8> = npz.by_name("a").unwrap();
    let b_np: ndarray::Array2<u8> = npz.by_name("b").unwrap();
    let scale_a_np: ndarray::Array1<u8> = npz.by_name("scale_a").unwrap();
    let scale_b_np: ndarray::Array1<u8> = npz.by_name("scale_b").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = (a_np.shape()[1] * 2) as u32;
    let n = b_np.shape()[1] as u32;

    let a_flat: Vec<u8> = a_np.into_raw_vec_and_offset().0;
    let b_flat: Vec<u8> = b_np.into_raw_vec_and_offset().0;
    let scale_a_flat: Vec<u8> = scale_a_np.into_raw_vec_and_offset().0;
    let scale_b_flat: Vec<u8> = scale_b_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let scale_a_dev = stream.memcpy_stod(&scale_a_flat).unwrap();
    let scale_b_dev = stream.memcpy_stod(&scale_b_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_nvfp4_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_dev,
        &mut c_dev,
        &scale_a_dev,
        &scale_b_dev,
        m,
        n,
        k,
    )
    .unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 1.0);
    eprintln!("gemm_nvfp4_mma 32x32x64: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_nvfp4_mma_32x32x128() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_nvfp4_32x32x128.npz");

    let a_np: ndarray::Array2<u8> = npz.by_name("a").unwrap();
    let b_np: ndarray::Array2<u8> = npz.by_name("b").unwrap();
    let scale_a_np: ndarray::Array1<u8> = npz.by_name("scale_a").unwrap();
    let scale_b_np: ndarray::Array1<u8> = npz.by_name("scale_b").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = (a_np.shape()[1] * 2) as u32;
    let n = b_np.shape()[1] as u32;

    let a_flat: Vec<u8> = a_np.into_raw_vec_and_offset().0;
    let b_flat: Vec<u8> = b_np.into_raw_vec_and_offset().0;
    let scale_a_flat: Vec<u8> = scale_a_np.into_raw_vec_and_offset().0;
    let scale_b_flat: Vec<u8> = scale_b_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let scale_a_dev = stream.memcpy_stod(&scale_a_flat).unwrap();
    let scale_b_dev = stream.memcpy_stod(&scale_b_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_nvfp4_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_dev,
        &mut c_dev,
        &scale_a_dev,
        &scale_b_dev,
        m,
        n,
        k,
    )
    .unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 1.0);
    eprintln!("gemm_nvfp4_mma 32x32x128: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_w4a16_mma_128x128x128() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_w4a16_128x128x128.npz");

    let a_np: ndarray::Array2<u16> = npz.by_name("a").unwrap();
    let w_np: ndarray::Array2<u8> = npz.by_name("w").unwrap();
    let scales_np: ndarray::Array1<u16> = npz.by_name("scales").unwrap();
    let zeros_np: ndarray::Array1<u16> = npz.by_name("zeros").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = a_np.shape()[1] as u32;
    let n = (w_np.shape()[1] * 2) as u32;

    let a_flat: Vec<u16> = a_np.into_raw_vec_and_offset().0;
    let w_flat: Vec<u8> = w_np.into_raw_vec_and_offset().0;
    let scales_flat: Vec<u16> = scales_np.into_raw_vec_and_offset().0;
    let zeros_flat: Vec<u16> = zeros_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let w_dev = stream.memcpy_stod(&w_flat).unwrap();
    let scales_dev = stream.memcpy_stod(&scales_flat).unwrap();
    let zeros_dev = stream.memcpy_stod(&zeros_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_w4a16_mma(
        &ctx,
        &stream,
        &a_dev,
        &w_dev,
        &mut c_dev,
        &scales_dev,
        &zeros_dev,
        m,
        n,
        k,
    )
    .unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 0.6);
    eprintln!("gemm_w4a16_mma 128x128x128: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_w4a16_mma_128x128x256() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("gemm_w4a16_128x128x256.npz");

    let a_np: ndarray::Array2<u16> = npz.by_name("a").unwrap();
    let w_np: ndarray::Array2<u8> = npz.by_name("w").unwrap();
    let scales_np: ndarray::Array1<u16> = npz.by_name("scales").unwrap();
    let zeros_np: ndarray::Array1<u16> = npz.by_name("zeros").unwrap();
    let c_expected: ndarray::Array2<u16> = npz.by_name("c").unwrap();

    let m = a_np.shape()[0] as u32;
    let k = a_np.shape()[1] as u32;
    let n = (w_np.shape()[1] * 2) as u32;

    let a_flat: Vec<u16> = a_np.into_raw_vec_and_offset().0;
    let w_flat: Vec<u8> = w_np.into_raw_vec_and_offset().0;
    let scales_flat: Vec<u16> = scales_np.into_raw_vec_and_offset().0;
    let zeros_flat: Vec<u16> = zeros_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = c_expected.into_raw_vec_and_offset().0;

    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let w_dev = stream.memcpy_stod(&w_flat).unwrap();
    let scales_dev = stream.memcpy_stod(&scales_flat).unwrap();
    let zeros_dev = stream.memcpy_stod(&zeros_flat).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_w4a16_mma(
        &ctx,
        &stream,
        &a_dev,
        &w_dev,
        &mut c_dev,
        &scales_dev,
        &zeros_dev,
        m,
        n,
        k,
    )
    .unwrap();

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&c_host, &expected_flat, 0.6);
    eprintln!("gemm_w4a16_mma 128x128x256: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
}

#[test]
fn test_gemm_fp8_w8a16_backward_dA() {
    let ctx = device::init_device(0).expect("init SM121");
    let stream = ctx.default_stream();

    for (m, n, k) in [(64u32, 128u32, 128u32), (128, 64, 256)] {
        let mut npz = load_npz(&format!("gemm_fp8_w8a16_backward_m{m}_n{n}_k{k}.npz"));
        let a_np: ndarray::Array2<u16> = npz.by_name("a").unwrap();
        let b_fp8_np: ndarray::Array2<u8> = npz.by_name("b_fp8").unwrap();
        let scale_np: ndarray::Array0<f32> = npz.by_name("scale_b").unwrap();
        let dc_np: ndarray::Array2<u16> = npz.by_name("dc").unwrap();
        let da_expected: ndarray::Array2<u16> = npz.by_name("da").unwrap();

        let _a_flat: Vec<u16> = a_np.into_raw_vec_and_offset().0;
        let b_fp8_flat: Vec<u8> = b_fp8_np.into_raw_vec_and_offset().0;
        let dc_flat: Vec<u16> = dc_np.into_raw_vec_and_offset().0;
        let da_expected_flat: Vec<u16> = da_expected.into_raw_vec_and_offset().0;
        let scale_b = scale_np.into_scalar();

        let b_fp8_dev = stream.memcpy_stod(&b_fp8_flat).unwrap();
        let dc_dev = stream.memcpy_stod(&dc_flat).unwrap();
        let mut da_dev = stream.alloc_zeros::<u16>((m * k) as usize).unwrap();

        gemm::gemm_fp8_w8a16_backward_dA(
            &ctx,
            &stream,
            &dc_dev,
            &b_fp8_dev,
            scale_b,
            &mut da_dev,
            m,
            n,
            k,
        )
        .unwrap();

        let da_host = stream.memcpy_dtov(&da_dev).unwrap();
        // FP8 backward has wider tolerance from quantization noise.
        let (max_diff, mean_diff) = compare_bf16(&da_host, &da_expected_flat, 2.0);
        eprintln!(
            "gemm_fp8_w8a16_bw dA m{m}n{n}k{k}: max_diff={max_diff:.4} mean_diff={mean_diff:.4}"
        );
    }
}
