mod common;

use sm121_kernels::{activation, device, gemm, quantization};

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn unbf16(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

fn cpu_gemv_bf16(x: &[u16], b: &[u16], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    for nn in 0..n {
        let mut acc = 0f32;
        for kk in 0..k {
            let bv = unbf16(b[kk * n + nn]);
            acc += unbf16(x[kk]) * bv;
        }
        out[nn] = acc;
    }
    out
}

#[test]
fn w8a16_gemv_qwen_in_proj_shape() {
    let ctx = device::init_device(0).expect("init SM121");
    let stream = ctx.default_stream();

    // Match GDN-hybrid GDN in_proj_fused shape: K=2048, N=12352
    // Use a smaller variant to keep the test fast: K=512, N=2048
    let k: u32 = 512;
    let n: u32 = 2048;

    let mut s = 0xCAFEBABE_u64;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let x_h: Vec<u16> = (0..k).map(|_| bf16(next() * 0.5)).collect();
    let b_h: Vec<u16> = (0..k * n).map(|_| bf16(next() * 0.3)).collect();

    // Compute per-tensor scale: max_abs(b) / 448.0
    let max_abs = b_h.iter().map(|&v| unbf16(v).abs()).fold(0f32, f32::max);
    let scale = (max_abs / 448.0).max(1e-12);
    eprintln!("test scale = {scale:.6}, max_abs = {max_abs:.6}");

    let x_dev = stream.memcpy_stod(&x_h).unwrap();
    let b_dev = stream.memcpy_stod(&b_h).unwrap();

    // Quantize B to FP8 on device using our kernel
    let mut b_fp8 = stream.alloc_zeros::<u8>((k * n) as usize).unwrap();
    quantization::quant_bf16_to_fp8_pertensor(&ctx, &stream, &b_dev, &mut b_fp8, k * n, scale)
        .expect("quantize");

    // Run W8A16 GEMV
    let mut out_f32 = stream.alloc_zeros::<f32>(n as usize).unwrap();
    let num_shards = 2;
    gemm::gemv_w8a16_split_k(
        &ctx,
        &stream,
        &x_dev,
        &b_fp8,
        scale,
        &mut out_f32,
        n,
        k,
        num_shards,
    )
    .expect("w8a16 gemv");
    let mut out_bf16 = stream.alloc_zeros::<u16>(n as usize).unwrap();
    activation::f32_to_bf16(&ctx, &stream, &out_f32, &mut out_bf16, n).unwrap();

    // CPU reference (BF16 math)
    let ref_out = cpu_gemv_bf16(&x_h, &b_h, n as usize, k as usize);

    let actual = stream.memcpy_dtov(&out_bf16).unwrap();

    // FP8 e4m3 has only 3 mantissa bits → relative error ~1/8 = 12% per element
    // worst case. With K=512 accumulation the errors partially cancel.
    // Tolerance: relative max ~10%, mean ~3% of max_abs(ref)
    let max_ref_abs = ref_out.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let mut max_diff = 0f32;
    let mut sum_diff = 0f32;
    for (i, (&a_b, &r)) in actual.iter().zip(ref_out.iter()).enumerate() {
        let a = unbf16(a_b);
        let d = (a - r).abs();
        if d > max_diff {
            max_diff = d;
        }
        sum_diff += d;
        if i < 4 {
            eprintln!("  out[{i}] actual={a:+.6} ref={r:+.6} diff={d:.6}");
        }
    }
    let mean_diff = sum_diff / n as f32;
    let rel_max = max_diff / max_ref_abs.max(1e-12);
    let rel_mean = mean_diff / max_ref_abs.max(1e-12);
    eprintln!("max_diff = {max_diff:.6}, mean_diff = {mean_diff:.6}");
    eprintln!(
        "rel_max = {:.4}%, rel_mean = {:.4}%",
        rel_max * 100.0,
        rel_mean * 100.0
    );

    // Loose tolerance: FP8 weight quant + BF16 round trip
    assert!(rel_max < 0.20, "relative max diff too large: {rel_max}");
    assert!(rel_mean < 0.05, "relative mean diff too large: {rel_mean}");
}
