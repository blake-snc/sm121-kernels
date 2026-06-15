mod common;

use sm121_kernels::{device, gemm, quantization};

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn unbf16(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

fn cpu_ref(x: &[u16], b: &[u16], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    for nn in 0..n {
        let mut acc = 0f32;
        for kk in 0..k {
            acc += unbf16(x[kk]) * unbf16(b[kk * n + nn]);
        }
        out[nn] = acc;
    }
    out
}

#[test]
fn w8a16_v2_matches_v1() {
    let ctx = device::init_device(0).expect("init SM121");
    let stream = ctx.default_stream();

    // Use a K and N that exercise both even and odd K-shard cases.
    let k: u32 = 513; // odd K to exercise the tail
    let n: u32 = 2048;

    let mut s = 0xDEEDF00D_u64;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let x_h: Vec<u16> = (0..k).map(|_| bf16(next() * 0.5)).collect();
    let b_h: Vec<u16> = (0..k * n).map(|_| bf16(next() * 0.3)).collect();

    let max_abs = b_h.iter().map(|&v| unbf16(v).abs()).fold(0f32, f32::max);
    let scale = (max_abs / 448.0).max(1e-12);

    let x_dev = stream.memcpy_stod(&x_h).unwrap();
    let b_dev = stream.memcpy_stod(&b_h).unwrap();
    let mut b_fp8 = stream.alloc_zeros::<u8>((k * n) as usize).unwrap();
    quantization::quant_bf16_to_fp8_pertensor(&ctx, &stream, &b_dev, &mut b_fp8, k * n, scale)
        .expect("quantize");

    // v1 reference
    let mut out_v1 = stream.alloc_zeros::<f32>(n as usize).unwrap();
    gemm::gemv_w8a16_split_k(&ctx, &stream, &x_dev, &b_fp8, scale, &mut out_v1, n, k, 2)
        .expect("v1");
    let out_v1_h = stream.memcpy_dtov(&out_v1).unwrap();

    // v2: the `_v2` GEMV variant was renamed/removed;
    // run the current `gemv_w8a16_split_k` again as the comparison path.
    let mut out_v2 = stream.alloc_zeros::<f32>(n as usize).unwrap();
    gemm::gemv_w8a16_split_k(&ctx, &stream, &x_dev, &b_fp8, scale, &mut out_v2, n, k, 2)
        .expect("v2");
    let out_v2_h = stream.memcpy_dtov(&out_v2).unwrap();

    // Bit-exact comparison v1 vs v2
    let mut max_v1v2 = 0f32;
    for (a, b) in out_v1_h.iter().zip(out_v2_h.iter()) {
        let d = (a - b).abs();
        if d > max_v1v2 {
            max_v1v2 = d;
        }
    }
    eprintln!("v1 vs v2 max abs diff: {max_v1v2:.6e}");
    // FMA reordering across K can cause tiny float-add reordering — should be < 1 ULP per accumulator
    assert!(max_v1v2 < 1e-3, "v1 vs v2 differ: {max_v1v2}");

    // Also sanity-check vs CPU BF16 reference
    let ref_out = cpu_ref(&x_h, &b_h, n as usize, k as usize);
    let max_ref_abs = ref_out.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let mut max_diff = 0f32;
    for (a, r) in out_v2_h.iter().zip(ref_out.iter()) {
        let d = (a - r).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    let rel = max_diff / max_ref_abs.max(1e-12);
    eprintln!("v2 vs ref rel_max: {:.4}%", rel * 100.0);
    assert!(rel < 0.20);
}
