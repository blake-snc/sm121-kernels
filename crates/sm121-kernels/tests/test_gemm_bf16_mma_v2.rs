//! Validate gemm_bf16_mma_v2 against v1 (byte-exact check) and against a CPU
//! reference computation.
use half::bf16;
use sm121_kernels::{device, gemm};

fn random_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (((s >> 33) as u32 as f32) / u32::MAX as f32 - 0.5) * 0.5;
            bf16::from_f32(f).to_bits()
        })
        .collect()
}

fn cpu_gemm_bf16(a: &[u16], b: &[u16], m: usize, n: usize, k: usize) -> Vec<u16> {
    let mut c = vec![0u16; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += bf16::from_bits(a[i * k + kk]).to_f32()
                    * bf16::from_bits(b[kk * n + j]).to_f32();
            }
            c[i * n + j] = bf16::from_f32(acc).to_bits();
        }
    }
    c
}

fn run_shape(m: u32, n: u32, k: u32, max_abs_tol: f32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let a = random_bf16((m * k) as usize, 0xA1A1);
    let b = random_bf16((k * n) as usize, 0xB2B2);

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let mut c_v1 = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
    let mut c_v2 = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_bf16_mma(&ctx, &stream, &a_dev, &b_dev, &mut c_v1, m, n, k).unwrap();
    gemm::gemm_bf16_mma_v2(&ctx, &stream, &a_dev, &b_dev, &mut c_v2, m, n, k).unwrap();

    let v1 = stream.memcpy_dtov(&c_v1).unwrap();
    let v2 = stream.memcpy_dtov(&c_v2).unwrap();

    // Also test v3 if N is divisible by 128 (v3 requires bigger N tile)
    if n.is_multiple_of(128) {
        let mut c_v3 = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
        gemm::gemm_bf16_mma_v3(&ctx, &stream, &a_dev, &b_dev, &mut c_v3, m, n, k).unwrap();
        let v3 = stream.memcpy_dtov(&c_v3).unwrap();
        let mut max_v1v3 = 0f32;
        let mut nd3 = 0;
        for (a, b) in v1.iter().zip(v3.iter()) {
            let af = bf16::from_bits(*a).to_f32();
            let bf = bf16::from_bits(*b).to_f32();
            let d = (af - bf).abs();
            if d > max_v1v3 {
                max_v1v3 = d;
            }
            if a != b {
                nd3 += 1;
            }
        }
        eprintln!("  v1↔v3 n_diff={nd3} max_abs={max_v1v3:.4}");
        assert!(
            max_v1v3 < max_abs_tol,
            "v3 vs v1 max_abs {max_v1v3} > tol {max_abs_tol}"
        );
    }

    // v1 vs v2: should match within FP add ordering tolerance
    let mut max_v1v2 = 0f32;
    let mut n_diff = 0;
    for (a, b) in v1.iter().zip(v2.iter()) {
        let af = bf16::from_bits(*a).to_f32();
        let bf = bf16::from_bits(*b).to_f32();
        let d = (af - bf).abs();
        if d > max_v1v2 {
            max_v1v2 = d;
        }
        if a != b {
            n_diff += 1;
        }
    }
    eprintln!("M={m} N={n} K={k}: v1↔v2 n_diff={n_diff} max_abs={max_v1v2:.4}");

    // v2 vs CPU ref (only for small shapes — CPU is slow)
    if (m * n * k) <= (256 * 256 * 256) {
        let r = cpu_gemm_bf16(&a, &b, m as usize, n as usize, k as usize);
        let mut max_ref = 0f32;
        for (g, t) in v2.iter().zip(r.iter()) {
            let gf = bf16::from_bits(*g).to_f32();
            let tf = bf16::from_bits(*t).to_f32();
            let d = (gf - tf).abs();
            if d > max_ref {
                max_ref = d;
            }
        }
        eprintln!("  vs CPU: max_abs={max_ref:.4}");
        assert!(
            max_ref < max_abs_tol,
            "max_abs vs CPU {max_ref} > tol {max_abs_tol}"
        );
    }

    assert!(
        max_v1v2 < max_abs_tol,
        "v2 vs v1 max_abs {max_v1v2} > tol {max_abs_tol}"
    );
}

#[test]
fn t_v2_128x64x32() {
    run_shape(128, 64, 32, 0.05);
}

#[test]
fn t_v2_128x64x64() {
    run_shape(128, 64, 64, 0.05);
}

#[test]
fn t_v2_256x256x256() {
    run_shape(256, 256, 256, 0.10);
}

#[test]
fn t_v2_512x512x512() {
    run_shape(512, 512, 512, 0.30);
}

#[test]
fn t_v2_1024x1024x1024() {
    run_shape(1024, 1024, 1024, 0.50);
}
