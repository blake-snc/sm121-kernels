//! Validate gemm_fp8_mma_v3 against v1 (FP8 reference).
use half::bf16;
use sm121_kernels::{device, gemm};

fn random_fp8(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = ((s >> 33) & 0xFF) as u8;
            if (b & 0x7F) == 0x7F {
                b ^ 0x10
            } else {
                b
            }
        })
        .collect()
}

fn run_shape(m: u32, n: u32, k: u32, max_abs_tol: f32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let a = random_fp8((m * k) as usize, 0xA1A1);
    let b = random_fp8((k * n) as usize, 0xB2B2);

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let mut c_v1 = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
    let mut c_v3 = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_fp8_mma(&ctx, &stream, &a_dev, &b_dev, &mut c_v1, m, n, k).unwrap();
    gemm::gemm_fp8_mma_v3(&ctx, &stream, &a_dev, &b_dev, &mut c_v3, m, n, k).unwrap();

    // Also test v3.5 if K large enough for the 3-stage prologue
    if k >= 96 {
        let mut c_v35 = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
        gemm::gemm_fp8_mma_v3_5(&ctx, &stream, &a_dev, &b_dev, &mut c_v35, m, n, k).unwrap();
        let v35 = stream.memcpy_dtov(&c_v35).unwrap();
        let v3_check = stream.memcpy_dtov(&c_v3).unwrap();
        let mut max_d = 0f32;
        let mut nd = 0;
        for (a, b) in v3_check.iter().zip(v35.iter()) {
            let af = bf16::from_bits(*a).to_f32();
            let bf = bf16::from_bits(*b).to_f32();
            let d = (af - bf).abs();
            if d > max_d {
                max_d = d;
            }
            if a != b {
                nd += 1;
            }
        }
        eprintln!("  v3↔v3.5 n_diff={nd} max_abs={max_d:.4}");
        assert!(
            max_d < max_abs_tol,
            "v3.5 vs v3 max_abs {max_d} > tol {max_abs_tol}"
        );
    }

    let v1 = stream.memcpy_dtov(&c_v1).unwrap();
    let v3 = stream.memcpy_dtov(&c_v3).unwrap();

    let mut max_diff = 0f32;
    let mut n_diff = 0;
    let mut max_mag = 0f32;
    for (a, b) in v1.iter().zip(v3.iter()) {
        let af = bf16::from_bits(*a).to_f32();
        let bf = bf16::from_bits(*b).to_f32();
        let d = (af - bf).abs();
        if d > max_diff {
            max_diff = d;
        }
        let mag = af.abs().max(bf.abs());
        if mag > max_mag {
            max_mag = mag;
        }
        if a != b {
            n_diff += 1;
        }
    }
    eprintln!(
        "M={m} N={n} K={k}: v1↔v3 n_diff={n_diff} max_abs={max_diff:.4} max_mag={max_mag:.2}"
    );
    assert!(
        max_diff < max_abs_tol,
        "v3 vs v1 max_abs {max_diff} > tol {max_abs_tol} (max_mag={max_mag})"
    );
}

#[test]
fn t_fp8_v3_128x128x32() {
    run_shape(128, 128, 32, 0.10);
}

#[test]
fn t_fp8_v3_128x128x128() {
    run_shape(128, 128, 128, 0.10);
}

#[test]
fn t_fp8_v3_256x256x256() {
    run_shape(256, 256, 256, 0.30);
}

#[test]
fn t_fp8_v3_512x512x512() {
    run_shape(512, 512, 512, 0.50);
}

#[test]
fn t_fp8_v3_1024x1024x1024() {
    run_shape(1024, 1024, 1024, 1.00);
}
