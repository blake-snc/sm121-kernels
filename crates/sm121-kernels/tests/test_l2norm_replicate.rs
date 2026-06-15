mod common;

use common::compare_bf16;
use sm121_kernels::{device, linear_attention};

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn unbf16(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

fn cpu_ref(x: &[u16], n_in: usize, hd: usize, factor: usize, scale: f32, eps: f32) -> Vec<u16> {
    let mut y = vec![0u16; n_in * factor * hd];
    for h in 0..n_in {
        let mut sumsq = 0f32;
        for d in 0..hd {
            let v = unbf16(x[h * hd + d]);
            sumsq += v * v;
        }
        let inv = scale / (sumsq + eps).sqrt();
        for d in 0..hd {
            let v = unbf16(x[h * hd + d]) * inv;
            let b = bf16(v);
            for r in 0..factor {
                y[(h * factor + r) * hd + d] = b;
            }
        }
    }
    y
}

fn run_case(num_heads_in: u32, head_dim: u32, factor: u32, scale: f32, seed: u64, tol: f32) {
    let ctx = device::init_device(0).expect("init SM121");
    let stream = ctx.default_stream();

    let n = (num_heads_in * head_dim) as usize;
    let mut s = seed;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let x: Vec<u16> = (0..n).map(|_| bf16(next() * 0.5)).collect();

    let exp = cpu_ref(
        &x,
        num_heads_in as usize,
        head_dim as usize,
        factor as usize,
        scale,
        1e-6,
    );

    let x_dev = stream.memcpy_stod(&x).unwrap();
    let mut y_dev = stream.alloc_zeros::<u16>(exp.len()).unwrap();

    linear_attention::l2norm_scale_replicate_bf16(
        &ctx,
        &stream,
        &x_dev,
        &mut y_dev,
        num_heads_in,
        head_dim,
        factor,
        scale,
        1e-6,
    )
    .expect("kernel launch");

    let y_host = stream.memcpy_dtov(&y_dev).unwrap();
    let (max, mean) = compare_bf16(&y_host, &exp, tol);
    eprintln!(
        "l2norm_scale_replicate (n_in={num_heads_in}, hd={head_dim}, f={factor}, s={scale}): max={max:.5} mean={mean:.5}"
    );
}

#[test]
fn l2_gdn_hybrid_q_path() {
    // GDN-hybrid GDN Q: n_qk=16 → n_v=32 (factor=2), hd_qk=128, scale=1/sqrt(128)
    run_case(16, 128, 2, 1.0 / (128.0_f32).sqrt(), 0xCAFEBABE, 0.02);
}

#[test]
fn l2_gdn_hybrid_k_path() {
    // GDN-hybrid GDN K: same shape, scale=1.0
    run_case(16, 128, 2, 1.0, 0xDEADBEEF, 0.02);
}

#[test]
fn l2_no_replicate() {
    run_case(8, 64, 1, 0.5, 0xAA55, 0.02);
}
