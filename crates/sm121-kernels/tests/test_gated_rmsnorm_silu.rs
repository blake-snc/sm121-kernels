mod common;

use common::compare_bf16;
use sm121_kernels::{device, norm};

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn unbf16(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

fn cpu_ref(y: &[u16], z: &[u16], w: &[u16], n_h: usize, hd: usize, eps: f32) -> Vec<u16> {
    let mut out = vec![0u16; n_h * hd];
    for h in 0..n_h {
        let off = h * hd;
        let mut sumsq = 0f32;
        for d in 0..hd {
            let v = unbf16(y[off + d]);
            sumsq += v * v;
        }
        let rms_inv = 1.0 / ((sumsq / hd as f32) + eps).sqrt();
        for d in 0..hd {
            let yv = unbf16(y[off + d]);
            let nv = unbf16(w[d]);
            let zv = unbf16(z[off + d]);
            let silu_z = zv * (1.0 / (1.0 + (-zv).exp()));
            out[off + d] = bf16(yv * rms_inv * nv * silu_z);
        }
    }
    out
}

fn run_case(num_heads: u32, head_dim: u32, seed: u64, tol: f32) {
    let ctx = device::init_device(0).expect("init SM121");
    let stream = ctx.default_stream();

    let n = (num_heads * head_dim) as usize;
    let mut s = seed;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let y: Vec<u16> = (0..n).map(|_| bf16(next() * 0.5)).collect();
    let z: Vec<u16> = (0..n).map(|_| bf16(next() * 0.5)).collect();
    let w: Vec<u16> = (0..head_dim as usize)
        .map(|_| bf16(next() * 0.3 + 1.0))
        .collect();

    let exp = cpu_ref(&y, &z, &w, num_heads as usize, head_dim as usize, 1e-6);

    let y_dev = stream.memcpy_stod(&y).unwrap();
    let z_dev = stream.memcpy_stod(&z).unwrap();
    let w_dev = stream.memcpy_stod(&w).unwrap();
    let mut out_dev = stream.alloc_zeros::<u16>(n).unwrap();

    norm::gated_rmsnorm_silu_bf16(
        &ctx,
        &stream,
        &y_dev,
        &z_dev,
        &w_dev,
        &mut out_dev,
        num_heads,
        head_dim,
        1e-6,
    )
    .expect("kernel launch");

    let out_host = stream.memcpy_dtov(&out_dev).unwrap();
    let (max, mean) = compare_bf16(&out_host, &exp, tol);
    eprintln!(
        "gated_rmsnorm_silu (n_h={num_heads}, hd={head_dim}): max_diff={max:.5} mean={mean:.5}"
    );
}

#[test]
fn gated_rmsnorm_silu_gdn_hybrid() {
    // GDN-hybrid GDN: n_v_heads=32, head_dim_v=128.
    // div.approx + rsqrt.approx + ex2.approx contribute small numerical drift; tol 0.02.
    run_case(32, 128, 0xDEADBEEF, 0.02);
}

#[test]
fn gated_rmsnorm_silu_small() {
    run_case(4, 64, 0xAA55AA55, 0.02);
}

#[test]
fn gated_rmsnorm_silu_wide() {
    run_case(2, 256, 0x1234CAFE, 0.03);
}
