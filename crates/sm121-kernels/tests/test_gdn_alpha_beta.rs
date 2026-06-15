mod common;

use sm121_kernels::{device, linear_attention};

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn unbf16(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

fn cpu_ref(
    a_logits: &[u16],
    b_logits: &[u16],
    dt_bias: &[u16],
    a_log: &[u16],
) -> (Vec<f32>, Vec<f32>) {
    let n = a_logits.len();
    let mut alpha = vec![0f32; n];
    let mut beta = vec![0f32; n];
    for i in 0..n {
        let dt_pre = unbf16(a_logits[i]) + unbf16(dt_bias[i]);
        let dt = if dt_pre > 20.0 {
            dt_pre
        } else {
            (1.0 + dt_pre.exp()).ln()
        };
        let a_pos = unbf16(a_log[i]).exp();
        alpha[i] = (-a_pos * dt).exp();
        let b_pre = unbf16(b_logits[i]);
        beta[i] = 1.0 / (1.0 + (-b_pre).exp());
    }
    (alpha, beta)
}

#[test]
fn gdn_alpha_beta_gdn_hybrid() {
    let ctx = device::init_device(0).expect("init SM121");
    let stream = ctx.default_stream();

    let n: u32 = 32;
    let mut s = 0xFEEDFACE_u64;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let al: Vec<u16> = (0..n).map(|_| bf16(next() * 4.0)).collect();
    let bl: Vec<u16> = (0..n).map(|_| bf16(next() * 2.0)).collect();
    let db: Vec<u16> = (0..n).map(|_| bf16(next() * 0.5)).collect();
    let alg: Vec<u16> = (0..n).map(|_| bf16(next() * 0.5 - 0.5)).collect(); // negative-ish

    let (alpha_ref, beta_ref) = cpu_ref(&al, &bl, &db, &alg);

    let al_dev = stream.memcpy_stod(&al).unwrap();
    let bl_dev = stream.memcpy_stod(&bl).unwrap();
    let db_dev = stream.memcpy_stod(&db).unwrap();
    let alg_dev = stream.memcpy_stod(&alg).unwrap();
    let mut alpha_dev = stream.alloc_zeros::<f32>(n as usize).unwrap();
    let mut beta_dev = stream.alloc_zeros::<f32>(n as usize).unwrap();

    linear_attention::gdn_alpha_beta_bf16(
        &ctx,
        &stream,
        &al_dev,
        &bl_dev,
        &db_dev,
        &alg_dev,
        &mut alpha_dev,
        &mut beta_dev,
        n,
    )
    .expect("kernel launch");

    let alpha_h = stream.memcpy_dtov(&alpha_dev).unwrap();
    let beta_h = stream.memcpy_dtov(&beta_dev).unwrap();
    let mut max_a = 0f32;
    let mut max_b = 0f32;
    for i in 0..n as usize {
        max_a = max_a.max((alpha_h[i] - alpha_ref[i]).abs());
        max_b = max_b.max((beta_h[i] - beta_ref[i]).abs());
    }
    eprintln!("alpha max_diff = {max_a:.6}, beta max_diff = {max_b:.6}");
    // ex2.approx + lg2.approx + rcp.approx contribute a few ULPs; tol modest.
    assert!(max_a < 5e-3, "alpha diff {max_a}");
    assert!(max_b < 5e-3, "beta diff {max_b}");
}
