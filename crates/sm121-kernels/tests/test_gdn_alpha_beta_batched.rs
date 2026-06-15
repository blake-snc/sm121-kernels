//! Correctness test: batched gdn_alpha_beta vs M
//! sequential single-seq calls. dt_bias and a_log are model weights
//! (no M offset); a_logits/b_logits and outputs are M-batched.

use sm121_kernels::{device, linear_attention as la};

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((self.0 >> 33) as u32) as f32;
        (v / (u32::MAX as f32)) * 2.0 - 1.0
    }
}
fn random_bf16_vec(rng: &mut Lcg, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(rng.next_f32()).to_bits())
        .collect()
}

#[test]
fn gdn_alpha_beta_batched_matches_sequential() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let n = 32u32; // 9B GDN-hybrid n_v = 32
    let m = 16u32;
    let mut rng = Lcg::new(0xBEEF_CAFE);

    let dt_bias_host = random_bf16_vec(&mut rng, n as usize);
    let a_log_host = random_bf16_vec(&mut rng, n as usize);
    let dt_bias = stream.memcpy_stod(&dt_bias_host).unwrap();
    let a_log = stream.memcpy_stod(&a_log_host).unwrap();

    let seq_a: Vec<Vec<u16>> = (0..m as usize)
        .map(|_| random_bf16_vec(&mut rng, n as usize))
        .collect();
    let seq_b: Vec<Vec<u16>> = (0..m as usize)
        .map(|_| random_bf16_vec(&mut rng, n as usize))
        .collect();

    // Path A: M sequential calls
    let mut seq_alphas: Vec<Vec<f32>> = Vec::with_capacity(m as usize);
    let mut seq_betas: Vec<Vec<f32>> = Vec::with_capacity(m as usize);
    for s in 0..m as usize {
        let a_d = stream.memcpy_stod(&seq_a[s]).unwrap();
        let b_d = stream.memcpy_stod(&seq_b[s]).unwrap();
        let mut alpha_d = stream.alloc_zeros::<f32>(n as usize).unwrap();
        let mut beta_d = stream.alloc_zeros::<f32>(n as usize).unwrap();
        la::gdn_alpha_beta_bf16(
            &ctx,
            &stream,
            &a_d,
            &b_d,
            &dt_bias,
            &a_log,
            &mut alpha_d,
            &mut beta_d,
            n,
        )
        .unwrap();
        seq_alphas.push(stream.memcpy_dtov(&alpha_d).unwrap());
        seq_betas.push(stream.memcpy_dtov(&beta_d).unwrap());
    }

    // Path B: ONE batched call
    let mut bat_a_host = Vec::with_capacity(m as usize * n as usize);
    let mut bat_b_host = Vec::with_capacity(m as usize * n as usize);
    for s in 0..m as usize {
        bat_a_host.extend_from_slice(&seq_a[s]);
        bat_b_host.extend_from_slice(&seq_b[s]);
    }
    let bat_a = stream.memcpy_stod(&bat_a_host).unwrap();
    let bat_b = stream.memcpy_stod(&bat_b_host).unwrap();
    let mut bat_alpha = stream.alloc_zeros::<f32>(m as usize * n as usize).unwrap();
    let mut bat_beta = stream.alloc_zeros::<f32>(m as usize * n as usize).unwrap();
    la::gdn_alpha_beta_bf16_batched(
        &ctx,
        &stream,
        &bat_a,
        &bat_b,
        &dt_bias,
        &a_log,
        &mut bat_alpha,
        &mut bat_beta,
        n,
        m,
    )
    .unwrap();
    let bat_alpha_host = stream.memcpy_dtov(&bat_alpha).unwrap();
    let bat_beta_host = stream.memcpy_dtov(&bat_beta).unwrap();

    // Compare
    let mut max_alpha_diff = 0.0f32;
    let mut max_beta_diff = 0.0f32;
    for s in 0..m as usize {
        let row_a = &bat_alpha_host[s * n as usize..(s + 1) * n as usize];
        let row_b = &bat_beta_host[s * n as usize..(s + 1) * n as usize];
        for i in 0..n as usize {
            let da = (row_a[i] - seq_alphas[s][i]).abs();
            let db = (row_b[i] - seq_betas[s][i]).abs();
            if da > max_alpha_diff {
                max_alpha_diff = da;
            }
            if db > max_beta_diff {
                max_beta_diff = db;
            }
        }
    }
    assert_eq!(max_alpha_diff, 0.0, "batched alpha mismatch");
    assert_eq!(max_beta_diff, 0.0, "batched beta mismatch");
    eprintln!("gdn_alpha_beta batched test passed: M={m}, n={n}, alpha + beta both exact match");
}
