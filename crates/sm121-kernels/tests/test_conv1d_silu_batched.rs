//! Correctness test: batched conv1d_silu vs M sequential
//! calls of the proven single-seq kernel. Both must produce identical output
//! and identical state advancement.

use sm121_kernels::{device, linear_attention};

// Tiny LCG for deterministic test inputs (avoid rand dependency).
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
fn conv1d_silu_batched_matches_sequential() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let total_qkv = 8192u32; // 9B GDN-hybrid GDN total_qkv = 8192
    let m = 16u32; // smaller M for test speed; behavior is M-independent
    let n = total_qkv as usize;

    let mut rng = Lcg::new(0xCAFE_BABE);

    // Generate input data: per-seq state [3, total_qkv], current [total_qkv]
    // Weight is shared across M.
    let weight_host = random_bf16_vec(&mut rng, 4 * n);
    let weight = stream.memcpy_stod(&weight_host).unwrap();

    // Per-seq starting states + currents
    let seq_states: Vec<Vec<u16>> = (0..m as usize)
        .map(|_| random_bf16_vec(&mut rng, 3 * n))
        .collect();
    let seq_currents: Vec<Vec<u16>> = (0..m as usize)
        .map(|_| random_bf16_vec(&mut rng, n))
        .collect();

    // === Path A: M sequential calls of the single-seq kernel ===
    let mut seq_outs: Vec<Vec<u16>> = Vec::with_capacity(m as usize);
    let mut seq_final_states: Vec<Vec<u16>> = Vec::with_capacity(m as usize);
    for s in 0..m as usize {
        let mut state_dev = stream.memcpy_stod(&seq_states[s]).unwrap();
        let curr_dev = stream.memcpy_stod(&seq_currents[s]).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(n).unwrap();
        linear_attention::conv1d_silu_bf16(
            &ctx,
            &stream,
            &mut state_dev,
            &curr_dev,
            &weight,
            &mut out_dev,
            total_qkv,
        )
        .expect("seq conv1d");
        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let final_state = stream.memcpy_dtov(&state_dev).unwrap();
        seq_outs.push(out_host);
        seq_final_states.push(final_state);
    }

    // === Path B: ONE batched call ===
    // Pack [M, 3, total_qkv] state and [M, total_qkv] current into single buffers.
    let mut batched_state_host: Vec<u16> = Vec::with_capacity(m as usize * 3 * n);
    let mut batched_current_host: Vec<u16> = Vec::with_capacity(m as usize * n);
    for s in 0..m as usize {
        batched_state_host.extend_from_slice(&seq_states[s]);
        batched_current_host.extend_from_slice(&seq_currents[s]);
    }
    let mut batched_state = stream.memcpy_stod(&batched_state_host).unwrap();
    let batched_current = stream.memcpy_stod(&batched_current_host).unwrap();
    let mut batched_out = stream.alloc_zeros::<u16>(m as usize * n).unwrap();
    linear_attention::conv1d_silu_bf16_batched(
        &ctx,
        &stream,
        &mut batched_state,
        &batched_current,
        &weight,
        &mut batched_out,
        total_qkv,
        m,
    )
    .expect("batched conv1d");
    let batched_out_host = stream.memcpy_dtov(&batched_out).unwrap();
    let batched_state_final = stream.memcpy_dtov(&batched_state).unwrap();

    // === Compare ===
    // Output: batched [s*n..(s+1)*n] should match seq_outs[s]
    let mut max_out_diff = 0i32;
    let mut diverged_seq = -1i32;
    for s in 0..m as usize {
        let row = &batched_out_host[s * n..(s + 1) * n];
        for i in 0..n {
            let d = row[i] as i32 - seq_outs[s][i] as i32;
            if d.abs() > max_out_diff {
                max_out_diff = d.abs();
                if d != 0 && diverged_seq < 0 {
                    diverged_seq = s as i32;
                }
            }
        }
    }
    assert_eq!(
        max_out_diff, 0,
        "batched conv1d output mismatch vs sequential (seq {} diverged)",
        diverged_seq
    );

    // State: batched [s*3*n..(s+1)*3*n] should match seq_final_states[s]
    let mut max_state_diff = 0i32;
    for s in 0..m as usize {
        let row = &batched_state_final[s * 3 * n..(s + 1) * 3 * n];
        for i in 0..(3 * n) {
            let d = row[i] as i32 - seq_final_states[s][i] as i32;
            if d.abs() > max_state_diff {
                max_state_diff = d.abs();
            }
        }
    }
    assert_eq!(
        max_state_diff, 0,
        "batched conv1d state mismatch vs sequential"
    );

    eprintln!("conv1d_silu_batched test passed: M={m}, total_qkv={total_qkv}, output exact match, state exact match");
}
