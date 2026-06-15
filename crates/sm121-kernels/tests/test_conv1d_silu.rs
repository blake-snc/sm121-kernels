mod common;

use common::compare_bf16;
use sm121_kernels::{device, linear_attention};

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn unbf16(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

/// CPU reference: depthwise causal conv1d (kernel=4) + SiLU.
/// state: [3, total_qkv]   — three prior tokens, channel-minor
/// current: [total_qkv]
/// weight: [total_qkv, 1, 4] = [total_qkv * 4]
/// returns (out [total_qkv], state_advanced [3, total_qkv])
// `0 * total_qkv` keeps the time-step index (t=0) explicit in the conv-window
// addressing so the layout matches the kernel's flat indexing; keep it.
#[allow(clippy::erasing_op)]
fn cpu_ref(
    state: &[u16],
    current: &[u16],
    weight: &[u16],
    total_qkv: usize,
) -> (Vec<u16>, Vec<u16>) {
    let mut out = vec![0u16; total_qkv];
    for c in 0..total_qkv {
        let wbase = c * 4;
        let s0 = unbf16(state[0 * total_qkv + c]);
        let s1 = unbf16(state[total_qkv + c]);
        let s2 = unbf16(state[2 * total_qkv + c]);
        let cur = unbf16(current[c]);
        let w0 = unbf16(weight[wbase]);
        let w1 = unbf16(weight[wbase + 1]);
        let w2 = unbf16(weight[wbase + 2]);
        let w3 = unbf16(weight[wbase + 3]);
        let y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * cur;
        let sig = 1.0 / (1.0 + (-y).exp());
        out[c] = bf16(y * sig);
    }
    let mut new_state = vec![0u16; 3 * total_qkv];
    for c in 0..total_qkv {
        new_state[0 * total_qkv + c] = state[total_qkv + c];
        new_state[total_qkv + c] = state[2 * total_qkv + c];
        new_state[2 * total_qkv + c] = current[c];
    }
    (out, new_state)
}

fn run_case(total_qkv: u32, seed: u64) {
    let ctx = device::init_device(0).expect("init SM121");
    let stream = ctx.default_stream();

    let n = total_qkv as usize;
    // Cheap deterministic LCG pseudo-random.
    let mut s = seed;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-1, 1]
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let state_h: Vec<u16> = (0..3 * n).map(|_| bf16(next() * 0.5)).collect();
    let current_h: Vec<u16> = (0..n).map(|_| bf16(next() * 0.5)).collect();
    let weight_h: Vec<u16> = (0..4 * n).map(|_| bf16(next() * 0.3)).collect();

    let (out_ref, state_ref) = cpu_ref(&state_h, &current_h, &weight_h, n);

    let mut state_dev = stream.memcpy_stod(&state_h).unwrap();
    let current_dev = stream.memcpy_stod(&current_h).unwrap();
    let weight_dev = stream.memcpy_stod(&weight_h).unwrap();
    let mut out_dev = stream.alloc_zeros::<u16>(n).unwrap();

    linear_attention::conv1d_silu_bf16(
        &ctx,
        &stream,
        &mut state_dev,
        &current_dev,
        &weight_dev,
        &mut out_dev,
        total_qkv,
    )
    .expect("conv1d_silu_bf16 launch");

    let out_host = stream.memcpy_dtov(&out_dev).unwrap();
    let state_host = stream.memcpy_dtov(&state_dev).unwrap();

    // SiLU + 4-tap accumulation in BF16 has accumulated rounding ~ 2-3 ULP per channel.
    let (max_y, mean_y) = compare_bf16(&out_host, &out_ref, 0.02);
    eprintln!("conv1d_silu n={n}: out max_diff={max_y:.5} mean={mean_y:.5}");

    // State advance is a pure copy — must be bit-exact.
    let (max_s, mean_s) = compare_bf16(&state_host, &state_ref, 0.0);
    eprintln!("conv1d_silu n={n}: state max_diff={max_s:.5} mean={mean_s:.5}");
}

#[test]
fn conv1d_silu_gdn_hybrid_total_qkv() {
    // GDN-hybrid: total_qkv = 6144 (Q) + 1024 (K) + 1024 (V) = 8192.
    run_case(8192, 0xCAFEBABE_DEADBEEF);
}

#[test]
fn conv1d_silu_partial_block() {
    // Non-multiple of BLOCK=128: exercises the bounds-guard.
    run_case(8192 + 17, 0x12345678_ABCDEF01);
}

#[test]
fn conv1d_silu_small() {
    // Small smoke: single block.
    run_case(64, 0xFEED_FACE_BAAD_F00D);
}
