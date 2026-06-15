//! Validate linear_attn_chunk_prefill against CPU reference.
//!
//! Recurrence:
//!   S_t = S_{t-1} + v_t k_t^T
//!   y_t = S_t q_t
//!
//! Chunk algorithm decomposes:
//!   Y_intra = lowerTri(Q K^T) V
//!   Y_inter = Q S_init^T
//!   Y = Y_intra + Y_inter
//!   S_new = S_init + V^T K
//!
//! Test: process N chunks via the kernel, compare final Y and S to a CPU
//! sequential reference using the same recurrence.

use half::bf16;
use sm121_kernels::{device, linear_attention};

const C: usize = 32;
const D: usize = 128;

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

/// CPU reference: process num_chunks chunks sequentially using the recurrence.
/// Returns (Y, S_final).
fn cpu_reference(
    k: &[u16],
    v: &[u16],
    q: &[u16],
    s_init: &[f32],
    num_chunks: usize,
) -> (Vec<u16>, Vec<f32>) {
    let total_tokens = num_chunks * C;
    let mut y = vec![0u16; total_tokens * D];
    let mut s = s_init.to_vec();
    for t in 0..total_tokens {
        // Read K[t], V[t], Q[t]
        let k_row: Vec<f32> = (0..D)
            .map(|d| bf16::from_bits(k[t * D + d]).to_f32())
            .collect();
        let v_row: Vec<f32> = (0..D)
            .map(|d| bf16::from_bits(v[t * D + d]).to_f32())
            .collect();
        let q_row: Vec<f32> = (0..D)
            .map(|d| bf16::from_bits(q[t * D + d]).to_f32())
            .collect();

        // S[t,j] += V[t,j] * K[t,j']  for all (t,j') — wait, that's S[i,j] += v[i] * k[j]
        // Actually S[d_out, d_in] += V[t, d_out] * K[t, d_in]
        for d_out in 0..D {
            for d_in in 0..D {
                s[d_out * D + d_in] += v_row[d_out] * k_row[d_in];
            }
        }

        // y[d] = S[d, :] · Q
        for d_out in 0..D {
            let mut acc = 0f32;
            for d_in in 0..D {
                acc += s[d_out * D + d_in] * q_row[d_in];
            }
            y[t * D + d_out] = bf16::from_f32(acc).to_bits();
        }
    }
    (y, s)
}

#[test]
fn t_linear_attn_chunk_b1_h1_1chunk() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let batch = 1u32;
    let num_heads = 1u32;
    let num_chunks = 1usize;
    let total_tokens = num_chunks * C;

    let k = random_bf16(batch as usize * num_heads as usize * total_tokens * D, 0xA);
    let v = random_bf16(batch as usize * num_heads as usize * total_tokens * D, 0xB);
    let q = random_bf16(batch as usize * num_heads as usize * total_tokens * D, 0xC);
    let s_init = vec![0f32; batch as usize * num_heads as usize * D * D];

    let (y_ref, s_ref) = cpu_reference(&k, &v, &q, &s_init, num_chunks);

    let k_dev = stream.memcpy_stod(&k).unwrap();
    let v_dev = stream.memcpy_stod(&v).unwrap();
    let q_dev = stream.memcpy_stod(&q).unwrap();
    let mut y_dev = stream.alloc_zeros::<u16>(y_ref.len()).unwrap();
    // State is FP16 now.
    let s_init_f16: Vec<u16> = s_init
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let s_in_dev = stream.memcpy_stod(&s_init_f16).unwrap();
    let mut s_out_dev = stream.memcpy_stod(&s_init_f16).unwrap();

    // Single chunk call
    linear_attention::linear_attn_chunk_prefill(
        &ctx,
        &stream,
        &k_dev,
        &v_dev,
        &q_dev,
        &mut y_dev,
        &s_in_dev,
        &mut s_out_dev,
        batch,
        num_heads,
    )
    .unwrap();

    let y_got = stream.memcpy_dtov(&y_dev).unwrap();
    let s_got_bits = stream.memcpy_dtov(&s_out_dev).unwrap();
    let s_got: Vec<f32> = s_got_bits
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect();

    // Compare Y
    let mut max_y_abs = 0f32;
    let mut max_y_rel = 0f32;
    let mut max_y_mag = 0f32;
    for (g, r) in y_got.iter().zip(y_ref.iter()) {
        let gf = bf16::from_bits(*g).to_f32();
        let rf = bf16::from_bits(*r).to_f32();
        let d = (gf - rf).abs();
        if d > max_y_abs {
            max_y_abs = d;
        }
        let m = gf.abs().max(rf.abs());
        if m > max_y_mag {
            max_y_mag = m;
        }
        if m > 0.01 {
            let rel = d / m;
            if rel > max_y_rel {
                max_y_rel = rel;
            }
        }
    }
    eprintln!("Y: max_abs={max_y_abs:.4} max_rel={max_y_rel:.4} max_mag={max_y_mag:.2}");

    // Compare S
    let mut max_s_abs = 0f32;
    let mut max_s_rel = 0f32;
    let mut max_s_mag = 0f32;
    for (g, r) in s_got.iter().zip(s_ref.iter()) {
        let d = (g - r).abs();
        if d > max_s_abs {
            max_s_abs = d;
        }
        let m = g.abs().max(r.abs());
        if m > max_s_mag {
            max_s_mag = m;
        }
        if m > 0.01 {
            let rel = d / m;
            if rel > max_s_rel {
                max_s_rel = rel;
            }
        }
    }
    eprintln!("S: max_abs={max_s_abs:.4} max_rel={max_s_rel:.4} max_mag={max_s_mag:.2}");

    assert!(max_y_rel < 0.10, "Y max_rel {max_y_rel} > 0.10");
    assert!(max_s_rel < 0.10, "S max_rel {max_s_rel} > 0.10");
}

#[test]
fn t_linear_attn_chunk_multi_chunk_iter() {
    // Iterate the chunk kernel 4 times, threading state. Compare to CPU reference of 4*C tokens.
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let batch = 1u32;
    let num_heads = 1u32;
    let num_chunks = 4usize;
    let total_tokens = num_chunks * C;

    let k = random_bf16(batch as usize * num_heads as usize * total_tokens * D, 0xA);
    let v = random_bf16(batch as usize * num_heads as usize * total_tokens * D, 0xB);
    let q = random_bf16(batch as usize * num_heads as usize * total_tokens * D, 0xC);
    let s_init = vec![0f32; batch as usize * num_heads as usize * D * D];

    let (y_ref, s_ref) = cpu_reference(&k, &v, &q, &s_init, num_chunks);

    // Per-chunk: state in/out, Y for this chunk
    let mut s_state = s_init.clone();
    let mut y_full = vec![0u16; total_tokens * D];

    for c in 0..num_chunks {
        let chunk_offset_elems = c * C * D;
        let k_chunk = &k[chunk_offset_elems..chunk_offset_elems + C * D];
        let v_chunk = &v[chunk_offset_elems..chunk_offset_elems + C * D];
        let q_chunk = &q[chunk_offset_elems..chunk_offset_elems + C * D];

        let k_dev = stream.memcpy_stod(k_chunk).unwrap();
        let v_dev = stream.memcpy_stod(v_chunk).unwrap();
        let q_dev = stream.memcpy_stod(q_chunk).unwrap();
        let mut y_dev = stream.alloc_zeros::<u16>(C * D).unwrap();
        // Thread state as FP16.
        let s_state_f16: Vec<u16> = s_state
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let s_in_dev = stream.memcpy_stod(&s_state_f16).unwrap();
        let mut s_out_dev = stream.memcpy_stod(&s_state_f16).unwrap();

        linear_attention::linear_attn_chunk_prefill(
            &ctx,
            &stream,
            &k_dev,
            &v_dev,
            &q_dev,
            &mut y_dev,
            &s_in_dev,
            &mut s_out_dev,
            batch,
            num_heads,
        )
        .unwrap();

        let y_chunk = stream.memcpy_dtov(&y_dev).unwrap();
        y_full[chunk_offset_elems..chunk_offset_elems + C * D].copy_from_slice(&y_chunk);
        let s_state_bits = stream.memcpy_dtov(&s_out_dev).unwrap();
        s_state = s_state_bits
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();
    }

    let mut max_y_rel = 0f32;
    let mut max_y_mag = 0f32;
    for (g, r) in y_full.iter().zip(y_ref.iter()) {
        let gf = bf16::from_bits(*g).to_f32();
        let rf = bf16::from_bits(*r).to_f32();
        let m = gf.abs().max(rf.abs());
        if m > max_y_mag {
            max_y_mag = m;
        }
        if m > 0.01 {
            let rel = (gf - rf).abs() / m;
            if rel > max_y_rel {
                max_y_rel = rel;
            }
        }
    }
    let mut max_s_rel = 0f32;
    let mut max_s_mag = 0f32;
    for (g, r) in s_state.iter().zip(s_ref.iter()) {
        let m = g.abs().max(r.abs());
        if m > max_s_mag {
            max_s_mag = m;
        }
        if m > 0.01 {
            let rel = (g - r).abs() / m;
            if rel > max_s_rel {
                max_s_rel = rel;
            }
        }
    }
    eprintln!("4-chunk: Y max_rel={max_y_rel:.4} (mag {max_y_mag:.2}), S max_rel={max_s_rel:.4} (mag {max_s_mag:.2})");
    assert!(max_y_rel < 0.10, "Y rel {max_y_rel} > 0.10");
    assert!(max_s_rel < 0.10, "S rel {max_s_rel} > 0.10");
}
