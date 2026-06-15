//! MLA split-K combine kernel: log-sum-exp reduce partial outputs.

mod common;

use sm121_kernels::{attention, device};

#[test]
fn test_mla_split_kv_combine_small() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let batch = 1u32;
    let num_heads = 2u32;
    let num_splits = 4u32;
    let d_c = 512u32;

    // Build partials: each split's O is all-ones of a distinct magnitude,
    // weighted by distinct LSE. Then check Σ weight * O matches.
    let n_partial = (num_splits * batch * num_heads * d_c) as usize;
    let mut o_partial = vec![0.0f32; n_partial];
    let mut lse = vec![0.0f32; (num_splits * batch * num_heads) as usize];

    for s in 0..num_splits as usize {
        for b in 0..batch as usize {
            for h in 0..num_heads as usize {
                for d in 0..d_c as usize {
                    let idx =
                        ((s * batch as usize + b) * num_heads as usize + h) * d_c as usize + d;
                    o_partial[idx] = s as f32 + 1.0; // split 0 = 1, split 1 = 2, ...
                }
                let lse_idx = (s * batch as usize + b) * num_heads as usize + h;
                lse[lse_idx] = s as f32 * 0.5; // lse = 0, 0.5, 1.0, 1.5
            }
        }
    }

    // Expected: weights = exp(lse - max_lse) = exp([-1.5, -1.0, -0.5, 0])
    //                   = [0.2231, 0.3679, 0.6065, 1.0]
    //         Z = sum = 2.1975
    //   o_final[d] = Σ w_i * val_i / Z
    //             = (0.2231*1 + 0.3679*2 + 0.6065*3 + 1.0*4) / 2.1975
    //             ≈ 3.147
    let weights: Vec<f32> = (0..num_splits as usize)
        .map(|s| ((s as f32 * 0.5) - 1.5).exp())
        .collect();
    let z: f32 = weights.iter().sum();
    let expected: f32 = weights
        .iter()
        .enumerate()
        .map(|(s, w)| w * (s as f32 + 1.0))
        .sum::<f32>()
        / z;

    let o_p_dev = stream.memcpy_stod(&o_partial).unwrap();
    let lse_dev = stream.memcpy_stod(&lse).unwrap();
    let mut o_f_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * d_c) as usize)
        .unwrap();

    attention::mla_split_kv_combine(
        &ctx,
        &stream,
        &o_p_dev,
        &lse_dev,
        &mut o_f_dev,
        batch,
        num_heads,
        num_splits,
    )
    .expect("combine failed");

    let o_host = stream.memcpy_dtov(&o_f_dev).unwrap();
    let mut max_diff: f32 = 0.0;
    for &bits in &o_host {
        let f = f32::from_bits((bits as u32) << 16);
        let d = (f - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("MLA combine: expected={expected:.4}, max_diff={max_diff:.4}");
    assert!(max_diff <= 0.02, "combine output drift: {}", max_diff);
}
