//! Bit-exactness gate for the deterministic split-K GEMM.
//!
//! `gemm_bf16_split_k_det` (batched, M rows) must produce, for every row m,
//! output BYTE-IDENTICAL to `gemv_bf16_split_k_det` (M=1) run on row m alone —
//! provided both use the same `num_shards`. That is the property that lets a
//! batched speculative-verify pass commit the SAME greedy tokens the M=1
//! decode path would.
//!
//! This test is input-agnostic: bit-identity must hold for arbitrary weights /
//! activations, so we use a deterministic xorshift PRNG (no golden vectors).

use sm121_kernels::{device, gemm};

/// Deterministic small-magnitude bf16 value from a 32-bit state.
fn rand_bf16(state: &mut u32) -> u16 {
    // xorshift32
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    // map to roughly [-1, 1)
    let f = ((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
    f32_to_bf16(f)
}

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

#[test]
fn split_k_det_batched_matches_m1_bytewise() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    // (n, k, num_shards) chosen to mirror real 9B GDN-hybrid decode projections:
    //   - attention/GDN via linear_decode_bf16: num_shards=((k+1023)/1024).clamp(1,8)
    //   - MLP via run_moe_layer:                num_shards=(k/256).clamp(1,16)
    //   - plus stress points (shards=1, last-shard-shorter via non-divisible k).
    let shapes: &[(u32, u32, u32)] = &[
        (512, 2048, 2),    // generic small
        (768, 4096, 4),    // attention qkv-ish (k=hidden), 4 shards
        (4096, 4096, 16),  // MLP gate/up shape, 16 shards
        (2048, 12288, 16), // MLP down shape (k=intermediate), 16 shards
        (256, 1024, 1),    // single shard (no split)
        (1024, 3000, 4),   // non-divisible k -> last shard shorter
        (512, 4096, 8),    // 8 shards
    ];
    let m_values: &[u32] = &[1, 2, 3, 4, 5, 8];

    let mut failures = 0usize;
    for &(n, k, num_shards) in shapes {
        // Weight B[k, n] — shared across all M and across both paths.
        let mut bstate = 0x1234_5678u32 ^ (n.wrapping_mul(2654435761)) ^ k;
        let b_host: Vec<u16> = (0..(k as usize * n as usize))
            .map(|_| rand_bf16(&mut bstate))
            .collect();
        let b_dev = stream.memcpy_stod(&b_host).unwrap();

        for &m in m_values {
            // Activations X[m, k].
            let mut xstate = 0x9E37_79B9u32 ^ m.wrapping_mul(40503) ^ k;
            let x_host: Vec<u16> = (0..(m as usize * k as usize))
                .map(|_| rand_bf16(&mut xstate))
                .collect();
            let x_dev = stream.memcpy_stod(&x_host).unwrap();

            // --- M=1 reference: run each row through the decode-path kernel. ---
            let mut ref_out: Vec<u16> = vec![0u16; m as usize * n as usize];
            for row in 0..m as usize {
                let xr = stream
                    .memcpy_stod(&x_host[row * k as usize..(row + 1) * k as usize])
                    .unwrap();
                let mut stage1 = stream
                    .alloc_zeros::<f32>((num_shards * n) as usize)
                    .unwrap();
                let mut out1 = stream.alloc_zeros::<u16>(n as usize).unwrap();
                gemm::gemv_bf16_split_k_det(
                    &ctx,
                    &stream,
                    &xr,
                    &b_dev,
                    &mut stage1,
                    n,
                    k,
                    num_shards,
                )
                .unwrap();
                gemm::f32_shard_reduce_to_bf16(&ctx, &stream, &stage1, &mut out1, n, num_shards)
                    .unwrap();
                let row_h = stream.memcpy_dtov(&out1).unwrap();
                ref_out[row * n as usize..(row + 1) * n as usize].copy_from_slice(&row_h);
            }

            // --- Batched: all M rows in one weight-stream. ---
            let mut stage_b = stream
                .alloc_zeros::<f32>((num_shards * m * n) as usize)
                .unwrap();
            let mut out_b = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
            gemm::gemm_bf16_split_k_det_managed(
                &ctx,
                &stream,
                &x_dev,
                &b_dev,
                &mut stage_b,
                &mut out_b,
                n,
                k,
                m,
                num_shards,
            )
            .unwrap();
            let batched_h = stream.memcpy_dtov(&out_b).unwrap();

            // Bytewise compare.
            let mut first_bad: Option<(usize, usize)> = None;
            for row in 0..m as usize {
                for j in 0..n as usize {
                    let idx = row * n as usize + j;
                    if batched_h[idx] != ref_out[idx] {
                        first_bad = Some((row, j));
                        break;
                    }
                }
                if first_bad.is_some() {
                    break;
                }
            }
            match first_bad {
                None => {
                    eprintln!(
                        "OK  n={n} k={k} shards={num_shards} m={m}: {} rows byte-identical",
                        m
                    );
                }
                Some((row, j)) => {
                    failures += 1;
                    let idx = row * n as usize + j;
                    eprintln!(
                        "BAD n={n} k={k} shards={num_shards} m={m}: row {row} col {j} batched=0x{:04x} ref=0x{:04x}",
                        batched_h[idx], ref_out[idx]
                    );
                }
            }
        }
    }

    assert_eq!(
        failures, 0,
        "{failures} shape/m combinations were NOT byte-identical"
    );
}
