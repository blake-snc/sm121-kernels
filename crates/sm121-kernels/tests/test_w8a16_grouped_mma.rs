//! Correctness for `gemm_w8a16_grouped_mma` vs `gemv_w8a16_grouped_split_k`.
//! Both compute the same math (per-slot: out[s, :] = bf16(b_scale * sum_k
//! a[s, k] * fp8_dequant(B[expert(s), k, :]))). The MMA kernel does it with
//! tile MMAs and FP32 accumulators; the split-K GEMV does it slot-by-slot
//! with sequential FMAs. Both round once at the end via cvt.rn.bf16.f32.
//! Expect close agreement (BF16 noise floor).

mod common;

use sm121_kernels::{device, moe};

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn unbf16(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

#[test]
fn w8a16_grouped_mma_matches_gemv_split_k() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    // Small but realistic shape: 4 experts, 16 tokens (avg 4 per expert),
    // K=128, N=64.
    let num_experts: u32 = 4;
    let m_max: u32 = 8;
    let n: u32 = 64;
    let k: u32 = 128;

    // Token-to-expert assignment: experts 0/1/2/3 get 5/3/4/4 tokens.
    let expert_token_counts: Vec<u32> = vec![5, 3, 4, 4];
    let total_tokens: u32 = expert_token_counts.iter().sum();
    assert_eq!(total_tokens, 16);

    let mut expert_offsets_host: Vec<u32> = vec![0];
    for &c in &expert_token_counts {
        let prev = *expert_offsets_host.last().unwrap();
        expert_offsets_host.push(prev + c);
    }
    // active_eids[s] = expert that owns slot s (for the GEMV reference).
    let mut active_eids_host: Vec<u32> = Vec::with_capacity(total_tokens as usize);
    for (e, &cnt) in expert_token_counts.iter().enumerate() {
        for _ in 0..cnt {
            active_eids_host.push(e as u32);
        }
    }

    // Random inputs (deterministic LCG).
    let mut s = 0xCAFE_BABE_u64;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5
    };
    let a_host: Vec<u16> = (0..(total_tokens * k) as usize)
        .map(|_| bf16(next() * 0.5))
        .collect();
    let b_fp8_host: Vec<u8> = (0..(num_experts * k * n) as usize)
        .map(|i| ((i * 31 + 17) % 240) as u8)
        .collect();
    let b_scale: f32 = 0.03125;

    let a_dev = stream.memcpy_stod(&a_host).unwrap();
    let b_fp8_dev = stream.memcpy_stod(&b_fp8_host).unwrap();
    let expert_offsets_dev = stream.memcpy_stod(&expert_offsets_host).unwrap();
    let active_eids_dev = stream.memcpy_stod(&active_eids_host).unwrap();

    // === Reference: gemv_w8a16_grouped_split_k over total_tokens slots ===
    let mut ref_f32 = stream
        .alloc_zeros::<f32>((total_tokens * n) as usize)
        .unwrap();
    let num_shards: u32 = (k / 256).max(1).min(8);
    moe::gemv_w8a16_grouped_split_k(
        &ctx,
        &stream,
        &a_dev,
        &b_fp8_dev,
        b_scale,
        &active_eids_dev,
        &mut ref_f32,
        total_tokens,
        n,
        k,
        num_shards,
        k, // x_stride = K (per-slot a)
    )
    .unwrap();
    let mut ref_bf16 = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();
    sm121_kernels::activation::f32_to_bf16(
        &ctx,
        &stream,
        &ref_f32,
        &mut ref_bf16,
        total_tokens * n,
    )
    .unwrap();

    // === Subject: gemm_w8a16_grouped_mma ===
    let mut sub_bf16 = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();
    moe::gemm_w8a16_grouped_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_fp8_dev,
        b_scale,
        &mut sub_bf16,
        &expert_offsets_dev,
        num_experts,
        m_max,
        n,
        k,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let ref_h = stream.memcpy_dtov(&ref_bf16).unwrap();
    let sub_h = stream.memcpy_dtov(&sub_bf16).unwrap();

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut n_bad = 0;
    for i in 0..(total_tokens * n) as usize {
        let r = unbf16(ref_h[i]);
        let s = unbf16(sub_h[i]);
        let d = (r - s).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = r.abs().max(1e-6);
        let rel = d / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        if d > 0.05 {
            n_bad += 1;
            if n_bad <= 4 {
                eprintln!("  bad@{i} ref={r:.4} sub={s:.4} diff={d:.4}");
            }
        }
    }
    eprintln!(
        "max_abs={max_abs:.4e}, max_rel={:.4}%, bad>{}={n_bad}",
        max_rel * 100.0,
        0.05
    );
    // BF16 noise floor at this scale (~1-2 ULP per element after FP32 accum).
    // The MMA and GEMV paths use different summation orders → expect ~ULP diff.
    assert!(max_rel < 0.05, "rel diff too large: {max_rel}");
}
