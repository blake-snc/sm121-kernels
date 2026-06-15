//! Device-side masked argmax matches the host result.
//!
//! Correctness: for a set of synthetic logits + allowed-token bitmasks,
//! the PTX kernel `masked_argmax_bf16` must produce the same token id as
//! a straightforward host reference (inlined below).
//!
//! Performance: micro-benchmark device kernel vs host implementation
//! for typical grammar states (sparse, dense, single-token, full vocab).

mod common;

use anyhow::{anyhow, Result};
use sm121_kernels::device;
use sm121_kernels::sampling::{
    allowed_to_bitmask, masked_argmax_bf16 as device_masked_argmax, masked_argmax_bf16_v2,
};

/// Host reference: argmax over the allowed token set on BF16 logits.
fn host_masked_argmax(logits: &[u16], allowed: &[u32]) -> u32 {
    let mut best: u32 = allowed[0];
    let mut best_v = f32::from_bits((logits[best as usize] as u32) << 16);
    for &tid in &allowed[1..] {
        let f = f32::from_bits((logits[tid as usize] as u32) << 16);
        if f > best_v {
            best_v = f;
            best = tid;
        }
    }
    best
}

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
    rounded as u16
}

#[test]
fn device_masked_argmax_matches_host_on_synthetic_cases() -> Result<()> {
    let ctx = device::init_device(0).map_err(|e| anyhow!("init: {e:?}"))?;
    let stream = ctx.new_stream().map_err(|e| anyhow!("stream: {e:?}"))?;

    let vocab: u32 = 248320;

    // Generate synthetic BF16 logits — deterministic PRNG so test is reproducible
    println!("===== build synthetic logits + 4 mask scenarios =====");
    let mut s = 0xC0FFEEu64;
    let mut next_f32 = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as i32 as f32) / (1i32 << 30) as f32 * 10.0
    };
    let logits_h: Vec<u16> = (0..vocab).map(|_| f32_to_bf16(next_f32())).collect();
    let logits_d = stream
        .memcpy_stod(&logits_h)
        .map_err(|e| anyhow!("htod: {e:?}"))?;

    // Four scenarios spanning typical grammar states
    let scenarios: Vec<(&str, Vec<u32>)> = vec![
        ("single-token (forced prefix)", vec![42_000u32]),
        (
            "3 tokens (close-quote, comma, brace)",
            vec![100u32, 1000, 10_000],
        ),
        (
            "sparse: 50 tokens (small enum + literals)",
            (0..50).map(|i| i * 5_000).collect::<Vec<_>>(),
        ),
        (
            "free string body: ~80K tokens (printable ASCII tokens)",
            (0..80_000u32).map(|i| (i * 3) % vocab).collect::<Vec<_>>(),
        ),
    ];

    let mut tokens_d = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| anyhow!("alloc: {e:?}"))?;
    for (name, mut allowed) in scenarios {
        allowed.sort_unstable();
        allowed.dedup();
        // -- HOST argmax --
        let host_pick = host_masked_argmax(&logits_h, &allowed);

        // -- DEVICE argmax --
        let mask_bytes = allowed_to_bitmask(&allowed, vocab);
        let mask_d = stream
            .memcpy_stod(&mask_bytes)
            .map_err(|e| anyhow!("htod mask: {e:?}"))?;
        device_masked_argmax(&ctx, &stream, &logits_d, &mask_d, &mut tokens_d, 1, vocab)
            .map_err(|e| anyhow!("device kernel: {e:?}"))?;
        let device_pick = stream
            .memcpy_dtov(&tokens_d)
            .map_err(|e| anyhow!("dtoh: {e:?}"))?[0];

        // Both picks must be in allowed AND have the same (max) logit value.
        // Tie-breaking can differ between sequential (host) and parallel
        // (device) reductions — both are mathematically valid argmaxes.
        assert!(
            allowed.binary_search(&host_pick).is_ok(),
            "host pick {} not in allowed",
            host_pick
        );
        assert!(
            allowed.binary_search(&device_pick).is_ok(),
            "device pick {} not in allowed",
            device_pick
        );
        let host_val = f32::from_bits((logits_h[host_pick as usize] as u32) << 16);
        let device_val = f32::from_bits((logits_h[device_pick as usize] as u32) << 16);
        println!(
            "  [{}] allowed={}  host={} (val={:.6})  device={} (val={:.6})",
            name,
            allowed.len(),
            host_pick,
            host_val,
            device_pick,
            device_val
        );
        assert_eq!(
            host_val, device_val,
            "scenario '{}': logit value mismatch — host={:.6} device={:.6}",
            name, host_val, device_val
        );
    }
    println!("  ✓ device kernel matches host across all 4 scenarios");
    Ok(())
}

#[test]
fn v2_matches_v1_on_synthetic_cases() -> Result<()> {
    let ctx = device::init_device(0).map_err(|e| anyhow!("init: {e:?}"))?;
    let stream = ctx.new_stream().map_err(|e| anyhow!("stream: {e:?}"))?;
    let vocab: u32 = 248320;
    let mut s = 0xC0FFEEu64;
    let mut next_f32 = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as i32 as f32) / (1i32 << 30) as f32 * 10.0
    };
    let logits_h: Vec<u16> = (0..vocab).map(|_| f32_to_bf16(next_f32())).collect();
    let logits_d = stream
        .memcpy_stod(&logits_h)
        .map_err(|e| anyhow!("htod: {e:?}"))?;

    let scenarios: Vec<(&str, Vec<u32>, u32)> = vec![
        ("single-token", vec![42_000u32], 4),
        ("3 tokens", vec![100u32, 1000, 10_000], 4),
        ("50 sparse", (0..50).map(|i| i * 5_000).collect(), 8),
        (
            "80K dense",
            (0..80_000u32).map(|i| (i * 3) % vocab).collect(),
            16,
        ),
    ];

    let mut tokens_v1 = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| anyhow!("alloc: {e:?}"))?;
    let mut tokens_v2 = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| anyhow!("alloc: {e:?}"))?;
    for (name, mut allowed, n_blocks) in scenarios {
        allowed.sort_unstable();
        allowed.dedup();
        let mask_bytes = allowed_to_bitmask(&allowed, vocab);
        let mask_d = stream
            .memcpy_stod(&mask_bytes)
            .map_err(|e| anyhow!("htod: {e:?}"))?;
        let mut scratch = stream
            .alloc_zeros::<u32>((n_blocks * 2) as usize)
            .map_err(|e| anyhow!("alloc scratch: {e:?}"))?;

        device_masked_argmax(&ctx, &stream, &logits_d, &mask_d, &mut tokens_v1, 1, vocab)?;
        masked_argmax_bf16_v2(
            &ctx,
            &stream,
            &logits_d,
            &mask_d,
            &mut scratch,
            &mut tokens_v2,
            1,
            vocab,
            n_blocks,
        )?;
        let p1 = stream
            .memcpy_dtov(&tokens_v1)
            .map_err(|e| anyhow!("dtoh: {e:?}"))?[0];
        let p2 = stream
            .memcpy_dtov(&tokens_v2)
            .map_err(|e| anyhow!("dtoh: {e:?}"))?[0];
        let v1 = f32::from_bits((logits_h[p1 as usize] as u32) << 16);
        let v2 = f32::from_bits((logits_h[p2 as usize] as u32) << 16);
        println!(
            "  [{}] n_blocks={}: v1 tok={} (v={:.4}), v2 tok={} (v={:.4})",
            name, n_blocks, p1, v1, p2, v2
        );
        // Both must be in allowed, and must have same logit value (tie-tolerant)
        assert!(
            allowed.binary_search(&p2).is_ok(),
            "v2 pick {} not in allowed for '{}'",
            p2,
            name
        );
        assert_eq!(
            v1, v2,
            "scenario '{}': v1 val {} != v2 val {}",
            name, v1, v2
        );
    }
    println!("  ✓ v2 matches v1 on all scenarios (tie-tolerant)");
    Ok(())
}

#[test]
fn device_vs_host_micro_benchmark() -> Result<()> {
    let ctx = device::init_device(0).map_err(|e| anyhow!("init: {e:?}"))?;
    let stream = ctx.new_stream().map_err(|e| anyhow!("stream: {e:?}"))?;
    let vocab: u32 = 248320;

    let mut s = 0xCABBA6Eu64;
    let mut next_f32 = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as i32 as f32) / (1i32 << 30) as f32 * 10.0
    };
    let logits_h: Vec<u16> = (0..vocab).map(|_| f32_to_bf16(next_f32())).collect();
    let logits_d = stream
        .memcpy_stod(&logits_h)
        .map_err(|e| anyhow!("htod: {e:?}"))?;

    // Use a realistic-sized allowed set: ~50K tokens (free-string region size)
    let allowed: Vec<u32> = (0..50_000u32)
        .map(|i| (i * 5) % vocab)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mask_bytes = allowed_to_bitmask(&allowed, vocab);
    let mask_d = stream
        .memcpy_stod(&mask_bytes)
        .map_err(|e| anyhow!("htod: {e:?}"))?;
    let mut tokens_d = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| anyhow!("alloc: {e:?}"))?;

    let n_iter = 100;
    // --- DEVICE benchmark ---
    // Warmup
    for _ in 0..10 {
        device_masked_argmax(&ctx, &stream, &logits_d, &mask_d, &mut tokens_d, 1, vocab)?;
    }
    stream.synchronize().map_err(|e| anyhow!("sync: {e:?}"))?;
    let t = std::time::Instant::now();
    for _ in 0..n_iter {
        device_masked_argmax(&ctx, &stream, &logits_d, &mask_d, &mut tokens_d, 1, vocab)?;
    }
    stream.synchronize().map_err(|e| anyhow!("sync: {e:?}"))?;
    let device_us = t.elapsed().as_micros() / n_iter as u128;

    // --- HOST benchmark: includes the dtoh roundtrip + scan, as the
    //     production caller would actually pay ---
    for _ in 0..10 {
        let logits_h2: Vec<u16> = stream
            .memcpy_dtov(&logits_d)
            .map_err(|e| anyhow!("dtoh: {e:?}"))?;
        let _ = host_masked_argmax(&logits_h2, &allowed);
    }
    let t = std::time::Instant::now();
    for _ in 0..n_iter {
        let logits_h2: Vec<u16> = stream
            .memcpy_dtov(&logits_d)
            .map_err(|e| anyhow!("dtoh: {e:?}"))?;
        let _ = host_masked_argmax(&logits_h2, &allowed);
    }
    let host_us = t.elapsed().as_micros() / n_iter as u128;

    // --- DEVICE V2 benchmark (multi-block) ---
    let n_blocks = 8u32;
    let mut scratch_d = stream
        .alloc_zeros::<u32>((n_blocks * 2) as usize)
        .map_err(|e| anyhow!("alloc scratch: {e:?}"))?;
    let mut tokens_v2_d = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| anyhow!("alloc: {e:?}"))?;
    for _ in 0..10 {
        masked_argmax_bf16_v2(
            &ctx,
            &stream,
            &logits_d,
            &mask_d,
            &mut scratch_d,
            &mut tokens_v2_d,
            1,
            vocab,
            n_blocks,
        )?;
    }
    stream.synchronize().map_err(|e| anyhow!("sync: {e:?}"))?;
    let t = std::time::Instant::now();
    for _ in 0..n_iter {
        masked_argmax_bf16_v2(
            &ctx,
            &stream,
            &logits_d,
            &mask_d,
            &mut scratch_d,
            &mut tokens_v2_d,
            1,
            vocab,
            n_blocks,
        )?;
    }
    stream.synchronize().map_err(|e| anyhow!("sync: {e:?}"))?;
    let v2_us = t.elapsed().as_micros() / n_iter as u128;

    println!("===== masked_argmax micro-benchmark =====");
    println!(
        "  vocab={}, allowed={} ({:.1}% of vocab)",
        vocab,
        allowed.len(),
        allowed.len() as f64 * 100.0 / vocab as f64
    );
    println!("  HOST    (dtoh + scan)              : {} μs/call", host_us);
    println!(
        "  DEVICE  v1 (1 block × 128 threads) : {} μs/call",
        device_us
    );
    println!(
        "  DEVICE  v2 ({} blocks × 128 + stage2): {} μs/call",
        n_blocks, v2_us
    );
    if v2_us > 0 {
        println!(
            "  v2 vs v1 speedup: {:.2}×",
            device_us as f64 / v2_us as f64
        );
        println!("  v2 vs host:       {:.2}×", host_us as f64 / v2_us as f64);
    }

    // Correctness cross-check on the same logits/mask: v2 should produce a
    // token id whose logit value equals v1's result (tie-tolerant).
    let v1_pick = stream
        .memcpy_dtov(&tokens_d)
        .map_err(|e| anyhow!("dtoh: {e:?}"))?[0];
    let v2_pick = stream
        .memcpy_dtov(&tokens_v2_d)
        .map_err(|e| anyhow!("dtoh: {e:?}"))?[0];
    let v1_val = f32::from_bits((logits_h[v1_pick as usize] as u32) << 16);
    let v2_val = f32::from_bits((logits_h[v2_pick as usize] as u32) << 16);
    println!(
        "  v1 picked tok={} (val={:.4}), v2 picked tok={} (val={:.4})",
        v1_pick, v1_val, v2_pick, v2_val
    );
    assert_eq!(v1_val, v2_val, "v1 and v2 picked different-valued tokens");
    Ok(())
}
