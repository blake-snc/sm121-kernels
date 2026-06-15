//! Numerical validation of the FP8-KV chunked-prefill flash
//! attention kernel `flash_attn_bf16_fp8kv_v3_d256_gqa_causal_pos_dev`.
//!
//! Oracle = the production BF16 d256 kernel run on the DEQUANTIZED (fp8 * scale)
//! K/V. Both kernels then attend the SAME effective K/V, so their outputs must
//! match within FP8 rounding-order tolerance — this isolates kernel-arithmetic
//! correctness from the (expected) FP8 quantization error.

use anyhow::{anyhow, Result};
use sm121_kernels::attention::{
    flash_attn_bf16_fp8kv_v3_d256_gqa_causal_pos_dev, flash_attn_bf16_v3_d256_gqa_causal_pos_dev,
};
use sm121_kernels::device;
use sm121_kernels::quantization::{dequant_fp8_bf16_pertensor, quant_bf16_to_fp8_pertensor};

const D: usize = 256;

fn f32_to_bf16(x: f32) -> u16 {
    // round-to-nearest-even bf16
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(round)) >> 16) as u16
}
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map high bits to [-1, 1)
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn run_case(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    label: &str,
    nq: u32,
    nkv: u32,
    sq: u32,
    skv: u32,
) -> Result<(f32, f32)> {
    let batch = 1u32;
    let qn = (batch * nq * sq) as usize * D;
    let kvn = (batch * nkv * skv) as usize * D;
    let mut rng = Lcg(0x1234_5678_9abc_def0 ^ (sq as u64) << 8 ^ (nq as u64));

    // random Q / K / V in [-1, 1), bf16
    let q_h: Vec<u16> = (0..qn).map(|_| f32_to_bf16(rng.next_f32())).collect();
    let k_h: Vec<u16> = (0..kvn).map(|_| f32_to_bf16(rng.next_f32())).collect();
    let v_h: Vec<u16> = (0..kvn).map(|_| f32_to_bf16(rng.next_f32())).collect();

    let q_d = stream
        .memcpy_stod(&q_h)
        .map_err(|e| anyhow!("htod q: {e:?}"))?;
    let k_d = stream
        .memcpy_stod(&k_h)
        .map_err(|e| anyhow!("htod k: {e:?}"))?;
    let v_d = stream
        .memcpy_stod(&v_h)
        .map_err(|e| anyhow!("htod v: {e:?}"))?;

    // per-tensor scale over K & V (e4m3 max ~= 448)
    let maxabs = k_h
        .iter()
        .chain(v_h.iter())
        .map(|&b| bf16_to_f32(b).abs())
        .fold(0f32, f32::max)
        .max(1e-6);
    let kv_scale = maxabs / 448.0;

    // quantize K,V -> fp8 e4m3 (q = round(x / scale))
    let mut k_fp8 = stream
        .alloc_zeros::<u8>(kvn)
        .map_err(|e| anyhow!("alloc kfp8: {e:?}"))?;
    let mut v_fp8 = stream
        .alloc_zeros::<u8>(kvn)
        .map_err(|e| anyhow!("alloc vfp8: {e:?}"))?;
    quant_bf16_to_fp8_pertensor(ctx, stream, &k_d, &mut k_fp8, kvn as u32, kv_scale)?;
    quant_bf16_to_fp8_pertensor(ctx, stream, &v_d, &mut v_fp8, kvn as u32, kv_scale)?;

    // dequantize back to bf16 (oracle inputs: out = scale * f32(fp8))
    let mut k_deq = stream
        .alloc_zeros::<u16>(kvn)
        .map_err(|e| anyhow!("alloc kdeq: {e:?}"))?;
    let mut v_deq = stream
        .alloc_zeros::<u16>(kvn)
        .map_err(|e| anyhow!("alloc vdeq: {e:?}"))?;
    dequant_fp8_bf16_pertensor(ctx, stream, &k_fp8, &mut k_deq, kvn as u32, kv_scale)?;
    dequant_fp8_bf16_pertensor(ctx, stream, &v_fp8, &mut v_deq, kvn as u32, kv_scale)?;

    let pos = stream
        .memcpy_stod(&vec![0u32])
        .map_err(|e| anyhow!("htod pos: {e:?}"))?;
    let scale = 1.0 / (D as f32).sqrt();

    // oracle: BF16 kernel on dequantized K/V
    let mut o_ref = stream
        .alloc_zeros::<u16>(qn)
        .map_err(|e| anyhow!("alloc oref: {e:?}"))?;
    flash_attn_bf16_v3_d256_gqa_causal_pos_dev(
        ctx, stream, &q_d, &k_deq, &v_deq, &mut o_ref, &pos, batch, nq, nkv, sq, skv, scale,
    )
    .map_err(|e| anyhow!("bf16 ref: {e:?}"))?;

    // test: FP8-KV kernel on fp8 K/V
    let mut o_test = stream
        .alloc_zeros::<u16>(qn)
        .map_err(|e| anyhow!("alloc otest: {e:?}"))?;
    flash_attn_bf16_fp8kv_v3_d256_gqa_causal_pos_dev(
        ctx,
        stream,
        &q_d,
        &k_fp8,
        &v_fp8,
        &mut o_test,
        &pos,
        batch,
        nq,
        nkv,
        sq,
        skv,
        scale,
        kv_scale,
    )
    .map_err(|e| anyhow!("fp8kv: {e:?}"))?;

    stream.synchronize().map_err(|e| anyhow!("sync: {e:?}"))?;
    let or: Vec<u16> = stream
        .memcpy_dtov(&o_ref)
        .map_err(|e| anyhow!("dtoh ref: {e:?}"))?;
    let ot: Vec<u16> = stream
        .memcpy_dtov(&o_test)
        .map_err(|e| anyhow!("dtoh test: {e:?}"))?;

    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    for (a, b) in or.iter().zip(ot.iter()) {
        let d = (bf16_to_f32(*a) - bf16_to_f32(*b)).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    let mean_abs = (sum_abs / or.len() as f64) as f32;
    println!(
        "  [{label}] nq={nq} nkv={nkv} sq={sq} skv={skv} kv_scale={kv_scale:.5}  max|Δ|={max_abs:.5}  mean|Δ|={mean_abs:.5}"
    );
    Ok((max_abs, mean_abs))
}

#[test]
fn fp8kv_prefill_matches_bf16_on_dequant() -> Result<()> {
    let ctx = device::init_device(0).map_err(|e| anyhow!("init: {e:?}"))?;
    let stream = ctx.new_stream().map_err(|e| anyhow!("stream: {e:?}"))?;
    println!("\n=== FP8-KV prefill FA kernel vs BF16-on-dequant oracle ===");

    // (non-GQA square), (GQA 2:1), (prefill-chunk shape)
    let cases: &[(&str, u32, u32, u32, u32)] = &[
        ("non-gqa", 2, 2, 64, 64),
        ("gqa-2to1", 4, 2, 64, 128),
        ("chunk-128x256", 2, 2, 128, 256),
    ];
    // FP8 tolerance: the two kernels attend identical FP8 K/V; only rounding order
    // differs, so max|Δ| should be well under 0.1 (e4m3 has ~3 mantissa bits).
    const TOL: f32 = 0.10;
    let mut worst = 0f32;
    for (label, nq, nkv, sq, skv) in cases {
        let (mx, _) = run_case(&ctx, &stream, label, *nq, *nkv, *sq, *skv)?;
        worst = worst.max(mx);
    }
    println!("  worst max|Δ| = {worst:.5}  (tol {TOL})");
    assert!(
        worst < TOL,
        "FP8-KV prefill kernel diverges from BF16-on-dequant oracle: max|Δ|={worst:.5} >= {TOL}"
    );
    println!("✓ GATE 1 PASS: FP8-KV prefill kernel numerically matches the oracle");
    Ok(())
}
