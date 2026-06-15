//! Honest benchmark of the new reference kernels landed this session.
//! These are CORRECTNESS-FIRST scalar-style references — expect much lower
//! TFLOPS than the MMA-optimized V21 / V12c kernels. This benchmark exists
//! to document the starting point so the MMA-optimized variants (future
//! work) can be measured against it.

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaStream};

use sm121_kernels::{attention, device, linear_attention};

fn time_us(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    iters: usize,
    mut run: impl FnMut(),
) -> f64 {
    for _ in 0..5 {
        run();
    }
    stream.synchronize().unwrap();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        let e = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        s.record(stream).unwrap();
        run();
        e.record(stream).unwrap();
        times.push(s.elapsed_ms(&e).unwrap() as f64 * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn main() {
    let ctx = device::init_device(0).expect("SM121 init");
    let stream = ctx.default_stream();

    println!("================================================================");
    println!("  New reference kernels — honest baseline (pre-MMA-optimization)");
    println!("  CUDA events, 5 warmup + 50 measured, median reported");
    println!("================================================================");
    println!();

    // MLA BF16 decode at DeepSeek V3 dims (active = 37B: H=128, D_C=512, D_R=64)
    {
        let b = 2u32;
        let h = 128u32;
        let skv = 1024u32;
        let qc_len = (b * h * attention::MLA_D_C) as usize;
        let qr_len = (b * h * attention::MLA_D_R) as usize;
        let kv_len = (b * skv * attention::MLA_D_C) as usize;
        let kr_len = (b * skv * attention::MLA_D_R) as usize;
        let qc = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let qr = stream.alloc_zeros::<u16>(qr_len).unwrap();
        let ckv = stream.alloc_zeros::<u16>(kv_len).unwrap();
        let kr = stream.alloc_zeros::<u16>(kr_len).unwrap();
        let mut o = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let scale = 1.0f32 / ((attention::MLA_D_C + attention::MLA_D_R) as f32).sqrt();

        let us = time_us(&ctx, &stream, 50, || {
            attention::mla_decode_bf16(
                &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o, b, h, skv, scale,
            )
            .unwrap();
        });
        // Rough FLOPS: per (b,h): 2 * skv * (D_C + D_R) for scores, + 2*skv*D_C for output
        let flops = b as f64
            * h as f64
            * skv as f64
            * (2.0 * (attention::MLA_D_C + attention::MLA_D_R) as f64
                + 2.0 * attention::MLA_D_C as f64);
        let tflops = flops / (us * 1e-6) / 1e12;
        println!("MLA BF16 decode   B={b} H={h} Skv={skv}  median={us:.1} us  {tflops:.2} TFLOPS  (scalar ref)");
    }

    // MLA BF16 decode at smaller dims
    {
        let b = 1u32;
        let h = 16u32;
        let skv = 512u32;
        let qc_len = (b * h * attention::MLA_D_C) as usize;
        let qr_len = (b * h * attention::MLA_D_R) as usize;
        let kv_len = (b * skv * attention::MLA_D_C) as usize;
        let kr_len = (b * skv * attention::MLA_D_R) as usize;
        let qc = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let qr = stream.alloc_zeros::<u16>(qr_len).unwrap();
        let ckv = stream.alloc_zeros::<u16>(kv_len).unwrap();
        let kr = stream.alloc_zeros::<u16>(kr_len).unwrap();
        let mut o = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let scale = 1.0f32 / ((attention::MLA_D_C + attention::MLA_D_R) as f32).sqrt();

        let us = time_us(&ctx, &stream, 50, || {
            attention::mla_decode_bf16(
                &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o, b, h, skv, scale,
            )
            .unwrap();
        });
        println!("MLA BF16 decode   B={b} H={h} Skv={skv}   median={us:.1} us  (scalar ref)");
    }

    // GDN decode at Qwen3-Next dims (heads ~16, D=128)
    {
        let b = 2u32;
        let h = 16u32;
        let qkv = (b * h * linear_attention::GDN_HEAD_DIM) as usize;
        let sc = (b * h) as usize;
        let state =
            (b * h * linear_attention::GDN_HEAD_DIM * linear_attention::GDN_HEAD_DIM) as usize;
        let q = stream.alloc_zeros::<u16>(qkv).unwrap();
        let k = stream.alloc_zeros::<u16>(qkv).unwrap();
        let v = stream.alloc_zeros::<u16>(qkv).unwrap();
        let alpha = stream.alloc_zeros::<f32>(sc).unwrap();
        let beta = stream.alloc_zeros::<f32>(sc).unwrap();
        let mut st = stream.alloc_zeros::<u16>(state).unwrap();
        let mut y = stream.alloc_zeros::<u16>(qkv).unwrap();

        let us = time_us(&ctx, &stream, 50, || {
            linear_attention::gdn_decode(
                &ctx, &stream, &q, &k, &v, &alpha, &beta, &mut st, &mut y, b, h,
            )
            .unwrap();
        });
        // GDN decode: 3 * D^2 ops per (b,h) roughly (temp, update, output)
        let flops = b as f64 * h as f64 * 3.0 * (linear_attention::GDN_HEAD_DIM as f64).powi(2);
        let tflops = flops / (us * 1e-6) / 1e12;
        println!("GDN decode        B={b} H={h}            median={us:.1} us  {tflops:.3} TFLOPS  (scalar ref)");
    }

    // Mamba2 decode
    {
        let b = 2u32;
        let h = 16u32;
        let sc = (b * h) as usize;
        let state = (b * h * linear_attention::MAMBA2_D_STATE) as usize;
        let x = stream.alloc_zeros::<f32>(sc).unwrap();
        let dd = stream.alloc_zeros::<f32>(sc).unwrap();
        let a = stream.alloc_zeros::<f32>(state).unwrap();
        let bb = stream.alloc_zeros::<f32>(state).unwrap();
        let cc = stream.alloc_zeros::<f32>(state).unwrap();
        let mut st = stream.alloc_zeros::<f32>(state).unwrap();
        let mut y = stream.alloc_zeros::<f32>(sc).unwrap();

        let us = time_us(&ctx, &stream, 50, || {
            linear_attention::mamba2_selective_scan_decode(
                &ctx, &stream, &x, &dd, &a, &bb, &cc, &mut st, &mut y, b, h,
            )
            .unwrap();
        });
        println!(
            "Mamba2 decode     B={b} H={h} D={}         median={us:.1} us",
            linear_attention::MAMBA2_D_STATE
        );
    }

    // Tree attention at Medusa-typical dims
    {
        let b = 1u32;
        let h = 16u32;
        let sq = 64u32; // 64 draft candidates
        let skv = 1024u32; // 1024 context
        let d = 128u32;
        let q = stream
            .alloc_zeros::<u16>((b * sq * h * d) as usize)
            .unwrap();
        let k = stream
            .alloc_zeros::<u16>((b * skv * h * d) as usize)
            .unwrap();
        let v = stream
            .alloc_zeros::<u16>((b * skv * h * d) as usize)
            .unwrap();
        let mask = stream.alloc_zeros::<u8>((sq * skv) as usize).unwrap();
        let mut o = stream
            .alloc_zeros::<u16>((b * sq * h * d) as usize)
            .unwrap();
        let scale = 1.0f32 / (d as f32).sqrt();

        let us = time_us(&ctx, &stream, 50, || {
            attention::tree_attention_bf16(
                &ctx, &stream, &q, &k, &v, &mask, &mut o, b, h, sq, skv, scale,
            )
            .unwrap();
        });
        let flops = 2.0 * 2.0 * b as f64 * h as f64 * sq as f64 * skv as f64 * d as f64;
        let tflops = flops / (us * 1e-6) / 1e12;
        println!("Tree attn         B={b} H={h} Sq={sq} Skv={skv}  median={us:.1} us  {tflops:.2} TFLOPS (dense)");
    }

    // NSA at sparse config (6 blocks × 64 = 384 selected of 2048 total)
    {
        let b = 1u32;
        let h = 16u32;
        let sq = 64u32;
        let skv = 2048u32;
        let d = 128u32;
        let block_size = 64u32;
        let k_top = 6u32;
        let num_blocks = skv / block_size;

        let q = stream
            .alloc_zeros::<u16>((b * sq * h * d) as usize)
            .unwrap();
        let k = stream
            .alloc_zeros::<u16>((b * skv * h * d) as usize)
            .unwrap();
        let v = stream
            .alloc_zeros::<u16>((b * skv * h * d) as usize)
            .unwrap();
        let idx: Vec<u32> = (0..(b * sq * h * k_top)).map(|i| i % num_blocks).collect();
        let idx_dev = stream.memcpy_stod(&idx).unwrap();
        let mut o = stream
            .alloc_zeros::<u16>((b * sq * h * d) as usize)
            .unwrap();
        let scale = 1.0f32 / (d as f32).sqrt();

        let us = time_us(&ctx, &stream, 50, || {
            attention::nsa_attention_bf16(
                &ctx, &stream, &q, &k, &v, &idx_dev, &mut o, b, h, sq, skv, k_top, block_size,
                scale,
            )
            .unwrap();
        });
        let effective_skv = k_top * block_size;
        let flops = 2.0 * 2.0 * b as f64 * h as f64 * sq as f64 * effective_skv as f64 * d as f64;
        let tflops = flops / (us * 1e-6) / 1e12;
        let sparsity = 100.0 * effective_skv as f64 / skv as f64;
        println!("NSA sparse        B={b} H={h} Sq={sq} Skv={skv} (eff {effective_skv}={sparsity:.0}%)  median={us:.1} us  {tflops:.2} TFLOPS");
    }

    println!();
    println!("These numbers are a STARTING POINT. MMA-optimized variants are");
    println!("future work and should bring MLA → tens of TFLOPS, matching V21.");
}
