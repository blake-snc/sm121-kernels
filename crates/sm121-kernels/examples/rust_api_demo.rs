//! sm121-kernels Rust API demo
//!
//! Demonstrates the complete kernel library API: flash attention (BF16/FP8),
//! GEMM, RMSNorm, RoPE, SiLU, top-k sampling, and MoE routing.
//!
//! Run: cargo run --release --example rust_api_demo

use sm121_kernels::{activation, attention, device, gemm, moe, norm, rope, sampling};

fn main() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    println!("sm121-kernels Rust API Demo");
    println!("==========================");
    println!("Device: SM121a (DGX Spark)");
    println!();

    // --- Flash Attention (BF16) ---
    let (batch, num_heads, seq_q, seq_kv, d) = (1u32, 8u32, 256u32, 256u32, 128u32);
    let scale = 1.0f32 / (d as f32).sqrt();
    let total_q = (batch * num_heads * seq_q * d) as usize;
    let total_kv = (batch * num_heads * seq_kv * d) as usize;

    let q = stream.alloc_zeros::<u16>(total_q).unwrap();
    let k = stream.alloc_zeros::<u16>(total_kv).unwrap();
    let v = stream.alloc_zeros::<u16>(total_kv).unwrap();
    let mut o = stream.alloc_zeros::<u16>(total_q).unwrap();

    // Non-causal V3
    attention::flash_attn_bf16_v3_d128(
        &ctx, &stream, &q, &k, &v, &mut o, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();
    println!("✓ BF16 Flash Attention V3 (non-causal): B={batch}, H={num_heads}, S={seq_q}");

    // Causal V3
    attention::flash_attn_bf16_v3_d128_causal(
        &ctx, &stream, &q, &k, &v, &mut o, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();
    println!("✓ BF16 Flash Attention V3 (causal)");

    // TMA V11 (experimental, superseded by V21)
    #[cfg(feature = "experimental")]
    {
        attention::flash_attn_bf16_v11_fused_scale(
            &ctx, &stream, &q, &k, &v, &mut o, batch, num_heads, seq_q, seq_kv, scale,
        )
        .unwrap();
        println!("✓ BF16 Flash Attention V11 TMA (warp-specialized)");
    }

    // --- Flash Attention (FP8) ---
    let q8 = stream.alloc_zeros::<u8>(total_q).unwrap();
    let k8 = stream.alloc_zeros::<u8>(total_kv).unwrap();
    let v8 = stream.alloc_zeros::<u8>(total_kv).unwrap();

    attention::flash_attn_fp8_v12c_vt(
        &ctx, &stream, &q8, &k8, &v8, &mut o, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();
    println!("✓ FP8 Flash Attention V12c VT-GMEM (100 TFLOPS kernel)");

    // --- Paged KV Cache ---
    let page_size = 64u32;
    let num_pages = batch * (seq_kv / page_size);
    let max_pages = seq_kv / page_size;
    let paged_kv_size = (num_pages * page_size * num_heads * d) as usize;

    let k_paged = stream.alloc_zeros::<u16>(paged_kv_size).unwrap();
    let v_paged = stream.alloc_zeros::<u16>(paged_kv_size).unwrap();
    let page_table = stream
        .memcpy_stod(&vec![0u32; (batch * max_pages) as usize])
        .unwrap();

    attention::flash_attn_bf16_v3_paged_kv(
        &ctx,
        &stream,
        &q,
        &k_paged,
        &v_paged,
        &mut o,
        &page_table,
        batch,
        num_heads,
        num_heads,
        seq_q,
        seq_kv,
        page_size,
        max_pages,
        scale,
    )
    .unwrap();
    println!("✓ BF16 Paged KV Attention (page_size={page_size})");

    // --- GEMM ---
    let (m, n, gk) = (512u32, 512u32, 512u32);
    let mat_size = (m * gk) as usize;
    let a = stream.alloc_zeros::<u16>(mat_size).unwrap();
    let b = stream.alloc_zeros::<u16>(mat_size).unwrap();
    let mut c = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_bf16_mma(&ctx, &stream, &a, &b, &mut c, m, n, gk).unwrap();
    println!("✓ BF16 GEMM (MMA m16n8k16): {m}x{n}x{gk}");

    // --- RMSNorm ---
    let (hidden, rows) = (4096u32, 128u32);
    let x = stream.alloc_zeros::<u16>((rows * hidden) as usize).unwrap();
    let w = stream.alloc_zeros::<u16>(hidden as usize).unwrap();
    let mut y = stream.alloc_zeros::<u16>((rows * hidden) as usize).unwrap();

    norm::rmsnorm_bf16(&ctx, &stream, &x, &mut y, &w, hidden, 1e-5f32, rows).unwrap();
    println!("✓ RMSNorm BF16: {rows}x{hidden}");

    // --- RoPE ---
    let (rope_b, rope_s, rope_h, rope_d2) = (1u32, 128u32, 32u32, 64u32);
    let rope_size = (rope_b * rope_s * rope_h * rope_d2 * 2) as usize;
    let mut rope_qk = stream.alloc_zeros::<u16>(rope_size).unwrap();
    let cos = stream
        .alloc_zeros::<f32>(((rope_b * rope_s) * rope_d2) as usize)
        .unwrap();
    let sin = stream
        .alloc_zeros::<f32>(((rope_b * rope_s) * rope_d2) as usize)
        .unwrap();

    rope::rope_bf16(
        &ctx,
        &stream,
        &mut rope_qk,
        &cos,
        &sin,
        rope_b,
        rope_s,
        rope_h,
        rope_d2,
    )
    .unwrap();
    println!("✓ RoPE BF16: B={rope_b}, S={rope_s}, H={rope_h}, D/2={rope_d2}");

    // --- Fused Activations ---
    let act_n = 1024u32;
    let act_d = 2048u32;
    let gate_up = stream
        .alloc_zeros::<u16>((act_n * act_d * 2) as usize)
        .unwrap();
    let mut act_out = stream.alloc_zeros::<u16>((act_n * act_d) as usize).unwrap();

    activation::silu_mul_bf16(&ctx, &stream, &gate_up, &mut act_out, act_n, act_d).unwrap();
    println!("✓ Fused SiLU×Mul BF16: {act_n}x{act_d}");

    // --- Top-k Sampling ---
    let vocab = 32000u32;
    let logits = stream.alloc_zeros::<u16>(vocab as usize).unwrap();
    let mut token = stream.alloc_zeros::<u32>(1).unwrap();

    let mut token_val = stream.alloc_zeros::<u16>(1).unwrap();
    sampling::topk_sampling(
        &ctx,
        &stream,
        &logits,
        &mut token,
        &mut token_val,
        1,
        vocab,
        1,
        1.0,
    )
    .unwrap();
    println!("✓ Top-k Sampling: vocab={vocab}, k=1");

    // --- MoE Routing ---
    let (moe_tokens, num_experts, top_k_experts) = (128u32, 8u32, 2u32);
    let router_logits = stream
        .alloc_zeros::<u16>((moe_tokens * num_experts) as usize)
        .unwrap();
    let mut expert_ids = stream
        .alloc_zeros::<u32>((moe_tokens * top_k_experts) as usize)
        .unwrap();
    let mut expert_weights = stream
        .alloc_zeros::<u16>((moe_tokens * top_k_experts) as usize)
        .unwrap();

    moe::moe_routing(
        &ctx,
        &stream,
        &router_logits,
        &mut expert_ids,
        &mut expert_weights,
        moe_tokens,
        num_experts,
        top_k_experts,
    )
    .unwrap();
    println!("✓ MoE Routing: {moe_tokens} tokens, {num_experts} experts, top-{top_k_experts}");

    println!();
    println!("All 11 kernel types launched successfully.");
    println!("Zero Python. Zero CUDA toolkit. Just libcuda.so.");
}
