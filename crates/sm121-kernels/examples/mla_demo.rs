//! DeepSeek V3 MLA decode/prefill demo.
//!
//! Demonstrates weight-absorbed Multi-head Latent Attention for DeepSeek V3-class
//! models at realistic dimensions: D_C=512 (compressed latent), D_R=64 (RoPE).
//!
//! Shows: BF16 MLA decode, BF16 MLA prefill (causal), FP8 KV MLA decode,
//! and paged MLA decode (vLLM/SGLang style).
//!
//! Run: cargo run --release --example mla_demo

use sm121_kernels::{attention, device};

fn main() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    println!("================================================================");
    println!("  DeepSeek V3 MLA Demo");
    println!("  Platform: DGX Spark (SM121a)");
    println!("  Dims: D_C=512 (compressed), D_R=64 (RoPE)");
    println!("================================================================");
    println!();

    let batch = 2u32;
    let num_heads = 64u32;
    let seq_kv = 1024u32;
    let scale = 1.0f32 / ((attention::MLA_D_C + attention::MLA_D_R) as f32).sqrt();

    // ===== BF16 MLA decode =====
    let q_c_len = (batch * num_heads * attention::MLA_D_C) as usize;
    let q_r_len = (batch * num_heads * attention::MLA_D_R) as usize;
    let ckv_len = (batch * seq_kv * attention::MLA_D_C) as usize;
    let krope_len = (batch * seq_kv * attention::MLA_D_R) as usize;

    let q_c = stream.alloc_zeros::<u16>(q_c_len).unwrap();
    let q_r = stream.alloc_zeros::<u16>(q_r_len).unwrap();
    let c_kv = stream.alloc_zeros::<u16>(ckv_len).unwrap();
    let k_rope = stream.alloc_zeros::<u16>(krope_len).unwrap();
    let mut o = stream.alloc_zeros::<u16>(q_c_len).unwrap();

    attention::mla_decode_bf16(
        &ctx, &stream, &q_c, &q_r, &c_kv, &k_rope, &mut o, batch, num_heads, seq_kv, scale,
    )
    .unwrap();
    println!("✓ MLA BF16 decode: B={batch}, H={num_heads}, S={seq_kv}");

    // ===== BF16 MLA prefill (causal) =====
    let seq_q = 32u32;
    let q_c_p_len = (batch * seq_q * num_heads * attention::MLA_D_C) as usize;
    let q_r_p_len = (batch * seq_q * num_heads * attention::MLA_D_R) as usize;
    let ckv_p_len = (batch * seq_q * attention::MLA_D_C) as usize; // matching prefill
    let krope_p_len = (batch * seq_q * attention::MLA_D_R) as usize;

    let q_c_p = stream.alloc_zeros::<u16>(q_c_p_len).unwrap();
    let q_r_p = stream.alloc_zeros::<u16>(q_r_p_len).unwrap();
    let c_kv_p = stream.alloc_zeros::<u16>(ckv_p_len).unwrap();
    let k_rope_p = stream.alloc_zeros::<u16>(krope_p_len).unwrap();
    let mut o_p = stream.alloc_zeros::<u16>(q_c_p_len).unwrap();

    attention::mla_prefill_bf16(
        &ctx, &stream, &q_c_p, &q_r_p, &c_kv_p, &k_rope_p, &mut o_p, batch, num_heads, seq_q,
        seq_q, scale,
    )
    .unwrap();
    println!("✓ MLA BF16 prefill: B={batch}, H={num_heads}, Sq=Skv={seq_q} (causal)");

    // ===== FP8 KV MLA decode =====
    let c_kv_fp8 = stream.alloc_zeros::<u8>(ckv_len).unwrap();
    let k_rope_fp8 = stream.alloc_zeros::<u8>(krope_len).unwrap();
    let mut o_fp8 = stream.alloc_zeros::<u16>(q_c_len).unwrap();

    attention::mla_decode_fp8(
        &ctx,
        &stream,
        &q_c,
        &q_r,
        &c_kv_fp8,
        &k_rope_fp8,
        &mut o_fp8,
        batch,
        num_heads,
        seq_kv,
        scale,
        1.0,
    )
    .unwrap();
    println!("✓ MLA FP8 KV decode: B={batch}, H={num_heads}, S={seq_kv} (~50% KV memory)");

    // ===== Paged MLA decode (vLLM/SGLang style) =====
    let page_size = 16u32;
    let pages_per_batch = seq_kv.div_ceil(page_size);
    let total_pages = batch * pages_per_batch;
    let c_kv_pool = stream
        .alloc_zeros::<u16>((total_pages * page_size * attention::MLA_D_C) as usize)
        .unwrap();
    let k_rope_pool = stream
        .alloc_zeros::<u16>((total_pages * page_size * attention::MLA_D_R) as usize)
        .unwrap();
    let page_table: Vec<u32> = (0..(batch * pages_per_batch)).collect();
    let seq_lens: Vec<u32> = vec![seq_kv; batch as usize];
    let page_table_dev = stream.memcpy_stod(&page_table).unwrap();
    let seq_lens_dev = stream.memcpy_stod(&seq_lens).unwrap();
    let mut o_paged = stream.alloc_zeros::<u16>(q_c_len).unwrap();

    attention::mla_decode_bf16_paged(
        &ctx,
        &stream,
        &q_c,
        &q_r,
        &c_kv_pool,
        &k_rope_pool,
        &page_table_dev,
        &seq_lens_dev,
        &mut o_paged,
        batch,
        num_heads,
        pages_per_batch,
        page_size,
        scale,
    )
    .unwrap();
    println!("✓ MLA paged decode: B={batch}, H={num_heads}, S={seq_kv} (page_size={page_size})");

    println!();
    println!("All MLA variants executed successfully.");
}
