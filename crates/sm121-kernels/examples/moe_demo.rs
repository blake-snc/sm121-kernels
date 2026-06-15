//! MoE (Mixture of Experts) routing demo.
//!
//! Demonstrates the top-k expert selection and softmax gating kernels
//! at Mixtral-8x7B dimensions.
//!
//! Run: cargo run --release --example moe_demo

use sm121_kernels::{activation, device, gemm, moe};

fn main() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    println!("================================================================");
    println!("  MoE Routing Demo (Mixtral-8x7B dimensions)");
    println!("  Platform: DGX Spark (SM121a)");
    println!("================================================================");
    println!();

    // Mixtral-8x7B config
    let num_tokens = 512u32; // batch of tokens
    let num_experts = 8u32; // 8 experts
    let top_k = 2u32; // top-2 routing
    let hidden_dim = 4096u32;
    let expert_dim = 14336u32; // intermediate dim per expert

    println!("Config: {num_tokens} tokens, {num_experts} experts, top-{top_k}");
    println!("Expert dims: {hidden_dim} → {expert_dim} → {hidden_dim}");
    println!();

    // Router logits: [num_tokens, num_experts]
    let router_logits = stream
        .alloc_zeros::<u16>((num_tokens * num_experts) as usize)
        .unwrap();
    let mut expert_ids = stream
        .alloc_zeros::<u32>((num_tokens * top_k) as usize)
        .unwrap();
    let mut expert_weights = stream
        .alloc_zeros::<u16>((num_tokens * top_k) as usize)
        .unwrap();

    // Run routing
    moe::moe_routing(
        &ctx,
        &stream,
        &router_logits,
        &mut expert_ids,
        &mut expert_weights,
        num_tokens,
        num_experts,
        top_k,
    )
    .unwrap();
    println!("✓ MoE routing: {num_tokens} tokens → top-{top_k} experts");

    // Simulate expert FFN (gate_up + SiLU + down per expert)
    // In practice, this would dispatch tokens to their assigned experts
    let gate_up = stream
        .alloc_zeros::<u16>((num_tokens * expert_dim * 2) as usize)
        .unwrap();
    let mut silu_out = stream
        .alloc_zeros::<u16>((num_tokens * expert_dim) as usize)
        .unwrap();

    activation::silu_mul_bf16(
        &ctx,
        &stream,
        &gate_up,
        &mut silu_out,
        num_tokens,
        expert_dim,
    )
    .unwrap();
    println!("✓ Fused SiLU×Mul: {num_tokens}×{expert_dim}");

    // Expert GEMM (simplified — real MoE groups tokens by expert)
    let _w_expert = stream
        .alloc_zeros::<u16>((expert_dim * hidden_dim) as usize)
        .unwrap();
    let _expert_out = stream
        .alloc_zeros::<u16>((num_tokens * hidden_dim) as usize)
        .unwrap();

    // Use smaller GEMM for demo (full expert GEMM is 512×4096×14336)
    let demo_m = 128u32;
    let demo_n = 128u32;
    let demo_k = 128u32;
    let a = stream
        .alloc_zeros::<u16>((demo_m * demo_k) as usize)
        .unwrap();
    let b = stream
        .alloc_zeros::<u16>((demo_k * demo_n) as usize)
        .unwrap();
    let mut c = stream
        .alloc_zeros::<u16>((demo_m * demo_n) as usize)
        .unwrap();

    gemm::gemm_bf16_mma(&ctx, &stream, &a, &b, &mut c, demo_m, demo_n, demo_k).unwrap();
    println!("✓ Expert GEMM (BF16 MMA): {demo_m}×{demo_n}×{demo_k}");

    println!();
    println!("MoE pipeline: route → gate_up → SiLU×Mul → down_proj → combine");
    println!("All kernels are hand-written PTX, compiled to SASS at build time.");
}
