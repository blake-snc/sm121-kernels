use sm121_kernels::{attention, device};

fn main() {
    let ctx = device::init_device(0).expect("failed to init SM121 device");
    let stream = ctx.default_stream();

    let batch: u32 = 1;
    let num_heads: u32 = 4;
    let seq: u32 = 256;
    let head_dim: u32 = 128;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let total = (batch * num_heads * seq * head_dim) as usize;

    // Zero-filled BF16 tensors
    let q_host = vec![0u16; total];
    let k_host = vec![0u16; total];
    let v_host = vec![0u16; total];

    let q_dev = stream.memcpy_stod(&q_host).unwrap();
    let k_dev = stream.memcpy_stod(&k_host).unwrap();
    let v_dev = stream.memcpy_stod(&v_host).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(total).unwrap();

    // V3 non-causal
    attention::flash_attn_bf16_v3_d128(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq, seq, scale,
    )
    .expect("V3 non-causal FA failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    println!(
        "V3 non-causal: batch={}, heads={}, seq={}, d={} -> output[0] = {:#06x}",
        batch, num_heads, seq, head_dim, o_host[0]
    );

    // V3 causal
    attention::flash_attn_bf16_v3_d128_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq, seq, scale,
    )
    .expect("V3 causal FA failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    println!(
        "V3 causal:     batch={}, heads={}, seq={}, d={} -> output[0] = {:#06x}",
        batch, num_heads, seq, head_dim, o_host[0]
    );
}
