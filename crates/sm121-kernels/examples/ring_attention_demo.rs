//! Ring attention demo — show single-GPU correctness + perf vs monolithic, and
//! sketch the multi-node code path (pseudocode in comments at bottom).
use cudarc::driver::sys::{cuEventCreate, cuEventElapsedTime, cuEventRecord, cuEventSynchronize};
use half::bf16;
use sm121_kernels::{attention, device, distributed};

fn random_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (((s >> 33) as u32 as f32) / u32::MAX as f32 - 0.5) * 0.5;
            bf16::from_f32(f).to_bits()
        })
        .collect()
}

unsafe fn time_fn<F: FnMut()>(s: cudarc::driver::sys::CUstream, mut f: F, iters: usize) -> f32 {
    let mut start = std::ptr::null_mut();
    let mut stop = std::ptr::null_mut();
    unsafe {
        cuEventCreate(&mut start, 0).result().unwrap();
        cuEventCreate(&mut stop, 0).result().unwrap();
        cuEventRecord(start, s).result().unwrap();
    }
    for _ in 0..iters {
        f();
    }
    let mut ms = 0f32;
    unsafe {
        cuEventRecord(stop, s).result().unwrap();
        cuEventSynchronize(stop).result().unwrap();
        cuEventElapsedTime(&mut ms, start, stop).result().unwrap();
    }
    ms / iters as f32
}

fn main() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let raw = stream.cu_stream();

    let configs: &[(u32, u32, u32, u32, &[u32])] = &[
        // (B, H, Sq, Skv, chunk options)
        (1, 8, 128, 1024, &[1, 2, 4, 8]),
        (1, 8, 128, 4096, &[1, 4, 8, 16]),
        (1, 16, 128, 8192, &[1, 8, 16, 32]),
        (2, 16, 128, 16384, &[1, 16, 32]),
    ];

    println!("Ring attention demo (single-GPU, simulates multi-node topology)\n");
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>10}",
        "shape", "mono(us)", "ring(us)", "chunks", "ring/mono"
    );
    println!("{}", "-".repeat(72));

    for &(b, h, sq, skv, chunks) in configs {
        let q = random_bf16((b * h * sq * 128) as usize, 0xA);
        let k = random_bf16((b * h * skv * 128) as usize, 0xB);
        let v = random_bf16((b * h * skv * 128) as usize, 0xC);
        let q_d = stream.memcpy_stod(&q).unwrap();
        let k_d = stream.memcpy_stod(&k).unwrap();
        let v_d = stream.memcpy_stod(&v).unwrap();
        let mut o_mono = stream
            .alloc_zeros::<u16>((b * h * sq * 128) as usize)
            .unwrap();
        let mut o_ring = stream
            .alloc_zeros::<u16>((b * h * sq * 128) as usize)
            .unwrap();
        let scale = 1.0 / (128f32).sqrt();

        // Warmup
        for _ in 0..3 {
            attention::flash_attn_bf16_v3_d128(
                &ctx,
                &stream,
                &q_d,
                &k_d,
                &v_d,
                &mut o_mono,
                b,
                h,
                sq,
                skv,
                scale,
            )
            .unwrap();
        }
        stream.synchronize().unwrap();

        let iters = 50;
        let ms_mono = unsafe {
            time_fn(
                raw,
                || {
                    attention::flash_attn_bf16_v3_d128(
                        &ctx,
                        &stream,
                        &q_d,
                        &k_d,
                        &v_d,
                        &mut o_mono,
                        b,
                        h,
                        sq,
                        skv,
                        scale,
                    )
                    .unwrap();
                },
                iters,
            )
        };

        for &nc in chunks {
            for _ in 0..3 {
                distributed::ring_attention_bf16(
                    &ctx,
                    &stream,
                    &q_d,
                    &k_d,
                    &v_d,
                    &mut o_ring,
                    b,
                    h,
                    sq,
                    skv,
                    scale,
                    nc,
                )
                .unwrap();
            }
            stream.synchronize().unwrap();
            let ms_ring = unsafe {
                time_fn(
                    raw,
                    || {
                        distributed::ring_attention_bf16(
                            &ctx,
                            &stream,
                            &q_d,
                            &k_d,
                            &v_d,
                            &mut o_ring,
                            b,
                            h,
                            sq,
                            skv,
                            scale,
                            nc,
                        )
                        .unwrap();
                    },
                    iters,
                )
            };

            println!(
                "{:<28} {:>10.1} {:>10.1} {:>10} {:>10.2}",
                format!("B={} H={} Sq={} Skv={}", b, h, sq, skv),
                ms_mono * 1000.0,
                ms_ring * 1000.0,
                nc,
                ms_ring / ms_mono
            );
        }
        println!();
    }

    println!("--- Multi-node ring attention (sketch) ---");
    println!("On N DGX Sparks with NCCL Comm of world_size=N:");
    println!("  let local_kv_slice = ...;   // [B, H, Skv/N, D] held by this rank");
    println!("  let mut q_partial_buf = ...;");
    println!("  let mut lse_partial_buf = ...;");
    println!("  let mut send_buf = local_kv_slice.clone();");
    println!("  let mut recv_buf = ...;");
    println!("  for step in 0..N {{");
    println!("      // Compute partial against current KV in send_buf");
    println!("      flash_attn_bf16_v3_split_kv(.., &send_buf, .., split_idx=step, num_splits=N);");
    println!("      // Ring rotate: send to (rank+1)%N, recv from (rank-1)%N");
    println!("      comm.send(&send_buf, dst=(rank+1)%N).unwrap();");
    println!("      comm.recv(&mut recv_buf, src=(rank-1+N)%N).unwrap();");
    println!("      std::mem::swap(&mut send_buf, &mut recv_buf);");
    println!("  }}");
    println!("  flash_decoding_combine(.., num_splits=N);");
    println!();
    println!("Kernel-level changes for that future port: ZERO. Same split-K +");
    println!("combine kernels. Only the orchestrator (above) needs to swap the");
    println!("loop body for an NCCL send/recv between compute steps.");
}
