//! Head-to-head: scalar vs TMA vs MMA MLA decode.

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaStream};
use sm121_kernels::{attention, device};

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
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    println!("MLA decode: scalar vs TMA vs MMA (median of 50)");
    println!("----------------------------------------------------------------");

    let configs = [
        (1u32, 16u32, 256u32),
        (1, 16, 1024),  // small batch, long context — split-K win case
        (1, 128, 1024), // B=1 DeepSeek V3 decode
        (2, 32, 512),
        (2, 64, 1024),
        (2, 128, 1024), // DeepSeek V3-active scale
        (4, 128, 2048),
    ];
    for (b, h, skv) in configs {
        let qc_len = (b * h * attention::MLA_D_C) as usize;
        let qr_len = (b * h * attention::MLA_D_R) as usize;
        let ckv_len = (b * skv * attention::MLA_D_C) as usize;
        let kr_len = (b * skv * attention::MLA_D_R) as usize;
        let qc = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let qr = stream.alloc_zeros::<u16>(qr_len).unwrap();
        let ckv = stream.alloc_zeros::<u16>(ckv_len).unwrap();
        let kr = stream.alloc_zeros::<u16>(kr_len).unwrap();
        let mut o1 = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let mut o2 = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let mut o3 = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let mut o4 = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let scale = 1.0f32 / ((attention::MLA_D_C + attention::MLA_D_R) as f32).sqrt();

        let scalar_us = time_us(&ctx, &stream, 50, || {
            attention::mla_decode_bf16(
                &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o1, b, h, skv, scale,
            )
            .unwrap();
        });
        let tma_us = time_us(&ctx, &stream, 50, || {
            attention::mla_decode_bf16_tma(
                &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o2, b, h, skv, scale,
            )
            .unwrap();
        });
        let mma_us = time_us(&ctx, &stream, 50, || {
            attention::mla_decode_bf16_mma(
                &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o3, b, h, skv, scale,
            )
            .unwrap();
        });
        let mma_tma_us = time_us(&ctx, &stream, 50, || {
            attention::mla_decode_bf16_mma_tma(
                &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o4, b, h, skv, scale,
            )
            .unwrap();
        });
        let mut o_pq = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let pq_us = time_us(&ctx, &stream, 50, || {
            attention::mla_decode_bf16_mma_tma_pq(
                &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o_pq, b, h, skv, scale,
            )
            .unwrap();
        });
        let mut o5 = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let _s_t = scalar_us / tma_us;
        let _s_m = scalar_us / mma_us;
        let s_mt = scalar_us / mma_tma_us;
        let mut best_sp_us = f64::INFINITY;
        let mut best_splits: u32 = 1;
        for &splits in &[2u32, 4, 8, 16] {
            let num_chunks = skv.div_ceil(8);
            if splits > num_chunks {
                continue;
            }
            let t = time_us(&ctx, &stream, 30, || {
                attention::mla_decode_bf16_mma_split(
                    &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o5, b, h, skv, splits, scale,
                )
                .unwrap();
            });
            if t < best_sp_us {
                best_sp_us = t;
                best_splits = splits;
            }
        }
        let s_sp = scalar_us / best_sp_us;
        let mut o6 = stream.alloc_zeros::<u16>(qc_len).unwrap();
        let auto_us = time_us(&ctx, &stream, 50, || {
            attention::mla_decode_bf16_auto(
                &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o6, b, h, skv, scale,
            )
            .unwrap();
        });
        let s_au = scalar_us / auto_us;
        let s_pq = scalar_us / pq_us;
        println!("B={b:2} H={h:3} Skv={skv:5}  scalar={scalar_us:7.1}us  MMA+TMA={mma_tma_us:7.1}us ({s_mt:.2}x)  +PQr={pq_us:7.1}us ({s_pq:.2}x)  BestSplit={best_sp_us:7.1}us ({s_sp:.2}x, n={best_splits})  Auto={auto_us:7.1}us ({s_au:.2}x)");
    }
}
