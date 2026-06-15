//! Head-to-head: scalar GDN decode vs TMA-accelerated GDN decode.

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaStream};
use sm121_kernels::{device, linear_attention};

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
    let ctx = device::init_device(0).expect("SM121");
    let stream = ctx.default_stream();
    println!("GDN decode: scalar vs TMA (CUDA events, 5 warmup + 100 measured, median)");
    println!("----------------------------------------------------------------");
    for (b, h) in [(1u32, 4u32), (2, 8), (4, 16), (8, 32)] {
        let d = linear_attention::GDN_HEAD_DIM;
        let qkv = (b * h * d) as usize;
        let sc = (b * h) as usize;
        let state_n = (b * h * d * d) as usize;
        let q = stream.alloc_zeros::<u16>(qkv).unwrap();
        let k = stream.alloc_zeros::<u16>(qkv).unwrap();
        let v = stream.alloc_zeros::<u16>(qkv).unwrap();
        let alpha = stream.alloc_zeros::<f32>(sc).unwrap();
        let beta = stream.alloc_zeros::<f32>(sc).unwrap();
        let mut st1 = stream.alloc_zeros::<u16>(state_n).unwrap();
        let mut st2 = stream.alloc_zeros::<f32>(state_n).unwrap();
        let mut y1 = stream.alloc_zeros::<u16>(qkv).unwrap();
        let mut y2 = stream.alloc_zeros::<u16>(qkv).unwrap();

        let scalar_us = time_us(&ctx, &stream, 100, || {
            linear_attention::gdn_decode(
                &ctx, &stream, &q, &k, &v, &alpha, &beta, &mut st1, &mut y1, b, h,
            )
            .unwrap();
        });
        let tma_us = time_us(&ctx, &stream, 100, || {
            linear_attention::gdn_decode_tma(
                &ctx, &stream, &q, &k, &v, &alpha, &beta, &mut st2, &mut y2, b, h,
            )
            .unwrap();
        });
        let speedup = scalar_us / tma_us;
        println!(
            "B={b} H={h}  scalar={scalar_us:7.1}us   TMA={tma_us:7.1}us   speedup={speedup:.2}x"
        );
    }
}
