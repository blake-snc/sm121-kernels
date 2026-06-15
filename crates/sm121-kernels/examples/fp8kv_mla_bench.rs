//! FP8-KV MLA decode: scalar vs BF16 MMA (FP8→BF16 dequant inline).
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaStream};
use sm121_kernels::{attention, device};
use std::sync::Arc;

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
    let mut t = Vec::with_capacity(iters);
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
        t.push(s.elapsed_ms(&e).unwrap() as f64 * 1000.0);
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}

fn main() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    println!("FP8-KV MLA decode: scalar vs MMA (median of 50)");
    println!("----------------------------------------------------------------");
    for (b, h, skv) in [
        (1u32, 16u32, 256u32),
        (2, 32, 512),
        (2, 64, 1024),
        (2, 128, 1024),
        (4, 128, 2048),
    ] {
        let qc = stream
            .alloc_zeros::<u16>((b * h * attention::MLA_D_C) as usize)
            .unwrap();
        let qr = stream
            .alloc_zeros::<u16>((b * h * attention::MLA_D_R) as usize)
            .unwrap();
        let ckv = stream
            .alloc_zeros::<u8>((b * skv * attention::MLA_D_C) as usize)
            .unwrap();
        let kr = stream
            .alloc_zeros::<u8>((b * skv * attention::MLA_D_R) as usize)
            .unwrap();
        let mut o1 = stream
            .alloc_zeros::<u16>((b * h * attention::MLA_D_C) as usize)
            .unwrap();
        let mut o2 = stream
            .alloc_zeros::<u16>((b * h * attention::MLA_D_C) as usize)
            .unwrap();
        let scale = 1.0f32 / ((attention::MLA_D_C + attention::MLA_D_R) as f32).sqrt();
        let kv_scale = 0.1f32;

        let s1 = time_us(&ctx, &stream, 50, || {
            attention::mla_decode_fp8(
                &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o1, b, h, skv, scale, kv_scale,
            )
            .unwrap();
        });
        let s2 = time_us(&ctx, &stream, 50, || {
            attention::mla_decode_fp8kv_mma(
                &ctx, &stream, &qc, &qr, &ckv, &kr, &mut o2, b, h, skv, scale, kv_scale,
            )
            .unwrap();
        });
        println!(
            "B={b:2} H={h:3} Skv={skv:5}  fp8_scalar={s1:7.1}us  fp8kv_mma={s2:7.1}us ({:.2}x)",
            s1 / s2
        );
    }
}
