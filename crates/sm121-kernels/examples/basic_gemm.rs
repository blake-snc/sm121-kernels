use sm121_kernels::{device, gemm};

fn main() {
    let ctx = device::init_device(0).expect("failed to init SM121 device");
    let stream = ctx.default_stream();

    let m: u32 = 128;
    let n: u32 = 128;
    let k: u32 = 128;

    // Random BF16 data (here we use zeros for simplicity)
    let a_host = vec![0u16; (m * k) as usize];
    let b_host = vec![0u16; (k * n) as usize];

    let a_dev = stream.memcpy_stod(&a_host).unwrap();
    let b_dev = stream.memcpy_stod(&b_host).unwrap();
    let mut c_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    // BF16 MMA GEMM
    gemm::gemm_bf16_mma(&ctx, &stream, &a_dev, &b_dev, &mut c_dev, m, n, k)
        .expect("BF16 MMA GEMM failed");

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    println!(
        "BF16 MMA GEMM: {}x{}x{} -> output[0] = {:#06x}",
        m, n, k, c_host[0]
    );

    // FP8 MMA GEMM
    let a_fp8 = vec![0u8; (m * k) as usize];
    let b_fp8 = vec![0u8; (k * n) as usize];
    let a_fp8_dev = stream.memcpy_stod(&a_fp8).unwrap();
    let b_fp8_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let mut c_fp8_dev = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_fp8_mma(
        &ctx,
        &stream,
        &a_fp8_dev,
        &b_fp8_dev,
        &mut c_fp8_dev,
        m,
        n,
        k,
    )
    .expect("FP8 MMA GEMM failed");

    let c_fp8_host = stream.memcpy_dtov(&c_fp8_dev).unwrap();
    println!(
        "FP8 MMA GEMM:  {}x{}x{} -> output[0] = {:#06x}",
        m, n, k, c_fp8_host[0]
    );
}
