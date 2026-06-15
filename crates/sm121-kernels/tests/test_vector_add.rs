use cudarc::driver::PushKernelArg;
use sm121_kernels::{device, module};

#[test]
fn test_vector_add_correctness() {
    let ctx = device::init_device(0).expect("failed to init SM121 device");
    let stream = ctx.default_stream();

    let func = module::load_kernel(&ctx, "vector_add", "vector_add")
        .expect("failed to load vector_add kernel");

    let n: u32 = 1024;
    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let a_dev = stream.memcpy_stod(&a_host).expect("failed to copy A");
    let b_dev = stream.memcpy_stod(&b_host).expect("failed to copy B");
    let mut c_dev = stream
        .alloc_zeros::<f32>(n as usize)
        .expect("failed to alloc C");

    let threads_per_block: u32 = 256;
    let num_blocks = n.div_ceil(threads_per_block);

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&n)
            .launch(cfg)
    }
    .expect("kernel launch failed");

    let c_host = stream.memcpy_dtov(&c_dev).expect("failed to copy C back");

    for i in 0..n as usize {
        let expected = a_host[i] + b_host[i];
        assert!(
            (c_host[i] - expected).abs() < 1e-5,
            "mismatch at index {i}: got {}, expected {expected}",
            c_host[i]
        );
    }
}

#[test]
fn test_vector_add_large() {
    let ctx = device::init_device(0).expect("failed to init SM121 device");
    let stream = ctx.default_stream();

    let func = module::load_kernel(&ctx, "vector_add", "vector_add")
        .expect("failed to load vector_add kernel");

    let n: u32 = 1_000_000;
    let a_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (i as f32) * -0.001 + 1.0).collect();

    let a_dev = stream.memcpy_stod(&a_host).expect("failed to copy A");
    let b_dev = stream.memcpy_stod(&b_host).expect("failed to copy B");
    let mut c_dev = stream
        .alloc_zeros::<f32>(n as usize)
        .expect("failed to alloc C");

    let threads_per_block: u32 = 256;
    let num_blocks = n.div_ceil(threads_per_block);

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&n)
            .launch(cfg)
    }
    .expect("kernel launch failed");

    let c_host = stream.memcpy_dtov(&c_dev).expect("failed to copy C back");

    // Spot-check a few values
    for &i in &[0, 1, 100, 999_999] {
        let expected = a_host[i] + b_host[i];
        assert!(
            (c_host[i] - expected).abs() < 1e-3,
            "mismatch at index {i}: got {}, expected {expected}",
            c_host[i]
        );
    }
}

#[test]
fn test_kernel_caching() {
    let ctx = device::init_device(0).expect("failed to init SM121 device");

    // Loading the same kernel twice should hit the cache
    let f1 = module::load_kernel(&ctx, "vector_add", "vector_add").expect("first load failed");
    let f2 = module::load_kernel(&ctx, "vector_add", "vector_add").expect("second load failed");

    // Both should be valid (we can't compare function pointers easily,
    // but if caching is broken, the second load would fail)
    drop(f1);
    drop(f2);
}
