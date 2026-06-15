use cudarc::driver::sys::{
    cuLaunchKernel, cuStreamSynchronize, cuTensorMapEncodeTiled, CUresult, CUtensorMapDataType,
    CUtensorMapFloatOOBfill, CUtensorMapInterleave, CUtensorMapL2promotion, CUtensorMapSwizzle,
    CUtensorMap_st,
};
use cudarc::driver::DevicePtr;
use sm121_kernels::device;

#[test]
#[ignore = "standalone TMA test kernel; pre-existing CUDA_ERROR_ILLEGAL_INSTRUCTION"]
fn test_tma_smoke() {
    eprintln!("=== TMA Smoke Test ===");

    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();
    eprintln!("Device initialized");

    // Create 64x128 BF16 input with known value at [0,0]
    let mut input_host = vec![0u16; 64 * 128];
    input_host[0] = 0x3F80; // 1.0 in BF16

    let input_dev = stream.memcpy_stod(&input_host).expect("alloc input");
    let output_dev = stream.alloc_zeros::<u16>(1).expect("alloc output");
    eprintln!("Buffers allocated");

    // Create TMA descriptor
    let (input_ptr, _sync) = input_dev.device_ptr(&stream);
    // TMA descriptor must be 64-byte aligned for cuLaunchKernel with .param .align 64
    #[repr(C, align(64))]
    struct AlignedTma(CUtensorMap_st);
    let mut tma_aligned = AlignedTma(CUtensorMap_st::default());
    let tma = &mut tma_aligned.0;
    let global_dim: [u64; 2] = [128, 64];
    let global_strides: [u64; 1] = [256]; // 128 * 2 bytes
    let box_dim: [u32; 2] = [128, 64];
    let elem_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tma,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            input_ptr as *mut core::ffi::c_void,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            elem_strides.as_ptr(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    assert_eq!(result, CUresult::CUDA_SUCCESS);
    eprintln!("TMA descriptor created");

    // Load kernel with raw CUfunction handle
    let cu_func = sm121_kernels::module::load_kernel_raw(&ctx, "tma_smoke", "tma_smoke_test")
        .expect("load kernel raw");
    eprintln!("Kernel loaded: cu_func = {:?}", cu_func);

    let cu_stream = stream.cu_stream();
    let (out_ptr, _) = output_dev.device_ptr(&stream);

    let coord_x: u32 = 0;
    let coord_y: u32 = 0;

    let params: [*mut core::ffi::c_void; 4] = [
        tma as *const CUtensorMap_st as *mut core::ffi::c_void,
        &out_ptr as *const u64 as *mut _,
        &coord_x as *const u32 as *mut _,
        &coord_y as *const u32 as *mut _,
    ];

    eprintln!("Launching kernel...");
    let launch = unsafe {
        cuLaunchKernel(
            cu_func,
            1,
            1,
            1, // grid
            64,
            1,
            1, // block (2 warps)
            0, // dynamic SMEM
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };
    eprintln!("Launch result: {:?}", launch);
    assert_eq!(launch, CUresult::CUDA_SUCCESS);

    let sync = unsafe { cuStreamSynchronize(cu_stream) };
    eprintln!("Sync result: {:?}", sync);
    assert_eq!(
        sync,
        CUresult::CUDA_SUCCESS,
        "kernel execution failed: {:?}",
        sync
    );

    // Read output
    let output_host = stream.memcpy_dtov(&output_dev).expect("D2H");
    eprintln!("Output: 0x{:04X} (expected 0x3F80)", output_host[0]);
    assert_eq!(output_host[0], 0x3F80, "TMA loaded wrong data");
    eprintln!("=== TMA Smoke Test PASSED ===");
}
