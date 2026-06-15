use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::error::{Result, SparkError};

/// Required SM version: 12.1 (DGX Spark GB10 / Blackwell GeForce).
const REQUIRED_MAJOR: u32 = 12;
const REQUIRED_MINOR: u32 = 1;

/// Initialize a CUDA context and verify it is SM121.
///
/// Returns the context handle if the GPU has compute capability 12.1.
/// Returns `SparkError::UnsupportedArch` otherwise.
pub fn init_device(ordinal: usize) -> Result<Arc<CudaContext>> {
    let ctx = match CudaContext::new(ordinal) {
        Ok(c) => c,
        // The DGX Spark GPU shares LPDDR5x with the host, so a CUDA OOM at context
        // init is usually page-cache fragmentation (high Buffers, low MemFree) rather
        // than true exhaustion. Turn the cryptic CUDA_ERROR_OUT_OF_MEMORY into an
        // actionable message; the success path is unchanged.
        Err(e) if format!("{e:?}").contains("OUT_OF_MEMORY") => {
            return Err(SparkError::Other(diagnose_init_oom(&format!("{e:?}"))));
        }
        Err(e) => return Err(e.into()),
    };

    let major = ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )? as u32;
    let minor = ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    )? as u32;

    if major != REQUIRED_MAJOR || minor != REQUIRED_MINOR {
        return Err(SparkError::UnsupportedArch { major, minor });
    }

    // Stamp this context with a fresh generation so the raw-kernel cache can
    // tell a NEW context apart from a destroyed one whose address was reused
    // (see module::load_kernel_raw).
    crate::module::register_context(&ctx);
    Ok(ctx)
}

/// Check whether the given device ordinal is SM121 without fully initializing it.
pub fn is_sm121(ordinal: usize) -> bool {
    init_device(ordinal).is_ok()
}

/// Build an actionable message for a CUDA-OOM at context init. On Linux it inspects
/// `/proc/meminfo`: if the page cache (Buffers) dominates while MemFree is low, the
/// unified-memory allocator simply cannot find contiguous pages, which a cache drop fixes.
fn diagnose_init_oom(raw: &str) -> String {
    let hint = read_meminfo_hint().unwrap_or_default();
    format!(
        "CUDA out of memory initializing the GPU context.{hint} \
         On the DGX Spark this is usually page-cache fragmentation of the shared LPDDR5x, \
         not true exhaustion; free it with: sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'. \
         (driver error: {raw})"
    )
}

/// Returns a ` MemFree=.. GiB, Buffers=.. GiB, MemAvailable=.. GiB.` snippet from
/// `/proc/meminfo`, or `None` if it cannot be read (non-Linux or parse failure).
fn read_meminfo_hint() -> Option<String> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb = |key: &str| -> Option<f64> {
        text.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .map(|v| v / 1_048_576.0) // kB -> GiB
    };
    let free = kb("MemFree:")?;
    let buffers = kb("Buffers:")?;
    let avail = kb("MemAvailable:")?;
    Some(format!(
        " MemFree={free:.1} GiB, Buffers={buffers:.1} GiB, MemAvailable={avail:.1} GiB."
    ))
}
