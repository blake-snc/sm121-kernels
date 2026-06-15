use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::{Result, SparkError};
use crate::module;

/// Launch BF16 token embedding lookup kernel.
///
/// For each token id in `token_ids`, copy the corresponding row from
/// `embedding_table` into `out`.
///
/// `token_ids`: [num_tokens] u32 (clamped to [0, vocab_size))
/// `embedding_table`: [vocab_size, hidden_dim] BF16
/// `out`: [num_tokens, hidden_dim] BF16
#[allow(clippy::too_many_arguments)]
pub fn embedding_lookup_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    token_ids: &CudaSlice<u32>,
    embedding_table: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    num_tokens: u32,
    vocab_size: u32,
    hidden_dim: u32,
) -> Result<()> {
    if num_tokens == 0 || vocab_size == 0 || hidden_dim == 0 {
        return Err(SparkError::InvalidArgument(
            "num_tokens, vocab_size, hidden_dim must be > 0".into(),
        ));
    }
    if token_ids.len() < num_tokens as usize {
        return Err(SparkError::InvalidArgument("token_ids too small".into()));
    }
    let need_table = vocab_size as usize * hidden_dim as usize;
    if embedding_table.len() < need_table {
        return Err(SparkError::InvalidArgument(format!(
            "embedding_table too small: {} < {need_table}",
            embedding_table.len()
        )));
    }
    let need_out = num_tokens as usize * hidden_dim as usize;
    if out.len() < need_out {
        return Err(SparkError::InvalidArgument(format!(
            "out too small: {} < {need_out}",
            out.len()
        )));
    }

    let func = module::load_kernel(ctx, "embedding_lookup_bf16", "embedding_lookup_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(token_ids)
            .arg(embedding_table)
            .arg(out)
            .arg(&hidden_dim)
            .arg(&vocab_size)
            .launch(cfg)?;
    }
    Ok(())
}
