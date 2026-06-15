#!/usr/bin/env bash
# sm121-kernels reproducer entrypoint
#
# Runs the kernel benchmark suite and prints the headline numbers in a
# format suitable for capturing in a CI log. Ships in the Docker image as
# `spark-reproducer`; can also be run directly from a source checkout:
#
#   ./scripts/reproducer.sh
#
# Optional env vars:
#   SPARK_BENCH_FILTER  — pattern to pass to spark-bench (default: all)
#
# Exit codes:
#   0 — all steps succeeded
#   1 — kernel benchmark failed (the headline-blocking failure)

set -euo pipefail

banner() {
    echo
    echo "=============================================================="
    echo "  $1"
    echo "=============================================================="
}

# ---- 0. Environment ---------------------------------------------------------
banner "Environment"
nvidia-smi --query-gpu=name,compute_cap,driver_version,cuda_version --format=csv 2>/dev/null \
    || echo "(nvidia-smi unavailable — running outside --gpus all?)"
echo
echo "spark-bench binary: $(command -v spark-bench || echo 'not on PATH')"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- 1. Kernel-level headline -----------------------------------------------
banner "Kernel benchmark suite (spark-bench)"
if command -v spark-bench >/dev/null 2>&1; then
    # Bench prints CUDA event timing: median/mean/min/max latency, TFLOPS,
    # GB/s, % theoretical peak.
    spark-bench ${SPARK_BENCH_FILTER:-} || {
        echo
        echo "ERROR: spark-bench failed. The kernel headline cannot be reproduced."
        exit 1
    }
else
    echo "spark-bench not found; running 'cargo run --release --example benchmark' instead."
    cargo run --release --example benchmark ${SPARK_BENCH_FILTER:-}
fi

banner "Reproducer complete"
echo "See docs/reproducer.md for the full headline table and expected numbers."
