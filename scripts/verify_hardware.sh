#!/usr/bin/env bash
# Verify SM121 hardware and toolchain are available.

set -e

echo "=== sm121-kernels hardware verification ==="

echo ""
echo "CUDA devices:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo ""
echo "Compute capability: $CC"
if [[ "$CC" != "12.1" ]]; then
    echo "WARNING: Expected 12.1 (SM121), got $CC"
fi

echo ""
PTXAS=${PTXAS:-$(which ptxas 2>/dev/null || echo "")}
if [[ -z "$PTXAS" ]]; then
    PTXAS="/usr/local/cuda/bin/ptxas"
fi
if [[ -x "$PTXAS" ]]; then
    echo "ptxas: $PTXAS"
    $PTXAS --version
else
    echo "ERROR: ptxas not found"
    exit 1
fi

echo ""
CPP=${CPP:-$(which cpp 2>/dev/null || echo "/usr/bin/cpp")}
if [[ -x "$CPP" ]]; then
    echo "cpp: $CPP"
    $CPP --version 2>&1 | head -1
else
    echo "ERROR: cpp (C preprocessor) not found"
    exit 1
fi

echo ""
echo "Rust toolchain:"
rustc --version
cargo --version

echo ""
echo "CUDA driver version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1

echo ""
echo "=== All checks passed ==="
