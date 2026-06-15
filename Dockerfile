# sm121-kernels kernel benchmark image
#
# Zero runtime CUDA toolkit dependency — only libcuda.so from the driver is
# needed at run time. The PTX → SASS compilation happens at build time via
# ptxas; the cubins are embedded into the Rust binary as `include_bytes!`.
#
# Build (multi-stage, ~15 min on Spark cold cache):
#   docker build -t sm121-kernels:latest .
#
# Run the kernel benchmark suite (the default CMD):
#   docker run --rm --gpus all sm121-kernels:latest
#
# Run (interactive shell with all kernels available):
#   docker run --rm --gpus all -it sm121-kernels:latest bash
#
# Pinned versions (also documented in docs/reproducer.md):
#   - Base:  ubuntu:24.04 (arm64 supported; this image targets sbsa for DGX Spark)
#   - Rust:  1.93.1 (matches the dev environment that produced the headline)
#   - CUDA:  13.0    (cuda-toolkit-13-0; ptxas-13.0 is what we ship cubins from)
#   - cudarc: 0.15   (driver-only loading; no cuRT, no cuBLAS, no cuDNN)
#
# Host driver: SM121-capable Blackwell GPU + NVIDIA driver that ships
# CUDA 13.0 user-mode driver (libcuda.so r580+). Verified on DGX Spark GB10.

# ----------------------------------------------------------------------------
# Stage 1: builder — compiles the Rust binaries + embeds PTX→SASS cubins
# ----------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

ARG RUST_VERSION=1.93.1
ARG CUDA_KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust at the pinned toolchain version
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain "${RUST_VERSION}" --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# Install CUDA toolkit 13.0 (build-time only; runtime stage drops it)
RUN curl -fsSL "${CUDA_KEYRING_URL}" -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-13-0 \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb
ENV PATH="/usr/local/cuda/bin:${PATH}"

WORKDIR /app
COPY . .

# Build the binaries we ship in the runtime image. The release profile is
# pinned in Cargo.toml (lto, opt-level=3, codegen-units=1).
RUN cargo build --release \
        --example benchmark \
        --example rust_api_demo

# ----------------------------------------------------------------------------
# Stage 2: runtime — only libcuda.so from the driver, no CUDA toolkit
# ----------------------------------------------------------------------------
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# The only runtime dependency is the NVIDIA user-mode driver (libcuda.so),
# and it must NOT be installed in the image: it has to match the host's
# kernel driver, so the nvidia-container-toolkit injects it at run time
# (`docker run --gpus all`). Nothing CUDA-related is baked in here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/examples/benchmark          /usr/local/bin/spark-bench
COPY --from=builder /app/target/release/examples/rust_api_demo      /usr/local/bin/spark-demo
COPY scripts/reproducer.sh                                          /usr/local/bin/spark-reproducer

# Default to the kernel benchmark suite; override with `bash`, spark-demo, or
# spark-reproducer (banner-wrapped benchmark run).
ENTRYPOINT []
CMD ["/usr/local/bin/spark-bench"]
