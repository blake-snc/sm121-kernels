# sm121-kernels — developer & CI-mirror targets.
# The no-GPU gate (verify) needs only CUDA Toolkit 13.0+ (ptxas) + Rust.
# The GPU gate (golden, test, bench) needs SM121a hardware.

.PHONY: help build build-capi header ptx-check codegen-check lint docs verify golden test sanitize bench ci-gpu api-snapshot api-check clean

help:
	@echo "sm121-kernels make targets:"
	@echo "  make verify     — no-GPU gate: ptx-check + build + lint + docs (mirrors CI lint-build-ptx)"
	@echo "  make golden     — regenerate golden vectors from the PyTorch reference (deterministic, seed=42)"
	@echo "  make test       — run the kernel test suite vs goldens (needs SM121a + 'make golden' first)"
	@echo "  make ci-gpu     — golden + test + sanitize + bench (mirrors CI gpu-test)"
	@echo "  make ptx-check  — cross-assemble every PTX for sm_121a (no GPU needed)"

# ---- No-GPU gate (free CI runners; ptxas cross-assembles sm_121a without an SM121a GPU) ----
ptx-check:
	bash scripts/ptx_syntax_check.sh sm_121a

# Prove the templated FA variants assemble byte-identical to the hand-written PTX
# (ptxas is deterministic -> identical cubin proves the kernel is unchanged).
codegen-check:
	bash scripts/gen_ptx_variants.sh
	bash scripts/check_codegen_identity.sh

build:
	cargo build --release

build-capi:
	cargo build --release --features c-api

# Generate the C header for the extern "C" surface (needed by examples/c_api_demo.c).
# Requires `cargo install cbindgen`. The crate name here is the PRIVATE crate name
# (sm121-kernels); the export mirror renames it to sm121-kernels.
header:
	cbindgen --crate sm121-kernels --lang c --config cbindgen.toml --output include/sm121_kernels.h

lint:
	cargo fmt --check
	cargo clippy --release -- -D warnings

docs:
	cargo doc --no-deps --release

verify: ptx-check codegen-check build build-capi lint docs
	@echo "no-GPU gate PASSED"

# ---- GPU gate (SM121a) ----
golden:
	pip install -r tests/reference/requirements.txt
	python tests/reference/generate_golden.py
	@echo "golden vectors regenerated (deterministic, torch seed=42)"

test:
	cargo test --release -- --test-threads=1

# memcheck on a core kernel test binary (build first via `make test` or cargo test --no-run)
sanitize:
	@BIN=$$(ls -t target/release/deps/test_attention-* 2>/dev/null | grep -v '\.d$$' | head -1); \
	 if [ -n "$$BIN" ]; then compute-sanitizer --tool memcheck --error-exitcode 1 "$$BIN"; \
	 else echo "no test binary — run 'cargo test --release --no-run' first"; exit 1; fi

bench:
	cargo run --release --example benchmark

ci-gpu: golden test sanitize bench
	@echo "GPU gate PASSED"

# ---- Public API surface guard (no GPU) ----
# Requires a nightly toolchain and `cargo install cargo-public-api`.
# See STABILITY.md for the stability/versioning policy this guards.
# api-snapshot writes the committed baseline; api-check diffs against it.
api-snapshot:
	cargo +nightly public-api --simplified > public-api.txt
	@echo "wrote public-api.txt baseline"

api-check:
	cargo +nightly public-api --simplified diff public-api.txt

clean:
	cargo clean
