#!/usr/bin/env bash
# ptx_syntax_check.sh — assemble every PTX kernel for sm_121a and report failures.
#
# This is the no-GPU correctness gate for the kernel sources: `ptxas` is a
# cross-assembler, so it validates sm_121a PTX on any host (no SM121a GPU
# required), catching register/encoding/ISA regressions in CI on free runners.
# Mirrors what build.rs does at build time, but as an explicit, per-file pass
# with a summary — usable locally and in CI.
#
# Requires: CUDA Toolkit 13.0+ (ptxas with sm_121a support) and cpp.
# Usage:  scripts/ptx_syntax_check.sh [ARCH]   (ARCH default: sm_121a)
set -uo pipefail

ARCH="${1:-sm_121a}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PTXAS="${PTXAS:-$(command -v ptxas || echo /usr/local/cuda/bin/ptxas)}"
CPP="${CPP:-$(command -v cpp || echo /usr/bin/cpp)}"
INC="$ROOT/ptx/common"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ ! -x "$PTXAS" ]; then echo "FATAL: ptxas not found (set PTXAS=)"; exit 2; fi

pass=0; fail=0; failed_files=()
while IFS= read -r ptx; do
  stem="$(basename "${ptx%.ptx}")"
  pp="$TMP/$stem.pp.ptx"
  # Match build.rs: cpp -P preprocessing with the common/ include path.
  if ! "$CPP" -P -I "$INC" "$ptx" "$pp" 2>"$TMP/$stem.cpp.err"; then
    echo "CPP-FAIL  $ptx"; cat "$TMP/$stem.cpp.err"; fail=$((fail+1)); failed_files+=("$ptx"); continue
  fi
  # Allow a per-file `// BUILD_ARCH: sm_xxx` override (build.rs honors this).
  arch="$ARCH"
  if a="$(grep -m1 -oE 'BUILD_ARCH:[[:space:]]*sm_[0-9a-z]+' "$ptx" 2>/dev/null | grep -oE 'sm_[0-9a-z]+')"; then
    [ -n "$a" ] && arch="$a"
  fi
  if "$PTXAS" --gpu-name "$arch" -O3 --warn-on-spills -o /dev/null "$pp" 2>"$TMP/$stem.ptxas.err"; then
    pass=$((pass+1))
    # surface spill warnings (non-fatal) for visibility
    grep -q "spill" "$TMP/$stem.ptxas.err" 2>/dev/null && echo "WARN-SPILL $stem: $(grep spill "$TMP/$stem.ptxas.err" | head -1)"
  else
    echo "PTXAS-FAIL $ptx ($arch)"; cat "$TMP/$stem.ptxas.err"; fail=$((fail+1)); failed_files+=("$ptx")
  fi
done < <(find "$ROOT/ptx" -name '*.ptx' | sort)

echo "----------------------------------------"
echo "PTX syntax check ($ARCH): $pass passed, $fail failed"
if [ "$fail" -gt 0 ]; then printf '  FAILED: %s\n' "${failed_files[@]}"; exit 1; fi
exit 0
