#!/usr/bin/env bash
# check_codegen_identity.sh — prove generated PTX variants are behavior-identical
# to the hand-written sources via cubin-identity.
#
# ptxas is a deterministic cross-assembler: assembling identical PTX yields a
# byte-identical cubin on any host (no SM121a GPU needed). So for each variant we
# assemble BOTH the generated PTX and the original hand-written .ptx with the same
# flags and `cmp` the cubins. Byte-identical cubins prove the refactor changed no
# kernel behavior — the maintainability collapse is provably inert.
#
# NOTE: comments differ between the template output and the hand files (filenames,
# inline notes); cpp -P strips // comments and ptxas ignores them, so only the
# emitted instructions/params affect the cubin. That is exactly what this gate
# checks.
#
# Runs gen_ptx_variants.sh first, then gates. Exits non-zero on ANY mismatch.
# Usage: scripts/check_codegen_identity.sh [ARCH]   (ARCH default: sm_121a)
set -uo pipefail

ARCH="${1:-sm_121a}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PTXAS="${PTXAS:-$(command -v ptxas || echo /usr/local/cuda/bin/ptxas)}"
CPP="${CPP:-$(command -v cpp || echo /usr/bin/cpp)}"
INC="$ROOT/ptx/common"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ ! -x "$PTXAS" ]; then echo "FATAL: ptxas not found (set PTXAS=)"; exit 2; fi

# Regenerate variants so the gate always reflects the current template.
echo "== regenerating variants =="
if ! bash "$ROOT/scripts/gen_ptx_variants.sh"; then
  echo "FATAL: generation failed"; exit 2
fi
echo

# Generated variants live here; the original hand-written sources are frozen in
# archive/ as the byte-identity reference (the build generates from templates).
# Every templated family is discovered from templates/*.variants.
GENDIR="$ROOT/ptx/attention/generated"
HANDDIR="$ROOT/ptx/attention/archive"
TEMPLATE_DIR="$ROOT/ptx/attention/templates"
shopt -s nullglob
MANIFESTS=( "$TEMPLATE_DIR"/*.variants )
shopt -u nullglob
if [ "${#MANIFESTS[@]}" -eq 0 ]; then echo "FATAL: no *.variants manifests in $TEMPLATE_DIR"; exit 2; fi

assemble() { # <src.ptx> <out.cubin>
  local src="$1" out="$2" stem; stem="$(basename "${src%.ptx}")"
  if ! "$CPP" -P -I "$INC" "$src" "$TMP/$stem.pp.ptx" 2>"$TMP/$stem.cpp.err"; then
    echo "CPP-FAIL $src"; cat "$TMP/$stem.cpp.err"; return 1
  fi
  if ! "$PTXAS" --gpu-name "$ARCH" -O3 -o "$out" "$TMP/$stem.pp.ptx" 2>"$TMP/$stem.ptxas.err"; then
    echo "PTXAS-FAIL $src"; cat "$TMP/$stem.ptxas.err"; return 1
  fi
  return 0
}

pass=0; fail=0; total=0; failed=()
echo "== cubin-identity gate ($ARCH) =="
for MANIFEST in "${MANIFESTS[@]}"; do
  echo "-- family: $(basename "${MANIFEST%.variants}") --"
  while read -r stem entry flags; do
    [ -z "${stem:-}" ] && continue
    case "$stem" in \#*) continue;; esac
    total=$((total+1))

    gen="$GENDIR/$stem.ptx"
    hand="$HANDDIR/$stem.ptx"
    if [ ! -f "$gen" ];  then echo "MISS-GEN   $stem"; fail=$((fail+1)); failed+=("$stem"); continue; fi
    if [ ! -f "$hand" ]; then echo "MISS-HAND  $stem"; fail=$((fail+1)); failed+=("$stem"); continue; fi

    if ! assemble "$gen"  "$TMP/$stem.gen.cubin";  then fail=$((fail+1)); failed+=("$stem"); continue; fi
    if ! assemble "$hand" "$TMP/$stem.hand.cubin"; then fail=$((fail+1)); failed+=("$stem"); continue; fi

    if cmp -s "$TMP/$stem.gen.cubin" "$TMP/$stem.hand.cubin"; then
      echo "IDENTICAL  $stem"
      pass=$((pass+1))
    else
      echo "MISMATCH   $stem  (generated cubin != hand cubin)"
      fail=$((fail+1)); failed+=("$stem")
    fi
  done < "$MANIFEST"
done

echo "----------------------------------------"
echo "cubin-identity: $pass/$total identical, $fail failed"
if [ "$fail" -gt 0 ]; then printf '  FAILED: %s\n' "${failed[@]}"; exit 1; fi
exit 0
