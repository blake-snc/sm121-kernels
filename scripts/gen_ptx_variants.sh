#!/usr/bin/env bash
# gen_ptx_variants.sh — emit concrete PTX variants from cpp-#ifdef templates.
#
# Collapse near-identical hand-PTX variant families
# into ONE template + a flag-driven generator. Each variant is produced by running
# the same `cpp -P -I ptx/common` preprocessing the build already uses, with a
# variant-specific -D flag set (and -DSPARK_ENTRY=<name> for the entry symbol).
#
# Generated files land in ptx/attention/generated/ and are proven byte-identical
# (at the cubin level) to the hand-written sources by scripts/check_codegen_identity.sh.
#
# This is standalone for now — it is NOT wired into build.rs. The hand-written .ptx
# files remain the build's source of truth until the identity gate is green and a
# follow-up migrates the build over.
#
# Usage: scripts/gen_ptx_variants.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CPP="${CPP:-$(command -v cpp || echo /usr/bin/cpp)}"
INC="$ROOT/ptx/common"

# Families are discovered from ptx/attention/templates/*.variants. Each manifest
# <family>.variants pairs with the template <family>.ptx.in next to it, and all
# variants are emitted into ptx/attention/generated.
TEMPLATE_DIR="$ROOT/ptx/attention/templates"
OUTDIR="$ROOT/ptx/attention/generated"

shopt -s nullglob
MANIFESTS=( "$TEMPLATE_DIR"/*.variants )
shopt -u nullglob
if [ "${#MANIFESTS[@]}" -eq 0 ]; then echo "FATAL: no *.variants manifests in $TEMPLATE_DIR"; exit 2; fi

rc=0
total=0
for manifest in "${MANIFESTS[@]}"; do
  template="${manifest%.variants}.ptx.in"
  outdir="$OUTDIR"
  if [ ! -f "$template" ]; then echo "FATAL: missing template $template"; exit 2; fi
  if [ ! -f "$manifest" ]; then echo "FATAL: missing manifest $manifest"; exit 2; fi
  echo "== family: $(basename "${manifest%.variants}") =="
  mkdir -p "$outdir"

  while read -r stem entry flags; do
    # skip blanks and comments
    [ -z "${stem:-}" ] && continue
    case "$stem" in \#*) continue;; esac

    defs=( "-DSPARK_ENTRY=$entry" )
    if [ "$flags" != "-" ] && [ -n "${flags:-}" ]; then
      for f in $flags; do
        [ "$f" = "-" ] && continue
        defs+=( "-D$f" )
      done
    fi

    out="$outdir/$stem.ptx"
    if "$CPP" -P -I "$INC" "${defs[@]}" "$template" "$out" 2>/tmp/gen_${stem}.cpp.err; then
      echo "GEN  $stem  (${defs[*]})"
      total=$((total+1))
    else
      echo "CPP-FAIL  $stem"; cat /tmp/gen_${stem}.cpp.err; rc=1
    fi
  done < "$manifest"
done

echo "----------------------------------------"
echo "generated $total variant(s)"
exit $rc
