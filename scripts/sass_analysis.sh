#!/usr/bin/env bash
# SASS Analysis Tool for sm121-kernels
# Disassembles all cubins and extracts instruction mix, register usage,
# MMA utilization, and scheduling insights.
#
# Usage: ./scripts/sass_analysis.sh [cubin_dir] [--json] [--verbose]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find cubin directory
CUBIN_DIR=""
VERBOSE=false
for arg in "$@"; do
    case "$arg" in
        --verbose) VERBOSE=true ;;
        --json) ;; # reserved
        *) [ -d "$arg" ] && CUBIN_DIR="$arg" ;;
    esac
done

if [ -z "$CUBIN_DIR" ]; then
    CUBIN_DIR=$(find "$PROJECT_ROOT/target/release/build" -name "fa_bf16_v11_fused_scale.cubin" -printf "%h" 2>/dev/null | head -1)
fi

if [ -z "$CUBIN_DIR" ] || [ ! -d "$CUBIN_DIR" ]; then
    echo "ERROR: No cubin directory found. Build first: cargo build --release"
    exit 1
fi

NVDISASM=$(which nvdisasm 2>/dev/null || echo "")
if [ -z "$NVDISASM" ]; then
    echo "ERROR: nvdisasm not found. Install CUDA toolkit."
    exit 1
fi

OUTPUT_DIR="$PROJECT_ROOT/analysis"
mkdir -p "$OUTPUT_DIR/sass" "$OUTPUT_DIR/summary"

echo "=============================================="
echo "  sm121-kernels SASS Analysis"
echo "  cubins: $(ls "$CUBIN_DIR"/*.cubin 2>/dev/null | wc -l) files"
echo "  nvdisasm: $($NVDISASM --version 2>/dev/null | head -1)"
echo "=============================================="
echo ""

# Summary CSV
SUMMARY_CSV="$OUTPUT_DIR/summary/instruction_mix.csv"
echo "kernel,total_instructions,mma_count,mma_pct,load_count,store_count,mov_count,fma_count,branch_count,barrier_count,tma_count,registers,smem_bytes,cubin_bytes" > "$SUMMARY_CSV"

TOTAL_KERNELS=0
TOTAL_MMA=0
TOTAL_INSN=0

for cubin in "$CUBIN_DIR"/*.cubin; do
    name=$(basename "$cubin" .cubin)
    TOTAL_KERNELS=$((TOTAL_KERNELS + 1))

    # Disassemble
    sass_file="$OUTPUT_DIR/sass/${name}.sass"
    $NVDISASM "$cubin" > "$sass_file" 2>/dev/null || {
        echo "  WARN: Failed to disassemble $name"
        continue
    }

    # Count instructions (lines with /*hex_addr*/ followed by an instruction mnemonic)
    total=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "$sass_file" 2>/dev/null || true)
    total=${total:-0}

    # Instruction categories
    mma=$(grep -cE 'HMMA|QMMA' "$sass_file" 2>/dev/null || true)
    loads=$(grep -cE '\bLDG\b|\bLDS\b|\bLDSM\b|\bLDL\b|\bLDGSTS\b' "$sass_file" 2>/dev/null || true)
    stores=$(grep -cE '\bSTG\b|\bSTS\b|\bSTL\b' "$sass_file" 2>/dev/null || true)
    movs=$(grep -cE '\bMOV\b|\bPRMT\b|\bSHFL\b|\bS2R\b|\bCS2R\b' "$sass_file" 2>/dev/null || true)
    fmas=$(grep -cE '\bFFMA\b|\bHFMA2\b|\bFADD\b|\bFMUL\b|\bF2FP\b' "$sass_file" 2>/dev/null || true)
    branches=$(grep -cE '\bBRA\b|\bEXIT\b|\bRET\b|\bCALL\b|\bBREAK\b' "$sass_file" 2>/dev/null || true)
    barriers=$(grep -cE '\bBAR\b|\bMBARRIER\b|\bDEPBAR\b|\bWARPSYNC\b' "$sass_file" 2>/dev/null || true)
    tma=$(grep -cE 'UTMALDG|UTMASTG|LDGSTS' "$sass_file" 2>/dev/null || true)

    # Sanitize — ensure integers
    mma=${mma:-0}; loads=${loads:-0}; stores=${stores:-0}; movs=${movs:-0}
    fmas=${fmas:-0}; branches=${branches:-0}; barriers=${barriers:-0}; tma=${tma:-0}

    # Resource usage from cuobjdump
    res_line=$(cuobjdump --dump-resource-usage "$cubin" 2>/dev/null | grep "REG:" | head -1 || true)
    registers=$(echo "$res_line" | grep -oP 'REG:\K[0-9]+' || echo "0")
    smem=$(echo "$res_line" | grep -oP 'SHARED:\K[0-9]+' || echo "0")
    registers=${registers:-0}; smem=${smem:-0}

    cubin_bytes=$(stat -c%s "$cubin" 2>/dev/null || echo "0")

    # MMA percentage
    if [ "$total" -gt 0 ] && [ "$mma" -gt 0 ]; then
        mma_pct=$(awk "BEGIN { printf \"%.1f\", ($mma / $total) * 100 }")
    else
        mma_pct="0.0"
    fi

    TOTAL_MMA=$((TOTAL_MMA + mma))
    TOTAL_INSN=$((TOTAL_INSN + total))

    # CSV
    echo "$name,$total,$mma,$mma_pct,$loads,$stores,$movs,$fmas,$branches,$barriers,$tma,$registers,$smem,$cubin_bytes" >> "$SUMMARY_CSV"

    # Console output
    if [ "$mma" -gt 0 ]; then
        printf "  %-45s %5d insn  %3d MMA (%5s%%)  %3d ld  %3d st  %3d bar  REG:%-3s SMEM:%-5s" \
            "$name" "$total" "$mma" "$mma_pct" "$loads" "$stores" "$barriers" "$registers" "$smem"
        [ "$tma" -gt 0 ] && printf "  TMA:%d" "$tma"
        echo ""
    else
        printf "  %-45s %5d insn  --- no MMA ---    %3d ld  %3d st  %3d bar  REG:%-3s SMEM:%-5s\n" \
            "$name" "$total" "$loads" "$stores" "$barriers" "$registers" "$smem"
    fi

    if $VERBOSE; then
        echo "    MOV/PRMT/SHFL/S2R: $movs  FMA/FADD/FMUL/F2FP: $fmas  Branches: $branches  Cubin: $cubin_bytes bytes"
    fi
done

echo ""
echo "=============================================="
echo "  Summary: $TOTAL_KERNELS kernels analyzed"
echo "  Total instructions: $TOTAL_INSN"
echo "  Total MMA instructions: $TOTAL_MMA"
if [ "$TOTAL_INSN" -gt 0 ]; then
    echo "  Overall MMA density: $(awk "BEGIN { printf \"%.1f\", ($TOTAL_MMA / $TOTAL_INSN) * 100 }")%"
fi
echo ""
echo "  CSV: $SUMMARY_CSV"
echo "  SASS: $OUTPUT_DIR/sass/"
echo "=============================================="

# === Sorted comparisons ===

echo ""
echo "=== Flash Attention — sorted by MMA density ==="
echo ""
printf "  %-45s %5s %5s %6s %5s %5s %4s\n" "Kernel" "Insn" "MMA" "MMA%" "REG" "SMEM" "TMA"
echo "  $(printf '%.0s-' {1..85})"
grep -E "^fa_" "$SUMMARY_CSV" | sort -t, -k4 -rn | while IFS=, read -r kn tot mm pct ld st mv fm br ba tm rg sm cb; do
    tma_str=""
    [ "$tm" -gt 0 ] 2>/dev/null && tma_str="$tm"
    printf "  %-45s %5s %5s %5s%% %5s %5s %4s\n" "$kn" "$tot" "$mm" "$pct" "$rg" "$sm" "$tma_str"
done

echo ""
echo "=== GEMM — sorted by MMA density ==="
echo ""
printf "  %-45s %5s %5s %6s %5s %5s\n" "Kernel" "Insn" "MMA" "MMA%" "REG" "SMEM"
echo "  $(printf '%.0s-' {1..75})"
grep -E "^gemm_" "$SUMMARY_CSV" | sort -t, -k4 -rn | while IFS=, read -r kn tot mm pct ld st mv fm br ba tm rg sm cb; do
    printf "  %-45s %5s %5s %5s%% %5s %5s\n" "$kn" "$tot" "$mm" "$pct" "$rg" "$sm"
done

echo ""
echo "=== Elementwise / Other ==="
echo ""
printf "  %-45s %5s %5s %5s\n" "Kernel" "Insn" "REG" "SMEM"
echo "  $(printf '%.0s-' {1..65})"
grep -vE "^fa_|^gemm_|^kernel" "$SUMMARY_CSV" | while IFS=, read -r kn tot mm pct ld st mv fm br ba tm rg sm cb; do
    printf "  %-45s %5s %5s %5s\n" "$kn" "$tot" "$rg" "$sm"
done
