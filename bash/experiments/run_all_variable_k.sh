#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <outputs_prefix_dir> [surrogate_arch] [max_jobs]"
    echo "Example: $0 outputs cnn 3"
    exit 1
fi

OUTPUTS_PREFIX_DIR="$1"
SURROGATE_ARCH="${2:-cnn}"
MAX_JOBS="${3:-3}"

# Benchmark tasks only (AAV and LGK intentionally omitted: already run)
TASKS=(AMIE TEM E4B Pab1 GFP UBE2I)
# AAV
# LGK

for TASK in "${TASKS[@]}"; do
    TASK_LOWER="$(echo "$TASK" | tr '[:upper:]' '[:lower:]')"
    OUT_DIR="${OUTPUTS_PREFIX_DIR}/out_240226_${TASK_LOWER}_${SURROGATE_ARCH}"
    LOG_FILE="${OUT_DIR}/out_240226.log"

    mkdir -p "$OUT_DIR"

    echo "============================================================"
    echo "Running TASK=${TASK} | SURROGATE=${SURROGATE_ARCH} | MAX_JOBS=${MAX_JOBS}"
    echo "Output dir: ${OUT_DIR}"
    echo "Log file:   ${LOG_FILE}"
    echo "============================================================"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        ./bash/experiments/run_variable_k.sh "$OUT_DIR" "$SURROGATE_ARCH" \
        --task "$TASK" --max-jobs "$MAX_JOBS" 2>&1 | tee "$LOG_FILE"
done
