#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_results_dir> [surrogate_arch] [--task TASK] [--max-jobs N]"
    exit 1
fi

TASK="AAV"
SEEDS=(1 2 3 4 5)
N_SAMPLES_VALUES=(8 16 32 64 128)
BASE_RESULTS_DIRPATH="$1"
SURROGATE_ARCH="cnn"
MAX_JOBS=5
shift

if [[ -n "${1:-}" && "$1" != --* ]]; then
    SURROGATE_ARCH="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            if [[ -z "${2:-}" ]]; then
                echo "Missing value for --task"
                exit 1
            fi
            TASK="$2"
            shift 2
            ;;
        --max-jobs)
            if [[ -z "${2:-}" ]]; then
                echo "Missing value for --max-jobs"
                exit 1
            fi
            MAX_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 <path_to_results_dir> [surrogate_arch] [--task TASK] [--max-jobs N]"
            exit 1
            ;;
    esac
done

N_ITERS=10
MIN_CORR=3
MAX_CORR=10

if [[ "$SURROGATE_ARCH" != "cnn" && "$SURROGATE_ARCH" != "esm_transformer" ]]; then
    echo "Invalid surrogate_arch: $SURROGATE_ARCH"
    echo "Allowed values: cnn, esm_transformer"
    exit 1
fi

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid --max-jobs value: $MAX_JOBS"
    echo "--max-jobs must be a positive integer"
    exit 1
fi

for N_SAMPLES in "${N_SAMPLES_VALUES[@]}"; do
    RESULTS_DIRPATH="${BASE_RESULTS_DIRPATH}/n_samples_${N_SAMPLES}"
    mkdir -p "$RESULTS_DIRPATH"

    ACTIVE_JOBS=0
    FAILED=0

    for SEED in "${SEEDS[@]}"; do
        echo "Running ${TASK} with seed=${SEED}, n_samples=${N_SAMPLES}, surrogate_arch=${SURROGATE_ARCH}"
        python src/prospero/runners/run_protein.py \
            --task "$TASK" \
            --seed "$SEED" \
            --results_dirpath "$RESULTS_DIRPATH" \
            --n_queries "$N_SAMPLES" \
            --n_iters "$N_ITERS" \
            --min_corruptions "$MIN_CORR" \
            --max_corruptions "$MAX_CORR" \
            --surrogate_arch "$SURROGATE_ARCH" \
            --full_deterministic &

        ((ACTIVE_JOBS+=1))
        if (( ACTIVE_JOBS >= MAX_JOBS )); then
            if ! wait -n; then
                FAILED=1
            fi
            ((ACTIVE_JOBS-=1))
        fi
    done

    while (( ACTIVE_JOBS > 0 )); do
        if ! wait -n; then
            FAILED=1
        fi
        ((ACTIVE_JOBS-=1))
    done

    if (( FAILED != 0 )); then
        echo "One or more runs failed for n_samples=${N_SAMPLES}; skipping ETL and exiting."
        exit 1
    fi

    python src/prospero/runners/etl_results.py \
        --task "$TASK" \
        --results_dirpath "$RESULTS_DIRPATH" \
        --n_iters "$N_ITERS"
done
