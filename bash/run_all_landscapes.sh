#!/bin/bash

set -uo pipefail

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_results_dir>"
    exit 1
fi

TASKS=(E4B Pab1 AAV GFP TEM UBE2I AMIE LGK D_SHIFT D_SHIFT_SMALL D_SHIFT_HARD)
SEEDS=(1 2 3 4 5)
RESULTS_DIRPATH="$1"

run_task() {
    local TASK="$1"
    local MIN_CORR
    local MAX_CORR
    local N_ITERS

    if [[ $TASK == "AAV" || $TASK == "E4B" || $TASK == "Pab1" || $TASK == D_SHIFT* ]]; then
        MIN_CORR=3
        MAX_CORR=10
    else
        MIN_CORR=5
        MAX_CORR=15
    fi

    if [[ $TASK == D_SHIFT* ]]; then
        N_ITERS=4
    else
        N_ITERS=10
    fi

    for SEED in "${SEEDS[@]}"; do
        echo "Running $TASK with seed=$SEED"
        python src/prospero/runners/run_protein.py \
            --task "$TASK" \
            --seed "$SEED" \
            --results_dirpath "$RESULTS_DIRPATH" \
            --n_iters "$N_ITERS" \
            --min_corruptions "$MIN_CORR" \
            --max_corruptions "$MAX_CORR" \
            --full_deterministic
    done

    python src/prospero/runners/etl_results.py \
        --task "$TASK" \
        --results_dirpath "$RESULTS_DIRPATH" \
        --n_iters "$N_ITERS"
}

PIDS=()
TASK_NAMES=()

for TASK in "${TASKS[@]}"; do
    run_task "$TASK" &
    PIDS+=("$!")
    TASK_NAMES+=("$TASK")
done

FAILURES=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "Task ${TASK_NAMES[$i]} failed" >&2
        FAILURES=1
    fi
done

exit "$FAILURES"
