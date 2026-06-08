#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

export CUDA_VISIBLE_DEVICES=GPU-025e10e6-263f-d814-6dd5-added86fc8af
export MPLCONFIGDIR=/tmp/matplotlib
export UV_CACHE_DIR=.uv_cache

BASE=outputs/aav_zero_shot_fixed_mask_k4_20260529
TASK=AAV
MASK_BUDGET=4
BUDGETS=(8 16 32 64)
STRATEGIES=(random middle_entropy seed_grow)
SEEDS=(1 2 3 4 5)

echo "=== AAV zero-shot fixed-mask variable-query run started $(date -Is) ==="
echo "BASE=${BASE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for n_queries in "${BUDGETS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    out_dir="${BASE}/n_samples_${n_queries}/${strategy}"
    mkdir -p "${out_dir}/${TASK}"
    echo "=== n_queries=${n_queries} strategy=${strategy} started $(date -Is) ==="
    for seed in "${SEEDS[@]}"; do
      save_path="${out_dir}/${TASK}/seed_${seed}.pkl"
      if [[ -s "${save_path}" ]]; then
        echo "--- skip existing n_queries=${n_queries} strategy=${strategy} seed=${seed}: ${save_path} ---"
        continue
      fi
      echo "--- run n_queries=${n_queries} strategy=${strategy} seed=${seed} $(date -Is) ---"
      uv run python -m prospero.runners.run_zero_shot_protein \
        --results_dirpath "${out_dir}" \
        --task "${TASK}" \
        --seed "${seed}" \
        --n_queries "${n_queries}" \
        --n_iters 10 \
        --batch_size 256 \
        --resampling_steps 1 \
        --mask_strategy "${strategy}" \
        --mask_budget "${MASK_BUDGET}" \
        --entropy_quantile 0.5 \
        --seed_grow_alpha 1.0 \
        --seed_grow_beta 1.0 \
        --seed_grow_coupling_tau 4.0 \
        --full_deterministic
    done
    echo "=== n_queries=${n_queries} strategy=${strategy} finished $(date -Is) ==="
  done
done

echo "=== AAV zero-shot fixed-mask variable-query run finished $(date -Is) ==="
