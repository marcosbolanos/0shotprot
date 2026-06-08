#!/usr/bin/env bash
set -euo pipefail
cd /home/lamsade/mbolanos/code/ProSpero
export CUDA_VISIBLE_DEVICES=GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26
export MPLCONFIGDIR=/tmp/matplotlib
export UV_CACHE_DIR=.uv_cache
BASE=outputs/aav_zero_shot_fixed_mask_k4_20260529
for strategy in random middle_entropy seed_grow; do
  echo "=== strategy=${strategy} started $(date -Is) ==="
  for seed in 1 2 3 4 5; do
    echo "--- strategy=${strategy} seed=${seed} $(date -Is) ---"
    uv run python -m prospero.runners.run_zero_shot_protein \
      --results_dirpath "${BASE}/${strategy}" \
      --task AAV \
      --seed "${seed}" \
      --n_queries 128 \
      --n_iters 10 \
      --batch_size 256 \
      --resampling_steps 1 \
      --mask_strategy "${strategy}" \
      --mask_budget 4 \
      --entropy_quantile 0.5 \
      --seed_grow_alpha 1.0 \
      --seed_grow_beta 1.0 \
      --seed_grow_coupling_tau 4.0 \
      --full_deterministic
  done
  echo "=== strategy=${strategy} finished $(date -Is) ==="
done
