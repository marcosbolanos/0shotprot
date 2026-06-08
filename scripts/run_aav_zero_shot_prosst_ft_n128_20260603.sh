#!/usr/bin/env bash
set -euo pipefail

GPU_UUID="GPU-025e10e6-263f-d814-6dd5-added86fc8af"
OUT="outputs/aav_zero_shot_prosst_ft_mixed_k4_n128_kl2_lr3e5_20260603"
mkdir -p "$OUT/logs"

for seed in 1 2 3 4 5; do
  echo "[$(date -Is)] starting ProSST FT AAV seed ${seed}"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" uv run python scripts/run_aav_zero_shot_prosst.py \
    --results_dirpath "$OUT" \
    --seed "$seed" \
    --n_queries 128 \
    --n_iters 10 \
    --batch_size 256 \
    --mask_budget 4 \
    --mask_strategy mixed_explore_exploit \
    --smc_vocab cluster \
    --finetune_prosst \
    --finetune_epochs 5 \
    --finetune_lr 3e-5 \
    --lambda_kl 2 \
    --finetune_batch_size 16 \
    --finetune_replay all \
    --device cuda \
    --debug_generation_trace \
    2>&1 | tee "$OUT/logs/seed_${seed}.log"
  echo "[$(date -Is)] finished ProSST FT AAV seed ${seed}"
done
