#!/usr/bin/env bash
set -euo pipefail

N_QUERIES="$1"
GPU_UUID="$2"
ROOT="outputs/prosst_ft_all_landscapes_n${N_QUERIES}_grpo_cluster_20260604"
TASKS=(AAV AMIE E4B GFP LGK Pab1 TEM UBE2I)
mkdir -p "$ROOT/logs"

for task in "${TASKS[@]}"; do
  for seed in 1 2 3 4 5; do
    OUT_PKL="$ROOT/$task/seed_${seed}.pkl"
    if [[ -s "$OUT_PKL" ]]; then
      echo "[$(date -Is)] n=${N_QUERIES} task=${task} seed=${seed} skip existing $OUT_PKL"
      continue
    fi
    echo "[$(date -Is)] n=${N_QUERIES} task=${task} seed=${seed} start"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$GPU_UUID" uv run python scripts/run_aav_zero_shot_prosst.py \
      --task "$task" \
      --results_dirpath "$ROOT" \
      --seed "$seed" \
      --n_queries "$N_QUERIES" \
      --n_iters 10 \
      --batch_size 64 \
      --mask_budget 4 \
      --mask_strategy mixed_explore_exploit \
      --smc_vocab cluster \
      --entropy_chunk_size 16 \
      --finetune_prosst \
      --finetune_epochs 5 \
      --finetune_lr 3e-5 \
      --lambda_kl 2 \
      --finetune_batch_size 1 \
      --finetune_replay all \
      --reward_mode grpo_advantage \
      --device cuda \
      --debug_generation_trace \
      2>&1 | tee "$ROOT/logs/${task}_seed_${seed}.log"
    echo "[$(date -Is)] n=${N_QUERIES} task=${task} seed=${seed} done"
  done
done
