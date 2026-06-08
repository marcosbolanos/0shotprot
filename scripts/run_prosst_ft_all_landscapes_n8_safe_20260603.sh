#!/usr/bin/env bash
set -euo pipefail
GPU_UUID="GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7"
ROOT="outputs/prosst_ft_all_landscapes_n8_rank_cluster_20260603"
TASKS=(AAV AMIE E4B GFP LGK Pab1 TEM UBE2I)
mkdir -p "$ROOT/logs"
for task in "${TASKS[@]}"; do
  for seed in 1 2 3 4 5; do
    echo "[$(date -Is)] n=8 task=${task} seed=${seed} start"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$GPU_UUID" uv run python scripts/run_aav_zero_shot_prosst.py \
      --task "$task" --results_dirpath "$ROOT" --seed "$seed" \
      --n_queries 8 --n_iters 10 --batch_size 64 \
      --mask_budget 4 --mask_strategy mixed_explore_exploit --smc_vocab cluster \
      --entropy_chunk_size 16 \
      --finetune_prosst --finetune_epochs 5 --finetune_lr 3e-5 --lambda_kl 2 \
      --finetune_batch_size 1 --finetune_replay all --reward_mode rank \
      --device cuda --debug_generation_trace \
      2>&1 | tee "$ROOT/logs/${task}_seed_${seed}.log"
    echo "[$(date -Is)] n=8 task=${task} seed=${seed} done"
  done
done
