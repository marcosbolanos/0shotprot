#!/usr/bin/env bash
set -euo pipefail

GPU_UUID="GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26"
ROOT="outputs/prosst_ft_soft_full_vocab_AAV_n128_20260604"
mkdir -p "$ROOT/logs"

# Avoid colliding with the current AAV ablation queue on GPU0.
while tmux has-session -t prosst_aav_ablate_n128 2>/dev/null; do
  echo "[$(date -Is)] waiting for prosst_aav_ablate_n128 to finish before soft-vocab sweep"
  sleep 300
done

run_cfg() {
  local cfg="$1"; shift
  local out="$ROOT/$cfg"
  mkdir -p "$out/logs"
  for seed in 1 2 3 4 5; do
    echo "[$(date -Is)] cfg=${cfg} seed=${seed} start"
    CUDA_VISIBLE_DEVICES="$GPU_UUID" uv run python scripts/run_aav_zero_shot_prosst.py \
      --task AAV \
      --results_dirpath "$out" \
      --seed "$seed" \
      --n_queries 128 \
      --n_iters 10 \
      --batch_size 256 \
      --mask_budget 4 \
      --mask_strategy mixed_explore_exploit \
      --smc_vocab full \
      --finetune_prosst \
      --finetune_epochs 5 \
      --finetune_lr 3e-5 \
      --lambda_kl 2 \
      --finetune_batch_size 16 \
      --finetune_replay all \
      --device cuda \
      --debug_generation_trace \
      "$@" \
      2>&1 | tee "$out/logs/seed_${seed}.log"
    echo "[$(date -Is)] cfg=${cfg} seed=${seed} done"
  done
}

for penalty in 0.5 1.0 2.0; do
  safe_penalty="${penalty/./p}"
  run_cfg "rank_penalty_${safe_penalty}" --reward_mode rank --non_cluster_logit_penalty "$penalty"
  run_cfg "grpo_penalty_${safe_penalty}" --reward_mode grpo_advantage --negative_weight 0.25 --advantage_clip 2 --non_cluster_logit_penalty "$penalty"
done
