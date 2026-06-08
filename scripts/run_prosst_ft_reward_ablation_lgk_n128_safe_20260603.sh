#!/usr/bin/env bash
set -euo pipefail
GPU_UUID="GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea"
TASK="LGK"
ROOT="outputs/prosst_ft_reward_vocab_ablation_LGK_n128_20260603"
mkdir -p "$ROOT/logs"
run_cfg() {
  local cfg="$1"; shift
  local out="$ROOT/$cfg"
  mkdir -p "$out/logs"
  for seed in 1 2 3 4 5; do
    echo "[$(date -Is)] task=${TASK} cfg=${cfg} seed=${seed} start"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$GPU_UUID" uv run python scripts/run_aav_zero_shot_prosst.py \
      --task "$TASK" --results_dirpath "$out" --seed "$seed" \
      --n_queries 128 --n_iters 10 --batch_size 64 \
      --mask_budget 4 --mask_strategy mixed_explore_exploit \
      --entropy_chunk_size 16 \
      --finetune_prosst --finetune_epochs 5 --finetune_lr 3e-5 --lambda_kl 2 \
      --finetune_batch_size 1 --finetune_replay all --device cuda --debug_generation_trace \
      "$@" 2>&1 | tee "$out/logs/seed_${seed}.log"
    echo "[$(date -Is)] task=${TASK} cfg=${cfg} seed=${seed} done"
  done
}
run_cfg full_vocab_rank --smc_vocab full --reward_mode rank
run_cfg grpo_cluster --smc_vocab cluster --reward_mode grpo_advantage --negative_weight 0.25 --advantage_clip 2
run_cfg standardized_cluster --smc_vocab cluster --reward_mode standardized_advantage --negative_weight 0.25 --advantage_clip 2
run_cfg bottomq_neg05_cluster --smc_vocab cluster --reward_mode bottom_quantile_negative --negative_weight 0.5 --bottom_quantile 0.25
