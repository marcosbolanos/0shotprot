#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

GPU_UUID="${GPU_UUID:-GPU-025e10e6-263f-d814-6dd5-added86fc8af}"
RESULTS_DIR="${RESULTS_DIR:-outputs_latent_campaign}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-5}"
LOG_PREFIX="${LOG_PREFIX:-/tmp/latent_campaign_lgk_phase1c}"

declare -a TOP_FEATURES=("1" "3" "10" "30")

for TOPF in "${TOP_FEATURES[@]}"; do
  PHASE="phase1c_topfeat_${TOPF}"
  LOG_PATH="${LOG_PREFIX}_topfeat_${TOPF}.log"
  echo "[latent-phase1c] phase=${PHASE} top_features=${TOPF} gpu_uuid=${GPU_UUID} seeds=${SEED_START}-${SEED_END}"
  CUDA_VISIBLE_DEVICES="${GPU_UUID}" UV_CACHE_DIR=.uv_cache \
    uv run python src/prospero/runners/run_latent_campaign.py \
      --results-dirpath "${RESULTS_DIR}" \
      --task LGK \
      --phase "${PHASE}" \
      --seed-start "${SEED_START}" \
      --seed-end "${SEED_END}" \
      --batch-size 256 \
      --top-k 256 \
      --top-features "${TOPF}" \
      --steering-layer 2 \
      --steering-direction-mode signed \
      --steering-scalars 0.05 0.2 0.7 \
      --combo-chunk-size 4 \
      > "${LOG_PATH}" 2>&1
done

echo "[latent-phase1c] done"
