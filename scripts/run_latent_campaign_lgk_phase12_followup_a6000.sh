#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

GPU_UUID="${GPU_UUID:-GPU-025e10e6-263f-d814-6dd5-added86fc8af}"
RESULTS_DIR="${RESULTS_DIR:-outputs_latent_campaign_v7}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-20}"
LOG_PREFIX="${LOG_PREFIX:-/tmp/latent_campaign_lgk_phase12_followup}"

# Phase 12: confirm strongest constrained setting at larger seed count.
declare -a ARMS=(
  "phase12_confirm_topfeat3_lambda0p1_maxmut2_20seed --top-features 3 --mutation-penalty-lambda 0.1 --max-mutations 2"
  "phase12_confirm_topfeat3_lambda0p2_maxmut2_20seed --top-features 3 --mutation-penalty-lambda 0.2 --max-mutations 2"
)

for ARM in "${ARMS[@]}"; do
  PHASE="${ARM%% *}"
  EXTRA_ARGS="${ARM#* }"
  LOG_PATH="${LOG_PREFIX}_${PHASE}.log"
  echo "[latent-phase12] phase=${PHASE} gpu_uuid=${GPU_UUID} seeds=${SEED_START}-${SEED_END}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${GPU_UUID}" UV_CACHE_DIR=.uv_cache \
    uv run python src/prospero/runners/run_latent_campaign.py \
      --results-dirpath "${RESULTS_DIR}" \
      --task LGK \
      --phase "${PHASE}" \
      --seed-start "${SEED_START}" \
      --seed-end "${SEED_END}" \
      --batch-size 256 \
      --top-k 256 \
      --steering-layer 2 \
      --steering-direction-mode signed \
      --steering-scalars 0.02 0.05 0.1 0.2 0.4 0.7 1.2 1.8 \
      --combo-chunk-size 4 \
      ${EXTRA_ARGS} \
      > "${LOG_PATH}" 2>&1
done

echo "[latent-phase12] done"
