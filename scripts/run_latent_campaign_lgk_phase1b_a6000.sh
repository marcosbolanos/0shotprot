#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

GPU_UUID="${GPU_UUID:-GPU-025e10e6-263f-d814-6dd5-added86fc8af}"
RESULTS_DIR="${RESULTS_DIR:-outputs_latent_campaign}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-10}"
LOG_PATH="${LOG_PATH:-/tmp/latent_campaign_lgk_phase1b.log}"

echo "[latent-phase1b] gpu_uuid=${GPU_UUID} seeds=${SEED_START}-${SEED_END} log=${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${GPU_UUID}" UV_CACHE_DIR=.uv_cache \
  uv run python src/prospero/runners/run_latent_campaign.py \
    --results-dirpath "${RESULTS_DIR}" \
    --task LGK \
    --phase phase1b_sign_test \
    --seed-start "${SEED_START}" \
    --seed-end "${SEED_END}" \
    --batch-size 256 \
    --top-k 256 \
    --top-features 3 \
    --steering-layer 2 \
    --steering-direction-mode both \
    --steering-scalars 0.05 0.2 0.7 \
    --combo-chunk-size 4 \
    > "${LOG_PATH}" 2>&1

echo "[latent-phase1b] done"
