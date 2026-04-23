#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

GPU_UUID="${GPU_UUID:-GPU-025e10e6-263f-d814-6dd5-added86fc8af}"
RESULTS_DIR="${RESULTS_DIR:-outputs_latent_campaign_v3}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-10}"
LOG_PREFIX="${LOG_PREFIX:-/tmp/latent_campaign_lgk_phase8_followup}"

# Phase 8: broaden constraints around strongest family while preserving separation.
declare -a ARMS=(
  "phase8_topfeat3_lambda0p2_maxmut4 --top-features 3 --mutation-penalty-lambda 0.2 --max-mutations 4"
  "phase8_topfeat3_lambda0p2_maxmut6_ppl_mean6_max10 --top-features 3 --mutation-penalty-lambda 0.2 --max-mutations 6 --max-masked-mean-ppl 6 --max-masked-token-ppl 10"
  "phase8_topfeat10_lambda0p2_maxmut6 --top-features 10 --mutation-penalty-lambda 0.2 --max-mutations 6"
  "phase8_topfeat10_lambda0p2_maxmut8_ppl_mean8_max12 --top-features 10 --mutation-penalty-lambda 0.2 --max-mutations 8 --max-masked-mean-ppl 8 --max-masked-token-ppl 12"
)

for ARM in "${ARMS[@]}"; do
  PHASE="${ARM%% *}"
  EXTRA_ARGS="${ARM#* }"
  LOG_PATH="${LOG_PREFIX}_${PHASE}.log"
  echo "[latent-phase8] phase=${PHASE} gpu_uuid=${GPU_UUID} seeds=${SEED_START}-${SEED_END}"
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

echo "[latent-phase8] done"
