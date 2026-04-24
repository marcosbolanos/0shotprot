#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSperoSAE

output_dir="${1:-outputs/interplm_gpu_$(date +%Y%m%d_%H%M%S)}"
gpu_id="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$output_dir" /tmp/mplconfig

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] start CUDA_VISIBLE_DEVICES=${gpu_id}" > "${output_dir}/run.log"

MPLCONFIGDIR=/tmp/mplconfig \
CUDA_VISIBLE_DEVICES="${gpu_id}" \
PYTHONPATH=src \
uv run --no-sync python -u -m prospero.runners.run_representation_proxies_interplm \
  --device cuda \
  --output-dir "$output_dir" \
  --embedding-batch-size "${INTERPLM_EMBED_BATCH_SIZE:-64}" \
  --sae-token-chunk-size "${INTERPLM_SAE_TOKEN_CHUNK_SIZE:-4096}" \
  >> "${output_dir}/run.log" 2>&1

rc=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] exit rc=${rc}" >> "${output_dir}/run.log"
exit "${rc}"
