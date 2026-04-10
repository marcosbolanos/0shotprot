#!/usr/bin/env bash
set -uo pipefail

cd /home/lamsade/mbolanos/code/ProSperoReprs
log_path="outputs/representation_proxies_onehot_combo.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] start CUDA_VISIBLE_DEVICES=3 batch=512 onehot+combo" > "$log_path"

CUDA_VISIBLE_DEVICES=3 UV_CACHE_DIR=.uv_cache uv run python -u -m prospero.runners.run_representation_proxies_onehot_combo \
  --output-dir outputs/representation_proxies_onehot_combo \
  --embedding-batch-size 512 \
  >> "$log_path" 2>&1
rc=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] exit rc=$rc" >> "$log_path"
exit $rc
