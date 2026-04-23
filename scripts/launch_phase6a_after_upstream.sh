#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

while ps -eo cmd | grep -E "run_latent_campaign.py" | grep -E "phase1g_confirm_lambda_0p2_20seed|phase_combo_topfeat30_lambda_" >/dev/null; do
  sleep 60
done

GPU_UUID="${GPU_UUID:-GPU-025e10e6-263f-d814-6dd5-added86fc8af}" \
LOG_PREFIX="${LOG_PREFIX:-/tmp/latent_campaign_lgk_phase6a}" \
./scripts/run_latent_campaign_lgk_phase6a_mutation_cap_a6000.sh
