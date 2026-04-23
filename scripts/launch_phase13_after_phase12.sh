#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

# Wait until both Phase 12 confirm runs are complete, and no Phase 12 process remains.
while [ ! -f outputs_latent_campaign_v7/LGK/phase12_confirm_topfeat3_lambda0p1_maxmut2_20seed/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign_v7/LGK/phase12_confirm_topfeat3_lambda0p2_maxmut2_20seed/campaign_summary.json ] \
  || ps -eo cmd | grep -E "run_latent_campaign.py" | grep -E "phase12_" >/dev/null; do
  sleep 60
done

GPU_UUID="${GPU_UUID:-GPU-025e10e6-263f-d814-6dd5-added86fc8af}" \
RESULTS_DIR="${RESULTS_DIR:-outputs_latent_campaign_v8}" \
LOG_PREFIX="${LOG_PREFIX:-/tmp/latent_campaign_lgk_phase13_followup}" \
./scripts/run_latent_campaign_lgk_phase13_followup_a6000.sh
