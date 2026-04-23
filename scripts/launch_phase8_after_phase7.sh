#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

# Wait until all Phase 7 outputs exist, and no Phase 7 process remains.
while [ ! -f outputs_latent_campaign_v2/LGK/phase7_lambda0p2_maxmut6/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign_v2/LGK/phase7_lambda0p2_maxmut8/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign_v2/LGK/phase7_lambda0p2_ppl_mean6_max10/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign_v2/LGK/phase7_lambda0p2_maxmut8_ppl_mean6_max10/campaign_summary.json ] \
  || ps -eo cmd | grep -E "run_latent_campaign.py" | grep -E "phase7_" >/dev/null; do
  sleep 60
done

GPU_UUID="${GPU_UUID:-GPU-025e10e6-263f-d814-6dd5-added86fc8af}" \
RESULTS_DIR="${RESULTS_DIR:-outputs_latent_campaign_v3}" \
LOG_PREFIX="${LOG_PREFIX:-/tmp/latent_campaign_lgk_phase8_followup}" \
./scripts/run_latent_campaign_lgk_phase8_followup_a6000.sh
