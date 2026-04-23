#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

# Wait until all Phase 10 outputs exist, and no Phase 10 process remains.
while [ ! -f outputs_latent_campaign_v5/LGK/phase10_topfeat3_lambda0p1_maxmut4_ppl_mean6_max10/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign_v5/LGK/phase10_topfeat3_lambda0p2_maxmut4_ppl_mean6_max10/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign_v5/LGK/phase10_topfeat3_lambda0p1_maxmut6_ppl_mean8_max12/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign_v5/LGK/phase10_topfeat3_lambda0p2_maxmut6_ppl_mean8_max12/campaign_summary.json ] \
  || ps -eo cmd | grep -E "run_latent_campaign.py" | grep -E "phase10_" >/dev/null; do
  sleep 60
done

GPU_UUID="${GPU_UUID:-GPU-025e10e6-263f-d814-6dd5-added86fc8af}" \
RESULTS_DIR="${RESULTS_DIR:-outputs_latent_campaign_v6}" \
LOG_PREFIX="${LOG_PREFIX:-/tmp/latent_campaign_lgk_phase11_followup}" \
./scripts/run_latent_campaign_lgk_phase11_followup_a6000.sh
