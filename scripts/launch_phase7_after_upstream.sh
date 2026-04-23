#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

# Wait for combo runs to stop, then require Phase 6A/2C completion artifacts.
while ps -eo cmd | grep -E "run_latent_campaign.py" | grep -E "phase1g_confirm_lambda_0p2_20seed|phase_combo_topfeat30_lambda_" >/dev/null \
  || [ ! -f outputs_latent_campaign/LGK/phase6a_maxmut_2/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign/LGK/phase6a_maxmut_4/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign/LGK/phase6a_maxmut_6/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign/LGK/phase6a_maxmut_8/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign/LGK/phase2c_ppl_mean4_max8/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign/LGK/phase2c_ppl_mean6_max10/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign/LGK/phase2c_ppl_mean8_max12/campaign_summary.json ] \
  || ps -eo cmd | grep -E "run_latent_campaign.py" | grep -E "phase6a_maxmut_|phase2c_ppl_" >/dev/null; do
  sleep 60
done

GPU_UUID="${GPU_UUID:-GPU-025e10e6-263f-d814-6dd5-added86fc8af}" \
RESULTS_DIR="${RESULTS_DIR:-outputs_latent_campaign_v2}" \
LOG_PREFIX="${LOG_PREFIX:-/tmp/latent_campaign_lgk_phase7_followup}" \
./scripts/run_latent_campaign_lgk_phase7_followup_a6000.sh
