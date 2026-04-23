#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

# Wait until all Phase 6A outputs exist, and no Phase 6A process remains.
while [ ! -f outputs_latent_campaign/LGK/phase6a_maxmut_2/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign/LGK/phase6a_maxmut_4/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign/LGK/phase6a_maxmut_6/campaign_summary.json ] \
  || [ ! -f outputs_latent_campaign/LGK/phase6a_maxmut_8/campaign_summary.json ] \
  || ps -eo cmd | grep -E "run_latent_campaign.py" | grep -E "phase6a_maxmut_" >/dev/null; do
  sleep 60
done

GPU_UUID="${GPU_UUID:-GPU-025e10e6-263f-d814-6dd5-added86fc8af}" \
LOG_PREFIX="${LOG_PREFIX:-/tmp/latent_campaign_lgk_phase2c}" \
./scripts/run_latent_campaign_lgk_phase2c_ppl_filter_a6000.sh
