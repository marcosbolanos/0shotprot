#!/usr/bin/env bash
set -u
cd /home/lamsade/mbolanos/code/ProSpero
LOG=outputs/aav_zero_shot_fixed_mask_k4_20260529/watchdog_1h.log
{
  echo "=== watchdog $(date -Is) ==="
  echo "--- tmux ---"
  tmux list-sessions || true
  echo "--- gpu ---"
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo "--- processes ---"
  ps -eo pid,ppid,stat,etime,cmd | grep -E 'run_zero_shot_protein|run_aav_zero_shot_fixed_mask|aav_zs_fixed_mask' | grep -v grep || true
  echo "--- latest files ---"
  find outputs/aav_zero_shot_fixed_mask_k4_20260529 -maxdepth 4 -type f -printf '%TY-%Tm-%Td %TH:%TM %p %s\n' | sort | tail -n 80 || true
  echo "--- route log tail ---"
  tail -n 160 outputs/aav_zero_shot_fixed_mask_k4_20260529/tmux.stdout.log 2>/dev/null || true
  echo "--- errors ---"
  grep -R -i -n 'traceback\|error\|exception\|killed\|cuda out of memory' outputs/aav_zero_shot_fixed_mask_k4_20260529 outputs/lgk_zero_shot_prospero_20260528 outputs/aav_onehot_random_mask_variable_k_20260528 outputs/aav_onehot_targeted_zero_score_variable_k_20260528 2>/dev/null | tail -n 120 || true
} >> "$LOG" 2>&1
