#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to a GPU UUID before running.}"
export MPLCONFIGDIR=/tmp/matplotlib
export UV_CACHE_DIR=.uv_cache

BASE="${BASE:-outputs/aav_zero_shot_fixed_mask_k4_trace_20260602}"
TASK="${TASK:-AAV}"
MASK_BUDGET="${MASK_BUDGET:-4}"
BUDGETS="${BUDGETS:-8 16 32 64 128}"
SEEDS="${SEEDS:-1 2 3 4 5}"
N_ITERS="${N_ITERS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"

seed_complete() {
  local save_path="$1"
  [[ -s "${save_path}" ]] || return 1
  uv run python - "${save_path}" "${N_ITERS}" <<'PY'
import pickle
import sys

path = sys.argv[1]
expected = int(sys.argv[2])
try:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    keys = [int(k) for k in data.keys() if isinstance(k, int)]
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if keys and max(keys) >= expected else 1)
PY
}

echo "=== traced AAV no-FT zero-shot run started $(date -Is) ==="
echo "BASE=${BASE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "BUDGETS=${BUDGETS}"

for n_queries in ${BUDGETS}; do
  out_dir="${BASE}/n_samples_${n_queries}/seed_grow"
  mkdir -p "${out_dir}/${TASK}"
  echo "=== no-FT n_queries=${n_queries} seed_grow started $(date -Is) ==="
  for seed in ${SEEDS}; do
    save_path="${out_dir}/${TASK}/seed_${seed}.pkl"
    if seed_complete "${save_path}"; then
      echo "--- skip completed no-FT n_queries=${n_queries} seed=${seed}: ${save_path} ---"
      continue
    fi
    echo "--- run no-FT n_queries=${n_queries} seed=${seed} $(date -Is) ---"
    uv run python -m prospero.runners.run_zero_shot_protein \
      --results_dirpath "${out_dir}" \
      --task "${TASK}" \
      --seed "${seed}" \
      --n_queries "${n_queries}" \
      --n_iters "${N_ITERS}" \
      --batch_size "${BATCH_SIZE}" \
      --resampling_steps 1 \
      --mask_strategy seed_grow \
      --mask_budget "${MASK_BUDGET}" \
      --entropy_quantile 0.5 \
      --seed_grow_alpha 1.0 \
      --seed_grow_beta 1.0 \
      --seed_grow_coupling_tau 4.0 \
      --debug_generation_trace \
      --full_deterministic
  done
  echo "=== no-FT n_queries=${n_queries} seed_grow finished $(date -Is) ==="
done

echo "=== traced AAV no-FT zero-shot run finished $(date -Is) ==="
