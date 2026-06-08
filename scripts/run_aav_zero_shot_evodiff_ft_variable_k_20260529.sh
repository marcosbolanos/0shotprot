#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to a GPU UUID before running.}"
export MPLCONFIGDIR=/tmp/matplotlib
export UV_CACHE_DIR=.uv_cache

BASE="${BASE:-outputs/aav_zero_shot_evodiff_ft_k4_variable_k_20260529}"
TASK="${TASK:-AAV}"
MASK_BUDGET="${MASK_BUDGET:-4}"
BUDGETS="${BUDGETS:-8 16 32 64 128}"
STRATEGIES="${STRATEGIES:-seed_grow}"
SEEDS="${SEEDS:-1 2 3 4 5}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
FINETUNE_LR="${FINETUNE_LR:-1e-5}"
LAMBDA_KL="${LAMBDA_KL:-2}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-16}"
N_ITERS="${N_ITERS:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ARGS_ARRAY=()
if [[ -n "${EXTRA_ARGS}" ]]; then
  # Keep launcher-provided optional flags from breaking the wrapper's quoting.
  read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS}"
fi

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

echo "=== AAV zero-shot EvoDiff fine-tune variable-k run started $(date -Is) ==="
echo "BASE=${BASE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "BUDGETS=${BUDGETS}"
echo "STRATEGIES=${STRATEGIES}"
echo "FINETUNE_EPOCHS=${FINETUNE_EPOCHS} FINETUNE_LR=${FINETUNE_LR} LAMBDA_KL=${LAMBDA_KL}"

for n_queries in ${BUDGETS}; do
  for strategy in ${STRATEGIES}; do
    out_dir="${BASE}/n_samples_${n_queries}/${strategy}"
    mkdir -p "${out_dir}/${TASK}"
    echo "=== n_queries=${n_queries} strategy=${strategy} started $(date -Is) ==="
    for seed in ${SEEDS}; do
      save_path="${out_dir}/${TASK}/seed_${seed}.pkl"
      if seed_complete "${save_path}"; then
        echo "--- skip completed n_queries=${n_queries} strategy=${strategy} seed=${seed}: ${save_path} ---"
        continue
      fi
      if [[ -s "${save_path}" ]]; then
        echo "--- rerun incomplete n_queries=${n_queries} strategy=${strategy} seed=${seed}: ${save_path} ---"
      else
        echo "--- run n_queries=${n_queries} strategy=${strategy} seed=${seed} $(date -Is) ---"
      fi
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
      fi
      uv run python -m prospero.runners.run_zero_shot_finetune_evodiff \
        --results_dirpath "${out_dir}" \
        --task "${TASK}" \
        --seed "${seed}" \
        --n_queries "${n_queries}" \
        --n_iters "${N_ITERS}" \
        --batch_size 256 \
        --resampling_steps 1 \
        --mask_strategy "${strategy}" \
        --mask_budget "${MASK_BUDGET}" \
        --entropy_quantile 0.5 \
        --seed_grow_alpha 1.0 \
        --seed_grow_beta 1.0 \
        --seed_grow_coupling_tau 4.0 \
        --finetune_epochs "${FINETUNE_EPOCHS}" \
        --finetune_lr "${FINETUNE_LR}" \
        --lambda_kl "${LAMBDA_KL}" \
        --finetune_batch_size "${FINETUNE_BATCH_SIZE}" \
        --finetune_replay all \
        --full_deterministic \
        "${EXTRA_ARGS_ARRAY[@]}"
    done
    echo "=== n_queries=${n_queries} strategy=${strategy} finished $(date -Is) ==="
  done
done

echo "=== AAV zero-shot EvoDiff fine-tune variable-k run finished $(date -Is) ==="
