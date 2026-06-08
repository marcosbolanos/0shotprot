#!/usr/bin/env bash
set -euo pipefail

cd /home/lamsade/mbolanos/code/ProSpero

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to a GPU UUID before running.}"
: "${BASE:?Set BASE output dir.}"
: "${REWARD_MODE:?Set REWARD_MODE.}"
: "${NEGATIVE_WEIGHT:?Set NEGATIVE_WEIGHT.}"

export MPLCONFIGDIR=/tmp/matplotlib
export UV_CACHE_DIR=.uv_cache

TASK="${TASK:-AAV}"
N_QUERIES="${N_QUERIES:-128}"
SEEDS="${SEEDS:-1 2 3 4 5}"
N_ITERS="${N_ITERS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MASK_BUDGET="${MASK_BUDGET:-4}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
FINETUNE_LR="${FINETUNE_LR:-1e-5}"
LAMBDA_KL="${LAMBDA_KL:-2}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-16}"
ADVANTAGE_CLIP="${ADVANTAGE_CLIP:-2}"
BOTTOM_QUANTILE="${BOTTOM_QUANTILE:-0.25}"

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

echo "=== AAV mixed negative ablation started $(date -Is) ==="
echo "BASE=${BASE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "REWARD_MODE=${REWARD_MODE} NEGATIVE_WEIGHT=${NEGATIVE_WEIGHT} BOTTOM_QUANTILE=${BOTTOM_QUANTILE}"
echo "N_QUERIES=${N_QUERIES} SEEDS=${SEEDS}"

out_dir="${BASE}/n_samples_${N_QUERIES}/mixed_explore_exploit"
mkdir -p "${out_dir}/${TASK}"
for seed in ${SEEDS}; do
  save_path="${out_dir}/${TASK}/seed_${seed}.pkl"
  if seed_complete "${save_path}"; then
    echo "--- skip completed ${REWARD_MODE} neg=${NEGATIVE_WEIGHT} seed=${seed}: ${save_path} ---"
    continue
  fi
  echo "--- run ${REWARD_MODE} neg=${NEGATIVE_WEIGHT} n_queries=${N_QUERIES} seed=${seed} $(date -Is) ---"
  uv run python -m prospero.runners.run_zero_shot_finetune_evodiff \
    --results_dirpath "${out_dir}" \
    --task "${TASK}" \
    --seed "${seed}" \
    --n_queries "${N_QUERIES}" \
    --n_iters "${N_ITERS}" \
    --batch_size "${BATCH_SIZE}" \
    --resampling_steps 1 \
    --mask_strategy mixed_explore_exploit \
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
    --reward_mode "${REWARD_MODE}" \
    --advantage_clip "${ADVANTAGE_CLIP}" \
    --negative_weight "${NEGATIVE_WEIGHT}" \
    --bottom_quantile "${BOTTOM_QUANTILE}" \
    --debug_generation_trace \
    --full_deterministic
done

echo "=== AAV mixed negative ablation finished $(date -Is) ==="
