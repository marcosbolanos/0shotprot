#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-eu-west-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-p5.4xlarge}"
INSTANCE_PROFILE_NAME="${INSTANCE_PROFILE_NAME:-s3_access_for_ec2}"
SUBNET_ID="${SUBNET_ID:-}"
AMI_ID="${AMI_ID:-ami-0ebd4f1bfe8d7d4f9}"
TASK="${TASK:-LGK}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-5}"
WAIT_FOR_TERMINATION="${WAIT_FOR_TERMINATION:-0}"
HF_TOKEN="${HF_TOKEN:-}"
ROOT_VOLUME_GB="${ROOT_VOLUME_GB:-500}"

if [[ -z "${BUCKET:-}" ]]; then
  BUCKET="$(
    aws s3api list-buckets \
      --query "Buckets[?contains(Name, 'prospero') || contains(Name, 'propsero')].Name | [0]" \
      --output text
  )"
fi
if [[ -z "${BUCKET}" || "${BUCKET}" == "None" ]]; then
  echo "Could not auto-discover S3 bucket containing prospero/propsero. Set BUCKET explicitly."
  exit 1
fi

TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
S3_PREFIX="${S3_PREFIX:-ec2_lgk_structure_feature_ablation/${TIMESTAMP_UTC}}"
S3_URI="s3://${BUCKET}/${S3_PREFIX}"

declare -a CANDIDATE_SUBNETS=()
if [[ -z "${SUBNET_ID:-}" ]]; then
  # First try AWS automatic placement (no subnet/AZ pinning), then fallback by subnet.
  CANDIDATE_SUBNETS+=("__AUTO__")
  mapfile -t OFFERED_AZS < <(
    aws ec2 describe-instance-type-offerings \
      --region "${REGION}" \
      --location-type availability-zone \
      --filters "Name=instance-type,Values=${INSTANCE_TYPE}" \
      --query 'InstanceTypeOfferings[].Location' \
      --output text | tr '\t' '\n'
  )

  while read -r sid saz; do
    if printf '%s\n' "${OFFERED_AZS[@]}" | grep -qx "${saz}"; then
      CANDIDATE_SUBNETS+=("${sid}")
    fi
  done < <(
    aws ec2 describe-subnets \
      --region "${REGION}" \
      --filters Name=default-for-az,Values=true \
      --query 'Subnets[].[SubnetId,AvailabilityZone]' \
      --output text
  )
else
  CANDIDATE_SUBNETS+=("${SUBNET_ID}")
fi

if [[ "${#CANDIDATE_SUBNETS[@]}" -eq 0 ]]; then
  echo "No candidate subnet found in region ${REGION}. Set SUBNET_ID explicitly."
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

SRC_TARBALL="${TMP_DIR}/prospero_source_${TIMESTAMP_UTC}.tar.gz"
USER_DATA_FILE="${TMP_DIR}/user_data.sh"


echo "Preparing source tarball..."
tar \
  --exclude './.git' \
  --exclude './.venv' \
  --exclude './.goodvenv' \
  --exclude './.badvenv' \
  --exclude './.cache' \
  --exclude './.uv_cache' \
  --exclude './.pytest_cache' \
  --exclude './.ruff_cache' \
  --exclude './.mypy_cache' \
  --exclude './.idea' \
  --exclude './.vscode' \
  --exclude '__pycache__' \
  --exclude './outputs' \
  --exclude './outputs*' \
  --exclude './oracles' \
  -czf "${SRC_TARBALL}" .

echo "Ensuring S3 bucket exists: ${BUCKET}"
if ! aws s3api head-bucket --bucket "${BUCKET}" >/dev/null 2>&1; then
  aws s3api create-bucket \
    --region "${REGION}" \
    --bucket "${BUCKET}" \
    --create-bucket-configuration "LocationConstraint=${REGION}" >/dev/null
fi

echo "Uploading source bundle to ${S3_URI}/source/"
aws s3 cp "${SRC_TARBALL}" "${S3_URI}/source/prospero_source.tar.gz"

cat > "${USER_DATA_FILE}" <<'EOF_USERDATA'
#!/usr/bin/env bash
set -euo pipefail
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

export DEBIAN_FRONTEND=noninteractive
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_TELEMETRY=1

retry_apt() {
  local max_attempts=20
  local attempt=1
  local sleep_seconds=15
  while (( attempt <= max_attempts )); do
    if apt-get update && apt-get install -y awscli curl git jq build-essential time procps; then
      return 0
    fi
    echo "apt retry ${attempt}/${max_attempts} failed; waiting ${sleep_seconds}s"
    sleep "${sleep_seconds}"
    attempt=$((attempt + 1))
  done
  echo "apt bootstrap failed after ${max_attempts} attempts"
  return 1
}

retry_apt

if [[ ! -x /root/.local/bin/uv ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

if [[ -n "__HF_TOKEN__" ]]; then
  export HF_TOKEN="__HF_TOKEN__"
fi

WORKDIR="/opt/prospero"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

aws s3 cp "__S3_URI__/source/prospero_source.tar.gz" "${WORKDIR}/source.tar.gz"
mkdir -p repo

tar -xzf source.tar.gz -C repo
cd repo

echo "Syncing deterministic project environment from uv.lock"
uv sync --frozen

echo "Enforcing ESM runtime surface (esm==3.2.3, no fair-esm collision)"
uv pip uninstall --python .venv/bin/python fair-esm || true
uv pip install --python .venv/bin/python --reinstall "esm==3.2.3"

echo "Verifying esm.pretrained API surface"
.venv/bin/python - <<'PY'
import importlib
import importlib.metadata as md
import sys

try:
    import esm
    p = importlib.import_module("esm.pretrained")
except Exception as exc:
    print(f"[esm-check] import failure: {exc!r}")
    raise SystemExit(11)

print(f"[esm-check] esm.__file__={getattr(esm, '__file__', None)}")
print(f"[esm-check] esm.pretrained.__file__={getattr(p, '__file__', None)}")
for dist in ("esm", "fair-esm"):
    try:
        print(f"[esm-check] {dist}=={md.version(dist)}")
    except Exception:
        print(f"[esm-check] {dist}=not-installed")

if not hasattr(p, "load_local_model"):
    print("[esm-check] missing required symbol: esm.pretrained.load_local_model")
    raise SystemExit(12)
PY

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="outputs/0424_experiments/ec2_lgk_structure_feature_ablation_${RUN_TS}"
mkdir -p "${RUN_DIR}/metrics"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L > "${RUN_DIR}/metrics/nvidia_smi_devices.txt" || true
  nvidia-smi > "${RUN_DIR}/metrics/nvidia_smi_snapshot.txt" || true
fi

echo "Re-checking esm.pretrained API surface right before run"
.venv/bin/python - <<'PY'
import importlib
p = importlib.import_module("esm.pretrained")
print(f"[esm-pre-run] esm.pretrained.__file__={getattr(p, '__file__', None)}")
if not hasattr(p, "load_local_model"):
    raise SystemExit("[esm-pre-run] missing required symbol: esm.pretrained.load_local_model")
PY

set +e
LOG_FILE="${RUN_DIR}/run_lgk_structure_feature_ablation.stdout.log"
CHECKPOINT_DIR="outputs/0424_experiments/lgk_structure_feature_ablation_checkpoints"
CHECKPOINT_S3_URI="__S3_URI__/results/checkpoints/"
SYNC_INTERVAL_SECONDS=90

echo "[launcher] extract phase started at $(date -u --iso-8601=seconds)" > "${LOG_FILE}"
/usr/bin/time -v -o "${RUN_DIR}/metrics/extract_time.txt" \
  .venv/bin/python src/prospero/runners/run_lgk_structure_token_ridge.py \
    --task __TASK__ \
    --esm-model-name EvolutionaryScale/esm3-sm-open-v1 \
    --device cuda \
    --mode extract \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-json outputs/0424_experiments/lgk_structure_feature_ablation_compact.json \
    --output-json-full outputs/0424_experiments/lgk_structure_feature_ablation_full.json \
    >> "${LOG_FILE}" 2>&1 &
EXTRACT_PID=$!

while kill -0 "${EXTRACT_PID}" 2>/dev/null; do
  aws s3 sync "${CHECKPOINT_DIR}/" "${CHECKPOINT_S3_URI}" >/dev/null 2>&1 || true
  sleep "${SYNC_INTERVAL_SECONDS}"
done
wait "${EXTRACT_PID}"
EXTRACT_EXIT=$?

echo "[launcher] extract phase exited with code ${EXTRACT_EXIT}" >> "${LOG_FILE}"
aws s3 sync "${CHECKPOINT_DIR}/" "${CHECKPOINT_S3_URI}" >/dev/null 2>&1 || true

if [[ "${EXTRACT_EXIT}" -eq 0 && -f "${CHECKPOINT_DIR}/extraction_complete.json" ]]; then
  echo "[launcher] waiting for checkpoint sync completion before ridge" >> "${LOG_FILE}"
  aws s3 sync "${CHECKPOINT_DIR}/" "${CHECKPOINT_S3_URI}" >/dev/null
  if ! aws s3 ls "${CHECKPOINT_S3_URI}extraction_complete.json" >/dev/null 2>&1; then
    echo "[launcher] checkpoint sync verification failed (missing extraction_complete.json in S3)" >> "${LOG_FILE}"
    EXTRACT_EXIT=98
  fi
fi

if [[ "${EXTRACT_EXIT}" -ne 0 ]]; then
  RUN_EXIT="${EXTRACT_EXIT}"
else
  echo "[launcher] fit phase started at $(date -u --iso-8601=seconds)" >> "${LOG_FILE}"
  /usr/bin/time -v -o "${RUN_DIR}/metrics/fit_time.txt" \
    .venv/bin/python src/prospero/runners/run_lgk_structure_token_ridge.py \
      --task __TASK__ \
      --esm-model-name EvolutionaryScale/esm3-sm-open-v1 \
      --device cuda \
      --mode fit \
      --checkpoint-dir "${CHECKPOINT_DIR}" \
      --output-json outputs/0424_experiments/lgk_structure_feature_ablation_compact.json \
      --output-json-full outputs/0424_experiments/lgk_structure_feature_ablation_full.json \
      >> "${LOG_FILE}" 2>&1
  RUN_EXIT=$?
fi
set -e

echo "${RUN_EXIT}" > "${RUN_DIR}/metrics/exit_code.txt"

echo "Syncing outputs to __S3_URI__/results/"
aws s3 sync outputs/0424_experiments/ "__S3_URI__/results/outputs_0424_experiments/"
aws s3 cp /var/log/user-data.log "__S3_URI__/results/user-data.log" || true

shutdown -h now
EOF_USERDATA

sed -i "s|__S3_URI__|${S3_URI}|g" "${USER_DATA_FILE}"
sed -i "s|__TASK__|${TASK}|g" "${USER_DATA_FILE}"
SAFE_HF_TOKEN="${HF_TOKEN//|/\\|}"
sed -i "s|__HF_TOKEN__|${SAFE_HF_TOKEN}|g" "${USER_DATA_FILE}"

INSTANCE_ID=""
USED_SUBNET=""
for subnet in "${CANDIDATE_SUBNETS[@]}"; do
  if [[ "${subnet}" == "__AUTO__" ]]; then
    echo "Launching instance with automatic subnet/AZ placement..."
    SUBNET_ARGS=()
  else
    echo "Launching instance in subnet ${subnet}..."
    SUBNET_ARGS=(--subnet-id "${subnet}")
  fi
  set +e
  RUN_OUTPUT="$(aws ec2 run-instances \
    --region "${REGION}" \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --iam-instance-profile Name="${INSTANCE_PROFILE_NAME}" \
    "${SUBNET_ARGS[@]}" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${ROOT_VOLUME_GB},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=prospero-lgk-structure-feature-ablation}]" \
    --user-data "file://${USER_DATA_FILE}" \
    --query 'Instances[0].InstanceId' \
    --output text 2>&1)"
  STATUS=$?
  set -e
  if [[ ${STATUS} -eq 0 && -n "${RUN_OUTPUT}" && "${RUN_OUTPUT}" != "None" ]]; then
    INSTANCE_ID="${RUN_OUTPUT}"
    if [[ "${subnet}" == "__AUTO__" ]]; then
      USED_SUBNET="auto"
    else
      USED_SUBNET="${subnet}"
    fi
    break
  fi
  if [[ "${subnet}" == "__AUTO__" ]]; then
    echo "Launch failed with automatic subnet/AZ placement: ${RUN_OUTPUT}"
  else
    echo "Launch failed in subnet ${subnet}: ${RUN_OUTPUT}"
  fi
done

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Failed to launch EC2 instance in candidate subnets."
  exit 1
fi

echo "Launched instance: ${INSTANCE_ID}"
echo "Region: ${REGION}"
echo "Subnet: ${USED_SUBNET}"
echo "S3 results prefix: ${S3_URI}/results/"

aws ec2 wait instance-running --region "${REGION}" --instance-ids "${INSTANCE_ID}"

echo "Instance is running."
if [[ "${WAIT_FOR_TERMINATION}" == "1" ]]; then
  echo "Waiting for termination..."
  aws ec2 wait instance-terminated --region "${REGION}" --instance-ids "${INSTANCE_ID}"
  echo "Instance terminated."
fi
