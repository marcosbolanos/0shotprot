#!/usr/bin/env bash
set -euo pipefail

# Launch one EC2 GPU instance, run LGK single-mutant test for failed models
# (EvolutionaryScale esmc-300m and esm3-sm), sync outputs/logs to S3,
# then always shut down (instance-initiated shutdown behavior = terminate).

REGION="${AWS_REGION:-eu-west-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-p3.2xlarge}"
INSTANCE_PROFILE_NAME="${INSTANCE_PROFILE_NAME:-s3_access_for_ec2}"
SUBNET_ID="${SUBNET_ID:-}"
AMI_ID="${AMI_ID:-ami-0ebd4f1bfe8d7d4f9}"
TASK="${TASK:-LGK}"
PROXY_BATCH_SIZE="${PROXY_BATCH_SIZE:-4}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-5}"
WAIT_FOR_TERMINATION="${WAIT_FOR_TERMINATION:-0}"
ORACLES_S3_PREFIX="${ORACLES_S3_PREFIX:-oracles}"
HF_TOKEN="${HF_TOKEN:-}"

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

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
S3_PREFIX="${S3_PREFIX:-ec2_lgk_failed_esm_retry/${TIMESTAMP_UTC}}"
S3_URI="s3://${BUCKET}/${S3_PREFIX}"
ORACLES_S3_URI="s3://${BUCKET}/${ORACLES_S3_PREFIX}"

declare -a CANDIDATE_SUBNETS=()
if [[ -z "${SUBNET_ID:-}" ]]; then
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
MANIFEST_FILE="${TMP_DIR}/run_manifest.json"

echo "Preparing source tarball..."
tar \
  --exclude .git \
  --exclude .venv \
  --exclude .cache \
  --exclude .uv_cache \
  --exclude .pytest_cache \
  --exclude __pycache__ \
  --exclude out.zip2541wezr.part \
  --exclude out_cuda \
  --exclude outputs* \
  --exclude outputs \
  --exclude oracles \
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

if aws s3 ls "__ORACLES_S3_URI__/" >/dev/null 2>&1; then
  echo "Syncing oracle assets from __ORACLES_S3_URI__ to ${WORKDIR}/repo/oracles"
  aws s3 sync "__ORACLES_S3_URI__/" "${WORKDIR}/repo/oracles/"
fi

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="outputs/ec2_lgk_failed_esm_retry_${RUN_TS}"
mkdir -p "${RUN_DIR}/metrics"

cat > "${RUN_DIR}/metrics/monitor.sh" <<'MON'
#!/usr/bin/env bash
set -euo pipefail
OUT_CSV="$1"
INTERVAL="$2"
echo "timestamp_utc,cpu_percent,mem_used_mib,mem_total_mib,gpu_util_percent,gpu_mem_used_mib,gpu_mem_total_mib,python_proc_count" > "${OUT_CSV}"
read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
prev_idle=$((idle + iowait))
prev_total=$((user + nice + system + idle + iowait + irq + softirq + steal))
while true; do
  sleep "${INTERVAL}"
  read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
  idle_now=$((idle + iowait))
  total_now=$((user + nice + system + idle + iowait + irq + softirq + steal))
  total_delta=$((total_now - prev_total))
  idle_delta=$((idle_now - prev_idle))
  cpu_pct="0.00"
  if (( total_delta > 0 )); then
    cpu_pct=$(awk -v td="${total_delta}" -v id="${idle_delta}" 'BEGIN { printf "%.2f", 100 * (td - id) / td }')
  fi
  prev_total="${total_now}"
  prev_idle="${idle_now}"
  mem_total_kib=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
  mem_avail_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  mem_used_mib=$(awk -v t="${mem_total_kib}" -v a="${mem_avail_kib}" 'BEGIN { printf "%.2f", (t-a)/1024 }')
  mem_total_mib=$(awk -v t="${mem_total_kib}" 'BEGIN { printf "%.2f", t/1024 }')
  gpu_util=""
  gpu_mem_used=""
  gpu_mem_total=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)
    if [[ -n "${gpu_line}" ]]; then
      gpu_util=$(echo "${gpu_line}" | awk -F',' '{gsub(/ /,""); print $1}')
      gpu_mem_used=$(echo "${gpu_line}" | awk -F',' '{gsub(/ /,""); print $2}')
      gpu_mem_total=$(echo "${gpu_line}" | awk -F',' '{gsub(/ /,""); print $3}')
    fi
  fi
  py_count=$(ps -eo comm= | grep -E '^python(3(\.[0-9]+)?)?$' | wc -l | tr -d ' ' || true)
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "${ts},${cpu_pct},${mem_used_mib},${mem_total_mib},${gpu_util},${gpu_mem_used},${gpu_mem_total},${py_count}" >> "${OUT_CSV}"
done
MON
chmod +x "${RUN_DIR}/metrics/monitor.sh"
"${RUN_DIR}/metrics/monitor.sh" "${RUN_DIR}/metrics/system_metrics.csv" "__MONITOR_INTERVAL_SECONDS__" &
MONITOR_PID=$!
echo "${MONITOR_PID}" > "${RUN_DIR}/metrics/monitor.pid"

MODEL_STATUS_JSON="${RUN_DIR}/model_status.jsonl"
touch "${MODEL_STATUS_JSON}"

write_summaries() {
  python3 - "${RUN_DIR}/metrics/system_metrics.csv" "${MODEL_STATUS_JSON}" "${RUN_DIR}/metrics/per_model_metrics.json" "${RUN_DIR}/metrics/per_family_metrics.json" <<'PY'
import csv
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path

metrics_csv = Path(sys.argv[1])
status_jsonl = Path(sys.argv[2])
per_model_out = Path(sys.argv[3])
per_family_out = Path(sys.argv[4])

def parse_ts(ts: str) -> dt.datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return dt.datetime.fromisoformat(ts)

def series_stats(values):
    if not values:
        return {"n": 0}
    vals = sorted(values)
    p95_idx = int(math.floor(0.95 * (len(vals) - 1)))
    return {
        "n": len(vals),
        "avg": float(statistics.mean(vals)),
        "max": float(max(vals)),
        "p95": float(vals[p95_idx]),
    }

samples = []
if metrics_csv.exists():
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = (row.get("timestamp_utc") or "").strip()
            if not ts:
                continue
            def num(key):
                v = (row.get(key) or "").strip()
                if not v:
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None
            samples.append(
                {
                    "ts": parse_ts(ts),
                    "gpu_util_percent": num("gpu_util_percent"),
                    "gpu_mem_used_mib": num("gpu_mem_used_mib"),
                    "cpu_percent": num("cpu_percent"),
                    "python_proc_count": num("python_proc_count"),
                }
            )

records = []
if status_jsonl.exists():
    for line in status_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

per_model = []
for rec in records:
    started = parse_ts(rec["started_utc"])
    ended = parse_ts(rec["ended_utc"])
    model_samples = [s for s in samples if started <= s["ts"] <= ended]
    gpu_util = [s["gpu_util_percent"] for s in model_samples if s["gpu_util_percent"] is not None]
    gpu_mem = [s["gpu_mem_used_mib"] for s in model_samples if s["gpu_mem_used_mib"] is not None]
    cpu = [s["cpu_percent"] for s in model_samples if s["cpu_percent"] is not None]
    pcount = [s["python_proc_count"] for s in model_samples if s["python_proc_count"] is not None]
    runtime_s = (ended - started).total_seconds()
    family = "other"
    mid = rec.get("model_id", "")
    if "/esmc-" in mid:
        family = "esmc"
    elif "/esm3-" in mid:
        family = "esm3"
    elif "/esm2_" in mid:
        family = "esm2"
    per_model.append(
        {
            "model_id": mid,
            "family": family,
            "slug": rec.get("slug"),
            "started_utc": rec.get("started_utc"),
            "ended_utc": rec.get("ended_utc"),
            "runtime_seconds": runtime_s,
            "exit_code": rec.get("exit_code"),
            "gpu_util_percent": series_stats(gpu_util),
            "gpu_mem_used_mib": series_stats(gpu_mem),
            "cpu_percent": series_stats(cpu),
            "python_proc_count": series_stats(pcount),
        }
    )

per_model_out.write_text(json.dumps(per_model, indent=2), encoding="utf-8")

by_family = {}
for rec in per_model:
    fam = rec["family"]
    by_family.setdefault(
        fam,
        {
            "models": [],
            "runtime_seconds": [],
            "gpu_util_percent_avg": [],
            "gpu_mem_used_mib_avg": [],
            "exit_codes": [],
        },
    )
    by_family[fam]["models"].append(rec["model_id"])
    by_family[fam]["runtime_seconds"].append(rec["runtime_seconds"])
    if rec["gpu_util_percent"].get("n", 0) > 0:
        by_family[fam]["gpu_util_percent_avg"].append(rec["gpu_util_percent"]["avg"])
    if rec["gpu_mem_used_mib"].get("n", 0) > 0:
        by_family[fam]["gpu_mem_used_mib_avg"].append(rec["gpu_mem_used_mib"]["avg"])
    by_family[fam]["exit_codes"].append(rec["exit_code"])

family_summary = {}
for fam, data in by_family.items():
    family_summary[fam] = {
        "num_models": len(data["models"]),
        "models": data["models"],
        "runtime_seconds": series_stats(data["runtime_seconds"]),
        "gpu_util_percent_avg_over_models": series_stats(data["gpu_util_percent_avg"]),
        "gpu_mem_used_mib_avg_over_models": series_stats(data["gpu_mem_used_mib_avg"]),
        "exit_codes": data["exit_codes"],
        "all_success": all(code == 0 for code in data["exit_codes"]),
    }

per_family_out.write_text(json.dumps(family_summary, indent=2), encoding="utf-8")
PY
}

finalize_and_shutdown() {
  local rc="${1:-1}"
  if kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
    kill "${MONITOR_PID}" || true
    wait "${MONITOR_PID}" || true
  fi
  write_summaries || true
  aws s3 sync "${RUN_DIR}" "__S3_URI__/results/${RUN_TS}/" || true
  aws s3 cp /var/log/user-data.log "__S3_URI__/logs/user-data_${RUN_TS}.log" || true
  aws s3 cp /var/log/cloud-init-output.log "__S3_URI__/logs/cloud-init-output_${RUN_TS}.log" || true
  echo "${rc}" > "${RUN_DIR}/metrics/overall_exit_code.txt"
  shutdown -h now
}

run_model() {
  local model_id="$1"
  local slug="$2"
  local model_dir="${RUN_DIR}/${slug}"
  mkdir -p "${model_dir}"
  local started
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  /usr/bin/time -v uv run python -m prospero.runners.run_single_mutant_energy_test \
    --tasks "__TASK__" \
    --seed 1 \
    --surrogate_arch frozen_esm_flat_ridge_no_onehot \
    --ensemble_size 1 \
    --proxy_batch_size "__PROXY_BATCH_SIZE__" \
    --esm_model_name "${model_id}" \
    --output-json "${model_dir}/single_mutant_energy_compact.json" \
    --output-json-full "${model_dir}/single_mutant_energy_full.json" \
    --oracle-cache-json "outputs/0424_experiments/oracle_single_mutant_cache.json" \
    --results_dirpath "${model_dir}" \
    > "${model_dir}/run.log" 2>&1
  local rc=$?
  set -e
  local ended
  ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"model_id":"%s","slug":"%s","started_utc":"%s","ended_utc":"%s","exit_code":%d}\n' \
    "${model_id}" "${slug}" "${started}" "${ended}" "${rc}" >> "${MODEL_STATUS_JSON}"
  return "${rc}"
}

OVERALL_RC=0
run_model "EvolutionaryScale/esmc-300m-2024-12" "EvolutionaryScale__esmc-300m-2024-12" || OVERALL_RC=1
run_model "EvolutionaryScale/esm3-sm-open-v1" "EvolutionaryScale__esm3-sm-open-v1" || OVERALL_RC=1

finalize_and_shutdown "${OVERALL_RC}"
EOF_USERDATA

python3 - "${USER_DATA_FILE}" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
mapping = {
    "__S3_URI__": "S3_URI_PLACEHOLDER",
    "__ORACLES_S3_URI__": "ORACLES_S3_URI_PLACEHOLDER",
    "__HF_TOKEN__": "HF_TOKEN_PLACEHOLDER",
    "__TASK__": "TASK_PLACEHOLDER",
    "__PROXY_BATCH_SIZE__": "PROXY_BATCH_SIZE_PLACEHOLDER",
    "__MONITOR_INTERVAL_SECONDS__": "MONITOR_INTERVAL_PLACEHOLDER",
}
for key, value in mapping.items():
    text = text.replace(key, value)
path.write_text(text, encoding="utf-8")
PY

python3 - "${USER_DATA_FILE}" <<PY
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
text = text.replace("S3_URI_PLACEHOLDER", ${S3_URI@Q})
text = text.replace("ORACLES_S3_URI_PLACEHOLDER", ${ORACLES_S3_URI@Q})
text = text.replace("HF_TOKEN_PLACEHOLDER", ${HF_TOKEN@Q})
text = text.replace("TASK_PLACEHOLDER", ${TASK@Q})
text = text.replace("PROXY_BATCH_SIZE_PLACEHOLDER", ${PROXY_BATCH_SIZE@Q})
text = text.replace("MONITOR_INTERVAL_PLACEHOLDER", ${MONITOR_INTERVAL_SECONDS@Q})
path.write_text(text, encoding="utf-8")
PY

chmod +x "${USER_DATA_FILE}"

echo "Launching ${INSTANCE_TYPE} in ${REGION}..."
INSTANCE_ID=""
SUBNET_ID_SELECTED=""
for subnet in "${CANDIDATE_SUBNETS[@]}"; do
  echo "Trying subnet ${subnet}..."
  set +e
  LAUNCH_OUTPUT="$(
    aws ec2 run-instances \
      --region "${REGION}" \
      --image-id "${AMI_ID}" \
      --instance-type "${INSTANCE_TYPE}" \
      --iam-instance-profile "Name=${INSTANCE_PROFILE_NAME}" \
      --subnet-id "${subnet}" \
      --associate-public-ip-address \
      --instance-initiated-shutdown-behavior terminate \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=prospero-lgk-failed-esm-retry}]" \
      --user-data "file://${USER_DATA_FILE}" \
      --query 'Instances[0].InstanceId' \
      --output text 2>&1
  )"
  LAUNCH_RC=$?
  set -e
  if [[ ${LAUNCH_RC} -eq 0 ]]; then
    INSTANCE_ID="${LAUNCH_OUTPUT}"
    SUBNET_ID_SELECTED="${subnet}"
    break
  else
    echo "Launch failed in ${subnet}: ${LAUNCH_OUTPUT}"
  fi
done

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Failed to launch ${INSTANCE_TYPE} in any candidate subnet: ${CANDIDATE_SUBNETS[*]}"
  exit 1
fi

cat > "${MANIFEST_FILE}" <<EOF
{
  "region": "${REGION}",
  "account_id": "${ACCOUNT_ID}",
  "instance_id": "${INSTANCE_ID}",
  "instance_type": "${INSTANCE_TYPE}",
  "ami_id": "${AMI_ID}",
  "subnet_id": "${SUBNET_ID_SELECTED}",
  "bucket": "${BUCKET}",
  "s3_prefix": "${S3_PREFIX}",
  "s3_uri": "${S3_URI}",
  "oracles_s3_prefix": "${ORACLES_S3_PREFIX}",
  "oracles_s3_uri": "${ORACLES_S3_URI}",
  "task": "${TASK}",
  "proxy_batch_size": ${PROXY_BATCH_SIZE},
  "hf_token_provided": $([[ -n "${HF_TOKEN}" ]] && echo "true" || echo "false"),
  "models": [
    "EvolutionaryScale/esmc-300m-2024-12",
    "EvolutionaryScale/esm3-sm-open-v1"
  ],
  "submitted_at_utc": "${TIMESTAMP_UTC}"
}
EOF

aws s3 cp "${MANIFEST_FILE}" "${S3_URI}/run_manifest.json"

if [[ "${WAIT_FOR_TERMINATION}" == "1" ]]; then
  echo "Waiting for instance to terminate..."
  while true; do
    INSTANCE_STATE="$(aws ec2 describe-instances --region "${REGION}" --instance-ids "${INSTANCE_ID}" --query 'Reservations[0].Instances[0].State.Name' --output text)"
    echo "Instance ${INSTANCE_ID} state=${INSTANCE_STATE}"
    if [[ "${INSTANCE_STATE}" == "terminated" ]]; then
      break
    fi
    sleep 30
  done
  echo "Instance terminated."
else
  echo "Launched ${INSTANCE_ID}; skipping wait (WAIT_FOR_TERMINATION=${WAIT_FOR_TERMINATION})."
fi

echo "Run manifest: ${S3_URI}/run_manifest.json"
echo "Results root: ${S3_URI}/results/"
echo "Logs root: ${S3_URI}/logs/"
