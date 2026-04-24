#!/usr/bin/env bash
set -euo pipefail

# Launches one EC2 GPU instance, runs D_SHIFT one_hot_ridge variable-k benchmark,
# captures runtime/system usage, syncs results to S3, then auto-terminates.

REGION="${AWS_REGION:-eu-west-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
INSTANCE_PROFILE_NAME="${INSTANCE_PROFILE_NAME:-s3_access_for_ec2}"
TASK="${TASK:-D_SHIFT}"
SURROGATE_ARCH="${SURROGATE_ARCH:-one_hot_ridge}"
N_SAMPLES="${N_SAMPLES:-8,16,32,64,128}"
SEEDS="${SEEDS:-1,2,3,4,5}"
N_ITERS="${N_ITERS:-10}"
MAX_WORKERS="${MAX_WORKERS:-5}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-5}"
AMI_PARAM="${AMI_PARAM:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${BUCKET:-prospero-${ACCOUNT_ID}-${REGION}}"
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
S3_PREFIX="${S3_PREFIX:-ec2_variable_k_dshift_one_hot_ridge/${TIMESTAMP_UTC}}"
S3_URI="s3://${BUCKET}/${S3_PREFIX}"

AMI_ID="$(aws ssm get-parameter --region "${REGION}" --name "${AMI_PARAM}" --query 'Parameter.Value' --output text)"

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
  echo "No default subnet found in region ${REGION}. Set SUBNET_ID explicitly."
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

SRC_TARBALL="${TMP_DIR}/0shotprot_source_${TIMESTAMP_UTC}.tar.gz"
USER_DATA_FILE="${TMP_DIR}/user_data.sh"
MANIFEST_FILE="${TMP_DIR}/run_manifest.json"

echo "Preparing source tarball..."
tar \
  --exclude .git \
  --exclude .venv \
  --exclude .pytest_cache \
  --exclude __pycache__ \
  --exclude outputs \
  -czf "${SRC_TARBALL}" .

echo "Ensuring S3 bucket exists: ${BUCKET}"
if ! aws s3api head-bucket --bucket "${BUCKET}" >/dev/null 2>&1; then
  aws s3api create-bucket \
    --region "${REGION}" \
    --bucket "${BUCKET}" \
    --create-bucket-configuration "LocationConstraint=${REGION}" >/dev/null
fi

echo "Uploading source bundle to ${S3_URI}/source/"
aws s3 cp "${SRC_TARBALL}" "${S3_URI}/source/0shotprot_source.tar.gz"

cat > "${USER_DATA_FILE}" <<'EOF_USERDATA'
#!/usr/bin/env bash
set -euo pipefail
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

export DEBIAN_FRONTEND=noninteractive

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

WORKDIR="/opt/prospero"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

aws s3 cp "__S3_URI__/source/0shotprot_source.tar.gz" "${WORKDIR}/source.tar.gz"
mkdir -p repo
tar -xzf source.tar.gz -C repo
cd repo

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="outputs/ec2_run_${RUN_TS}"
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

  py_count=$(ps -eo comm= | grep -E '^python(3(\.[0-9]+)?)?$' | wc -l | tr -d ' ')
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "${ts},${cpu_pct},${mem_used_mib},${mem_total_mib},${gpu_util},${gpu_mem_used},${gpu_mem_total},${py_count}" >> "${OUT_CSV}"
done
MON

chmod +x "${RUN_DIR}/metrics/monitor.sh"

"${RUN_DIR}/metrics/monitor.sh" "${RUN_DIR}/metrics/system_metrics.csv" "__MONITOR_INTERVAL_SECONDS__" &
MONITOR_PID=$!
echo "${MONITOR_PID}" > "${RUN_DIR}/metrics/monitor.pid"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L > "${RUN_DIR}/metrics/nvidia_smi_devices.txt" || true
  nvidia-smi > "${RUN_DIR}/metrics/nvidia_smi_snapshot.txt" || true
fi

START_EPOCH=$(date +%s)
echo "${START_EPOCH}" > "${RUN_DIR}/metrics/start_epoch.txt"

set +e
/usr/bin/time -v uv run python src/prospero/runners/run_variable_k.py "${RUN_DIR}" \
  --task "__TASK__" \
  --surrogate-arch "__SURROGATE_ARCH__" \
  --n-samples "__N_SAMPLES__" \
  --seeds "__SEEDS__" \
  --n-iters "__N_ITERS__" \
  --max-workers "__MAX_WORKERS__" \
  2>&1 | tee "${RUN_DIR}/run_variable_k.stdout.log"
RUN_EXIT=${PIPESTATUS[0]}
set -e

END_EPOCH=$(date +%s)
echo "${END_EPOCH}" > "${RUN_DIR}/metrics/end_epoch.txt"
echo "$((END_EPOCH-START_EPOCH))" > "${RUN_DIR}/metrics/runtime_seconds.txt"
echo "${RUN_EXIT}" > "${RUN_DIR}/metrics/exit_code.txt"

if kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
  kill "${MONITOR_PID}" || true
  wait "${MONITOR_PID}" || true
fi

python3 - <<'PY' "${RUN_DIR}/metrics/system_metrics.csv" "${RUN_DIR}/metrics/summary.txt"
import csv
import math
import statistics
import sys

src, out = sys.argv[1], sys.argv[2]
rows = []
with open(src, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def numbers(key):
    vals = []
    for r in rows:
        v = (r.get(key) or "").strip()
        if not v:
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    return vals

cpu = numbers("cpu_percent")
mem = numbers("mem_used_mib")
gpu_u = numbers("gpu_util_percent")
gpu_m = numbers("gpu_mem_used_mib")
py_c = numbers("python_proc_count")

def line(name, vals):
    if not vals:
        return f"{name}: n=0"
    return (
        f"{name}: n={len(vals)} avg={statistics.mean(vals):.2f} "
        f"p95={vals[math.floor(0.95*(len(vals)-1))]:.2f} max={max(vals):.2f}"
    )

for arr in (cpu, mem, gpu_u, gpu_m, py_c):
    arr.sort()

with open(out, "w", encoding="utf-8") as w:
    w.write(line("cpu_percent", cpu) + "\n")
    w.write(line("mem_used_mib", mem) + "\n")
    w.write(line("gpu_util_percent", gpu_u) + "\n")
    w.write(line("gpu_mem_used_mib", gpu_m) + "\n")
    w.write(line("python_proc_count", py_c) + "\n")
PY

aws s3 sync "${RUN_DIR}" "__S3_URI__/results/${RUN_TS}/"
aws s3 cp /var/log/user-data.log "__S3_URI__/logs/user-data_${RUN_TS}.log" || true
aws s3 cp /var/log/cloud-init-output.log "__S3_URI__/logs/cloud-init-output_${RUN_TS}.log" || true

shutdown -h now
EOF_USERDATA

python3 - "${USER_DATA_FILE}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
mapping = {
    "__S3_URI__": """S3_URI_PLACEHOLDER""",
    "__TASK__": """TASK_PLACEHOLDER""",
    "__SURROGATE_ARCH__": """SURROGATE_ARCH_PLACEHOLDER""",
    "__N_SAMPLES__": """N_SAMPLES_PLACEHOLDER""",
    "__SEEDS__": """SEEDS_PLACEHOLDER""",
    "__N_ITERS__": """N_ITERS_PLACEHOLDER""",
    "__MAX_WORKERS__": """MAX_WORKERS_PLACEHOLDER""",
    "__MONITOR_INTERVAL_SECONDS__": """MONITOR_INTERVAL_PLACEHOLDER""",
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
text = text.replace("TASK_PLACEHOLDER", ${TASK@Q})
text = text.replace("SURROGATE_ARCH_PLACEHOLDER", ${SURROGATE_ARCH@Q})
text = text.replace("N_SAMPLES_PLACEHOLDER", ${N_SAMPLES@Q})
text = text.replace("SEEDS_PLACEHOLDER", ${SEEDS@Q})
text = text.replace("N_ITERS_PLACEHOLDER", ${N_ITERS@Q})
text = text.replace("MAX_WORKERS_PLACEHOLDER", ${MAX_WORKERS@Q})
text = text.replace("MONITOR_INTERVAL_PLACEHOLDER", ${MONITOR_INTERVAL_SECONDS@Q})
path.write_text(text, encoding="utf-8")
PY

chmod +x "${USER_DATA_FILE}"

echo "Launching ${INSTANCE_TYPE} in ${REGION}..."
INSTANCE_ID=""
SUBNET_ID=""
for subnet in "${CANDIDATE_SUBNETS[@]}"; do
  echo "Trying subnet ${subnet}..."
  LAUNCH_OUTPUT="$(
    aws ec2 run-instances \
      --region "${REGION}" \
      --image-id "${AMI_ID}" \
      --instance-type "${INSTANCE_TYPE}" \
      --iam-instance-profile "Name=${INSTANCE_PROFILE_NAME}" \
      --subnet-id "${subnet}" \
      --associate-public-ip-address \
      --instance-initiated-shutdown-behavior terminate \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=prospero-variable-k-dshift-onehot}]" \
      --user-data "file://${USER_DATA_FILE}" \
      --query 'Instances[0].InstanceId' \
      --output text 2>&1
  )"
  if [[ $? -eq 0 ]]; then
    INSTANCE_ID="${LAUNCH_OUTPUT}"
    SUBNET_ID="${subnet}"
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
  "subnet_id": "${SUBNET_ID}",
  "bucket": "${BUCKET}",
  "s3_prefix": "${S3_PREFIX}",
  "s3_uri": "${S3_URI}",
  "task": "${TASK}",
  "surrogate_arch": "${SURROGATE_ARCH}",
  "n_samples": "${N_SAMPLES}",
  "seeds": "${SEEDS}",
  "n_iters": ${N_ITERS},
  "max_workers": ${MAX_WORKERS},
  "monitor_interval_seconds": ${MONITOR_INTERVAL_SECONDS},
  "submitted_at_utc": "${TIMESTAMP_UTC}"
}
EOF

aws s3 cp "${MANIFEST_FILE}" "${S3_URI}/run_manifest.json"

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
echo "Run manifest: ${S3_URI}/run_manifest.json"
echo "Results root: ${S3_URI}/results/"
echo "Logs root: ${S3_URI}/logs/"
