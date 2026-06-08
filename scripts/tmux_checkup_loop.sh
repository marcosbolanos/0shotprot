#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tmux_checkup_loop.sh [--target PANE] [--reply-target PANE] [--delay MINUTES] [--poll-seconds N]

Behavior:
  - Waits --delay minutes.
  - Sends the checkup message to --target.
  - Instructs target Codex to reply via tmux_send_message.sh to --reply-target with:
      STATUS=SUCCESS CHECKUP_ID=<id>
    or
      STATUS=FAILED CHECKUP_ID=<id>
  - If SUCCESS is observed, loop exits.
  - If FAILED is observed, schedules another checkup after --delay minutes (repeats forever).
EOF
}

target="cx:2.2"
reply_target="cx:2.4"
delay_minutes=30
poll_seconds=20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sender_script="${SCRIPT_DIR}/tmux_send_message.sh"
log_file="/tmp/tmux_checkup_loop.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="${2:-}"
      shift 2
      ;;
    --reply-target)
      reply_target="${2:-}"
      shift 2
      ;;
    --delay)
      delay_minutes="${2:-}"
      shift 2
      ;;
    --poll-seconds)
      poll_seconds="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$delay_minutes" =~ ^[0-9]+$ ]]; then
  echo "--delay must be a non-negative integer number of minutes." >&2
  exit 1
fi

if ! [[ "$poll_seconds" =~ ^[0-9]+$ ]] || [[ "$poll_seconds" -lt 1 ]]; then
  echo "--poll-seconds must be a positive integer." >&2
  exit 1
fi

if [[ ! -x "$sender_script" ]]; then
  echo "Required sender script is missing or not executable: $sender_script" >&2
  exit 1
fi

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$log_file"
}

send_checkup() {
  local checkup_id="$1"
  local checkup_msg="hey, did the job finish properly ? If not, you know what to do, keep retrying"
  local instruction_msg

  instruction_msg="CHECKUP_ID=${checkup_id}. If run is good, reply using: ${sender_script} --target ${reply_target} --delay 0 --message \"STATUS=SUCCESS CHECKUP_ID=${checkup_id}\". If run failed, kill/fix/retry now and then reply using: ${sender_script} --target ${reply_target} --delay 0 --message \"STATUS=FAILED CHECKUP_ID=${checkup_id}\"."

  "$sender_script" --target "$target" --delay 0 --message "$checkup_msg" --enters 2
  "$sender_script" --target "$target" --delay 0 --message "$instruction_msg" --enters 2
  log "Sent checkup CHECKUP_ID=${checkup_id} to ${target}"
}

wait_for_status() {
  local checkup_id="$1"
  local success_pattern="STATUS=SUCCESS CHECKUP_ID=${checkup_id}"
  local failed_pattern="STATUS=FAILED CHECKUP_ID=${checkup_id}"

  while true; do
    local pane_tail
    pane_tail="$(tmux capture-pane -p -t "$reply_target" | tail -n 400)"
    if grep -Fq "$success_pattern" <<<"$pane_tail"; then
      log "Received SUCCESS for CHECKUP_ID=${checkup_id} on ${reply_target}"
      return 0
    fi
    if grep -Fq "$failed_pattern" <<<"$pane_tail"; then
      log "Received FAILED for CHECKUP_ID=${checkup_id} on ${reply_target}"
      return 1
    fi
    sleep "$poll_seconds"
  done
}

log "Loop started target=${target} reply_target=${reply_target} delay=${delay_minutes}m poll=${poll_seconds}s"

while true; do
  sleep "$((delay_minutes * 60))"
  checkup_id="$(date +%s)"
  send_checkup "$checkup_id"
  if wait_for_status "$checkup_id"; then
    log "Stopping loop after SUCCESS."
    exit 0
  fi
  log "Scheduling next checkup in ${delay_minutes} minutes due to FAILED status."
done
