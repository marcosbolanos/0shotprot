#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tmux_send_message.sh --message TEXT [--delay MINUTES] [--target PANE] [--enters N]

Options:
  --message TEXT   Message to send (required).
  --delay MINUTES Delay before sending, in minutes (default: 0).
  --target PANE   tmux target pane (default: cx:2.2).
  --enters N      Number of Enter keystrokes after message (default: 2).
  --help          Show this help.
EOF
}

message=""
delay_minutes=0
target="cx:2.2"
enters=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --message)
      message="${2:-}"
      shift 2
      ;;
    --delay)
      delay_minutes="${2:-}"
      shift 2
      ;;
    --target)
      target="${2:-}"
      shift 2
      ;;
    --enters)
      enters="${2:-}"
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

if [[ -z "$message" ]]; then
  echo "--message is required." >&2
  usage >&2
  exit 1
fi

if ! [[ "$delay_minutes" =~ ^[0-9]+$ ]]; then
  echo "--delay must be a non-negative integer number of minutes." >&2
  exit 1
fi

if ! [[ "$enters" =~ ^[0-9]+$ ]]; then
  echo "--enters must be a non-negative integer." >&2
  exit 1
fi

sleep_seconds=$((delay_minutes * 60))
if (( sleep_seconds > 0 )); then
  sleep "$sleep_seconds"
fi

tmux send-keys -t "$target" -l "$message"
for ((i = 0; i < enters; i++)); do
  tmux send-keys -t "$target" C-m
  sleep 0.15
done
