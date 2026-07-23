#!/bin/bash
set -euo pipefail

file_id="1UiKVdnNDlqeHMHvc12qd6kbw5QltQm9D"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
download_dir="$(mktemp -d)"
archive_path="$download_dir/oracles.zip"
trap 'rm -rf "$download_dir"' EXIT

uv run --with gdown gdown "$file_id" -O "$archive_path"
unzip -q "$archive_path" -d "$repo_root"

if [[ ! -d "$repo_root/oracles" ]]; then
  echo "The downloaded archive did not create the expected oracles directory." >&2
  exit 1
fi

echo "Oracles are available at $repo_root/oracles"
