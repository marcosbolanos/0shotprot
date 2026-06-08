#!/usr/bin/env bash
set -euo pipefail
cd /home/lamsade/mbolanos/code/ProSpero
CACHE="${PROTEINGYM_CACHE:-external/ProteinGym_data}"
VERSION="${PROTEINGYM_VERSION:-v1.3}"
mkdir -p "$CACHE"
files=(
  DMS_ProteinGym_substitutions.zip
  cv_folds_singles_substitutions.zip
  cv_folds_multiples_substitutions.zip
)
for f in "${files[@]}"; do
  url="https://marks.hms.harvard.edu/proteingym/ProteinGym_${VERSION}/${f}"
  dest="$CACHE/$f"
  stem="${f%.zip}"
  if [[ -d "$CACHE/$stem" ]]; then
    echo "skip extracted $stem"
    continue
  fi
  if [[ ! -s "$dest" ]]; then
    echo "download $url"
    curl -k -L --retry 5 --retry-delay 10 -o "$dest" "$url"
  else
    echo "skip existing zip $dest"
  fi
  echo "extract $dest"
  uv run python - <<PY
from pathlib import Path
import zipfile
p=Path('$dest')
out=Path('$CACHE')
with zipfile.ZipFile(p) as z:
    z.extractall(out)
print('extracted', p)
PY
done
