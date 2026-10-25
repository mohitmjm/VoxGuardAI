#!/usr/bin/env bash
# Native Python deployment; no Docker. Choose Python 3.11 or 3.12 on the host.
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Select Python 3.11 or 3.12 on the hosting provider"'
for voxguard_tool in ffmpeg ffprobe; do
  if ! command -v "$voxguard_tool" >/dev/null; then
    echo "$voxguard_tool is required on the host. Enable it using the provider-supported system package mechanism." >&2
    exit 1
  fi
done
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend/requirements.txt -r ml/requirements.txt
python -m ml.download_model
python -m pip check
python scripts/preflight.py
