#!/usr/bin/env bash
# Verify an already-installed development environment from any directory.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then
  voxguard_python=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then
  voxguard_python=.venv/Scripts/python.exe
else
  echo 'Missing .venv. Run scripts/setup.sh or scripts/setup.ps1 first.' >&2
  exit 1
fi
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
"$voxguard_python" -m pip check
"$voxguard_python" -m ml.download_model
"$voxguard_python" -m pytest tests research ml experiments -q
(cd frontend && npm test -- --run && npm run build)
