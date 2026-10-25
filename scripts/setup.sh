#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r backend/requirements.txt -r ml/requirements.txt pytest httpx scikit-learn scipy
.venv/bin/python -m ml.download_model
(cd frontend && npm ci)
