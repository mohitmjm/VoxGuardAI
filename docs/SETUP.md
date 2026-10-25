# Local setup

Prerequisites: Python 3.11 or 3.12, Node.js 20.19+ (or 22.12+), and FFmpeg with ffprobe on PATH. No API key, database, or paid service is required. Run commands from the repository root.

## Linux / macOS

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -r backend/requirements.txt
python -m pip install -r ml/requirements.txt
python -m ml.download_model
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```sh
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

## Windows PowerShell

Install Python, Node.js, and an FFmpeg build; verify `ffmpeg -version` and `ffprobe -version` first.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m pip install -r ml/requirements.txt
.\.venv\Scripts\python.exe -m ml.download_model
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal, run `cd frontend`, `npm ci`, then `npm run dev`.

The Vite development proxy forwards `/api` to port 8000. For separate deployment, configure `VITE_API_BASE_URL` at frontend build time and the backend's allowed CORS origins. Do not put secrets in frontend variables. Use HTTPS and review upload/resource limits before any public exposure. Public deployment is outside this setup.

## Verification

```sh
python -m pip install pytest httpx
python -m pytest tests -q
cd frontend
npm run build
npm test -- --run
```

Model tests use a generated tone to exercise real checkpoint loading, determinism, finite outputs, and segment bounds, plus locally synthesized Flite speech through the complete API when FFmpeg includes the optional flite filter. That speech test is explicitly skipped on FFmpeg builds without Flite. It is **not** evidence of speech detection accuracy. Dataset evaluation must use properly labeled speech; see `research/` and `experiments/`.

A missing checkpoint is a setup error; the backend must never manufacture a prediction. Keep the model checkpoint cache between runs. See `ml/` for the exact source, license, checksum, and downloader command if filenames change.
