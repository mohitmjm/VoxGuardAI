# VoxGuard AI

Research-oriented detection of AI-generated speech using genuine pretrained AASIST inference. No paid service or API key is required. Scores are uncalibrated model outputs, not authenticity probabilities. Classification remains **Uncertain** until thresholds are validated on representative held-out data. English benchmark training does not establish Hindi, Marathi, or unseen-generator performance.

## Local development

Python 3.11+, Node 20+, and FFmpeg/ffprobe must be available.

```sh
python -m venv .venv
.venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r backend/requirements.txt -r ml/requirements.txt pytest httpx scikit-learn scipy
.venv/bin/python -m ml.download_model
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
# In another terminal
cd frontend
npm ci
npm run dev
```

On Windows use `.venv\Scripts\python.exe` instead of `.venv/bin/python`. Install FFmpeg and add its bin directory to PATH. See `docs/` for detailed setup and research limitations. No uploaded recordings are retained or used for training. Do not expose this unauthenticated research service publicly without an appropriate deployment security review.

## Validation

```sh
.venv/bin/python -m pytest tests research -q
cd frontend
npm test -- --run
npm run build
```

Research tools and experiment protocols live in `research/` and `experiments/`. No project benchmark results are claimed. Model source and licensing are documented in `ml/`.
