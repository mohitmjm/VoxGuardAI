# VoxGuard AI

Research-oriented detection of AI-generated speech using genuine pretrained AASIST inference. No paid service or API key is required. Scores are uncalibrated model outputs, not authenticity probabilities. Classification remains **Uncertain** until thresholds are validated on representative held-out data. English benchmark training does not establish Hindi, Marathi, or unseen-generator performance.

Complete project report: [read online](docs/PROJECT_DOCUMENTATION.md), [PDF](docs/VoxGuard_AI_Project_Report.pdf), or [editable Word document](docs/VoxGuard_AI_Project_Report.docx).

## Local development

Python 3.11 or 3.12, Node 20+, and FFmpeg/ffprobe must be available.

```sh
python -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
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
.venv/bin/python -m pytest tests research ml experiments -q
cd frontend
npm test -- --run
npm run build
```

Research tools and experiment protocols live in `research/` and `experiments/`. No project benchmark results are claimed. Model source and licensing are documented in `ml/`.

Optional validation-derived decisions: see [decision policy](docs/DECISION_POLICY.md). No policy or benchmark figures are bundled. For hosting without Docker see [deployment](docs/DEPLOYMENT.md). Run `.venv/bin/python scripts/preflight.py` to check FFmpeg, checkpoint integrity, and real inference.

Research report dashboard: set `VOXGUARD_RESEARCH_REPORTS_PATH` to a trusted offline summary JSON (for example `research/results/pilot_dashboard.json`) before starting the backend. The dashboard always shows dataset, count, model and conditions; no metrics are generated from user uploads. The bundled six-recording pilot is not a benchmark. See `research/paper/manuscript.md` and `research/datasets/README.md`.

Optional research speech embeddings are extracted separately; they are not a second validated detector. See `ml/README.md`. Windows can run `powershell -File scripts/run.ps1` after setup; a separate terminal runs `cd frontend; npm run dev`.

Research validation: [single-command runner](research/validation/README.md), [dataset access and manual import](research/datasets/VALIDATION_ACCESS.md), and [reproducibility record](docs/RESEARCH_REPRODUCIBILITY.md). The IEEE-style manuscript source and compiled PDF are in `research/paper/`; representative validation/test data remain unavailable, and no deployment policy has been enabled.

The resumed dataset sprint adds [exact requested-source acquisition/transfer](research/datasets/REQUESTED_SOURCES.md), [clustered uncertainty and frozen protocols](research/validation/README.md), and [matched hybrid comparisons](experiments/README.md). No new independent recordings or calibrated deployment policy were obtained; the source licenses remain unverified behind the proxy.
