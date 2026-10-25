# Execution evidence

## Real browser integration

Executed `tests/browser.mjs` using Playwright Core 1.56.1 and system Chromium, against real Vite and FastAPI processes, with actual AASIST weights (no mocked API).

| Recording | Provenance | Duration | Actual AI-oriented score | Result | Model-call latency |
|---|---|---:|---:|---|---:|
| Human speech | CC-BY-4.0 LibriSpeech-derived sample, metadata from `scripts/fetch_evaluation_samples.py` retained in ignored artifacts | 14.84 s | 0.3065590553 | Uncertain | 946 ms |
| Synthetic speech | Locally generated FFmpeg Flite `slt` voice | 3.545 s | 1.0 | Uncertain | 290.58 ms |

Both scores were displayed correctly in the browser. Invalid-file handling, JSON download, and a 390-pixel mobile layout check passed. Screenshots are generated locally at `artifacts/qa/human-browser.png` and `artifacts/qa/synthetic-browser.png`; they are not committed. Two successful recordings are not evidence of reliable accuracy. Latencies are individual observations including applicable first-load work, not benchmark means.

## Mechanical CPU performance observation

A separate fresh Python process on AMD EPYC 9V45 hardware, Python 3.12.14, PyTorch 2.6 CPU, two model threads, and a one-second generated tone measured a cold model call of 275.74 ms and a subsequent warm call of 214.74 ms. Process peak resident memory was 476,280 KiB. The tone is not labeled speech and these measurements establish operation only. Full model/runtime imports and process allocations contribute to the memory observation.

## Test coverage

Persisted tests distinguish explicit unit doubles from actual pretrained inference. Added checks cover checksum rejection; actual short-input equivalence to upstream repeat padding; unsupported rate/stereo/empty/nonfinite inputs; research-summary metadata validation and nonfinite rejection; and inference-timeout upload deletion plus concurrency-gate release. Frontend tests include readiness, report summaries, upload/results/errors, cancellation, segment seeking and report downloads. Run the full current suite, rather than relying on stale counts:

```sh
python -m pytest tests research ml -q
cd frontend
npm test -- --run
npm run build
```

## Unverified external operations

Docker build could not install FFmpeg because `deb.debian.org` could not resolve within build containers despite supported host-network/proxy diagnosis. No TLS or artifact verification was bypassed. Windows setup, CUDA and hosted CI have not been executed here. No public deployment was performed.

## Research validation continuation

The complete `pytest tests research ml -q` suite passed 93 tests (one upstream TestClient deprecation warning) using Python3.12/PyTorch2.6CPU. Node20.20.2 ran16 frontend tests successfully and the production build passed. Fresh Chromium uploads through Vite5190 to FastAPI8020 exercised actual human and classical synthetic speech, model-score display, JSON download, and mobile layout without mocks. No frontend/backend application logic was changed by the research continuation. The new engineering run deliberately reused the same six pilot recordings; no new representative corpus or production thresholds were obtained. See `research/validation/` for preserved evidence and exact reproduction.

## Resumed dataset/hybrid sprint

The completed `pytest tests research ml experiments -q` suite passed **145 tests** using Python3.12/PyTorch2.6 CPU (one upstream TestClient deprecation warning). The frontend passed **16 tests** under Node20.20.2 and its production build passed. Dependency checks and diff whitespace checks passed. CI and the verification script now include experimental unit tests so the new comparison protocol is exercised. Numerical hybrid/uncertainty fixtures are not measured benchmark data.

No six-recording pilot experiment or new benchmark inference was rerun in this continuation. Zero new representative recordings were acquired; the exact requested-source rights and metadata remain unreadable behind the configured proxy. No production policy, reliability claim or public deployment was introduced. The updated PDF is four pages, uses the documented IEEE-style fallback, and explicitly states the unexecuted empirical comparisons. See `research/datasets/REQUESTED_SOURCES.md` and `experiments/README.md` for the remaining data prerequisite and runnable workflow.
