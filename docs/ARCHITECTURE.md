# System and final-year project guide

VoxGuard AI investigates detection of synthesized or cloned speech. It is a research prototype, not a forensic proof system.

## Data flow

1. React accepts a user-selected recording and provides local playback/waveform rendering.
2. FastAPI bounds the upload and validates the decoded container with ffprobe rather than trusting its extension.
3. FFmpeg converts audio to mono 16 kHz PCM under subprocess limits.
4. The ML service applies the pretrained anti-spoofing model's input preparation and evaluates windows on CPU (or optional CUDA).
5. The API returns an AI-oriented, uncalibrated model score, window scores, processing details, and limitations.
6. The frontend displays the result without presenting the score as a probability.

`frontend/` owns presentation; `backend/` owns transport and safe conversion; `ml/` owns model provenance, preprocessing, and inference; `experiments/` owns offline evaluation; `research/` owns methodology. Benchmark and experimental code must remain separate from stable inference.

## Interpretation

Without defensible validation-derived thresholds, the decision is **Uncertain**, even when a raw score is near an endpoint. A model score is not a calibrated probability or percentage confidence. Window scores are contextual model responses, not precise manipulation boundaries. Pure silence, tones, non-speech, noise, replayed audio, unseen synthesis methods, and mismatched languages may produce misleading outputs.

The pretrained baseline's training/evaluation domain does not establish Hindi or Marathi support, real-world microphone robustness, unseen-generator generalization, or broad modern TTS coverage. Those remain evaluation goals. No accuracy or benchmark is claimed by the application until a versioned, reproducible evaluation supports it.

## Privacy and security

Uploaded recordings are processed in temporary files and deleted after the request, including failures. Recordings must not enter logs, source control, analytics, or training corpora without explicit permission. The browser retains the selected file for local playback until replacement or page close. Resource bounds and concurrency controls reduce accidental exhaustion; application limits are not a replacement for a production reverse proxy or host sandbox. Do not share a development server publicly.

## Academic deliverables

Report the problem, threat model, dataset licenses, anti-spoofing literature, architecture, implementation, evaluation protocol, measured results, error cases, ethics, and limitations. Use the paper outline under `research/` as a starting point. Store split manifests and configuration hashes with results. Separate speaker, original source recording, and generator identities across the appropriate splits; select thresholds only on validation data. Report skipped and unavailable experiments explicitly.

Potential future work includes learned speech embeddings, LFCC/acoustic fusion, compression/noise augmentation, language-specific evaluation, and calibration. Each is a hypothesis until evaluated. Do not claim novelty or improvement from scaffolding alone.
