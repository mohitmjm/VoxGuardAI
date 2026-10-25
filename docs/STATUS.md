# Capability status

Implemented: official AASIST CPU inference, checksum-verified download, bounded private upload API, WAV/MP3/M4A/FLAC conversion, window scores, responsive upload/player UI, evaluation and experimental fusion tooling.

Research limitations: no representative labeled benchmark has been evaluated for this project. No validated deployment thresholds exist, so the application abstains with Uncertain and displays its actual uncalibrated score. An opt-in, model-bound validation policy can enable Human/AI decisions with an abstention interval; no reviewed policy or validation dataset is bundled. Hindi, Marathi, modern unseen generators, noise and compression accuracy remain unevaluated. Segment scores are not edit localization. No VAD is applied. Experimental fusion requires separately extracted, licensed features; it is not a trained production model.

Smoke tests establish software operation and deterministic genuine pretrained inference, not detection accuracy. A known local Flite TTS recording is a mechanical speech smoke fixture, not a benchmark. No consented human speech test corpus is bundled.

Docker files are provided; container image build and CUDA have not been validated here. Code has been pushed to the feature/voxguard-ai GitHub branch; no public deployment was performed. Frontend static assets can deploy independently with VITE_API_BASE_URL configured to the inference backend and the backend CORS allowlist adjusted.

Added: report JSON export, playback-linked timeline controls, request cancellation/timeout, Pydantic result contracts, preflight checks, validation-policy fitting/loading, CI and no-Docker deployment instructions. See DECISION_POLICY.md.
