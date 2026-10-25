# Capability status

Implemented: official AASIST CPU inference, checksum-verified download, bounded private upload API, WAV/MP3/M4A/FLAC conversion, window scores, responsive upload/player UI, evaluation and experimental fusion tooling.

Research limitations: no representative labeled benchmark has been evaluated for this project. No validated deployment thresholds exist, so the application abstains with Uncertain and displays its actual uncalibrated score. Human/AI result presentation is supported for future validated policies; the present detector does not issue those decisions. Hindi, Marathi, modern unseen generators, noise and compression accuracy remain unevaluated. Segment scores are not edit localization. No VAD is applied. Experimental fusion requires separately extracted, licensed features; it is not a trained production model.

Smoke tests establish software operation and deterministic genuine pretrained inference, not detection accuracy. A known local Flite TTS recording is a mechanical speech smoke fixture, not a benchmark. No consented human speech test corpus is bundled.

Docker files are provided; container image build and CUDA have not been validated here. No public deployment or push was performed. Frontend static assets can deploy independently with VITE_API_BASE_URL configured to the inference backend and the backend CORS allowlist adjusted.
