# Capability status

## Verified application

Implemented and exercised: checksum-verified official AASIST CPU inference; private, bounded WAV/MP3/M4A/FLAC uploads; FFmpeg mono 16 kHz conversion; short-audio repeat padding; long-audio window scores; responsive React upload/player/waveform; light/dark modes; playback-linked segment controls; cancel/timeout handling; original processing metadata; downloadable JSON reports; and model/readiness information. Public research summaries are displayed only from explicitly configured evaluation reports.

Real Chromium upload-to-inference tests passed for a licensed human speech recording (14.84 seconds, CC-BY-4.0 attribution retained locally) and known Flite synthetic speech (3.545 seconds). Both browser scores matched actual API responses, report downloads worked, invalid files produced actionable errors, and the mobile layout did not overflow. These checks establish application functionality, not reliable detection accuracy. Uploaded recordings are not retained by the API; test recordings remain only in ignored local artifacts.

## Research infrastructure and measured pilot

Available: audited dataset manifests; validation-only abstaining threshold fitting; model/protocol-bound optional policy loading; test reporting with confusion/error metrics; bootstrap evaluation; compression/noise experiments; source/speaker leakage checks; experiment ablations; a manuscript draft; and plots generated from actual results.

A six-recording pilot (three licensed human clips and three Flite-generated clips) and 36 paired clean/noise/MP3/AAC observations were measured. See `research/results/` for exact scores, metric definitions, sample counts, provenance and strong limitations. This small convenience pilot is not a representative benchmark, not evidence of modern-generator/Hindi/Marathi generalization, and not sufficient to choose production thresholds. Default classifications remain **Uncertain**. An optional, reviewed validation policy can enable domain-specific Human/AI decisions; no deployment policy is bundled.

Official pretrained Wav2Vec2 representation extraction has been exercised for research. It is an encoder, not a validated spoof classifier. Hybrid fusion training/ablation infrastructure requires legitimate split-audited features and proper training/evaluation; no hybrid improvement or production hybrid detector is claimed.

## Compatibility and deployment

Python 3.11/3.12 dependency pins were corrected; actual inference and tests were run using Python 3.12 and PyTorch 2.6 CPU. The Node20-compatible jsdom26 test dependency removes the previously reported newer-Node engine mismatch; 16 frontend unit/component tests and the production build pass. Windows PowerShell instructions/scripts are provided, but Windows was not available for executing the complete setup here.

Docker configuration is prepared, but an attempted image build is blocked by DNS/network access to `deb.debian.org` inside Docker build containers. Host-network and existing-proxy propagation did not restore that operation. No package/TLS verification was disabled. Container startup and CUDA remain unverified. Native Python deployment does not require Docker. No public deployment has occurred.

Latest work is on the feature branch; see Git history for local commits and remote state. CI is configured with an immutable, checksum-verified model cache; hosted CI results require a permitted GitHub push and actual runner execution.

## Remaining blockers

- Representative, licensed speaker/source/generator-disjoint validation and untouched test corpora for defensible deployment thresholds.
- Modern neural-generator, unseen-generator, cross-dataset, Hindi and Marathi evaluation.
- Empirical hybrid model training, matched baselines, controlled ablations and research-grade statistical conclusions.
- Docker build-network access, optional CUDA hardware validation, Windows end-to-end installation, and authorized always-on hosting.

The manuscript is a research draft with measured pilot findings and explicitly pending experiments; it is not evidence of a publishable improvement or acceptance readiness.
