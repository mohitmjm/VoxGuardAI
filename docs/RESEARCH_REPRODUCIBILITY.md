# Research reproduction and evidence boundaries

The functional application and a small recorded pilot are available. They do not establish reliable real-world detection. No representative held-out validation/test result or validated production threshold is bundled.

## Fixed baseline

- Official model: NAVER AASIST, source commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`.
- Checkpoint SHA-256: `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0`.
- Mono float32, 16 kHz; 64,600-sample contiguous windows; short/tail repeat padding; duration-weighted spoof-class softmax. Aggregation is an application extension requiring evaluation. Score is uncalibrated.
- Runtime inference must use the same checkpoint, decoder settings, aggregation and score direction as validation. Default output remains Uncertain without a reviewed artifact.

## Required run record

Retain release/license/consent records, source URLs, checksums, official protocol and original IDs, sample-selection rules, generator family, language and channel. Save frozen manifests, partition audit output, pretrained-overlap findings, model and score-protocol digests, score CSV, threshold artifact, software versions, Git revision, hardware/device/thread settings, seed and full commands. Keep original recordings and derivatives together. Never tune thresholds or select features using final test results.

The earlier convenience pilot contains three CC-BY-4.0 LibriSpeech excerpts and three project-text classical Flite recordings, all test rows. `research/results/` records six clean scores and 36 paired clean/noise/codec observations; these are six independent source recordings, not 36 independent samples. Their descriptive AUROC/EER cannot justify a calibrated probability or general-purpose decision boundary. LibriSpeech may overlap representation pretraining; source domains differ. No modern-generator, Hindi or Marathi performance follows from this pilot.

## Representative evaluation procedure

1. Inspect actual release-specific recording rights and original train/dev/eval protocol before acquisition. See `research/datasets/` for current host findings, bounded/resumable acquisition and manual import paths. Reachable source code is not proof of recording rights.
2. Import unchanged official speaker/source/generator identifiers; namespace identities only in a documented, consistent way. Audit source hashes, speaker/source overlap and generator holdout. If pretraining overlap cannot be ruled out, disclose it.
3. Score validation and untouched test partitions using frozen preprocessing; save explicit checkpoint/score-protocol identity. Keep validation and test score files separate.
4. Fit an abstaining threshold policy using validation only, with an explicit false-positive/false-negative objective and declared data-domain scope. Do not lower sample constraints merely to enable a UI verdict.
5. Freeze the policy and run independent test reporting, including coverage, class-specific abstention, confusion counts, FPR/FNR, precision, recall, F1, AUROC/EER and appropriate cluster-level uncertainty. Report unsupported metrics as unavailable rather than choosing test thresholds.
6. Reuse the same source recordings for paired MP3/AAC/noise analyses. Preserve condition names and don't count encoded/noisy derivatives as independent recordings. Generator-held-out training and validation must both exclude held-out synthesis families.
7. Train comparison heads only when legitimate identity-audited train/validation/test features exist. Frozen Wav2Vec2 features alone are not a competing anti-spoofing detector. Separate literature-reported baselines from project measurements.

Consult CLI `--help` and `research/README.md` for manifest scoring, policy fitting and reporting; newer research validation tools and dataset adapters have their own help. Do not reuse the six pilot test examples as validation or hybrid training data.

## Paper reproduction

The paper directory includes Markdown, LaTeX/BibTeX, a generated actual-score plot and a compiled draft. Run the commands in `research/paper/README.md`. The delivered PDF used a two-column article fallback because IEEEtran was absent. It is not final IEEE venue formatting or a submission-ready empirical paper. References were checked against official repositories with verification scope in `references_verified.md`; no published accuracy figures are represented as project measurements.

Windows PowerShell execution, optional CUDA, hosted CI and publication must be reported separately from local Linux tests. The latest research task does not authorize a Git push or public deployment.

## Exact requested sources in the final dataset sprint

On 2026-10-25, configured-proxy requests for the raw cards of `mueller91/MLAAD-tiny` and `mueller91/In-The-Wild` on Hugging Face, and Zenodo API record `14498691`, each failed with a tunnel HTTP 403. No license text, metadata body or recording was retrieved from those checks. Official publisher relationships and individual dataset/recording terms remain unverified here; do not fill a license field from memory or infer it from a repository name. See dataset acquisition records for exact requests and the permitted-machine download/transfer workflow. No pilot inference was repeated by this paper update.

For a future model comparison, freeze identical held-out recording IDs/content digests across AASIST and all feature-head ablations. Head-only timing excludes feature extraction and must not be advertised as total application latency. Use source/speaker clusters for intervals and keep repeated codec observations in their cluster; disclose whether intervals condition on validation-selected thresholds. Numerical fixture metrics validate code only. Real head training and representative comparison are unexecuted, pending legitimate identity-audited train/validation/test data.

The new uncertainty report uses connected speaker/source components, not individual augmented rows. Its minimum component/valid-resample guards are engineering checks rather than coverage guarantees; the six-recording pilot is below the reporting minimum. No new benchmark intervals were computed in this sprint.
