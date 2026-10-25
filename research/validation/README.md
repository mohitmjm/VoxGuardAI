# Reproducible frozen-model validation

No new representative licensed corpus was accessible in this execution. The
existing six-clip LibriSpeech/Flite pilot remains an engineering fixture, not a
modern-speech benchmark. No deployment policy was established or enabled.

## One-command experiment

Use a rights-reviewed local manifest with the original training/validation/test
partitions and original speaker, source and synthesis-system identifiers. The
required CSV schema remains
`audio_path,label,speaker_id,source_id,generator,split,language`; audio paths are
relative to the manifest. Use `human` and `ai` labels only. Do not split related
encodings, crops, speakers or source utterances into different partitions.

PowerShell, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m research.validation.run data\manifest.csv --output artifacts\validation\run-001 --dataset "Actual dataset/release and selection" --conditions "Language, domain, original protocol and sampling rule" --fit-policy --robustness
```

Only add `--held-out-generators` when synthesis-system/family identities are
known and preserved. It excludes test generators from both training and
validation and rejects unknown/generic identities; it cannot establish whether a
pretrained checkpoint saw undisclosed related data. Do not infer unseen status
from missing metadata. No automatic re-splitting or random record selection is
performed by this runner.

The command:

1. Audits the complete manifest before evaluating selected validation/test rows,
   including training rows for leakage checks. Repeated paths or byte-identical
   recordings are rejected rather than silently inflating counts. Re-encoded
   duplicates and incorrect metadata still need manual provenance review.
2. Records immutable input/content/model/protocol digests, code hashes,
   Git revision and dirty state, dependency versions, platform, CPU count,
   inference device/thread configuration, and seed.
3. Runs genuine frozen AASIST inference and saves complete and split-specific
   score CSVs. Training rows are not scored or fitted.
4. Optionally fits the two-threshold abstention policy using validation scores
   only (default minimum100 recordings and10 speakers **per class**, requested
   empirical misclassification cap1%). It refuses missing, insufficient or
   inseparable validation data. Minimum counts are checks, not statistical
   guarantees. The frozen policy is tested only on untouched test rows; no policy
   is automatically installed in the application.
5. Optionally creates six paired test-only conditions: clean, white noise at
   20/10dB SNR, MP3 at32/64kbps and AAC at64kbps. Any fitted clean-validation
   policy remains fixed across conditions. Multiple encodings are repeated
   observations of the same recording, not extra independent samples.

Successful runs write `summary.json`; failures retain `failure.json` and available
provenance. Output directories and result files are exclusive: choose a new run
identifier instead of overwriting earlier evidence. Research decoding caps input
at120seconds; use documented shorter benchmark utterances/segments and disclose
that cap. Decoding is excluded from reported model-call latency; the first score
can include checkpoint load time.

When no defensible policy can be fit, the report retains AUROC/EER as descriptive
ranking metrics, leaves binary precision/recall/F1/confusion decisions unavailable,
and records the reason. The application's existing Uncertain behavior persists.
For a valid frozen policy, test reporting includes precision, recall, F1,
confusion counts, FPR/FNR and coverage/abstention with their exact denominators.
Accepted-only accuracy is conditional on the subset receiving a decision.
`research.evaluate report` additionally supports speaker-cluster bootstrap
intervals for a frozen binary operating point. Neither descriptive EER nor a
bootstrap interval is a test-selected production threshold.

## Engineering regression executed here

The pipeline was exercised once on the same **previously used** six pilot clips:
three attributed human excerpts and three classical, non-neural Flite outputs.
No new data or independent representative benchmark was acquired. The actual
run completed6 clean scores plus36 paired robustness observations, reported
AUROC1.0/descriptive EER0.0, and correctly declined calibration because the
validation partition was empty. Precision/recall/F1 decision metrics remain
unavailable; no arbitrary decision threshold was substituted.

To reproduce on a machine whose FFmpeg includes Flite, first acquire the
checksummed fixtures (use a new output directory if already downloaded):

```powershell
.\.venv\Scripts\python.exe scripts\fetch_evaluation_samples.py --output artifacts\pilot-rerun
.\.venv\Scripts\python.exe -m research.validation.run artifacts\pilot-rerun\manifest.csv --output artifacts\validation\engineering-rerun --dataset "Existing six-recording engineering smoke only" --conditions "Three licensed LibriSpeech excerpts and three classical Flite recordings; not representative evidence; no validation partition" --fit-policy --robustness
```

The portable summary in `results/engineering_smoke.json` preserves this execution's
measured engineering evidence and its limitations. Raw scores, source attribution,
recordings and run provenance remain under ignored `artifacts/`; no uploaded
user audio is used. The engineering smoke preceded the final equivalent
threshold-search/streaming-hash hardening, whose behavior was checked with
numerical and leakage fixtures rather than repeating these inference runs.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest research\validation research\test_evaluation.py research\test_policy_report.py tests\test_policy.py -q
```

Numerical fixtures test policy boundaries, split separation, duplicate/content
leakage, unknown generator rejection and failure preservation; they are never
presented as measured detector accuracy. Authorized representative modern,
unseen-generator, multilingual and hybrid-comparison results remain pending data.

## Prospective uncertainty and frozen protocol (dataset sprint)

This sprint acquired **zero new evaluation recordings** because the requested
MLAAD-tiny/In-The-Wild Hugging Face and Zenodo14498691 sources were blocked.
It deliberately did not rerun the six-recording pilot. No new detector metrics,
validation policy, unseen-generator result or hybrid improvement is asserted.

Before inference, the runner now writes an exclusive `protocol.json` binding the
complete manifest, each recording's content hash, supplied identities/labels and
partitions, model/score identity, seed, threshold objective and uncertainty
configuration. Its digest is included in the summary. Changed input files or a
changed manifest invalidate a run; a frozen protocol still cannot prove that the
custodian supplied correct metadata or representative data.

For real future corpora, `--bootstrap-replicates 1000` (default) adds 95%
percentile intervals using connected components of **both** speaker and source
identities. This avoids treating one speaker, derived source utterances or
paired codec observations as independent samples. At least10 components and100
valid replicates per metric are required; otherwise intervals remain unavailable.
A ten-component cutoff is a pragmatic minimum, not a statistical guarantee.
Single-class draws and undefined precision/accepted accuracy are excluded only
from their respective metric, with valid replicate counts reported. Perfect or
rare-error rankings can give degenerate/unstable intervals even when an interval
is computable. These intervals do not repair selection bias or pretraining
contamination. The six-clip pilot would receive **no intervals** under this rule.

When a reviewed frozen validation policy exists, intervals include coverage,
abstention-sensitive recall/F1, FPR/FNR and accepted-only accuracy with explicitly
named denominators. No thresholds are re-fit during resampling or on degraded
test recordings. Per-generator ranking references the same supplied human
recordings for each synthetic family: these comparisons are dependent, not
independent experiments. Pretrained exposure remains `unknown`; naming a
modern generator does not establish that its family was unseen by AASIST.

The numerical-fixture tests exercise clustering, mutation detection, bootstrap
reproducibility, ties/undefined metrics and model-call protocol ordering. They
are software/statistical checks and are never reported as model evaluation.

The frozen protocol preserves an explicit `recording_id` when provided. Otherwise
`sha256:<audio-byte-digest>` is a derived byte identity, **not** an invented
speaker/source identifier or an assertion of original dataset provenance.
Repeated recording IDs/content are rejected. Scored row count, identities,
labels and partition metadata must still match the frozen protocol; editing the
protocol during inference invalidates the run rather than producing a success
summary. This also supplies an exact identity/hash join for separately extracted
hybrid features and baseline score CSVs.
