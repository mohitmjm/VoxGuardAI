# Frozen-feature comparison (research only)

No representative comparison has been executed. The six-clip engineering pilot is not a training or calibration corpus. Numerical tests exercise code, not detector performance.

Use a rights-reviewed manifest with original `train`, `validation`, and `test` partitions and known speaker/source identities. Preserve official recording IDs where available; content-derived IDs identify bytes and do not replace speaker/source provenance. For an unseen-generator study, inspect generator families and run the generator-held-out audit before extracting features. Unknown pretraining exposure remains a limitation.

From the repository root with the project Python environment:

```powershell
python -m research.evaluate audit data/manifest.csv --held-out-generators --output artifacts/audit.json
python -m research.evaluate score data/manifest.csv --output artifacts/aasist-scores.csv
# Optional official CPU speech encoder; production inference does not require it.
python -m pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m experiments.extract_features data/manifest.csv --output artifacts/features.npz
python -m experiments.hybrid_evaluate artifacts/features.npz --baseline-scores artifacts/aasist-scores.csv --output artifacts/comparison-seed42 --dataset "Verified release, subset and original protocol" --seed 42
```

Choose fresh output paths. Omit the held-out audit flag when generator identity is unavailable and explicitly decline an unseen-generator claim. The baseline CSV and features must match recording IDs, content hashes, labels, partitions, and speaker/source identities. Legacy features without identities must be regenerated on appropriate data; do not fill missing fields with invented metadata.

The comparison trains only small learned-only, LFCC-only and fusion heads. Normalization uses training rows; validation selects checkpoints and binary research operating points; untouched test rows supply final metrics. Default safeguards require at least 100 recordings per class in each partition. Those counts do not establish representativeness or statistical power. Do not reduce them simply to obtain demo results. Prespecify multiple seeds and report all runs and suitable clustered uncertainty.

Feature sidecars bind the official Wav2Vec2 artifact digest and NPZ checksum. Runtime/size fields distinguish full AASIST, frozen speech encoder and trainable head. Warm batched **head-only** time excludes expensive decoding/feature extraction and cannot support an end-to-end speed claim; extraction timing, when measured, is separate. Scores are not calibrated authenticity probabilities. Neither comparison thresholds nor experimental heads are installed in production.

Check CLI `--help` for early stopping and experiment parameters. Training/evaluation remains blocked until legitimate, identity-audited train/validation/test data is available. Use `python -m pytest experiments -q` for clearly synthetic numerical unit tests only.
