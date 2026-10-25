# Validation-derived decisions

Default inference abstains: the pretrained model supplies a genuine uncalibrated score, not a reliable universal authenticity decision. No deployment policy is bundled. Do not invent thresholds or describe unit-test fixtures as validation data.

For a domain-specific research policy:

1. Acquire licensed/consented human and synthetic recordings with language, generator, speaker and source metadata. Respect official partitions and audit training/validation/test leakage using `research.evaluate audit`.
2. Score the validation split using the frozen model and identical production preprocessing. Maintain CSV columns `label,score,split,speaker_id,source_id,language`. Record the checkpoint digest and score protocol.
3. Fit an abstaining policy (default requires 100 recordings and 10 speakers per class):

```sh
python -m research.fit_policy data/validation_scores.csv --checkpoint-sha256 51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0 --domain "Describe actual validated recording domain" --output policy.json
```

4. Freeze the artifact, evaluate on untouched test speakers/sources/generators, and review per-language false positives, false negatives, abstention coverage and uncertainty before deployment. Empirical validation error is not a statistical guarantee. The binary reporting CLI alone does not measure the two-threshold policy's abstentions; calculate and report them separately.
5. Explicitly enable a reviewed artifact using `VOXGUARD_POLICY_PATH=/absolute/path/policy.json`, then restart the backend. Missing/malformed or model-mismatched configured policies make inference unavailable rather than silently falling back. `/api/model/info` exposes provenance, scope and empirical validation rates.

The policy assigns Human below its low boundary, AI above its high boundary, and Uncertain between them. It does not calibrate probabilities, detect input language, or automatically verify that an uploaded recording belongs to its validated domain. Do not apply it to general public uploads as though it establishes universal accuracy. Segment scores remain exploratory and are not independently classified edit locations.
