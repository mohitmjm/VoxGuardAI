# Research protocol

**Proposed title:** Robust Detection of AI-Generated Speech Across Unseen Voice Synthesis Systems Using Hybrid Acoustic Representations.

The question is whether learned speech representations, frequency-domain descriptors, and compression-aware augmentation improve discrimination on held-out synthesis systems. This is a hypothesis, not an established contribution. No benchmark results are bundled. Production scores are not calibrated probabilities; successful inference does not establish detection accuracy.

## Data and rights

Acquire ASVspoof 2019 LA and/or ASVspoof 2021 LA/DF from the official [ASVspoof](https://www.asvspoof.org/) challenge distribution. Review the individual release license, access conditions, permitted research use, redistribution restrictions, and participant protocol before downloading. Do not redistribute recordings or infer dataset rights from the model code license. Respect official partitions and label semantics (bonafide → human, spoof → ai). Confirm whether a pretrained model has seen each partition before claiming held-out performance.

Optional external sets include WaveFake and Fake-or-Real, only after checking each release's dataset and underlying source-speech licenses and training overlap. For Hindi and Marathi, collect explicitly consented recordings and authorized synthesis outputs with speaker, source utterance, generator/version, language, channel, and consent metadata. These languages remain unevaluated goals. Do not use application uploads for training.

`manifest.example.csv` illustrates the schema, not distributed data. Paths are relative to the CSV. Namespaced globally stable speaker and source IDs must link each original utterance to all derived voices, crops, augmentations and encodings. Use a conservative shared source ID for uncertain provenance. Audit metadata manually: byte hashes do not detect re-encoded duplicates. Keep all derivatives in the same split; generator-held-out evaluation must also separate generators. Report pretrained overlap that cannot be ruled out as a limitation.

## Reproducible commands

Run from the project root with the project Python environment; NumPy, PyTorch, and FFmpeg are required. No dataset is downloaded automatically.

```bash
python -m research.evaluate audit data/manifest.csv --held-out-generators --output audit.json
python -m research.evaluate score data/manifest.csv --output scores.csv
# Export validation rows and test rows to separate CSVs, preserving the header.
python -m research.evaluate fit-threshold validation_scores.csv --output threshold.json
python -m research.evaluate report test_scores.csv --threshold threshold.json --output metrics.json
python -m experiments.robustness data/manifest.csv --output robustness.csv
python -m experiments.hybrid data/features.npz --ablation hybrid --output hybrid.pt
```

The audit runs before inference. Research decoding currently caps each recording at 120 seconds and resamples to mono 16 kHz; use shorter benchmark utterances or explicitly document this cap. The scoring runner records end-to-end model-call latency (excluding FFmpeg decoding); first-call load time is included, so report cold and warm timings separately for hardware comparisons. Save `GET /api/model/info`, model/checkpoint digest, dependency lock, Git revision, hardware, thread count, OS, seed, manifest digest and rights record with each run. Threshold fitting accepts only validation rows and writes a frozen artifact; reporting accepts only test rows. The application does not automatically trust or load experimental thresholds. Threshold transfer requires review of model identity, data domain and false-positive operating point.

Metrics include accuracy, precision, recall, F1, AUROC, interpolated EER, false-positive rate, latency median/p95, and speaker-cluster bootstrap 95% intervals. AI is the positive class. Binary metrics use the frozen validation threshold, independently of the application's abstention state. Small samples or single-class bootstrap draws may not support intervals; valid replicate counts are reported. EER is a descriptive curve statistic, not a test-selected deployment threshold. Also report class counts, language/generator breakdown, abstention coverage, false positives and failure cases. No test set may select hyperparameters, thresholds or checkpoint epochs.

## Experiments and ablations

| Experiment | Established baseline / proposed comparison | Controls |
|---|---|---|
| A | Frozen authentic pretrained anti-spoof checkpoint | Official clean benchmark, documented preprocessing |
| B | Same model on held-out synthesis systems | Generator-disjoint and source/speaker-disjoint splits |
| C | Clean versus MP3 32/64 kbps paired audio | Same recordings, paired speaker-bootstrap deltas |
| D | Clean versus seeded white noise 20/10 dB SNR | Same recordings; extend to licensed real noise separately |
| E | Proposed learned + acoustic fusion | Train-only normalization, validation-only selection |
| F | Learned only, acoustic only, hybrid; augmentation off/on | Fixed splits, same seeds, parameter/cost accounting |

`experiments.hybrid` trains a small experimental fusion head on precomputed frozen speech embeddings and LFCC summary features. It does not implement or claim a new backbone. Record exact encoder/version, layer, pooling, LFCC settings and provenance when producing the NPZ. No extractor, pretrained hybrid weights, fine-tuned detector, or evidence of improvement is supplied. Use at least three seeds for final comparisons and matched compute where feasible. Compression/noise harness outputs paired observations; evaluate each condition separately using the same frozen validation threshold. Do not mix repeated conditions as independent speakers.

## Literature review outline

Verify full citations and claims against primary papers before submission: ASVspoof 2019/2021 challenge protocols; Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks* (ICASSP 2022); Baevski et al., *wav2vec 2.0* (NeurIPS 2020); LFCC/GMM challenge baselines; RawNet2 anti-spoofing; cross-domain and codec robustness studies. Distinguish representation pretraining from spoof-specific supervision. AASIST and LFCC baselines are established methods; the fusion comparison here is a proposed experiment.

## Paper / final-year report structure

1. Abstract: question, datasets and measured results only after evaluation.
2. Introduction: threat model, probabilistic limits, unseen synthesis challenge.
3. Related work: anti-spoofing, pretrained representations, acoustic features, robustness.
4. Method: established baseline, proposed fusion, aggregation and abstention policy.
5. Data and ethics: rights, consent, partitions, language coverage and leakage prevention.
6. Experiments: reproducible setup, baselines, operating-point selection and metrics.
7. Results: measured tables with uncertainty, held-out generators, ablations and failures.
8. Discussion: domain shift, false accusations, privacy, invalid authenticity claims.
9. Conclusion: evidence-supported findings and remaining work.

Retain a result card per experiment: manifest/model/config hashes, run command, seed, hardware, runtime, metrics, confidence intervals, errors, and deviations. Do not fill tables with estimates or fabricate research results.
