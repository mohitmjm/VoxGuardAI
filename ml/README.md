# Genuine AASIST baseline

Source: NAVER's official https://github.com/clovaai/aasist at commit
`a04c9863f63d44471dde8a6abcb3b082b07cd1d1`. Architecture is vendored unchanged
in `vendor/aasist.py`; MIT license and upstream NOTICE accompany it. The official
repository distributes the pretrained weights under its repository license;
ASVspoof dataset licensing is separate and must be reviewed before data use.

`python -m ml.download_model` fetches the immutable official checkpoint and
checks SHA-256 `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0`.
The hash was recorded from the upstream Git checkout, not a claimed author signature.
No paid API or keys are required. PyTorch uses safe `weights_only=True` loading.

Preprocessing matches upstream evaluation: mono float audio, 16 kHz, 64600 samples,
repeat padding for short utterances. Backend FFmpeg handles decoding/resampling.
Long recordings use contiguous windows and duration-weighted mean spoof scores;
this aggregation is an application extension requiring evaluation. Class 0 is
spoof and class 1 bona fide (see upstream `data_utils.py`). Softmax class 0 is
reported as an **uncalibrated score**, never confidence. No project thresholds
have been validated, so every production result remains **Uncertain** until a
separate, versioned validation study supplies thresholds. This deliberately
avoids asserting detection reliability on modern generators or Hindi/Marathi.

CPU is default; set `VOXGUARD_DEVICE=cuda` only with compatible PyTorch/CUDA.
`VOXGUARD_TORCH_THREADS=2` bounds CPU parallelism. Missing/corrupt checkpoints fail
explicitly. No uploaded audio is used for training. Do not interpret silence or
music scores as speech authenticity evidence. Upstream reported benchmark figures
are not project evaluation results and are not displayed as such.

## Optional learned + acoustic research features

This component is separate from production anti-spoofing inference. It is not an
alternative detector and cannot classify audio by itself. Install matching CPU
TorchAudio in the research environment:

```sh
python -m pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m experiments.extract_features data/manifest.csv --output artifacts/features.npz
python -m experiments.hybrid artifacts/features.npz --ablation hybrid --output artifacts/hybrid.pt
```

`WAV2VEC2_BASE` downloads its official ~360 MB LibriSpeech-960h checkpoint from
`download.pytorch.org`. TLS verification remains enabled and PyTorch loads only
weights. The output sidecar records actual checkpoint SHA-256, TorchAudio/version,
manifest digest, extraction definition and feature digest. The recorded checksum
is a local artifact digest, not an upstream signature or a publisher-supplied
expected checksum. Review TorchAudio BSD-2-Clause, fairseq MIT and LibriSpeech CC
BY 4.0 rights for the intended experiment; do not redistribute speech datasets.
Official bundle input is mono 16 kHz, with no added input normalization. The final
(12th) transformer layer is pooled with mean/std inside fixed four-second windows,
then duration-weighted across windows. LFCC mean/std supply the acoustic branch.
This explicit experimental pooling is not a new algorithm or evidence of useful
spoof features. No encoder fine-tuning, trained hybrid production detector,
accuracy result, or Hindi/Marathi support is implied.

`python -m pytest ml/test_features.py -q` exercises feature definitions and, only
when the optional checkpoint is already cached, genuine learned features. Routine
tests do not download the large optional encoder.

`VOXGUARD_INFERENCE_TIMEOUT_SECONDS` defaults to 120 seconds. A cooperative
monotonic deadline is checked before and after each window. Expiration aborts the
whole result, never returning a partial score. It cannot interrupt a currently
executing native PyTorch operation or model load; enforce a process-level limit
at deployment if hard interruption is required. CPU windows remain bounded to
64600 samples and thread count is configurable.

The optional encoder's recorded official-artifact digest is
`1275f890a623b31b9915295dc01d5049c93dc6de8813406c16dfab9b6bd36056`;
subsequent loads enforce it. It was measured from the official TLS download,
not supplied as an author signature. TorchAudio and fairseq license text was
checked at their official repositories; no source from those projects is
vendored by this extractor.

Set `VOXGUARD_CHECKPOINT` to a custom AASIST checkpoint file path before running
both the downloader and API. The downloader uses that same path, verifies the
expected digest, installs atomically, and cleans temporary files on failure.
