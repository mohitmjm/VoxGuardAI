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
