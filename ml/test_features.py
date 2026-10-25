"""Mechanical feature checks, independent of dataset labels or accuracy claims."""
from pathlib import Path
import numpy as np
import pytest
import torch
from ml.features import lfcc, log_spectrogram


def test_lfcc_shape_and_spectral_sanity():
    signal = np.sin(np.arange(16000) * .1).astype(np.float32)
    coefficients = lfcc(signal)
    assert coefficients.shape[0] == 20
    assert torch.isfinite(coefficients).all()
    assert torch.equal(coefficients, lfcc(signal))


@pytest.mark.parametrize('signal', [np.zeros((2, 16000)), np.zeros(10), np.full(512, np.nan)])
def test_invalid_feature_audio(signal):
    with pytest.raises(ValueError):
        log_spectrogram(signal)


def test_optional_real_wav2vec_features():
    pytest.importorskip('torchaudio')
    cache = Path('.cache/representations')
    if not (cache / 'wav2vec2_fairseq_base_ls960.pth').is_file():
        pytest.skip('Optional official Wav2Vec2 checkpoint not cached; no download during routine tests')
    from ml.representations import Wav2Vec2Features
    torch.set_num_threads(2)
    extractor = Wav2Vec2Features(cache)
    signal = np.sin(np.arange(16000) * .1).astype(np.float32)
    learned, acoustic = extractor.extract(signal)
    assert learned.shape == (1536,) and acoustic.shape == (40,)
    assert np.isfinite(learned).all() and np.isfinite(acoustic).all()
    assert np.allclose(learned, extractor.extract(signal)[0], atol=1e-6)
