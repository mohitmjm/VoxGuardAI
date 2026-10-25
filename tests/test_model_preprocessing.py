"""Actual checkpoint tests for upstream short-audio handling and integrity."""
import numpy as np
import pytest
from ml.constants import WINDOW
from ml.service import ModelService, ModelUnavailable


def test_short_audio_matches_upstream_repeat_padding():
    service = ModelService()
    short = (.1 * np.sin(np.arange(16000) * .08)).astype(np.float32)
    repeated = np.tile(short, (WINDOW + len(short) - 1) // len(short))[:WINDOW]
    short_result = service.analyze(short)
    full_result = service.analyze(repeated)
    assert short_result['score'] == pytest.approx(full_result['score'], abs=1e-6)
    assert short_result['segments'][0]['end'] == 1
    assert full_result['segments'][0]['end'] == WINDOW / 16000


@pytest.mark.parametrize('samples,sample_rate', [(np.zeros(16000),8000),(np.zeros((2,16000)),16000),(np.array([float('nan')]),16000),(np.array([]),16000)])
def test_nonfinite_empty_stereo_or_wrong_rate_rejected(samples, sample_rate):
    with pytest.raises(ValueError):
        ModelService().analyze(samples, sample_rate=sample_rate)


def test_corrupt_checkpoint_is_never_loaded(monkeypatch, tmp_path):
    path=tmp_path/'corrupt.pth'; path.write_bytes(b'not the approved model')
    monkeypatch.setenv('VOXGUARD_CHECKPOINT',str(path))
    with pytest.raises(ModelUnavailable, match='checksum'):
        ModelService().analyze(np.ones(16000,dtype=np.float32)*.1)
