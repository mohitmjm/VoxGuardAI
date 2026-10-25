import numpy as np
import pytest
from ml.service import ModelService, ModelUnavailable


def test_invalid_deadline_configuration(monkeypatch):
    monkeypatch.setenv('VOXGUARD_INFERENCE_TIMEOUT_SECONDS', 'nan')
    with pytest.raises(ModelUnavailable):
        ModelService()


def test_deadline_fails_before_window_after_slow_load(monkeypatch):
    monkeypatch.setenv('VOXGUARD_INFERENCE_TIMEOUT_SECONDS', '1')
    service = ModelService()
    monkeypatch.setattr(service, '_load', lambda: None)
    ticks = iter([0., 2.])
    monkeypatch.setattr('ml.service.time.monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError):
        service.analyze(np.ones(16000, dtype=np.float32))
