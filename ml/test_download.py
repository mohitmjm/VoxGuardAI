import io
import hashlib
from pathlib import Path
import pytest
from ml import download_model


def test_configured_destination_and_verified_repeat(tmp_path, monkeypatch):
    payload = b'unit-test checkpoint payload, never a production model'
    destination = tmp_path / 'custom' / 'model.pth'
    monkeypatch.setenv('VOXGUARD_CHECKPOINT', str(destination))
    monkeypatch.setattr(download_model, 'SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(download_model.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(payload))
    download_model.main()
    assert destination.read_bytes() == payload
    monkeypatch.setattr(download_model.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Verified checkpoint must not download again'))
    download_model.main()


def test_failed_replace_cleans_temporary_file(tmp_path, monkeypatch):
    payload = b'unit fixture'
    monkeypatch.setenv('VOXGUARD_CHECKPOINT', str(tmp_path / 'model.pth'))
    monkeypatch.setattr(download_model, 'SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(download_model.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(payload))
    def fail(*args):
        raise OSError('simulated destination failure')
    monkeypatch.setattr(Path, 'replace', fail)
    with pytest.raises(OSError):
        download_model.main()
    assert list(tmp_path.iterdir()) == []
