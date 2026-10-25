"""API conversion tests use an explicit inference double, not fake production predictions."""
import importlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

api = importlib.import_module('backend.app')


class RecordingModel:
    def info(self):
        return {'ready': True, 'name': 'test-double'}

    def analyze(self, samples, sample_rate=16000):
        assert samples.dtype == np.float32
        assert samples.ndim == 1 and sample_rate == 16000
        return {'classification': 'uncertain', 'score': 0.5, 'segments': [], 'model': self.info(), 'limitations': ['unit test double']}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, '_model', RecordingModel())
    with TestClient(api.app) as client:
        yield client


def upload(client, name, data):
    return client.post('/api/analyze', files={'file': (name, data, 'application/octet-stream')})


def test_health_and_metadata(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['ffmpeg_available'] is True
    assert client.get('/api/model/info').json()['ready'] is True


@pytest.mark.parametrize('extension', ['wav', 'mp3', 'm4a', 'flac'])
def test_supported_formats_resampled(client, wav_bytes, tmp_path, extension):
    source = tmp_path / 'source.wav'
    source.write_bytes(wav_bytes(rate=22050, channels=2))
    encoded = tmp_path / ('fixture.' + extension)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source), str(encoded)], check=True)
    result = upload(client, encoded.name, encoded.read_bytes())
    assert result.status_code == 200, result.text
    body = result.json()
    assert body['classification'] == 'uncertain'
    assert body['sample_rate'] == 16000
    assert body['processing']['channels'] == 2
    assert body['processing']['output_channels'] == 1
    assert body['duration_seconds'] == pytest.approx(1, abs=0.1)


@pytest.mark.parametrize('name,data,status', [('file.txt', b'hello', 415), ('file.wav', b'not audio', 422), ('file.wav', b'', 422)])
def test_invalid_upload(client, name, data, status):
    result = upload(client, name, data)
    assert result.status_code == status
    assert '/tmp/' not in result.text
    assert 'Traceback' not in result.text


def test_extension_is_not_trusted(client, wav_bytes):
    assert upload(client, 'pretend.flac', wav_bytes()).status_code == 415


def test_short_and_silent_audio(client, wav_bytes):
    assert upload(client, 'short.wav', wav_bytes(duration=0.1)).status_code == 422
    silent = bytearray(wav_bytes())
    silent[44:] = bytes(len(silent) - 44)
    assert upload(client, 'silent.wav', bytes(silent)).status_code == 422


def test_size_limit(client, monkeypatch):
    monkeypatch.setattr(api, 'MAX_BYTES', 16)
    assert upload(client, 'large.wav', bytes(20)).status_code == 413


def test_duration_limit(client, monkeypatch, wav_bytes):
    monkeypatch.setattr(api, 'MAX_DURATION', 0.75)
    assert upload(client, 'long.wav', wav_bytes()).status_code == 422


def test_temp_cleanup_success_and_failure(client, wav_bytes, monkeypatch):
    original = tempfile.TemporaryDirectory
    created = []
    def tracked(*args, **kwargs):
        context = original(*args, **kwargs)
        created.append(Path(context.name))
        return context
    monkeypatch.setattr(api.tempfile, 'TemporaryDirectory', tracked)
    assert upload(client, 'tone.wav', wav_bytes()).status_code == 200
    assert upload(client, 'broken.wav', b'bad').status_code == 422
    assert created and all(not path.exists() for path in created)


def test_missing_model_is_explicit(client, wav_bytes, monkeypatch):
    from ml.service import ModelUnavailable
    def unavailable():
        raise ModelUnavailable('private details must not leak')
    monkeypatch.setattr(api, 'get_model', unavailable)
    result = upload(client, 'tone.wav', wav_bytes())
    assert result.status_code == 503
    assert 'private details' not in result.text


def test_busy_request(client, wav_bytes):
    api._gate.acquire()
    try:
        assert upload(client, 'tone.wav', wav_bytes()).status_code == 429
    finally:
        api._gate.release()


def test_missing_multipart_field(client):
    assert client.post('/api/analyze', data={}).status_code == 422
