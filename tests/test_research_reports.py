"""Dashboard consumes supplied research output; never invents benchmark values."""
import json

import pytest
from fastapi.testclient import TestClient
from backend.app import app


def test_no_report_configuration_is_explicit(monkeypatch):
    monkeypatch.delenv('VOXGUARD_RESEARCH_REPORTS_PATH', raising=False)
    response = TestClient(app).get('/api/research/results')
    assert response.status_code == 200
    assert response.json() == {'status': 'not_evaluated', 'reports': []}


def test_supplied_report_preserves_evaluation_conditions(monkeypatch, tmp_path):
    path = tmp_path / 'fixture-report.json'
    # This is a numerical API fixture, not measured detector accuracy.
    report = {'dataset': 'unit-fixture', 'sample_count': 4, 'model': 'fixture-model', 'conditions': 'test fixture only', 'metrics': {'auroc': .5}}
    path.write_text(json.dumps({'reports': [report]}))
    monkeypatch.setenv('VOXGUARD_RESEARCH_REPORTS_PATH', str(path))
    result = TestClient(app).get('/api/research/results').json()
    assert result['status'] == 'available'
    assert result['reports'] == [report]


@pytest.mark.parametrize('content', ['{"reports":[{"dataset":"bad"}]}', '{"reports":[{"dataset":"x","sample_count":1,"model":"x","conditions":"x","metrics":{"auc":NaN}}]}', '{broken', '[]'])
def test_invalid_report_fails_closed(monkeypatch, tmp_path, content):
    path = tmp_path / 'private-report.json'; path.write_text(content)
    monkeypatch.setenv('VOXGUARD_RESEARCH_REPORTS_PATH', str(path))
    response = TestClient(app).get('/api/research/results')
    assert response.json()['status'] == 'unavailable'
    assert response.json()['reports'] == []
    assert str(path) not in response.text


def test_missing_report_file_hides_server_path(monkeypatch, tmp_path):
    path = tmp_path / 'missing-private-file.json'
    monkeypatch.setenv('VOXGUARD_RESEARCH_REPORTS_PATH', str(path))
    response = TestClient(app).get('/api/research/results')
    assert response.json()['status'] == 'unavailable'
    assert str(path) not in response.text
