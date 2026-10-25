"""Synthetic numerical fixtures test policy mechanics, not detection accuracy."""
import copy
import json

import pytest

from ml.policy import load_policy, validate_artifact
from research.fit_policy import fit

SHA = 'a' * 64


def rows():
    return [dict(label=label, score=score, split='validation', speaker_id=f'{label}-{i}', source_id=f'{label}-{i}', language='English')
            for label, scores in [('human', [.1,.2,.3]), ('ai',[.7,.8,.9])] for i, score in enumerate(scores)]


def artifact():
    return fit(rows(), checkpoint_sha256=SHA, scores_sha256='b'*64, domain='Synthetic numerical unit test only', max_error=0, min_count=3, min_speakers=3)


def test_opt_in_and_boundary_abstention(tmp_path, monkeypatch):
    monkeypatch.delenv('VOXGUARD_POLICY_PATH', raising=False)
    assert load_policy(SHA) is None
    path = tmp_path/'policy.json'; path.write_text(json.dumps(artifact()))
    monkeypatch.setenv('VOXGUARD_POLICY_PATH', str(path))
    policy = load_policy(SHA)
    assert [policy.classify(s) for s in [.1,.3,.5,.7,.9]] == ['human','human','uncertain','ai','ai']
    assert 'not guarantees' in policy.metadata()['warning']
    assert policy.metadata()['scope']['languages'] == ['English']


@pytest.mark.parametrize('change', [
    {'checkpoint_sha256':'c'*64}, {'human_max':float('nan')},
    {'ai_min':float('inf')}, {'human_max':.9,'ai_min':.7},
    {'fit_split':'test'}, {'validation_scores_sha256':'missing'},
    {'score_protocol':'unrelated-model'}, {'scope':{}}, {'counts':{}},
    {'empirical_rates':{'human_as_ai':float('nan')}},
])
def test_rejects_untrusted_or_invalid_artifacts(change):
    value = copy.deepcopy(artifact()); value.update(change)
    with pytest.raises(ValueError): validate_artifact(value, SHA)


def test_fitting_rejects_test_rows_and_insufficient_data():
    value=rows();value[0]['split']='test'
    with pytest.raises(ValueError,match='validation'): fit(value,checkpoint_sha256=SHA,scores_sha256='b'*64,domain='test')
    with pytest.raises(ValueError,match='Insufficient'): fit(rows(),checkpoint_sha256=SHA,scores_sha256='b'*64,domain='test')


def test_unseparated_classes_do_not_invent_thresholds():
    value=rows()
    for row in value:row['score']=.5
    with pytest.raises(ValueError,match='No separated'): fit(value,checkpoint_sha256=SHA,scores_sha256='b'*64,domain='test',min_count=1,min_speakers=1)


def test_bad_score_and_missing_file_fail_closed(tmp_path):
    policy=validate_artifact(artifact(),SHA)
    with pytest.raises(ValueError):policy.classify(float('nan'))
    with pytest.raises(ValueError):load_policy(SHA,tmp_path/'missing.json')


def test_service_rejects_policy_for_another_checkpoint(tmp_path, monkeypatch):
    from ml.service import ModelService, ModelUnavailable
    path=tmp_path/'wrong-model.json';path.write_text(json.dumps(artifact()))
    monkeypatch.setenv('VOXGUARD_POLICY_PATH',str(path))
    with pytest.raises(ModelUnavailable):ModelService()


def test_service_opt_in_policy_metadata_and_real_inference(tmp_path, monkeypatch):
    import numpy as np
    from ml.service import ModelService, SHA256
    value=artifact();value['checkpoint_sha256']=SHA256
    path=tmp_path/'numeric-fixture.json';path.write_text(json.dumps(value))
    monkeypatch.setenv('VOXGUARD_POLICY_PATH',str(path))
    service=ModelService()
    result=service.analyze(np.sin(np.arange(16000)*.08).astype(np.float32)*.1)
    expected=validate_artifact(value,SHA256).classify(result['score'])
    assert result['classification']==expected
    assert result['model']['thresholds']['scope']['domain']=='Synthetic numerical unit test only'
