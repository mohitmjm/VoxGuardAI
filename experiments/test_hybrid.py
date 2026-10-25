"""Numerical unit fixtures ONLY. These are not audio, benchmarks or accuracy evidence."""
import csv
import hashlib
import json
import numpy as np
import pytest
import torch
from experiments.hybrid import train, validate_features, load_features, digest
from experiments.hybrid_evaluate import comparison, aligned_baseline, validation_threshold, size_accounting
from ml.constants import SHA256
from ml.policy import SCORE_PROTOCOL
from ml.representations import WAV2VEC2_SHA256


def fixture():
    # Unique identities prevent these synthetic numerical tests from exercising
    # any real speaker or recording; each split has two examples of each class.
    rng=np.random.default_rng(2);n=12
    return {'learned':rng.normal(size=(n,4)).astype(np.float32),'acoustic':rng.normal(size=(n,2)).astype(np.float32),
        'label':np.tile([0,1,0,1],3),'split':np.repeat(['train','validation','test'],4),
        'speaker_id':np.array([f'unit-speaker-{i}' for i in range(n)]),'source_id':np.array([f'unit-source-{i}' for i in range(n)]),
        'recording_id':np.array([f'unit-recording-{i}' for i in range(n)]),
        'audio_sha256':np.array([hashlib.sha256(str(i).encode()).hexdigest() for i in range(n)])}


def baseline(data):
    return [{'recording_id':str(data['recording_id'][i]),'audio_sha256':str(data['audio_sha256'][i]),
             'label':'ai' if data['label'][i] else 'human','split':str(data['split'][i]),'score':str(.8 if data['label'][i] else .2),
             'speaker_id':str(data['speaker_id'][i]),'source_id':str(data['source_id'][i]),
             'latency_ms':'10','checkpoint_sha256':SHA256,'score_protocol':SCORE_PROTOCOL} for i in range(4,12)]


def write_features(tmp_path,data):
    p=tmp_path/'unit-fixture.npz';np.savez(p,**data)
    p.with_suffix('.npz.json').write_text(json.dumps({'feature_sha256':digest(p),'extractor':{'checkpoint_sha256':WAV2VEC2_SHA256}}))
    return p


def test_requires_real_partitions_and_adequate_counts():
    data=fixture()
    with pytest.raises(ValueError,match='Insufficient'):validate_features(data)
    data['split'][:]='test'
    with pytest.raises(ValueError,match='Separate'):validate_features(data,1)


@pytest.mark.parametrize('field',['speaker_id','source_id','recording_id','audio_sha256'])
def test_cross_partition_identity_leakage(field):
    data=fixture();data[field][8]=data[field][0]
    with pytest.raises(ValueError,match='leaks|duplicates'):validate_features(data,1)


def test_nonbinary_labels_and_nonfinite_features_rejected():
    data=fixture();data['label']=data['label'].astype(float);data['label'][0]=.5
    with pytest.raises(ValueError,match='binary'):validate_features(data,1)
    data=fixture();data['learned'][0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):validate_features(data,1)


def test_normalization_and_training_do_not_use_test_rows():
    data=fixture();torch.set_num_threads(2)
    first=train(data,epochs=3,patience=2,min_per_class=1)
    data['learned'][8:]+=10000;data['acoustic'][8:]-=10000
    data['label'][8:]=1-data['label'][8:]
    second=train(data,epochs=3,patience=2,min_per_class=1)
    assert torch.equal(first['mean'],second['mean'])
    assert torch.equal(first['scale'],second['scale'])
    assert all(torch.equal(first['state_dict'][k],second['state_dict'][k]) for k in first['state_dict'])


def test_legacy_or_tampered_features_rejected(tmp_path):
    data=fixture();data.pop('recording_id');p=write_features(tmp_path,data)
    with pytest.raises(ValueError,match='identities'):load_features(p,1)
    p.with_suffix('.npz.json').write_text('{"feature_sha256":"wrong"}')
    with pytest.raises(ValueError,match='provenance'):load_features(p,1)


def test_baseline_requires_exact_recording_content_and_model():
    data=fixture();rows=baseline(data);aligned_baseline(rows,data)
    rows[0]['audio_sha256']='wrong'
    with pytest.raises(ValueError,match='content'):aligned_baseline(rows,data)
    rows=baseline(data);rows[0]['checkpoint_sha256']='wrong'
    with pytest.raises(ValueError,match='checkpoint'):aligned_baseline(rows,data)


def test_validation_threshold_uses_validation_labels_only():
    t=validation_threshold([0,0,1,1],[.1,.2,.8,.9])
    assert .2<t<=.8


def test_complete_comparison_numerical_fixture_only(tmp_path):
    data=fixture();features=write_features(tmp_path,data);rows=baseline(data)
    path=tmp_path/'baseline-unit-fixture.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    torch.set_num_threads(2)
    output=tmp_path/'unit-results'
    result=comparison(features,path,output,dataset='Numerical unit fixture; NOT measured audio evidence',epochs=2,min_per_class=1)
    assert set(result['methods'])=={'aasist','learned','acoustic','hybrid'}
    assert result['counts']=={'train':4,'validation':4,'test':4}
    assert all(result['methods'][x]['parameter_count']>0 for x in ('learned','acoustic','hybrid'))
    assert result['methods']['hybrid']['latency_ms']>=0
    with pytest.raises(FileExistsError):comparison(features,path,output,dataset='unit',epochs=1,min_per_class=1)


def test_unrelated_extractor_metadata_rejected(tmp_path):
    p=write_features(tmp_path,fixture())
    metadata=json.loads(p.with_suffix('.npz.json').read_text())
    metadata['extractor']['checkpoint_sha256']='a'*64
    p.with_suffix('.npz.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='official Wav2Vec2'):load_features(p,1)


def test_identity_whitespace_cannot_hide_leakage():
    data=fixture();data['speaker_id'][8]=' unit-spk '
    with pytest.raises(ValueError,match='identity'):validate_features(data,1)


def test_model_size_does_not_hide_encoder():
    checkpoint={'ablation':'hybrid','parameter_count':129}
    report=size_accounting(checkpoint,{'parameter_count':1000,'checkpoint_size_bytes':4000})
    assert report['head_parameter_count']==129
    assert report['total_parameter_count']==1129
    assert size_accounting(checkpoint,{})['total_parameter_count'] is None
    assert size_accounting({**checkpoint,'ablation':'acoustic'}, {})['total_parameter_count']==129


@pytest.mark.parametrize('field',['speaker_id','source_id'])
def test_baseline_identity_provenance_must_align(field):
    data=fixture();rows=baseline(data);rows[0][field]='different-provenance'
    with pytest.raises(ValueError,match='speaker/source'):aligned_baseline(rows,data)
