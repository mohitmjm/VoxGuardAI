"""Numerical fixtures test metric definitions and leakage rules, not model accuracy."""
import copy
import hashlib
import pytest
from ml.constants import SHA256
from ml.policy import SCORE_PROTOCOL
from research.fit_policy import fit
from research.report_policy import evaluate
from research.pilot_report import summarize
from research.evaluate import metrics, breakdowns


def rows(split='validation'):
    return [{'label':label,'score':score,'split':split,'speaker_id':f'{split}-speaker-{i}', 'source_id':f'{split}-source-{i}', 'language':'en', 'latency_ms':10, 'checkpoint_sha256':SHA256,'score_protocol':SCORE_PROTOCOL} for i,(label,score) in enumerate([('human',.1),('human',.2),('ai',.8),('ai',.9)])]


def policy():
    return fit(rows(),checkpoint_sha256=SHA256,scores_sha256='a'*64,domain='numerical unit fixture only',min_count=2,min_speakers=2)


def test_policy_report_keeps_abstentions_in_denominators():
    test=rows('test');test[0]['score']=.5
    result=evaluate(test,policy(),SHA256)
    assert result['coverage']==.75
    assert result['accuracy_on_accepted']==1
    assert result['accuracy_counting_abstention_as_incorrect']==.75
    assert result['confusion_matrix']['counts']==[[1,0,1],[0,2,0]]


def test_policy_report_rejects_leakage_and_provenance():
    test=rows('test');test[0]['source_id']=rows()[0]['source_id']
    with pytest.raises(ValueError,match='leakage'):evaluate(test,policy(),SHA256)
    test=rows('test');test[0]['checkpoint_sha256']='b'*64
    with pytest.raises(ValueError,match='provenance'):evaluate(test,policy(),SHA256)


def test_single_threshold_fnr_and_matrix():
    result=metrics([0,0,1,1],[.1,.9,.2,.8],.5)
    assert result['false_negative_rate']==.5
    assert result['confusion_matrix']['rows_true_columns_predicted']==[[1,1],[1,1]]
    with pytest.raises(ValueError):metrics([0,1],[.1,.9],float('nan'))


def test_rank_only_pilot_omits_decisions():
    result=summarize(rows('test'),'unit fixture','numerical only')
    assert result['metrics']=={'auroc':1.,'eer':0.}
    assert result['threshold'] is None and result['decision_metrics'] is None


def test_single_class_domain_breakdown_is_explicit():
    data=rows('test')
    for r in data:r['generator']='human' if r['label']=='human' else 'fixture'
    result=breakdowns(data,.5)
    assert result['generator']['fixture']['metrics'] is None
