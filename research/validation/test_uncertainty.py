"""Statistical/numerical fixtures only; never benchmark detector results."""
import json
import pytest
from research.validation.uncertainty import clusters, bootstrap, statistics, generator_breakdown


def rows(n=20):
    return [dict(label='human' if i%2==0 else 'ai',score=.1 if i%2==0 else .9,
                 speaker_id=f'speaker-{i}',source_id=f'source-{i}',generator='human' if i%2==0 else 'fixture-generator') for i in range(n)]


def test_joint_components_prevent_source_speaker_pseudoreplication():
    data=rows(4)
    data[1]['speaker_id']=data[0]['speaker_id']
    data[2]['source_id']=data[1]['source_id']
    assert sorted(map(len,clusters(data))) == [1,3]


def test_tiny_pilot_does_not_receive_intervals():
    result=bootstrap(rows(6),replicates=150)
    assert result['status']=='insufficient_clusters'
    assert result['intervals']=={} and result['cluster_count']==6


def test_bootstrap_reproducibility_and_valid_replicate_counts():
    first=bootstrap(rows(),replicates=150)
    assert first==bootstrap(rows(),replicates=150)
    assert first['status']=='estimated'
    assert first['intervals']['auroc']==[1.,1.]
    assert first['valid_replicates']['eer'] >=100
    # A degenerate interval on a fixture is not a generalization guarantee.
    assert 'degenerate' in ' '.join(first['limitations'])


def test_policy_intervals_include_abstention_and_undefined_precision():
    class Abstain:
        def classify(self,score): return 'uncertain'
    result=bootstrap(rows(),policy=Abstain(),replicates=150)
    assert result['point_estimates']['coverage']==0
    assert result['point_estimates']['precision_on_ai_decisions'] is None
    assert result['valid_replicates']['precision_on_ai_decisions']==0
    assert 'precision_on_ai_decisions' not in result['intervals']
    json.dumps(result,allow_nan=False)


def test_wrong_decisions_have_zero_f1_when_precision_defined():
    class Wrong:
        def classify(self,score): return 'ai' if score<.5 else 'human'
    assert statistics(rows(),Wrong())['f1_counting_abstention_as_miss']==0


@pytest.mark.parametrize('value',[float('nan'),-1,2])
def test_bad_scores_rejected(value):
    data=rows();data[0]['score']=value
    with pytest.raises(ValueError):bootstrap(data,replicates=150)


def test_unknown_identity_and_one_class_rejected():
    data=rows();data[0]['source_id']='unknown'
    with pytest.raises(ValueError):bootstrap(data,replicates=150)
    with pytest.raises(ValueError):bootstrap([r for r in rows() if r['label']=='human'],replicates=150)


def test_generator_reference_reuse_is_explicit_not_unseen_claim():
    result=generator_breakdown(rows())
    assert result['groups']['fixture-generator']['human_reference_count']==10
    assert result['groups']['fixture-generator']['pretrained_exposure']=='unknown'
    assert 'dependent' in result['warning']
