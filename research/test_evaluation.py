"""Numerical sanity checks, not detection-accuracy benchmarks."""
import pytest
from research.evaluate import metrics, audit, report


def test_perfect_and_inverse_rankings():
    perfect = metrics([0,0,1,1], [.1,.2,.8,.9], .5)
    assert perfect['auroc'] == 1 and perfect['eer'] == 0
    inverse = metrics([0,0,1,1], [.9,.8,.2,.1], .5)
    assert inverse['auroc'] == 0 and inverse['eer'] == 1


def test_ties_are_chance_rankings():
    result = metrics([0,1], [.5,.5], .5)
    assert result['auroc'] == .5 and result['eer'] == .5


def test_single_class_and_invalid_scores_rejected():
    with pytest.raises(ValueError): metrics([0,0], [.2,.3], .5)
    with pytest.raises(ValueError): metrics([0,1], [.2,float('nan')], .5)


def test_content_leakage_is_rejected(tmp_path):
    audio = tmp_path/'clip.wav'; audio.write_bytes(b'example')
    rows = [{'audio_path': str(audio), 'speaker_id':str(i), 'source_id':str(i), 'split':s} for i,s in enumerate(['train','test'])]
    with pytest.raises(ValueError, match='spans|Identical'): audit(rows)


def test_speaker_cluster_bootstrap():
    rows=[{'label':'ai' if i%2 else 'human','score':.9 if i%2 else .1,'latency_ms':10,'speaker_id':str(i)} for i in range(10)]
    result=report(rows,.5,150)
    assert result['bootstrap']['valid_replicates'] >= 100
    assert result['confidence_intervals']['accuracy'] == [1.,1.]


def test_sorted_curve_matches_bruteforce_with_ties():
    import numpy as np
    from research.evaluate import curves
    labels=np.array([0,1,0,1,1,0,1])
    scores=np.array([.2,.8,.5,.5,.9,.2,.1])
    thresholds,tpr,fpr=curves(labels,scores)
    reference=np.r_[np.inf,np.sort(np.unique(scores))[::-1],-np.inf]
    assert np.array_equal(thresholds,reference)
    assert np.allclose(tpr,[np.mean(scores[labels==1]>=t) for t in reference])
    assert np.allclose(fpr,[np.mean(scores[labels==0]>=t) for t in reference])
