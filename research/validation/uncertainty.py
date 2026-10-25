"""Speaker/source-cluster uncertainty for frozen, already-audited test scores.

Resampling units are connected components of speaker AND source identities.
Intervals are descriptive percentile intervals; they do not repair selection bias.
"""
from __future__ import annotations

import math
import numpy as np
from research.evaluate import curves


def clusters(rows):
    if not rows:
        raise ValueError('Nonempty test rows required')
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for field in ('speaker_id', 'source_id'):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip() or value.strip().lower() in {'unknown','-','n/a'}:
                raise ValueError('Known speaker and source identities required for clustered intervals')
            key = (field, value)
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def statistics(rows, policy=None):
    if any(r.get('label') not in {'human','ai'} for r in rows):
        raise ValueError('Only human/ai labels are supported')
    y = np.asarray([r['label'] == 'ai' for r in rows], dtype=int)
    scores = np.asarray([float(r['score']) for r in rows])
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Scores must be finite within [0,1]')
    result = {}
    if len(set(y)) == 2:
        _, tpr, fpr = curves(y, scores)
        diff = fpr-(1-tpr)
        i = int(np.flatnonzero(diff >= 0)[0])
        alpha = 0 if i == 0 else -diff[i-1]/(diff[i]-diff[i-1])
        result.update(auroc=float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2)),
                      eer=float(fpr[0] if i == 0 else fpr[i-1]+alpha*(fpr[i]-fpr[i-1])))
    if policy is not None:
        pred = [policy.classify(float(s)) for s in scores]
        tp = sum(p == 'ai' and label == 1 for p,label in zip(pred,y))
        fp = sum(p == 'ai' and label == 0 for p,label in zip(pred,y))
        fn = sum(p == 'human' and label == 1 for p,label in zip(pred,y))
        correct = sum(p == ('ai' if label else 'human') for p,label in zip(pred,y))
        accepted = sum(p != 'uncertain' for p in pred)
        precision = tp/(tp+fp) if tp+fp else None
        recall = tp/int(y.sum()) if y.sum() else None
        result.update(coverage=accepted/len(rows), accuracy_on_accepted=correct/accepted if accepted else None,
                      accuracy_counting_abstention_as_incorrect=correct/len(rows),
                      precision_on_ai_decisions=precision, recall_counting_abstention_as_miss=recall,
                      f1_counting_abstention_as_miss=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else (0.0 if precision is not None and recall is not None else None),
                      false_positive_rate=fp/int((y==0).sum()) if (y==0).any() else None,
                      false_negative_rate_excluding_abstention=fn/int(y.sum()) if y.sum() else None)
    return result


def bootstrap(rows, *, policy=None, replicates=1000, seed=42, min_clusters=10):
    if not isinstance(replicates,int) or isinstance(replicates,bool) or replicates < 100 or replicates > 10000:
        raise ValueError('Use 100..10000 bootstrap replicates')
    if not isinstance(min_clusters,int) or isinstance(min_clusters,bool) or min_clusters < 2:
        raise ValueError('At least two independent clusters are required')
    groups = clusters(rows)
    point = statistics(rows,policy)
    if not {'auroc','eer'}.issubset(point):
        raise ValueError('Both classes required for test uncertainty')
    result = {'method':'percentile connected speaker/source-cluster bootstrap', 'confidence':.95,
              'seed':seed,'requested_replicates':replicates,'cluster_count':len(groups),
              'minimum_clusters':min_clusters,'point_estimates':point,'intervals':{},
              'limitations':['Intervals condition on the supplied sample/provenance; they do not establish representativeness or pretrained independence.',
                             'Speaker/source connections are jointly clustered; incorrect or missing provenance can still invalidate uncertainty.',
                             'Percentile intervals may be unstable with few clusters, rare errors or degenerate/perfect rankings.']}
    if len(groups) < min_clusters:
        return {**result,'status':'insufficient_clusters','valid_replicates':{}}
    rng = np.random.default_rng(seed)
    values = {key:[] for key in point}
    for _ in range(replicates):
        indices = [i for group in rng.integers(0,len(groups),len(groups)) for i in groups[group]]
        estimate = statistics([rows[i] for i in indices],policy)
        for key in values:
            value = estimate.get(key)
            if value is not None and math.isfinite(value): values[key].append(value)
    result['valid_replicates'] = {k:len(v) for k,v in values.items()}
    result['intervals'] = {k:np.quantile(v,[.025,.975]).tolist() for k,v in values.items() if len(v)>=100}
    result['status'] = 'estimated' if result['intervals'] else 'insufficient_valid_replicates'
    return result


def generator_breakdown(rows):
    """Reference all supplied humans for each synthetic family, with no unseen claim."""
    humans = [r for r in rows if r['label']=='human']
    groups = sorted({r.get('generator','unknown') for r in rows if r['label']=='ai'})
    summaries = {}
    for group in groups:
        subset = humans+[r for r in rows if r['label']=='ai' and r.get('generator','unknown')==group]
        summaries[group] = {'human_reference_count':len(humans),'synthetic_count':len(subset)-len(humans),
                            'ranking':statistics(subset),'pretrained_exposure':'unknown'}
    return {'groups':summaries,'warning':'The same human reference recordings are reused across families; these are dependent descriptive comparisons. Generator names do not establish whether a pretrained model saw a related synthesis family.'}
