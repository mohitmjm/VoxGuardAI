"""Evaluate a frozen two-threshold policy on untouched test scores, including abstentions."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
from ml.policy import validate_artifact, SCORE_PROTOCOL
from research.evaluate import curves


def evaluate(rows, artifact, expected_checkpoint_sha256):
    policy = validate_artifact(artifact, expected_checkpoint_sha256)
    if not rows or any(r.get('split') != 'test' or r.get('label') not in {'human', 'ai'} for r in rows):
        raise ValueError('Policy evaluation accepts nonempty test rows only')
    for field in ('checkpoint_sha256', 'score_protocol'):
        expected = expected_checkpoint_sha256 if field == 'checkpoint_sha256' else SCORE_PROTOCOL
        if any(r.get(field) != expected for r in rows):
            raise ValueError('Test score provenance does not match the policy')
    ids = artifact.get('validation_id_hashes', {})
    for field in ('speaker_id', 'source_id'):
        if not ids.get(field):
            raise ValueError('Policy lacks validation identity hashes; refit with the current pipeline')
        if any(not r.get(field) for r in rows):
            raise ValueError('Test identity metadata is required')
        if any(hashlib.sha256(r[field].encode()).hexdigest() in ids[field] for r in rows):
            raise ValueError(f'Validation/test {field} leakage detected')
    validated_audio = set(artifact.get('validation_audio_sha256', []))
    if validated_audio:
        hashes = [r.get('audio_sha256') for r in rows]
        if any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value) for value in hashes):
            raise ValueError('Test audio digest provenance is required')
        if len(set(hashes)) != len(hashes):
            raise ValueError('Duplicate test recording content')
        if validated_audio.intersection(hashes):
            raise ValueError('Validation/test recording-content leakage detected')
    scores = [float(r['score']) for r in rows]
    predictions = [policy.classify(s) for s in scores]
    matrix = [[sum(r['label'] == label and p == prediction for r, p in zip(rows, predictions)) for prediction in ('human', 'ai', 'uncertain')] for label in ('human', 'ai')]
    human, ai = (sum(matrix[i]) for i in (0, 1))
    if not human or not ai:
        raise ValueError('Both test classes are required')
    accepted = sum(p != 'uncertain' for p in predictions)
    correct = matrix[0][0] + matrix[1][1]
    tp, fp, fn = matrix[1][1], matrix[0][1], matrix[1][0]
    precision = tp / (tp + fp) if tp + fp else None
    # Recall below treats abstaining AI recordings as unrecovered, and is clearly named.
    recall = tp / ai
    _, tpr, fpr = curves([int(r['label'] == 'ai') for r in rows], scores)
    difference = fpr - (1 - tpr)
    i = int(np.flatnonzero(difference >= 0)[0])
    alpha = 0 if i == 0 else -difference[i-1] / (difference[i] - difference[i-1])
    eer = float(fpr[0] if i == 0 else fpr[i-1] + alpha * (fpr[i] - fpr[i-1]))
    latencies = [float(r['latency_ms']) for r in rows]
    if any(not math.isfinite(x) or x < 0 for x in latencies):
        raise ValueError('Invalid latency')
    return {'sample_count': len(rows), 'class_counts': {'human': human, 'ai': ai},
            'coverage': accepted / len(rows), 'abstention_rate': 1 - accepted / len(rows),
            'accuracy_on_accepted': correct / accepted if accepted else None,
            'accuracy_counting_abstention_as_incorrect': correct / len(rows),
            'precision_on_ai_decisions': precision, 'recall_counting_abstention_as_miss': recall,
            'f1_counting_abstention_as_miss': 2*precision*recall/(precision+recall) if precision is not None and precision+recall else (0.0 if precision is not None else None),
            'false_positive_rate': fp/human, 'false_negative_rate_excluding_abstention': fn/ai,
            'human_abstention_rate': matrix[0][2]/human, 'ai_abstention_rate': matrix[1][2]/ai,
            'auroc': float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2)), 'eer': eer,
            'confusion_matrix': {'rows_true': ['human', 'ai'], 'columns_predicted': ['human', 'ai', 'uncertain'], 'counts': matrix},
            'latency_ms': {'median': float(np.median(latencies)), 'p95': float(np.quantile(latencies, .95))},
            'scope': artifact['scope'], 'checkpoint_sha256': expected_checkpoint_sha256,
            'limitations': ['Conditional accepted accuracy excludes abstentions. Report coverage alongside it.', 'No generalization or calibrated probability claim follows from this evaluation.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.scores.open(newline='', encoding='utf8') as handle:
        rows = list(csv.DictReader(handle))
    artifact = json.loads(args.policy.read_text())
    result = evaluate(rows, artifact, args.checkpoint_sha256)
    result['test_scores_sha256'] = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    result['policy_sha256'] = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')

if __name__ == '__main__':
    main()
