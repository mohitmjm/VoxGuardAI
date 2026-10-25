"""Exploratory ranking metrics for a licensed pilot, without selecting decision thresholds."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from research.evaluate import curves
from ml.policy import SCORE_PROTOCOL


def summarize(rows, dataset, conditions):
    if not rows or any(r.get('label') not in {'human', 'ai'} for r in rows):
        raise ValueError('Nonempty human/ai scores are required')
    scores = np.array([float(r['score']) for r in rows])
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Scores must be finite and within [0,1]')
    if any(r.get('score_protocol') != SCORE_PROTOCOL for r in rows):
        raise ValueError('Unexpected or missing score protocol')
    _, tpr, fpr = curves([int(r['label'] == 'ai') for r in rows], scores)
    diff = fpr - (1 - tpr)
    i = int(np.flatnonzero(diff >= 0)[0])
    alpha = 0 if i == 0 else -diff[i-1]/(diff[i]-diff[i-1])
    eer = float(fpr[0] if i == 0 else fpr[i-1]+alpha*(fpr[i]-fpr[i-1]))
    models = {r.get('checkpoint_sha256', '') for r in rows}
    if len(models) != 1 or not re.fullmatch(r'[0-9a-f]{64}', next(iter(models))):
        raise ValueError('A single recorded checkpoint hash is required')
    latency = np.array([float(r['latency_ms']) for r in rows])
    if not np.isfinite(latency).all() or (latency < 0).any():
        raise ValueError('Invalid latency measurements')
    return {'dataset': dataset, 'sample_count': len(rows), 'class_counts': {label: sum(r['label'] == label for r in rows) for label in ('human','ai')},
            'model': 'AASIST', 'checkpoint_sha256': next(iter(models)), 'conditions': conditions,
            'metrics': {'auroc': float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2)), 'eer': eer},
            'latency_ms': {'median': float(np.median(latency)), 'p95': float(np.quantile(latency,.95)), 'first_call': float(latency[0])},
            'threshold': None, 'decision_metrics': None,
            'limitations': ['Exploratory convenience pilot, not a benchmark or deployment validation set.',
                            'No decision threshold fitted; no binary accuracy, confidence, or generalization claim.',
                            'EER is a descriptive curve statistic, not an enabled operating threshold.',
                            'Very small sample size and synthesis/domain mismatch preclude reliable population estimates.'],
            'languages': sorted({r.get('language','unknown') for r in rows}),
            'generators': sorted({r.get('generator','unknown') for r in rows if r['label']=='ai'})}



def dashboard_report(result):
    return {key: result[key] for key in ('dataset','sample_count','model','metrics')} | {'conditions': result['conditions'] + '; Exploratory six-recording convenience pilot only: no binary accuracy, calibration, representative benchmark or modern-generator robustness claim.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--conditions', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dashboard-output', type=Path, help='Optional sanitized API dashboard report (separate from full provenance result)')
    args = parser.parse_args()
    with args.scores.open(newline='',encoding='utf8') as handle:
        rows = list(csv.DictReader(handle))
    result = summarize(rows,args.dataset,args.conditions)
    result['scores_sha256'] = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    with args.output.open('x',encoding='utf8') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    if args.dashboard_output:
        with args.dashboard_output.open('x',encoding='utf8') as handle:
            json.dump({'reports':[dashboard_report(result)]},handle,indent=2,allow_nan=False);handle.write('\n')

if __name__=='__main__': main()
