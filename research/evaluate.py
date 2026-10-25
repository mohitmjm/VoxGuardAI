"""Manifest auditing, real inference, validation-only thresholding, and metrics.
Run `python -m research.evaluate --help` from the repository root.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
import numpy as np

FIELDS = {'audio_path', 'label', 'speaker_id', 'source_id', 'generator', 'split', 'language'}


def manifest(path):
    path = Path(path)
    with path.open(newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        if not FIELDS.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing manifest columns: {sorted(FIELDS - set(reader.fieldnames or []))}')
        rows = list(reader)
    if not rows:
        raise ValueError('Manifest is empty')
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in FIELDS):
            raise ValueError('Required manifest fields cannot be empty')
        if row['label'] not in {'human', 'ai'} or row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('Labels must be human/ai and splits train/validation/test')
        row['audio_path'] = str((path.parent / row['audio_path']).resolve())
        if not Path(row['audio_path']).is_file():
            raise ValueError('A manifest recording is missing')
    return rows


def audit(rows, held_out_generators=False):
    """Conservative audit: global speaker/source IDs and exact file content."""
    errors = []
    for field in ('speaker_id', 'source_id', 'audio_path'):
        seen = {}
        for row in rows:
            previous = seen.setdefault(row[field], row['split'])
            if previous != row['split']:
                errors.append(f'{field} spans {previous} and {row["split"]}: {row[field]}')
    hashes = {}
    for row in rows:
        with Path(row['audio_path']).open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        previous = hashes.setdefault(digest, row['split'])
        if previous != row['split']:
            errors.append('Identical audio bytes span splits')
    if held_out_generators:
        placeholders = {'unknown','unspecified','unavailable','none','null','n/a','na','other','spoof','synthetic','ai','tts','voice-cloning'}
        for row in rows:
            if row.get('label') != 'ai':
                continue
            name = str(row.get('generator', '')).strip().lower().replace('_','-')
            final_component = name.replace('/',':').split(':')[-1]
            if not name or name in placeholders or final_component in placeholders or final_component.startswith(('unknown-','unspecified-')):
                errors.append('Held-out generator evaluation requires known synthesis-system identifiers; unknown/unspecified provenance cannot establish an unseen generator')
        train = {r['generator'] for r in rows if r['split'] == 'train' and r['label'] == 'ai'}
        test = {r['generator'] for r in rows if r['split'] == 'test' and r['label'] == 'ai'}
        validation = {r['generator'] for r in rows if r['split'] == 'validation' and r['label'] == 'ai'}
        if (train | validation) & test:
            errors.append('Test generators overlap training/validation generators: ' + ', '.join(sorted((train | validation) & test)))
    if errors:
        raise ValueError('\n'.join(sorted(set(errors))))
    return {'recordings': len(rows), 'splits': {s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test')}, 'audit': 'passed', 'caveat': 'Metadata and byte hashes cannot detect all re-encodings or undisclosed pretrained training overlap.'}


def curves(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not np.all(np.isfinite(scores)) or not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError('Expected matching one-dimensional labels and finite [0,1] scores')
    if set(y.tolist()) != {0, 1}:
        raise ValueError('Both human and AI classes are required')
    # A stable descending sort plus cumulative counts handles tied scores as one
    # operating point. O(n log n), instead of rescanning n scores per threshold.
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], y[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(y) - 1]
    positives = np.cumsum(sorted_labels == 1)
    negatives = np.cumsum(sorted_labels == 0)
    thresholds = np.r_[np.inf, sorted_scores[endpoints], -np.inf]
    tpr = np.r_[0., positives[endpoints] / positives[-1], 1.]
    fpr = np.r_[0., negatives[endpoints] / negatives[-1], 1.]
    return thresholds, tpr, fpr


def metrics(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and in [0,1]')
    if not np.all(np.isfinite(scores)) or not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError('Scores must be finite and in [0,1]')
    _, tpr, fpr = curves(y, scores)
    pred = scores >= threshold
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    diff = fpr - (1 - tpr)
    i = int(np.flatnonzero(diff >= 0)[0])
    if i == 0:
        eer = float(fpr[0])
    else:
        alpha = -diff[i-1] / (diff[i] - diff[i-1])
        eer = float(fpr[i-1] + alpha * (fpr[i] - fpr[i-1]))
    return {'accuracy': float(np.mean(pred == y)), 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'auroc': float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2)), 'eer': eer,
            'false_positive_rate': fp / (fp + tn), 'false_negative_rate': fn / (fn + tp),
            'confusion_matrix': {'labels': ['human', 'ai'], 'rows_true_columns_predicted': [[tn, fp], [fn, tp]]},
            'n': len(y), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def report(rows, threshold, bootstrap=1000, seed=42):
    y = np.array([int(r['label'] == 'ai') for r in rows])
    scores = np.array([float(r['score']) for r in rows])
    if bootstrap < 0:
        raise ValueError('Bootstrap count must be nonnegative')
    result = metrics(y, scores, threshold)
    latency = np.array([float(r['latency_ms']) for r in rows])
    result['latency_ms'] = {'median': float(np.median(latency)), 'p95': float(np.quantile(latency, .95))}
    # Cluster bootstrap preserves within-speaker correlation. Splits must already be audited.
    groups = sorted({r['speaker_id'] for r in rows})
    index = {g: np.array([i for i, r in enumerate(rows) if r['speaker_id'] == g]) for g in groups}
    rng, replicates = np.random.default_rng(seed), []
    for _ in range(bootstrap):
        ids = np.concatenate([index[g] for g in rng.choice(groups, len(groups), replace=True)])
        if len(np.unique(y[ids])) == 2:
            replicates.append(metrics(y[ids], scores[ids], threshold))
    result['bootstrap'] = {'unit': 'speaker', 'seed': seed, 'requested': bootstrap, 'valid_replicates': len(replicates), 'confidence': .95}
    result['confidence_intervals'] = {k: np.quantile([r[k] for r in replicates], [.025, .975]).tolist() for k in ('accuracy','precision','recall','f1','auroc','eer','false_positive_rate','false_negative_rate')} if len(replicates) >= 100 else {}
    result['breakdowns'] = breakdowns(rows, threshold)
    return result


def breakdowns(rows, threshold):
    output = {}
    for field in ('language', 'generator', 'dataset'):
        if not all(field in r for r in rows):
            continue
        output[field] = {}
        for value in sorted({r[field] for r in rows}):
            subset = [r for r in rows if r[field] == value]
            classes = {r['label'] for r in subset}
            output[field][value] = {'n': len(subset), 'human': sum(r['label'] == 'human' for r in subset), 'ai': sum(r['label'] == 'ai' for r in subset), 'metrics': metrics([int(r['label'] == 'ai') for r in subset], [float(r['score']) for r in subset], threshold) if classes == {'human', 'ai'} else None}
    return output


def decode(path):
    proc = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-t', '120', '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=45)
    if proc.returncode:
        raise ValueError('FFmpeg could not decode a research recording')
    audio = np.frombuffer(proc.stdout, dtype='<f4').copy()
    if not len(audio) or not np.all(np.isfinite(audio)):
        raise ValueError('Empty or invalid audio')
    return audio


def score(rows):
    from ml.service import ModelService
    from ml.constants import SHA256
    from ml.policy import SCORE_PROTOCOL
    service = ModelService()
    for row in rows:
        audio = decode(row['audio_path'])
        start = time.perf_counter()
        prediction = service.analyze(audio, sample_rate=16000)
        latency_ms = (time.perf_counter()-start)*1000
        with Path(row['audio_path']).open('rb') as handle:
            audio_digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        yield {**row, 'recording_id': row.get('recording_id') or 'sha256:'+audio_digest, 'audio_sha256': audio_digest, 'score': prediction['score'], 'latency_ms': latency_ms, 'duration_seconds': len(audio)/16000, 'checkpoint_sha256': SHA256, 'score_protocol': SCORE_PROTOCOL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit','score','fit-threshold','report'])
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=Path, help='Frozen validation threshold artifact for report')
    parser.add_argument('--held-out-generators', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.command in {'audit', 'score'}:
        rows = manifest(args.input)
        summary = audit(rows, args.held_out_generators)
        if args.command == 'audit':
            result = summary
        else:
            scored = list(score(rows))
            with args.output.open('w', newline='', encoding='utf8') as f:
                writer = csv.DictWriter(f, fieldnames=list(scored[0]))
                writer.writeheader(); writer.writerows(scored)
            return
    else:
        with args.input.open(newline='', encoding='utf8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError('No scores')
        if any(r.get('label') not in {'human', 'ai'} for r in rows):
            raise ValueError('Score labels must be human/ai')
        if args.command == 'fit-threshold':
            if any(r['split'] != 'validation' for r in rows):
                raise ValueError('Threshold fitting accepts ONLY validation rows; create a separate validation scores CSV')
            y, s = [int(r['label']=='ai') for r in rows], [float(r['score']) for r in rows]
            thresholds, tpr, fpr = curves(y, s)
            idx = int(np.argmin(np.abs(fpr-(1-tpr))))
            threshold = float(thresholds[idx])
            if not math.isfinite(threshold):
                raise ValueError('No finite threshold can be estimated')
            result = {'threshold': threshold, 'method': 'nearest empirical equal-error operating point', 'fit_split': 'validation', 'scores_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'n':len(rows), 'calibrated_probability': False, 'warning': 'Threshold applies only to the model and evaluated domain. This artifact does not automatically enable application decisions.'}
        else:
            if not args.threshold:
                parser.error('report requires --threshold from fit-threshold')
            artifact = json.loads(args.threshold.read_text())
            if artifact.get('fit_split') != 'validation':
                raise ValueError('Invalid threshold provenance')
            if any(r['split'] != 'test' for r in rows):
                raise ValueError('Final report accepts ONLY test rows')
            result = report(rows, float(artifact['threshold']), args.bootstrap)
            result['threshold_artifact'] = artifact
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

if __name__ == '__main__':
    main()
