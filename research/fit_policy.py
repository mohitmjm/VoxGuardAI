"""Fit abstaining thresholds from validation CSVs only; never calibrates probabilities.

Required CSV columns: label, score, split, speaker_id, source_id, language.
Score protocol must match frozen production inference. A separate untouched test
set must assess the frozen policy before deployment to its declared domain.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from ml.policy import SCORE_PROTOCOL, validate_artifact


def fit(rows, *, checkpoint_sha256, scores_sha256, domain, max_error=.01, min_count=100, min_speakers=10):
    if not 0 <= max_error < .5 or min_count < 1 or min_speakers < 1:
        raise ValueError('Invalid fitting constraints')
    if not rows:
        raise ValueError('Validation CSV is empty')
    required = {'label', 'score', 'split', 'speaker_id', 'source_id', 'language'}
    for row in rows:
        if not required.issubset(row) or any(not str(row[k]).strip() for k in required):
            raise ValueError('Missing required validation fields')
        if row['split'] != 'validation' or row['label'] not in {'human', 'ai'}:
            raise ValueError('Only validation rows labeled human/ai are accepted')
        score = float(row['score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Scores must be finite and within [0, 1]')
    human = [float(r['score']) for r in rows if r['label'] == 'human']
    ai = [float(r['score']) for r in rows if r['label'] == 'ai']
    counts = {label: sum(r['label'] == label for r in rows) for label in ('human','ai')}
    counts.update({label+'_speakers': len({r['speaker_id'] for r in rows if r['label'] == label}) for label in ('human','ai')})
    if min(counts['human'], counts['ai']) < min_count or min(counts['human_speakers'], counts['ai_speakers']) < min_speakers:
        raise ValueError('Insufficient validation recordings or speakers per class')
    # Inclusive rules match production. Require genuine accepted examples on both
    # sides; fail rather than invent a viable threshold for inseparable scores.
    lows = [s for s in sorted(set(human)) if sum(a <= s for a in ai)/len(ai) <= max_error]
    highs = [s for s in sorted(set(ai)) if sum(h >= s for h in human)/len(human) <= max_error]
    if not lows or not highs or max(lows) >= min(highs):
        raise ValueError('No separated thresholds meet the requested empirical error constraints')
    low, high = max(lows), min(highs)
    artifact = {'version': 1, 'fit_split': 'validation', 'checkpoint_sha256': checkpoint_sha256,
                'score_protocol': SCORE_PROTOCOL, 'validation_scores_sha256': scores_sha256,
                'human_max': low, 'ai_min': high, 'counts': counts,
                'scope': {'domain': domain, 'languages': sorted({r['language'] for r in rows})},
                'empirical_rates': {'human_as_ai': sum(h >= high for h in human)/len(human),
                                    'ai_as_human': sum(a <= low for a in ai)/len(ai),
                                    'coverage': sum(float(r['score']) <= low or float(r['score']) >= high for r in rows)/len(rows)},
                'fit_constraints': {'max_empirical_error_per_class': max_error, 'min_recordings_per_class': min_count, 'min_speakers_per_class': min_speakers},
                'limitations': ['Threshold selection reuses validation scores; validate the frozen policy on an untouched test set.',
                                'No statistical guarantee, probability calibration, or generalization to other languages/generators is established.']}
    validate_artifact(artifact, checkpoint_sha256)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-error', type=float, default=.01)
    parser.add_argument('--min-count', type=int, default=100)
    parser.add_argument('--min-speakers', type=int, default=10)
    args = parser.parse_args()
    if args.output.resolve() == args.scores.resolve():
        parser.error('Output must not overwrite validation scores')
    with args.scores.open(newline='', encoding='utf8') as handle:
        rows = list(csv.DictReader(handle))
    artifact = fit(rows, checkpoint_sha256=args.checkpoint_sha256,
                   scores_sha256=hashlib.sha256(args.scores.read_bytes()).hexdigest(), domain=args.domain,
                   max_error=args.max_error, min_count=args.min_count, min_speakers=args.min_speakers)
    # Exclusive creation preserves an existing reviewed policy.
    with args.output.open('x', encoding='utf8') as handle:
        json.dump(artifact, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('Validation policy saved. Untouched test evaluation and domain review are still required.')


if __name__ == '__main__':
    main()
