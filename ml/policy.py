"""Opt-in, validation-derived decisions. Empirical error is not a guarantee."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

SCORE_PROTOCOL = 'aasist-spoof-softmax-duration-weighted-64600-v1'


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class DecisionPolicy:
    artifact: dict

    def classify(self, score: float) -> str:
        if not _finite_number(score) or not 0 <= score <= 1:
            raise ValueError('Model score must be finite and within [0, 1]')
        if score <= self.artifact['human_max']:
            return 'human'
        if score >= self.artifact['ai_min']:
            return 'ai'
        return 'uncertain'

    def metadata(self) -> dict:
        return {**self.artifact, 'warning': 'Validation-set error rates are empirical, not guarantees. Only apply to the declared domain. Scores are not calibrated probabilities.'}


def validate_artifact(artifact: dict, expected_checkpoint_sha256: str) -> DecisionPolicy:
    if not isinstance(artifact, dict) or artifact.get('version') != 1 or artifact.get('fit_split') != 'validation':
        raise ValueError('Unsupported policy or non-validation provenance')
    if not re.fullmatch(r'[0-9a-f]{64}', expected_checkpoint_sha256) or artifact.get('checkpoint_sha256') != expected_checkpoint_sha256:
        raise ValueError('Policy checkpoint does not match the loaded model')
    if artifact.get('score_protocol') != SCORE_PROTOCOL:
        raise ValueError('Policy score protocol does not match inference')
    if not re.fullmatch(r'[0-9a-f]{64}', artifact.get('validation_scores_sha256', '')):
        raise ValueError('Policy requires a validation score digest')
    low, high = artifact.get('human_max'), artifact.get('ai_min')
    if not _finite_number(low) or not _finite_number(high) or not 0 <= low < high <= 1:
        raise ValueError('Policy requires finite, ordered, non-overlapping thresholds')
    scope = artifact.get('scope')
    if not isinstance(scope, dict) or not isinstance(scope.get('domain'), str) or not scope['domain'].strip() or not isinstance(scope.get('languages'), list) or not scope['languages'] or any(not isinstance(x, str) or not x.strip() for x in scope['languages']):
        raise ValueError('Policy requires explicit domain and language scope')
    counts = artifact.get('counts', {})
    if not isinstance(counts, dict) or any(not isinstance(counts.get(k), int) or isinstance(counts[k], bool) or counts[k] < 1 for k in ('human', 'ai', 'human_speakers', 'ai_speakers')):
        raise ValueError('Policy requires positive validation counts')
    rates = artifact.get('empirical_rates', {})
    if not isinstance(rates, dict) or any(not _finite_number(rates.get(k)) or not 0 <= rates[k] <= 1 for k in ('human_as_ai', 'ai_as_human', 'coverage')):
        raise ValueError('Policy requires finite empirical validation rates')
    return DecisionPolicy(artifact)


def load_policy(expected_checkpoint_sha256: str, path: str | Path | None = None) -> DecisionPolicy | None:
    configured = path if path is not None else os.getenv('VOXGUARD_POLICY_PATH')
    if not configured:
        return None
    try:
        with Path(configured).open(encoding='utf8') as handle:
            artifact = json.load(handle)
        return validate_artifact(artifact, expected_checkpoint_sha256)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError('Unable to read a valid configured decision policy') from exc
