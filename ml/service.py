"""Official pretrained AASIST, without unsupported confidence claims."""
import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np
import torch

from ml.vendor.aasist import Model
from ml.policy import load_policy

from ml.constants import ROOT, SHA256, COMMIT, WINDOW

class ModelUnavailable(RuntimeError):
    pass

class ModelService:
    def __init__(self):
        self.path = Path(os.getenv('VOXGUARD_CHECKPOINT', str(ROOT / 'checkpoints/AASIST.pth')))
        self.device = os.getenv('VOXGUARD_DEVICE', 'cpu')
        self._model = None
        self._lock = threading.Lock()
        try:
            self.policy = load_policy(expected_checkpoint_sha256=SHA256)
        except ValueError as exc:
            raise ModelUnavailable("Configured decision policy is invalid; review its model and validation provenance.") from exc
        torch.set_num_threads(max(1, int(os.getenv('VOXGUARD_TORCH_THREADS', '2'))))

    def info(self):
        return {
            'name': 'AASIST', 'version': COMMIT, 'sample_rate': 16000,
            'window_samples': WINDOW, 'license': 'MIT (NAVER official repository)',
            'source': 'https://github.com/clovaai/aasist',
            'checkpoint_sha256': SHA256, 'available': self.path.is_file(),
            'loaded': self._model is not None, 'device': self.device,
            'training_dataset': 'ASVspoof 2019 Logical Access',
            'score_type': 'uncalibrated spoof-class softmax score; not a probability of authenticity',
            'thresholds': self.policy.metadata() if self.policy else None,
            'decision_policy': 'validation-derived opt-in' if self.policy else 'abstain-until-validated',
            'evaluated_languages': [],
        }

    def _load(self):
        if self._model is not None:
            return
        if not self.path.is_file():
            raise ModelUnavailable('AASIST checkpoint missing. Run python -m ml.download_model.')
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != SHA256:
            raise ModelUnavailable('AASIST checkpoint checksum mismatch; download the verified checkpoint again.')
        try:
            config = json.loads((ROOT / 'aasist_config.json').read_text())['model_config']
            model = Model(config)
            model.load_state_dict(torch.load(self.path, map_location='cpu', weights_only=True), strict=True)
            self._model = model.to(self.device).eval()
        except Exception as exc:
            raise ModelUnavailable('Unable to load AASIST; verify dependencies and device configuration.') from exc

    def analyze(self, samples, sample_rate=16000):
        samples = np.asarray(samples, dtype=np.float32)
        if sample_rate != 16000 or samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise ValueError('Expected nonempty finite mono 16 kHz samples.')
        segments = []
        # Upstream repeats short utterances to exactly 64600 samples. Long recordings
        # use contiguous windows; the final short tail follows that same rule.
        with self._lock, torch.inference_mode():
            self._load()
            for start in range(0, len(samples), WINDOW):
                end = min(start + WINDOW, len(samples))
                window = samples[start:end]
                if len(window) < WINDOW:
                    window = np.tile(window, int(np.ceil(WINDOW / len(window))))[:WINDOW]
                tensor = torch.from_numpy(window.copy()).unsqueeze(0).to(self.device)
                _, logits = self._model(tensor)
                score = float(torch.softmax(logits, dim=-1)[0, 0].cpu())
                segments.append({'start': start / 16000, 'end': end / 16000, 'score': score})
        score = float(np.average([s['score'] for s in segments], weights=[s['end']-s['start'] for s in segments]))
        classification = self.policy.classify(score) if self.policy else 'uncertain'
        return {'classification': classification, 'score': score, 'segments': segments,
                'model': self.info(), 'limitations': [
                    ('Opt-in validation-derived thresholds apply only to the declared research domain; uploaded audio domain is not automatically verified.' if self.policy else 'No deployment validation thresholds are available; classification remains Uncertain.'),
                    'Scores are uncalibrated and cannot prove recording authenticity.',
                    'Modern unseen generators, Hindi and Marathi have not been evaluated.',
                    'Window scores do not locate exact synthetic edits; short windows are repeat-padded.',
                    'No speech activity detector is applied; silence, music and noise may produce misleading scores.',
                ]}
