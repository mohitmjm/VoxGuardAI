"""Optional research-only frozen speech representations; never an anti-spoof detector."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from ml.features import lfcc

WAV2VEC2_SHA256 = '1275f890a623b31b9915295dc01d5049c93dc6de8813406c16dfab9b6bd36056'


def _digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class Wav2Vec2Features:
    """Official TorchAudio WAV2VEC2_BASE, pretrained on LibriSpeech 960h.

    Uses the bundle's unnormalized 16 kHz waveform input and final transformer
    layer. Four-second nonoverlapping windows bound quadratic attention memory.
    Per-window temporal mean/std are duration-weighted across a recording. This
    pooling is our research feature definition, not an established spoof model.
    """
    def __init__(self, cache_dir=None, device='cpu'):
        try:
            import torchaudio
        except (ImportError, OSError) as exc:
            raise RuntimeError('Install the optional CPU torchaudio==2.6.0 package alongside torch==2.6.0.') from exc
        if torchaudio.__version__.split('+')[0] != '2.6.0':
            raise RuntimeError('Research extractor requires the documented torchaudio==2.6.0 version.')
        self.device = device
        self.cache = Path(cache_dir or os.getenv('VOXGUARD_REPRESENTATION_CACHE', '.cache/representations'))
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache / 'wav2vec2_fairseq_base_ls960.pth'
        if path.is_file() and _digest(path) != WAV2VEC2_SHA256:
            raise RuntimeError('Optional Wav2Vec2 checkpoint integrity mismatch; remove the corrupt cache and retry.')
        self.model = torchaudio.pipelines.WAV2VEC2_BASE.get_model(
            dl_kwargs={'model_dir': str(self.cache), 'weights_only': True, 'progress': False}
        ).to(device).eval()
        self.digest = _digest(path)
        if self.digest != WAV2VEC2_SHA256:
            raise RuntimeError('Official Wav2Vec2 download differs from the recorded checkpoint digest.')

    def info(self):
        return {'name': 'TorchAudio WAV2VEC2_BASE', 'torch_version': torch.__version__,
                'torchaudio_version': '2.6.0', 'training_data': 'LibriSpeech 960h (English)',
                'source': 'https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960.pth',
                'checkpoint_sha256': self.digest,
                'digest_status': 'enforced recorded official HTTPS artifact digest; not an author-signed checksum',
                'layer': 12, 'sample_rate': 16000, 'window_samples': 64000,
                'pooling': 'duration-weighted window temporal mean and population std',
                'learned_dimensions': 1536, 'acoustic_dimensions': 40,
                'lfcc': {'n_fft': 512, 'hop_length': 160, 'filters': 60, 'coefficients': 20,
                         'window': 'Hann centered STFT', 'dct': 'orthonormal DCT-II',
                         'deltas': False, 'pooling': 'window temporal mean/std then duration-weighted'},
                'purpose': 'frozen research features only; no spoof detection head or probability',
                'normalization': 'none, matching TorchAudio WAV2VEC2_BASE bundle',
                'license_notes': 'TorchAudio BSD-2-Clause; fairseq MIT; LibriSpeech CC BY 4.0. Review weight and dataset rights for your intended use.'}

    def extract(self, samples, sample_rate=16000):
        x = np.asarray(samples, dtype=np.float32)
        if sample_rate != 16000 or x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError('Expected finite nonempty mono 16 kHz waveform.')
        learned, acoustic, weights = [], [], []
        with torch.inference_mode():
            for start in range(0, len(x), 64000):
                chunk = x[start:start + 64000]
                weights.append(len(chunk))
                if len(chunk) < 512:
                    chunk = np.resize(chunk, 512)
                signal = torch.from_numpy(chunk.copy()).to(self.device)
                layers, _ = self.model.extract_features(signal[None], num_layers=12)
                final = layers[-1][0]
                learned.append(torch.cat([final.mean(0), final.std(0, unbiased=False)]).cpu().numpy())
                coefficients = lfcc(chunk)
                acoustic.append(torch.cat([coefficients.mean(1), coefficients.std(1, unbiased=False)]).numpy())
        return (np.average(learned, axis=0, weights=weights).astype(np.float32),
                np.average(acoustic, axis=0, weights=weights).astype(np.float32))
