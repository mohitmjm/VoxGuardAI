"""Create frozen learned+LFCC NPZ inputs for experiments.hybrid, never production predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from research.evaluate import manifest, audit, decode
from ml.representations import Wav2Vec2Features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + '.json')
    if args.output.exists() or sidecar.exists():
        parser.error('Output already exists; choose a new path to preserve existing experiment artifacts.')
    rows = manifest(args.manifest)
    audited = audit(rows)
    torch.set_num_threads(2)
    extractor = Wav2Vec2Features(args.cache_dir, args.device)
    learned, acoustic = [], []
    for row in rows:
        embedding, coefficients = extractor.extract(decode(row['audio_path']))
        learned.append(embedding); acoustic.append(coefficients)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, learned=np.stack(learned), acoustic=np.stack(acoustic),
                            label=np.array([int(row['label'] == 'ai') for row in rows]),
                            **{field: np.array([row[field] for row in rows]) for field in ('split', 'speaker_id', 'source_id', 'generator', 'language')})
    metadata = {'extractor': extractor.info(), 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'feature_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'audit': audited,
                'recording_decode_cap_seconds': 120,
                'limitations': ['Feature extraction is not model training or evaluation.',
                                'No spoof head or multilingual/generalization claim is provided.']}
    with sidecar.open('x', encoding='utf8') as handle:
        json.dump(metadata, handle, indent=2)
    print(f'Extracted {len(rows)} recordings; provenance saved to {sidecar}. No accuracy claim.')


if __name__ == '__main__':
    main()
