"""Create frozen learned+LFCC NPZ inputs for experiments.hybrid, never production predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

from research.evaluate import manifest, audit, decode
from ml.representations import Wav2Vec2Features


def recording_identities(rows):
    """Bind rows to bytes; a derived content ID never asserts source provenance."""
    identifiers, checksums, seen_ids, seen_bytes = [], [], set(), set()
    for row in rows:
        with Path(row['audio_path']).open('rb') as handle:
            checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
        identifier = row.get('recording_id') or 'sha256:' + checksum
        if not identifier.strip() or identifier != identifier.strip():
            raise ValueError('Recording identifiers must be nonempty without surrounding whitespace')
        if identifier in seen_ids or checksum in seen_bytes:
            raise ValueError('Repeated recording identity or audio content would inflate feature rows')
        seen_ids.add(identifier); seen_bytes.add(checksum)
        identifiers.append(identifier); checksums.append(checksum)
    return identifiers, checksums


def extract_verified(row, expected_sha256, extractor):
    """Refuse features whose source changed since the manifest identity audit."""
    embedding, coefficients = extractor.extract(decode(row['audio_path']))
    with Path(row['audio_path']).open('rb') as handle:
        observed = hashlib.file_digest(handle, 'sha256').hexdigest()
    if observed != expected_sha256:
        raise ValueError('Recording changed during feature extraction; no matching feature artifact can be trusted')
    return embedding, coefficients


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
    identifiers, checksums = recording_identities(rows)
    torch.set_num_threads(2)
    load_started = time.perf_counter()
    extractor = Wav2Vec2Features(args.cache_dir, args.device)
    extractor_initialization_ms = (time.perf_counter() - load_started) * 1000
    learned, acoustic, extraction_ms = [], [], []
    for row, checksum in zip(rows, checksums):
        started = time.perf_counter()
        embedding, coefficients = extract_verified(row, checksum, extractor)
        learned.append(embedding); acoustic.append(coefficients)
        extraction_ms.append((time.perf_counter() - started) * 1000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, learned=np.stack(learned), acoustic=np.stack(acoustic),
                            label=np.array([int(row['label'] == 'ai') for row in rows]),
                            recording_id=np.array(identifiers), audio_sha256=np.array(checksums),
                            **{field: np.array([row[field] for row in rows]) for field in ('split', 'speaker_id', 'source_id', 'generator', 'language')})
    metadata = {'extractor': extractor.info(), 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'feature_sha256': '', 'audit': audited,
                'recording_identity': 'Original recording_id if supplied; otherwise sha256:<audio bytes digest>, a derived byte identity, not a claimed source/speaker identifier.',
                'recording_decode_cap_seconds': 120,
                'timing': {'extractor_initialization_ms': extractor_initialization_ms,
                           'decode_and_feature_ms_by_recording': dict(zip(identifiers, extraction_ms)),
                           'conditions': 'Measured initialization separately; per-row wall time includes decoding, Wav2Vec2 and LFCC extraction, excludes fusion head.'},
                'limitations': ['Feature extraction is not model training or evaluation.',
                                'No spoof head or multilingual/generalization claim is provided.']}
    with args.output.open('rb') as handle:
        metadata['feature_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    with sidecar.open('x', encoding='utf8') as handle:
        json.dump(metadata, handle, indent=2)
    print(f'Extracted {len(rows)} recordings; provenance saved to {sidecar}. No accuracy claim.')


if __name__ == '__main__':
    main()
