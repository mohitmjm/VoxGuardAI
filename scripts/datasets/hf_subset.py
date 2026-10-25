"""Fetch explicitly reviewed HF files at an immutable revision, with a byte budget.

Optional dependency: huggingface_hub. No network requests occur during import.
This command does not infer recording labels, source IDs or license permission.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from scripts.datasets.source_catalog import SOURCES


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Use safe relative repository paths')
    return path


def selection(catalog, names, max_bytes):
    if catalog.get('status') != 'metadata_available' or catalog.get('source') not in {'mlaad-tiny', 'in-the-wild'}:
        raise ValueError('A successful catalog for an exact requested HF source is required')
    metadata = catalog['metadata']
    if not isinstance(metadata, dict): raise ValueError('Expected an inspected metadata object')
    expected_repo = SOURCES[catalog['source']][1]
    if metadata.get('id') != expected_repo:
        raise ValueError('Catalog repository identity does not match the selected source')
    revision = metadata.get('sha', '')
    if not re.fullmatch('[0-9a-f]{40}', revision): raise ValueError('Immutable HF revision SHA is required')
    if not names or len(names) > 32 or len(set(names)) != len(names): raise ValueError('Select 1..32 unique files explicitly')
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1: raise ValueError('Positive byte budget required')
    indexed = {item['rfilename']: item for item in metadata.get('siblings', [])}
    selected, total = [], 0
    for name in names:
        safe_relative(name)
        item = indexed.get(name)
        if item is None: raise ValueError('Selected file is not in the inspected official catalog')
        lfs = item.get('lfs') or {}
        size = lfs.get('size', item.get('size'))
        if not isinstance(size, int) or isinstance(size, bool) or size < 0: raise ValueError('Official file size is required before a bounded download')
        expected_digest = lfs.get('sha256')
        if expected_digest is not None and not re.fullmatch('[0-9a-f]{64}', str(expected_digest)):
            raise ValueError('Invalid official LFS SHA-256 in catalog')
        total += size
        selected.append((name, size, lfs.get('sha256')))
    if total > max_bytes: raise ValueError('Selection exceeds byte budget; choose a smaller shard or recording subset')
    return revision, selected, total


def fetch(catalog, names, output, rights, max_bytes, *, download=None):
    revision, selected, total = selection(catalog, names, max_bytes)
    rights = Path(rights)
    if not rights.is_file() or not rights.read_text(encoding='utf8').strip(): raise ValueError('Retained reviewed recording rights are required')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'acquisition.json').exists(): raise ValueError('Acquisition already complete; use a new directory')
    if shutil.disk_usage(root).free < total: raise ValueError('Insufficient free storage for the selected files')
    if download is None:
        try: from huggingface_hub import hf_hub_download as download
        except ImportError as exc: raise RuntimeError('Install optional huggingface_hub on a permitted acquisition computer') from exc
    if not (root/'files').resolve().is_relative_to(root.resolve()): raise ValueError('Acquisition files directory escapes output root')
    repo_id = SOURCES[catalog['source']][1]
    records = []
    for name, size, expected in selected:
        destination = root/'files'/Path(name)
        if not destination.resolve().is_relative_to(root.resolve()): raise ValueError('Selected destination escapes output root')
        if destination.exists() and destination.is_symlink(): raise ValueError('Do not download through existing symlinks')
        path = Path(download(repo_id=repo_id, repo_type='dataset', filename=name, revision=revision, local_dir=root/'files'))
        if not path.resolve().is_relative_to((root/'files').resolve()): raise ValueError('Downloaded file escapes acquisition directory')
        if path.resolve() != destination.resolve(): raise ValueError('Download returned a different file than the requested catalog entry')
        if path.stat().st_size != size: raise ValueError('Downloaded size differs from the inspected catalog')
        with path.open('rb') as handle: digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if expected and digest != expected: raise ValueError('HF LFS SHA-256 mismatch')
        records.append({'path':'files/'+name, 'bytes':size, 'sha256':digest,
                        'checksum_provenance':'HF LFS SHA-256' if expected else 'Observed HTTPS download digest; not an author-published SHA-256'})
    record = {'source_url':SOURCES[catalog['source']][2], 'repo_id':repo_id, 'revision':revision,
              'selected_bytes':total, 'files':records, 'rights_sha256':hashlib.sha256(rights.read_bytes()).hexdigest(),
              'limitations':['Publisher authority and rights were reviewed by the operator, not established by this script.',
                             'Selected files may be shards; preserve their original labels/IDs and official partitions during import.']}
    with (root/'acquisition.json').open('x', encoding='utf8') as handle: json.dump(record, handle, indent=2); handle.write('\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--files',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--license-file',type=Path,required=True)
    p.add_argument('--max-bytes',type=int,required=True);p.add_argument('--rights-reviewed',action='store_true',required=True)
    a=p.parse_args()
    record=fetch(json.loads(a.catalog.read_text()),[x.strip() for x in a.files.read_text().splitlines() if x.strip()],a.output,a.license_file,a.max_bytes)
    print(json.dumps({'files':len(record['files']),'bytes':record['selected_bytes'],'revision':record['revision']}))

if __name__=='__main__':main()
