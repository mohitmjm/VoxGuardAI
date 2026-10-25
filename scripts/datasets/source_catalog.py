"""Inspect exact requested public dataset metadata without downloading recordings."""
import argparse
import datetime
import json
import urllib.request
from pathlib import Path

SOURCES = {
    'mlaad-tiny': ('hf', 'mueller91/MLAAD-tiny', 'https://huggingface.co/datasets/mueller91/MLAAD-tiny'),
    'in-the-wild': ('hf', 'mueller91/In-The-Wild', 'https://huggingface.co/datasets/mueller91/In-The-Wild'),
    'zenodo-14498691': ('zenodo', '14498691', 'https://zenodo.org/records/14498691'),
}

def inspect(source, *, opener=None):
    kind, identity, page = SOURCES[source]
    url = f'https://huggingface.co/api/datasets/{identity}?blobs=true' if kind == 'hf' else f'https://zenodo.org/api/records/{identity}'
    record = {'source': source, 'source_url': page, 'metadata_url': url,
              'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'provenance_status': 'User-requested source; publisher authority and recording rights require review.',
              'recordings_downloaded': 0}
    try:
        with (opener or urllib.request.urlopen)(url, timeout=20) as response:
            body = response.read(2 * 1024 * 1024 + 1)
        if len(body) > 2 * 1024 * 1024:
            raise ValueError('Metadata exceeds bounded catalog size; use official selective API tooling')
        metadata = json.loads(body)
        if not isinstance(metadata, dict):
            raise ValueError('Expected official metadata object')
        record.update(status='metadata_available', metadata=metadata, license_status='requires_independent_recording_rights_review')
    except Exception as exc:
        record.update(status='unavailable', error=f'{type(exc).__name__}: {exc}', license_status='not_retrieved')
    return record

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=SOURCES, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = inspect(a.source)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(result['status'])
    if result['status'] != 'metadata_available': raise SystemExit(2)

if __name__ == '__main__': main()
