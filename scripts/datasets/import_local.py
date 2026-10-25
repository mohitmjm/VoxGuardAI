"""Validate local partial corpora without copying recordings or inventing identities.

Normalized CSV is preferred. ASVspoof5 six-field baseline protocol support requires
reviewed source-ID provenance because the baseline file alone omits source groups.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

REQUIRED = {'audio_path','label','speaker_id','source_id','generator','split','language'}


def import_rows(rows, audio_root, *, limit=None, allow_missing=False):
    root = Path(audio_root).resolve()
    if not root.is_dir(): raise ValueError('Audio root must be an existing directory')
    selected, missing, seen = [], 0, set()
    for row in rows:
        if not REQUIRED.issubset(row): raise ValueError('Manifest missing required columns')
        if any(not str(row[k]).strip() for k in REQUIRED):
            raise ValueError('Missing provenance: recording/speaker/source IDs must not be guessed')
        if row['source_id'].lower() in {'unknown','-','n/a'} or row['speaker_id'].lower() in {'unknown','-','n/a'}:
            raise ValueError('Unknown speaker/source identity cannot establish leakage-free partitions')
        if row['label'] not in {'human','ai'} or row['split'] not in {'train','validation','test'}:
            raise ValueError('Preserve explicit human/ai labels and official split mappings')
        relative = Path(row['audio_path'])
        if relative.is_absolute(): raise ValueError('Audio paths must be relative to the declared root')
        path = (root / relative).resolve()
        if not path.is_relative_to(root): raise ValueError('Audio path escapes declared root')
        if not path.is_file():
            if allow_missing: missing += 1; continue
            raise ValueError('Protocol references an unavailable recording; use --allow-missing for a declared partial import')
        recording_id = row.get('recording_id') or str(relative)
        if recording_id in seen: raise ValueError('Duplicate original recording ID')
        seen.add(recording_id)
        if limit is None or len(selected) < limit:
            selected.append({**row,'audio_path':str(path),'recording_id':recording_id})
    if not selected: raise ValueError('No recordings selected')
    # Check all provided identity metadata, not just the first-N selected subset.
    for field in ('speaker_id','source_id'):
        groups = {}
        for row in rows:
            groups.setdefault(row[field],set()).add(row['split'])
        if any(len(splits)>1 for splits in groups.values()):
            raise ValueError(f'{field} spans official splits; resolve provenance before evaluation')
    return selected, missing


def asvspoof5_rows(protocol, provenance, partition, language, audio_subdir=''):
    """Adapt the six-column format used by the official baseline data_utils.py.

    Metadata omissions are filled ONLY from supplied reviewed provenance, not
    guessed from filenames. Raw protocol fields and partition are retained.
    """
    split = {'train':'train','dev':'validation','eval':'test'}[partition]
    rows = []
    for line in Path(protocol).read_text().splitlines():
        if not line.strip(): continue
        fields=line.split()
        if len(fields)!=6 or fields[-1] not in {'bonafide','spoof'}:
            raise ValueError('Expected labeled six-field ASVspoof5 baseline protocol; do not guess an unknown release layout')
        speaker, recording = fields[:2]
        meta = provenance.get(recording)
        if not meta or not meta.get('source_id'):
            raise ValueError('Official baseline protocol lacks source grouping; supply reviewed per-recording provenance')
        rows.append({'audio_path':str(Path(audio_subdir)/(recording+'.flac')),
                     'label':'human' if fields[-1]=='bonafide' else 'ai',
                     'speaker_id':speaker,'source_id':meta['source_id'],
                     'generator':meta.get('generator') or 'unknown', 'split':split,'language':language,
                     'recording_id':recording,'original_partition':partition,'original_protocol_line':line})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--audio-root',type=Path,required=True)
    p.add_argument('--format',choices=['normalized','asvspoof5-baseline'],default='normalized')
    p.add_argument('--source-provenance',type=Path);p.add_argument('--official-partition',choices=['train','dev','eval'])
    p.add_argument('--language');p.add_argument('--audio-subdir',default='')
    p.add_argument('--license-file',type=Path,required=True);p.add_argument('--rights-reviewed',action='store_true',required=True)
    p.add_argument('--allow-missing',action='store_true');p.add_argument('--limit',type=int)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.limit is not None and a.limit<1:p.error('limit must be positive')
    if not a.license_file.is_file() or not a.license_file.read_text().strip():p.error('Reviewed recording/dataset rights are required')
    if a.output.resolve()==a.protocol.resolve() or a.output.exists():p.error('Refusing to overwrite protocol or existing manifest')
    if a.format=='normalized':
        with a.protocol.open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    else:
        if not a.source_provenance or not a.official_partition or not a.language:p.error('ASVspoof5 requires reviewed source provenance, official partition and language')
        with a.source_provenance.open(newline='',encoding='utf8') as f:
            records=list(csv.DictReader(f))
        provenance={r['recording_id']:r for r in records}
        if len(provenance)!=len(records):p.error('Duplicate provenance recording IDs')
        rows=asvspoof5_rows(a.protocol,provenance,a.official_partition,a.language,a.audio_subdir)
    selected,missing=import_rows(rows,a.audio_root,limit=a.limit,allow_missing=a.allow_missing)
    from research.evaluate import audit
    summary=audit(selected)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    metadata={'protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
              'license_sha256':hashlib.sha256(a.license_file.read_bytes()).hexdigest(),
              'format':a.format,'protocol_rows':len(rows),'selected_rows':len(selected),'missing_local_files':missing,
              'selection':'available protocol-order prefix; engineering subset, not representative' if a.limit else 'all locally available protocol entries',
              'audit':summary,'warning':'No partitions or speaker/source IDs are manufactured; unknown generator identity prevents generator-held-out claims.'}
    a.output.with_suffix(a.output.suffix+'.provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'selected_rows':len(selected),'missing_local_files':missing,'audit':'passed'}))

if __name__=='__main__': main()
