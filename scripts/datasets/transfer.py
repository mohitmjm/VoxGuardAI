"""Seal or verify a portable explicit-file inventory for authorized local transfer.

The inventory digest must be compared through a trusted separate channel.
Transfer integrity does not establish publisher provenance or recording rights.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from scripts.datasets.hf_subset import safe_relative


def checksum(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()


def resolve_file(root, name):
    safe_relative(name)
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Inventory path is absent or escapes transfer root')
    return path


def seal(root, names):
    root=Path(root).resolve()
    if not names or len(set(names)) != len(names): raise ValueError('Explicit nonempty unique transfer file list required')
    records=[]
    for name in names:
        path=resolve_file(root,name)
        records.append({'path':name,'bytes':path.stat().st_size,'sha256':checksum(path)})
    return {'version':1,'files':records,'warning':'Transfer hashes are observed local digests, not author-signed release checksums or recording permission.'}


def verify(root, inventory, expected_digest):
    if not re.fullmatch('[0-9a-f]{64}',expected_digest): raise ValueError('Trusted inventory SHA-256 required')
    if checksum(inventory)!=expected_digest:raise ValueError('Inventory digest mismatch')
    data=json.loads(Path(inventory).read_text(encoding='utf8'))
    if data.get('version')!=1 or not isinstance(data.get('files'),list) or not data['files']:raise ValueError('Unsupported or empty transfer inventory')
    root=Path(root).resolve();names=set()
    for record in data['files']:
        name=record['path']
        if name in names:raise ValueError('Duplicate transfer path')
        names.add(name);path=resolve_file(root,name)
        if not isinstance(record['bytes'],int) or isinstance(record['bytes'],bool) or record['bytes']<0 or path.stat().st_size!=record['bytes'] or checksum(path)!=record['sha256']:
            raise ValueError('Transferred file size or checksum mismatch')
    return len(names)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('seal');s.add_argument('--root',type=Path,required=True);s.add_argument('--files',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    v=sub.add_parser('verify');v.add_argument('--root',type=Path,required=True);v.add_argument('--inventory',type=Path,required=True);v.add_argument('--inventory-sha256',required=True)
    a=p.parse_args()
    if a.command=='seal':
        data=seal(a.root,[x.strip() for x in a.files.read_text().splitlines() if x.strip()])
        with a.output.open('x',encoding='utf8') as f:json.dump(data,f,indent=2);f.write('\n')
        print('inventory_sha256='+checksum(a.output))
    else:print('verified_files='+str(verify(a.root,a.inventory,a.inventory_sha256)))

if __name__=='__main__':main()
