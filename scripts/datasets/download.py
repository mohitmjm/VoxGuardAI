"""Bounded HTTPS download with verified resume and explicit rights provenance.

Never discovers mirrors or bypasses the configured proxy. Archives are not extracted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request


def download(url, output, *, sha256, max_bytes, license_file, license_url, allowed_hosts=(), opener=None):
    target = Path(output)
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use a public credential-free static HTTPS URL without query/fragment; signed/private URLs are not supported')
    if not re.fullmatch('[0-9a-f]{64}', sha256):
        raise ValueError('Expected official SHA-256 is required')
    if not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError('Positive byte limit is required')
    rights = Path(license_file)
    if not rights.is_file() or not rights.read_text(encoding='utf8').strip():
        raise ValueError('Retain the reviewed dataset/recording license before download')
    license_parts = urllib.parse.urlsplit(license_url)
    if license_parts.scheme != 'https' or not license_parts.hostname or license_parts.username or license_parts.password or license_parts.query or license_parts.fragment:
        raise ValueError('Authoritative license URL is required')
    hosts = {parsed.hostname, *allowed_hosts}
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    provenance = target.with_name(target.name + '.download.json')
    identity = {'url': url, 'sha256': sha256, 'max_bytes': max_bytes,
                'license_url': license_url, 'license_sha256': hashlib.sha256(rights.read_bytes()).hexdigest()}
    if target.exists():
        with target.open('rb') as handle:
            existing_digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if existing_digest != sha256:
            raise ValueError('Existing completed file has an unexpected checksum; not overwriting')
        return target
    saved = json.loads(provenance.read_text()) if provenance.exists() else None
    if part.exists():
        if not saved or any(saved.get(k) != v for k,v in identity.items()):
            raise ValueError('Partial download provenance differs; use a new output path')
    else:
        provenance.write_text(json.dumps(identity, indent=2)+'\n')
    offset = part.stat().st_size if part.exists() else 0
    if offset > max_bytes:
        raise ValueError('Existing partial file exceeds byte limit')
    if offset:
        with part.open('rb') as handle:
            already_complete = hashlib.file_digest(handle, 'sha256').hexdigest() == sha256
        if already_complete:
            part.replace(target)
            return target
    if shutil.disk_usage(target.parent).free < max_bytes - offset:
        raise ValueError('Insufficient storage for requested bounded download')
    headers = {'User-Agent': 'VoxGuard-research/1.0', 'Accept-Encoding': 'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
        if saved and saved.get('etag') and not saved['etag'].startswith('W/'):
            headers['If-Range'] = saved['etag']
    request = urllib.request.Request(url, headers=headers)
    # Reject redirects to an undeclared host before requesting it.
    class AuthorizedRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            redirect = urllib.parse.urlsplit(newurl)
            if redirect.scheme != 'https' or redirect.hostname not in hosts or redirect.username or redirect.password or redirect.query:
                raise ValueError('Redirect requires an explicitly reviewed allowed host')
            return super().redirect_request(req, fp, code, msg, hdrs, newurl)
    open_request = opener or urllib.request.build_opener(AuthorizedRedirect()).open
    with open_request(request, timeout=60) as response:
        status = response.status
        mode = 'wb'
        if offset and status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', response.headers.get('Content-Range', ''))
            if not match or int(match[1]) != offset:
                raise ValueError('Server returned an invalid resume range')
            if saved and saved.get('etag') and response.headers.get('ETag') != saved['etag']:
                raise ValueError('Resume ETag changed; refuse to combine different versions')
            if match[3] != '*' and int(match[3]) > max_bytes:
                raise ValueError('Remote resume total exceeds configured byte limit')
            mode = 'ab'
        elif status != 200:
            raise ValueError(f'Unexpected download response: {status}')
        else:
            offset = 0  # server ignored Range: safely restart this partial file
        declared = response.headers.get('Content-Length')
        if declared and int(declared) + offset > max_bytes:
            raise ValueError('Remote content exceeds configured byte limit')
        provenance.write_text(json.dumps({**identity, 'etag':response.headers.get('ETag')},indent=2)+'\n')
        total = offset
        with part.open(mode) as handle:
            while chunk := response.read(65536):
                if total + len(chunk) > max_bytes:
                    raise ValueError('Download exceeded configured byte limit')
                handle.write(chunk)
                total += len(chunk)
    with part.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != sha256:
        raise ValueError('Incomplete or mismatched SHA-256; partial retained for diagnosis')
    part.replace(target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sha256', required=True); p.add_argument('--max-bytes', type=int, required=True)
    p.add_argument('--license-file', type=Path, required=True); p.add_argument('--license-url', required=True)
    p.add_argument('--allowed-redirect-host', action='append', default=[])
    p.add_argument('--rights-reviewed', action='store_true', required=True,
                   help='Confirm that dataset/recording terms permit this intended evaluation; not just code terms')
    a = p.parse_args()
    result = download(a.url,a.output,sha256=a.sha256,max_bytes=a.max_bytes,
                      license_file=a.license_file,license_url=a.license_url,allowed_hosts=a.allowed_redirect_host)
    print(f'Verified download: {result}')

if __name__ == '__main__': main()
