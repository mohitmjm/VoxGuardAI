"""Fetch immutable official weights and enforce the recorded SHA-256."""
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path
from ml.constants import ROOT, SHA256, COMMIT

def main():
    url = f'https://raw.githubusercontent.com/clovaai/aasist/{COMMIT}/models/weights/AASIST.pth'
    destination = Path(os.getenv('VOXGUARD_CHECKPOINT', str(ROOT / 'checkpoints' / 'AASIST.pth')))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
        print('Verified existing AASIST checkpoint.')
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(10 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError('Checkpoint checksum mismatch; refusing installation.')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
            tmp.write(data)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print('Downloaded and verified official AASIST checkpoint.')

if __name__ == '__main__':
    main()
