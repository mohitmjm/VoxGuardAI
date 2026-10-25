"""Check local dependencies and verified inference before starting the service."""
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    for name in ('ffmpeg', 'ffprobe'):
        if not shutil.which(name):
            raise SystemExit(f'Missing {name}: install FFmpeg and add its bin directory to PATH.')
    from ml.service import ModelService, SHA256
    service = ModelService()
    if not service.path.exists():
        raise SystemExit('Missing checkpoint: run python -m ml.download_model')
    if hashlib.sha256(service.path.read_bytes()).hexdigest() != SHA256:
        raise SystemExit('Checkpoint integrity failed. Run python -m ml.download_model')
    import numpy as np
    result = service.analyze((.1 * np.sin(np.arange(16000) * .1)).astype(np.float32))
    print(f'FFmpeg, checkpoint integrity and real {result["model"]["device"]} inference passed.')
    print('Generated tone is a mechanical smoke check, not an accuracy measurement.')

if __name__ == '__main__':
    main()
