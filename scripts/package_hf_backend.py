"""Build a small upload package; excludes recordings, caches and model weights."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/deploy/voxguard-connected-backend.zip'

def main():
    files = [(p, p.name) for p in (ROOT / 'deploy/huggingface').iterdir() if p.is_file()]
    for folder in ('backend', 'ml'):
        for p in (ROOT / folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or 'checkpoints' in p.parts or p.name.startswith('test_'):
                continue
            if p.suffix not in ('.py', '.json', '.txt') and p.name not in ('LICENSE', 'NOTICE'):
                continue
            files.append((p, p.relative_to(ROOT).as_posix()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p, name in files:
            archive.write(p, name)
    print(OUT)

if __name__ == '__main__':
    main()
