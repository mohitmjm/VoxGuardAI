"""Download three attributed CC-BY-4.0 speech excerpts and generate classical TTS.

This six-recording convenience pilot is NOT a benchmark or calibration corpus.
No audio is added to Git; default output is in ignored artifacts/pilot/.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

COMMIT = '38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0'
SAMPLES = {
    '5703-47212-0000': ('a284612b46af0535f7e1873758c4387bb8369f6dbbe192ffdec1f171108f98dd', '0d3f9a04b764d9474a93959633b560ae3ab65163123773f58c98a21908b67901'),
    '3436-172162-0000': ('780e482b52f2c8a500356babf60178ce005449cc171a8537eb3ef2961eb3e855', 'f9b76b7a994db493f76fe2aacd5615d6b35f49b5674b87066ccb10daaadba08f'),
    '198-209-0000': ('0473ba9613befa1026db5e254546f105aeffba8366b5468fc5e8d474d844ca1d', 'b41a90dbe8d7ca36dc7498fef5524b95dd8dc9e8c8bef6110f207afed8b57b46'),
}
TEXTS = {
    'slt': 'This recording was generated for a software evaluation of synthetic speech.',
    'awb': 'The quiet laboratory contains a microphone and an audio analysis computer.',
    'rms': 'Please interpret every model score carefully because a small pilot cannot prove accuracy.',
}


def verified_fetch(url, destination, expected):
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Existing file fails verification: {destination.name}')
        return
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(2 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Source checksum mismatch; refusing sample')
    destination.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/pilot'))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    # Preserve user artifacts; create a new directory for an independent rerun.
    if (root / 'manifest.csv').exists() or (root / 'provenance.json').exists():
        raise SystemExit('Pilot manifest already exists; use a new --output directory.')
    rows, sources = [], []
    for sample, (audio_digest, license_digest) in SAMPLES.items():
        base = f'https://raw.githubusercontent.com/librosa/librosa-data/{COMMIT}/audio/{sample}'
        # Inspect the independently checksummed per-recording rights before audio.
        rights = root / f'{sample}.toml'
        verified_fetch(base + '.toml', rights, license_digest)
        if 'license = "CC-BY-4.0"' not in rights.read_text():
            raise ValueError('Expected CC-BY-4.0 attribution not found')
        path = root / f'{sample}.ogg'
        verified_fetch(base + '.ogg', path, audio_digest)
        rows.append(dict(audio_path=path.name, label='human', speaker_id='librispeech-' + sample.split('-')[0], source_id='librispeech-' + sample, generator='human-recording', split='test', language='en'))
        sources.append(dict(file=path.name, url=base + '.ogg', sha256=audio_digest, license='CC-BY-4.0', attribution=rights.read_text(), transformation='None; original upstream Ogg excerpt. FFmpeg resampling occurs only during model scoring.'))
    for voice, text in TEXTS.items():
        path = root / f'flite-{voice}.wav'
        if path.exists():
            raise ValueError('Refusing to overwrite pre-existing synthesized sample')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i', f'flite=text={text}:voice={voice}', '-ar', '16000', '-ac', '1', str(path)], check=True, timeout=30)
        rows.append(dict(audio_path=path.name, label='ai', speaker_id='flite-' + voice, source_id='voxguard-pilot-text-' + voice, generator='flite-classical-tts', split='test', language='en'))
        sources.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), generator='FFmpeg flite filter; classical non-neural TTS', voice=voice, text=text, rights='Original project-authored evaluation text, locally synthesized with installed Flite. Flite software license inspected at https://github.com/festvox/flite/blob/master/COPYING; no recording of a user is reused.'))
    with (root / 'manifest.csv').open('x', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    provenance = dict(dataset='VoxGuard convenience pilot: LibriSpeech excerpts + local classical Flite TTS', n=len(rows), sources=sources, ffmpeg_version=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0], caveats=['Only three human excerpts and three synthesized utterances; not representative.', 'Human and synthetic source domains differ and may confound scores.', 'Flite is classical TTS, not modern neural synthesis or voice cloning.', 'LibriSpeech overlap with unknown pretrained data cannot be ruled out.', 'All six are test rows; no validation thresholds, probabilities, or reliable language/generator accuracy can be established.'])
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Prepared {len(rows)} convenience pilot recordings. Not a calibration dataset.')

if __name__ == '__main__':
    main()
