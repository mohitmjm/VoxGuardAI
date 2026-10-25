"""Paired clean/noise/MP3 inference. Never changes source recordings."""
import argparse
import csv
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
from research.evaluate import manifest, audit, decode


def noise(audio, snr_db, rng):
    power = float(np.mean(audio.astype(np.float64)**2))
    if power <= 0:
        raise ValueError('Cannot define SNR for silent audio')
    perturbation = rng.normal(size=len(audio))
    perturbation *= np.sqrt(power / (10 ** (snr_db/10)) / np.mean(perturbation**2))
    # Preserve SNR rather than clipping; peak normalize the entire mixture.
    mixed = audio + perturbation
    return (mixed / max(1., np.max(np.abs(mixed)))).astype(np.float32)


def compressed(audio, bitrate, codec="mp3"):
    if codec not in {"mp3", "aac"}:
        raise ValueError("Supported research codecs are mp3 and aac")
    with tempfile.TemporaryDirectory(prefix='voxguard-research-') as folder:
        path = Path(folder)/('compressed.mp3' if codec == 'mp3' else 'compressed.m4a')
        r = subprocess.run(['ffmpeg','-nostdin','-v','error','-f','f32le','-ar','16000','-ac','1','-i','pipe:0','-c:a',('libmp3lame' if codec == 'mp3' else 'aac'),'-b:a',bitrate,str(path)], input=audio.astype('<f4').tobytes(),capture_output=True,timeout=45)
        if r.returncode:
            raise RuntimeError('Research codec conversion failed')
        return decode(path)


def observations(rows, model=None, seed=42):
    """Yield paired observations using one real detector; do not refit thresholds."""
    from ml.service import ModelService
    from ml.constants import SHA256
    from ml.policy import SCORE_PROTOCOL
    model = model or ModelService()
    rng = np.random.default_rng(seed)
    for row in rows:
        audio = decode(row['audio_path'])
        for condition, transformed in [('clean',audio),('white_noise_20db',noise(audio,20,rng)),('white_noise_10db',noise(audio,10,rng)),('mp3_32k',compressed(audio,'32k')),('mp3_64k',compressed(audio,'64k')),('aac_64k',compressed(audio,'64k','aac'))]:
            start=time.perf_counter(); prediction=model.analyze(transformed,sample_rate=16000)
            yield {**row,'condition':condition,'score':prediction['score'],'latency_ms':1000*(time.perf_counter()-start),'seed':seed,'checkpoint_sha256':SHA256,'score_protocol':SCORE_PROTOCOL}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    rows=manifest(a.manifest); audit(rows)
    # Exclusive creation protects earlier measured evidence.
    with a.output.open('x',newline='') as f:
        iterator=iter(observations(rows,seed=a.seed))
        first=next(iterator)
        w=csv.DictWriter(f,fieldnames=list(first));w.writeheader();w.writerow(first);w.writerows(iterator)

if __name__=='__main__': main()
