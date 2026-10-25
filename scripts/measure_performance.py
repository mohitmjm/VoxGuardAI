"""Measure cold/warm real inference. This is a performance run, not accuracy."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from research.evaluate import decode
    from ml.service import ModelService
    samples = decode(args.audio)
    service = ModelService()
    times = []
    for _ in range(2):
        start = time.perf_counter()
        service.analyze(samples)
        times.append((time.perf_counter() - start) * 1000)
    peak_rss_bytes = None
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = int(peak if sys.platform == 'darwin' else peak * 1024)
    except ImportError:
        pass
    report = {'model': service.info(), 'os': platform.platform(), 'python': platform.python_version(),
              'duration_seconds': len(samples)/16000, 'cold_inference_ms': times[0], 'warm_inference_ms': times[1],
              'peak_process_rss_bytes': peak_rss_bytes, 'cpu_count': __import__('os').cpu_count(),
              'conditions': 'Single recording; decoding excluded from timing; cold includes model load. Peak RSS includes Python and imports. Not an accuracy evaluation.'}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('duration_seconds','cold_inference_ms','warm_inference_ms','peak_process_rss_bytes')}))

if __name__ == '__main__': main()
