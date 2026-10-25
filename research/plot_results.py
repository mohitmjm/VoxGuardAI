"""Create a standalone scientific SVG/PDF/PNG from measured result JSON only.
Install optional plotting dependencies: pip install -r research/requirements.txt
"""
import argparse
import json
import math
from pathlib import Path

KEYS = ('accuracy', 'precision', 'recall', 'f1', 'auroc', 'eer', 'false_positive_rate', 'false_negative_rate', 'coverage', 'abstention_rate')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', required=True, help='Include the measured dataset and experiment conditions')
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    metrics = result.get('metrics', result)
    values = [(k, metrics[k]) for k in KEYS if isinstance(metrics.get(k), (int, float)) and math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1]
    if not values:
        raise ValueError('No measured scalar metrics in this result file')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('Install optional plotting dependencies: pip install -r research/requirements.txt') from exc
    fig, ax = plt.subplots(figsize=(10, max(3, .65*len(values)+1.8)), layout='constrained')
    positions = list(range(len(values)))
    bars = ax.barh(positions, [v for _,v in values], color='#087f8c', height=.6)
    ax.set_yticks(positions,[k for k,_ in values]);ax.invert_yaxis();ax.set_xlim(0,1.12)
    ax.set_xlabel('Measured metric (error rates: lower is better)')
    ax.set_title(args.title,wrap=True,fontsize=11)
    ax.bar_label(bars,labels=[f'{v:.4f}' for _,v in values],padding=5)
    ax.spines[['right','top']].set_visible(False)
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.text(.01,.005,f'Source: {args.results.name}. Sample count and conditions must accompany interpretation.',fontsize=8)
    fig.savefig(args.output,dpi=200,metadata={'Creator':'VoxGuard research.plot_results'})
    plt.close(fig)

if __name__ == '__main__': main()
