"""Plot recorded six-clip scores; does not create labels, thresholds or new metrics."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root.parent / 'results' / 'pilot_scores.csv').open(newline='') as handle:
    rows = list(csv.DictReader(handle))
fig, ax = plt.subplots(figsize=(5, 2.7), constrained_layout=True)
for i, row in enumerate(rows):
    ax.scatter(float(row['score']), i, color='#143444' if row['label']=='human' else '#087e8b', s=40)
ax.set_yticks(range(len(rows)), [f"{r['label']} {i+1}" for i,r in enumerate(rows)])
ax.set_xlim(0,1.02); ax.set_xlabel('Uncalibrated synthetic-oriented score')
ax.set_title('Six-recording convenience pilot; no fitted threshold', fontsize=10)
ax.grid(axis='x', alpha=.2)
fig.savefig(root / 'pilot_scores.pdf')
plt.close(fig)
