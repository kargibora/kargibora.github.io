"""Project-page comparison, generated from the adjacent verified results.csv.

Half-Truth is an equal-weight mean across the three evaluation sets.
Compositional accuracy is the reported mean over 16 independent benchmarks.
Selected methods include the strongest baseline on each displayed metric.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
BLUE, GRAY = '#185aa5', '#8994a3'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.linewidth': .8, 'legend.frameon': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'svg.hashsalt': 'half-truths-summary',
    'savefig.facecolor': '#f5f7f9', 'figure.facecolor': '#f5f7f9', 'axes.facecolor': '#f5f7f9',
})
rows = list(csv.DictReader((OUT / 'results.csv').open()))
all_values = {}
for model in dict.fromkeys(row['model'] for row in rows):
    scores = [row for row in rows if row['model'] == model]
    assert len(scores) == 3
    assert len(set(row['comp_i2t'] for row in scores)) == 1
    all_values[model] = [np.mean([float(row['ht_all']) for row in scores]), float(scores[0]['comp_i2t'])]
selected = ['CLIP', 'NegCLIP', 'DeGLA', 'ReadCLIP', 'FSC-CLIP-cc3m', 'FSC-CLIP-coco', 'CS-CLIP']
selected.sort(key=lambda model: all_values[model][0], reverse=True)
values = np.array([all_values[model] for model in selected])
for metric in [0, 1]:
    assert max(all_values, key=lambda model: all_values[model][metric]) == 'CS-CLIP'
    strongest_baseline = max((model for model in all_values if model != 'CS-CLIP'), key=lambda model: all_values[model][metric])
    assert strongest_baseline in selected
assert np.allclose(np.round(all_values['CS-CLIP'], 1), [69.9, 57.8])


def draw(mobile=False):
    fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                             figsize=(3.6, 6.5) if mobile else (10, 4.3))
    fig.subplots_adjust(left=.39 if mobile else .21, right=.97, top=.91 if mobile else .83,
                        bottom=.08 if mobile else .17, hspace=.7, wspace=.16)
    names = {'FSC-CLIP-cc3m': 'FSC-CLIP (CC3M)', 'FSC-CLIP-coco': 'FSC-CLIP (COCO)', 'CS-CLIP': 'CS-CLIP (ours)'}
    labels = [names.get(model, model) for model in selected]
    for metric, (ax, title) in enumerate(zip(axes, ['Half-Truth', 'Compositional'])):
        colors = [BLUE if model == 'CS-CLIP' else GRAY for model in selected]
        ax.barh(np.arange(len(selected)), values[:, metric], color=colors, height=.53, zorder=2)
        for row, (model, value) in enumerate(zip(selected, values[:, metric])):
            ax.text(value+1.2, row, f'{value:.1f}', va='center', fontsize=9,
                    color=BLUE if model == 'CS-CLIP' else '#454d57', weight='bold' if model == 'CS-CLIP' else 'normal')
        ax.set(yticks=range(len(selected)), yticklabels=labels if mobile or metric == 0 else [],
               ylim=(len(selected)-.4, -.7), xlim=(0, 83 if metric == 0 else 70),
               xticks=[0, 20, 40, 60, 80] if metric == 0 else [0, 20, 40, 60], xlabel='Accuracy (%)')
        ax.set_title(title, loc='left', fontsize=11, weight='bold', pad=24)
        ax.text(0, 1.04, '3-set average' if metric == 0 else '16-benchmark average',
                transform=ax.transAxes, fontsize=8.5, color='#555b62')
        ax.tick_params(axis='y', length=0, labelsize=8 if mobile else 10)
        ax.tick_params(axis='x', length=3, color=GRAY, labelsize=8 if mobile else 9)
        ax.grid(axis='x', color='#dce2e9', lw=.65)
        ax.set_axisbelow(True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b = text.get_window_extent(renderer)
            assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width + 1 and b.y1 <= fig.bbox.height + 1, text.get_text()
    stem = OUT / ('task-summary-mobile' if mobile else 'task-summary')
    for extension in (['svg', 'png'] if mobile else ['svg', 'png', 'pdf']):
        fig.savefig(stem.with_suffix('.' + extension), dpi=300)
    plt.close(fig)
    print(f'{stem.name}: all-model ranking, comparator selection, release values and label bounds checked.')

if __name__ == '__main__':
    draw()
    draw(True)
