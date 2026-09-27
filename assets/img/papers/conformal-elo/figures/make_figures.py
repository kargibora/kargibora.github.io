"""Reproduce the project-page figures: python make_figures.py (matplotlib, numpy).

Data: author-provided OpenReview responses, 28 July 2026.
Chronological evaluation and transfer coverage: vcvx-Q2.
Calibration budget: wFQW-Q6. All arrays below transcribe the reported tables.
Ranges across cutoffs, central 95% ranges, and bootstrap CIs are kept distinct.
Style: scientific-figure-making palette and vector-export conventions.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
BLUE, RED, GRAY = "#0F4D92", "#B64342", "#767676"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "svg.hashsalt": "conformal-elo-project",
})

# Rows: Hard-Elo, Soft-Elo; columns: mean, low, high across chronological cutoffs.
TEMPORAL = [
    ("Elo MAE ↓", [[67.2, 65.4, 69.2], [25.1, 23.3, 26.5]], (0, 80), [0, 40, 80]),
    ("Median interval width (Elo) ↓", [[207.7, 178.5, 229.9], [129.2, 105.6, 158.5]], (0, 260), [0, 100, 200]),
    ("Coverage (%)", [[77.8, 66.0, 88.8], [93.0, 89.6, 98.4]], (0, 105), [0, 50, 100]),
]
BUDGET = np.array([[5, 19.2, 16.8, 22.8], [10, 18.3, 17.0, 20.2], [20, 18.0, 17.0, 19.6],
                   [30, 17.9, 17.0, 19.3], [44, 17.8, 17.3, 18.8]])
# Coverage: source calibration, target recalibration; mean and bootstrap 95% CI.
TRANSFER = np.array([[[71.2, 61.9, 81.2], [94.0, 92.8, 95.2]],
                     [[82.4, 76.7, 88.1], [94.6, 93.9, 95.4]]])


def interval(values):
    mean, low, high = np.asarray(values).T
    assert np.all(low <= mean) and np.all(mean <= high)
    return np.array([mean - low, high - mean])


def clean(ax, axis="x"):
    ax.grid(axis=axis, color="#e9e9e9", lw=.65, zorder=0)
    ax.tick_params(length=3, color=GRAY, labelcolor="#333333")


def save(fig, name, mobile):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check text stays inside the exported canvas at its intended size.
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, text.get_text()
    stem = OUT / (name + ("-mobile" if mobile else ""))
    for extension in (["svg", "png"] if mobile else ["svg", "pdf", "png"]):
        fig.savefig(stem.with_suffix("." + extension), dpi=300)
    plt.close(fig)
    print(f"Exported {stem.name}; labels fit the canvas.")


def chronological(mobile=False):
    fig, axes = plt.subplots(3 if mobile else 1, 1 if mobile else 3,
                             figsize=(3.35, 6.6) if mobile else (6.75, 2.6), layout="constrained")
    for i, (ax, (label, rows, limits, ticks)) in enumerate(zip(np.ravel(axes), TEMPORAL)):
        for j, (values, color, marker) in enumerate(zip(rows, [RED, BLUE], ["s", "o"])):
            mean, lo, hi = values
            ax.errorbar(mean, 1-j, xerr=interval([values]), fmt=marker, color=color,
                        capsize=4, elinewidth=1.8, markersize=6, zorder=3)
            ax.text(mean, 1-j+.22, f"{mean:.1f}", ha="center", color=color, weight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .6})
        ax.set(yticks=[1, 0], yticklabels=["Hard-Elo", "Soft-Elo"], ylim=(-.55, 1.65),
               xlim=limits, xticks=ticks, xlabel=label)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if i == 2:
            ax.axvline(90, color=GRAY, ls="--", lw=1, zorder=1)
            ax.text(87, 1.52, "90% target", ha="right", va="center", fontsize=8, color=GRAY)
        clean(ax)
    save(fig, "future-models", mobile)


def calibration(mobile=False):
    fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                             figsize=(3.35, 6.5) if mobile else (6.75, 3.3), layout="constrained")
    ax, transfer = np.ravel(axes)
    x, mean, low, high = BUDGET.T
    ax.errorbar(x, mean, yerr=interval(BUDGET[:, 1:]), color=BLUE, fmt="o-", lw=1.5,
                markersize=5, capsize=3, elinewidth=1.2, label="Soft-Elo", zorder=3)
    for count, score, upper in zip(x, mean, high):
        ax.text(count, upper+1.8, f"{score:.1f}", ha="center", fontsize=8, color=BLUE)
    ax.axhline(50.1, ls="--", color=RED, lw=1.2)
    ax.text(45, 47.3, "Hard-Elo: 50.1", ha="right", va="top", color=RED, fontsize=8)
    ax.set(xlim=(1, 48), ylim=(0, 56), xticks=x, yticks=[0, 20, 40],
           xlabel="Models used to fit β", ylabel="Elo MAE ↓")
    ax.text(.05, .95, "(a) Calibration budget", transform=ax.transAxes, fontsize=9, weight="bold")
    clean(ax, "y")

    for i, (label, color, marker, offset) in enumerate([
        ("Source calibration", RED, "s", -.13), ("Target recalibration", BLUE, "o", .13)
    ]):
        values = TRANSFER[:, i]
        transfer.errorbar(np.arange(2)+offset, values[:, 0], yerr=interval(values),
                          fmt=marker, color=color, capsize=4, markersize=6, lw=1.5, label=label, zorder=3)
        for j, (point, lo, hi) in enumerate(values):
            transfer.text(j+offset, hi+2, f"{point:.1f}", color=color, ha="center", fontsize=8)
    transfer.axhline(90, color=GRAY, ls="--", lw=1)
    transfer.text(1.43, 88, "90% target", ha="right", va="top", fontsize=8, color=GRAY)
    transfer.set(xlim=(-.45, 1.45), ylim=(0, 113), xticks=[0, 1], xticklabels=["LMArena 140K", "ComparIA"],
                 yticks=[0, 25, 50, 75, 100], ylabel="Coverage (%)", xlabel="Target corpus")
    transfer.text(.04, .94, "(b) Distribution shift", transform=transfer.transAxes, fontsize=9, weight="bold")
    transfer.legend(loc="lower left", fontsize=8)
    clean(transfer, "y")
    save(fig, "calibration-data", mobile)


def judge_accuracy(mobile=False):
    """Table 2, arXiv:2606.13221v2. Point estimates; no uncertainty inferred."""
    names = ["DeepSeek-V3.2", "Gemma4-26B-A4B", "Qwen3.5-27B", "GPT-OSS-120B",
             "Gemma4-E4B", "Llama-3.3-70B", "GPT-OSS-20B", "Qwen3-32B"]
    hard = np.array([63.4, 55.9, 46.0, 47.4, 48.2, 43.9, 34.5, 27.5])
    soft = np.array([17.1, 15.7, 13.6, 14.4, 21.0, 24.5, 19.8, 16.7])
    assert np.all(soft < hard)
    fig, ax = plt.subplots(figsize=(3.35, 4.1) if mobile else (5.1, 3.85), layout="constrained")
    y = np.arange(len(names))
    ax.hlines(y, soft, hard, color="#c0c7d0", lw=2, zorder=2)
    ax.scatter(hard, y, color=RED, marker="s", s=27, label="Hard-Elo", zorder=3)
    ax.scatter(soft, y, color=BLUE, s=30, label="Soft-Elo", zorder=3)
    for row, (a, b) in enumerate(zip(hard, soft)):
        if mobile:
            continue
        ax.annotate(f"{b:.1f}", (b, row), xytext=(-7, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=BLUE)
        ax.annotate(f"{a:.1f}", (a, row), xytext=(7, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=RED)
    ax.set(yticks=y, yticklabels=names, xlim=(0, 76), ylim=(7.65, -1.25),
           xticks=[0, 20, 40, 60], xlabel="Elo mean absolute error ↓")
    ax.spines["left"].set_visible(False)
    clean(ax)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", ncol=1 if mobile else 2, fontsize=8, handletextpad=.4, columnspacing=1)
    if mobile:
        ax.set_ylim(7.65, -2)
    save(fig, "judge-accuracy", mobile)


def rating_results(mobile=False):
    """Tables 2–3, arXiv:2606.13221v2; identical judge order in both panels.

    Widths are means of median widths across five calibration/test splits,
    at 90% nominal coverage. These panels show reported point estimates.
    """
    names = ["DeepSeek-V3.2", "Gemma4-26B-A4B", "Qwen3.5-27B", "GPT-OSS-120B",
             "Gemma4-E4B", "Llama-3.3-70B", "GPT-OSS-20B", "Qwen3-32B"]
    hard = np.array([63.4, 55.9, 46.0, 47.4, 48.2, 43.9, 34.5, 27.5])
    soft = np.array([17.1, 15.7, 13.6, 14.4, 21.0, 24.5, 19.8, 16.7])
    hard_width = np.array([261.0, 219.6, 212.2, 243.9, 235.2, 249.2, 186.8, 131.9])
    soft_width = np.array([78.0, 78.2, 74.4, 75.1, 143.1, 132.6, 102.9, 78.0])
    soft_coverage = np.array([92.9, 95.7, 95.7, 92.9, 96.4, 94.3, 94.3, 92.1])
    assert np.all(soft < hard) and np.all(soft_width < hard_width)
    assert np.isclose(hard.mean(), 45.85) and np.isclose(soft.mean(), 17.85)
    assert np.all(soft_coverage >= 90)
    assert round(100 * (1 - soft.mean() / hard.mean())) == 61
    assert tuple(np.round(100 * np.array([min(1-soft_width/hard_width), max(1-soft_width/hard_width)]))) == (39, 70)
    fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                             figsize=(3.6, 7.5) if mobile else (10.4, 4.4))
    fig.subplots_adjust(left=.39 if mobile else .175, right=.98,
                        top=.86 if mobile else .80, bottom=.07 if mobile else .15,
                        hspace=.62 if mobile else .25, wspace=.15)
    y = np.arange(len(names))
    for i, (ax, before, after, title, ticks, limit) in enumerate(zip(
        axes, [hard, hard_width], [soft, soft_width],
        ["Rating error", "Prediction interval width"],
        [[0, 20, 40, 60], [0, 100, 200, 300]], [76, 315]
    )):
        for row in range(0, 8, 2):
            ax.axhspan(row-.5, row+.5, color="#f5f7fa", zorder=0)
        ax.hlines(y, after, before, color="#aeb8c5", lw=1.8, zorder=2)
        ax.scatter(before, y, color=RED, marker="s", s=28, label="Usual win/tie/loss (Hard-Elo)", zorder=3)
        ax.scatter(after, y, color=BLUE, marker="o", s=32, label="Soft-Elo", zorder=3)
        if not mobile:
            for row, (a, b) in enumerate(zip(before, after)):
                ax.annotate(f"{b:.1f}", (b, row), xytext=(-8, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=8, color=BLUE)
                ax.annotate(f"{a:.1f}", (a, row), xytext=(8, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=8, color=RED)
        ax.set(yticks=y, yticklabels=names if (mobile or i == 0) else [],
               xlim=(0, limit), ylim=(7.55, -.55), xticks=ticks,
               xlabel="Mean absolute error (Elo)" if i == 0 else "Width (Elo) · 90% target")
        ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=15)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=8 if mobile else 9)
        ax.tick_params(axis="x", length=3, color=GRAY, labelcolor="#444444")
        ax.grid(axis="x", color="#e4e8ed", lw=.6, zorder=0)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=1 if mobile else 2, fontsize=9, handletextpad=.5, columnspacing=2)
    save(fig, "rating-results", mobile)


if __name__ == "__main__":
    for mobile in [False, True]:
        rating_results(mobile)
        judge_accuracy(mobile)
        chronological(mobile)
        calibration(mobile)
