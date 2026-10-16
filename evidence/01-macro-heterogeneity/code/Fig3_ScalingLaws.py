"""
Fig2: Model Scale — CoT effect across Qwen2.5 0.5B->32B on GSM8K.
Highlights term-first as a distinct bar among CoT variants.

Reads from: evidence/01-macro-heterogeneity/data/fig2_model_scale_scores.csv
Output to: evidence/01-macro-heterogeneity/figures/
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
from pathlib import Path

# Ensure project root is on sys.path (robust against any cwd)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.plot_config import setup_style, save_figure, sequential_from_cmap, OKABE_ITO as OI

# ---- Paths ----
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
FIG_DIR = HERE.parent / "figures"
CSV_PATH = DATA_DIR / "fig2_model_scale_scores.csv"

# ---- Style ----
setup_style({"font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})

MODELS = ['0.5B', '1.5B', '3B', '7B', '32B']
# Sequential blue for model scale (single-hue gradient, perceptually uniform)
BAR_COLORS = sequential_from_cmap(5, "Blues")
TERMFIRST_COLOR = OI["vermillion"]  # stands out against blue gradient
RUNS_PER_MODEL = 4
XLIM_MAP = {'0.5B': (-2, 20), '1.5B': (-10, 2), '3B': (-5, 6),
            '7B': (-3, 12), '32B': (-3, 8)}

TERMFIRST_LABEL = "Term first, then reasoning."  # matches CSV instruction column
TERMFIRST_DISPLAY = "What is the query? Term first, then reasoning."  # y-axis display (consistent with Fig1)

# ---- Load ----
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def extract_model(rows, model_name):
    col_names = [f"{model_name}_R{i}" for i in range(1, RUNS_PER_MODEL + 1)]
    instructions = [r['Instruction'] for r in rows]
    runs = []
    for col in col_names:
        runs.append([float(r[col]) for r in rows])
    return instructions, runs

def process(runs, base_idx=0):
    arr = np.array(runs)
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0, ddof=1)
    base = means[base_idx]
    delta_pp = means - base
    sem = stds / np.sqrt(RUNS_PER_MODEL)
    return delta_pp, sem

# ---- Draw ----
def draw():
    rows = load_csv(CSV_PATH)
    instructions_full, _ = extract_model(rows, '0.5B')

    # Find indices
    # 'Standard' is the baseline; TERMFIRST_LABEL marks term-first;
    # all other rows are conventional CoT variants.
    termfirst_idx = None
    standard_indices = []
    for i, instr in enumerate(instructions_full):
        if instr == TERMFIRST_LABEL:
            termfirst_idx = i
        elif instr != 'Standard':
            standard_indices.append(i)

    # Standard prompt labels (for y-axis)
    standard_labels = [instructions_full[i] for i in standard_indices]
    y_pos = np.arange(len(standard_labels))[::-1]

    fig, axes = plt.subplots(1, 5, figsize=(7.2, 4.2), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, 1], 'wspace': 0.1})

    for i, name in enumerate(MODELS):
        ax = axes[i]
        _, runs = extract_model(rows, name)
        pc, err = process(runs, base_idx=0)

        # Standard prompts
        std_pc = [pc[j] for j in standard_indices]
        std_err = [err[j] for j in standard_indices]
        ax.barh(y_pos, std_pc, color=BAR_COLORS[i], height=0.65, edgecolor='none', alpha=0.9,
                label='CoT variants')

        # Error bars for standard prompts
        ax.errorbar(std_pc, y_pos, xerr=std_err, fmt='none', color=OI["gray"],
                     alpha=0.7, linewidth=0.5, capsize=1.5)

        # Term-first - highlight with distinct color + wider bar
        if termfirst_idx is None:
            continue
        tf_pc = pc[termfirst_idx]
        tf_err = err[termfirst_idx]
        # Place term-first at the top of the bars
        tf_y = len(standard_labels)  # position above all other bars
        ax.barh(tf_y, tf_pc, color=TERMFIRST_COLOR, height=0.75, edgecolor='white',
                linewidth=0.5, alpha=1.0, label='term-first', zorder=10)
        ax.errorbar([tf_pc], [tf_y], xerr=[tf_err], fmt='none', color='#333333',
                     alpha=0.7, linewidth=0.5, capsize=2.0)

        # Add text label for term-first value
        offset = 0.5 if tf_pc >= 0 else -0.5
        ax.annotate(f'{tf_pc:+.1f}', xy=(tf_pc, tf_y),
                     xytext=(offset, 0), textcoords='offset fontsize',
                     fontsize=6, fontweight='bold', color=TERMFIRST_COLOR,
                     va='center')

        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.yaxis.set_visible(False)
        ax.set_title(name, fontsize=8, fontweight='bold', pad=10, color=BAR_COLORS[i])
        # x-axis label matches Fig 1 ('ΔACC (pp)'); shown on every panel
        # for consistency with Fig 1's layout (one label per panel).
        ax.set_xlabel(r'$\Delta$ACC (pp)', fontsize=7)
        ax.set_xlim(XLIM_MAP[name])
        ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.3)

        if i == 0:
            # All y-ticks plus one for term-first
            all_y = list(y_pos) + [tf_y]
            all_labels = standard_labels + [TERMFIRST_DISPLAY]
            ax.set_yticks(all_y)
            ax.set_yticklabels(all_labels, fontsize=6.5, ha='right')
            ax.yaxis.set_visible(True)
            ax.yaxis.set_tick_params(width=0)

    save_figure(fig, "Fig3_ScalingLaws", FIG_DIR)

    # Note: paper/figures/ copy removed for release isolation
    plt.close(fig)

if __name__ == "__main__":
    draw()
