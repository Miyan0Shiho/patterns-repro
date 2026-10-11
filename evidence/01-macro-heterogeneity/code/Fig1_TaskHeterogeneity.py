"""
Fig1: Task Heterogeneity — CoT prompt variants across 4 datasets (GSM8K benchmark only).

Reads from: evidence/01-macro-heterogeneity/data/fig1_task_heterogeneity_scores.csv
Output to: evidence/01-macro-heterogeneity/figures/
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
from pathlib import Path

# ---- Plot config ----
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.plot_config import setup_style, save_figure, OKABE_ITO as OI

# ---- Paths ----
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
FIG_DIR = HERE.parent / "figures"
CSV_PATH = DATA_DIR / "fig1_task_heterogeneity_scores.csv"

# ---- Style ----
setup_style({"font.size": 7})

COLORS = {
    'GSM8K': OI["orange"],
    'MMLU': OI["sky_blue"],
    'CMMLU': OI["bluish_green"],
    'HumanEval': OI["reddish_purple"],
}

DATASET_COLS = ['GSM8K', 'MMLU', 'CMMLU', 'HumanEval']
RUNS_PER_DATASET = 4

# ---- Load data ----
def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def extract_dataset(rows, dataset_name):
    """Return (instructions, runs_matrix) where runs_matrix is list of 4 lists of scores."""
    col_names = [f"{dataset_name}_R{i}" for i in range(1, RUNS_PER_DATASET + 1)]
    instructions = [r['Instruction'] for r in rows]
    runs = []
    for col in col_names:
        runs.append([float(r[col]) for r in rows])
    return instructions, runs

def process_dataset(runs):
    """runs: list of 4 lists of scores. Returns (pct_change, pct_error)."""
    arr = np.array(runs)
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    base = means[0]
    pct_change = ((means - base) / base) * 100
    pct_error = (stds / base) * 100
    return pct_change, pct_error

# ---- Draw ----
def draw():
    rows = load_csv(CSV_PATH)

    instructions_full, _ = extract_dataset(rows, 'GSM8K')
    instructions = instructions_full[1:]  # skip "Standard" (baseline)

    datasets_data = []
    for name in DATASET_COLS:
        _, runs = extract_dataset(rows, name)
        pc, err = process_dataset(runs)
        pc, err = pc[1:], err[1:]  # skip baseline
        datasets_data.append((name, pc, err))

    xlim_map = {'GSM8K': (-8, 2), 'MMLU': (-3, 2), 'CMMLU': (0, 9), 'HumanEval': (-11, 11)}

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.0), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 1, 1, 1], 'wspace': 0.1})
    y_pos = np.arange(len(instructions))[::-1]

    for i, (name, pc, err) in enumerate(datasets_data):
        ax = axes[i]
        color = COLORS[name]
        ax.barh(y_pos, pc, color=color, height=0.7, edgecolor='none', alpha=0.9)
        ax.errorbar(pc, y_pos, xerr=err, fmt='none', color='#333333',
                     alpha=0.7, linewidth=0.5, capsize=1.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.yaxis.set_visible(False)
        ax.set_title(name, fontsize=8, fontweight='bold', pad=10, color=color)
        ax.set_xlabel(r'$\Delta$ACC (%)', fontsize=7)
        ax.set_xlim(xlim_map[name])
        ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.3)
        if i == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(instructions, fontsize=7, ha='right')
            ax.yaxis.set_visible(True)
            ax.yaxis.set_tick_params(width=0)

    save_figure(fig, "Fig1_BarChart_TaskHeterogeneity", FIG_DIR)
    print(f"Saved figures to {FIG_DIR}")

if __name__ == "__main__":
    draw()
