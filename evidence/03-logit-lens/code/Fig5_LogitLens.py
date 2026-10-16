"""
Fig4: 4-condition Logit Lens trajectories (updated with term_first).
5 AIME 2024 questions × 4 prompt conditions × 29-layer probability trajectories.
Okabe-Ito colorblind-safe palette.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path (robust against any cwd)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from scripts.plot_config import setup_style, save_figure, OKABE_ITO as OI

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "logit_lens_5q_4cond.json")
DATA_PATH = os.path.normpath(DATA_PATH)

EVIDENCE_FIG_DIR = os.path.join(HERE, "..", "figures")
EVIDENCE_FIG_DIR = os.path.normpath(EVIDENCE_FIG_DIR)
# PAPER_FIG_DIR removed for release isolation

CONDITIONS = ["under_thinking", "normal_thinking", "over_thinking", "term_first"]
LABELS = {
    "under_thinking": "Premature convergence",
    "normal_thinking": "Delayed commitment",
    "over_thinking": "Persistent oscillation",
    "term_first": "Term-first",
}
# Okabe-Ito — 论文专属四态语义映射
COLORS = {
    "under_thinking": OI["vermillion"],   # 警示
    "normal_thinking": OI["bluish_green"], # 正向
    "over_thinking":  OI["blue"],          # 冷调
    "term_first":     OI["sky_blue"],      # 区分但正向
}

# ---- Load data ----
with open(DATA_PATH) as f:
    data = json.load(f)

# Aggregate trajectories per condition across all 5 questions
trajectories = {c: [] for c in CONDITIONS}
for qid, conds in data.items():
    for cname in CONDITIONS:
        traj = conds[cname]["layer_probs"]
        trajectories[cname].append(traj)

# Compute mean and std across questions
mean_traj = {}
std_traj = {}
for cname in CONDITIONS:
    arr = np.array(trajectories[cname])
    mean_traj[cname] = np.mean(arr, axis=0)
    std_traj[cname] = np.std(arr, axis=0)

num_layers = len(mean_traj["under_thinking"])
layers = np.arange(num_layers)

# ---- Style ----
setup_style({
    "font.size": 7.6,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.color": "#D9D9D9",
    "grid.alpha": 0.45,
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "axes.edgecolor": "#333333",
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
})

# ---- Draw ----
fig, axes = plt.subplots(
    2, 2, figsize=(7.2, 3.55), dpi=300,
    sharex=True, sharey=True,
    gridspec_kw={"wspace": 0.14, "hspace": 0.26},
)

for i, cname in enumerate(CONDITIONS):
    ax = axes.flat[i]

    mean = mean_traj[cname]
    std = std_traj[cname]
    color_main = COLORS[cname]
    band_low = np.clip(mean - std, 0, 1)
    band_high = np.clip(mean + std, 0, 1)

    ax.fill_between(layers, band_low, band_high, color=color_main, alpha=0.095, linewidth=0)
    for traj in trajectories[cname]:
        ax.plot(layers, traj, color=color_main, alpha=0.155, linewidth=0.65)

    ax.plot(layers, mean, color=color_main, alpha=1.0, linewidth=2.25)

    peak_idx = np.argmax(mean)
    peak_val = mean[peak_idx]
    ax.vlines(peak_idx, 0, peak_val, color=color_main, alpha=0.22,
              linewidth=0.70, linestyles=(0, (2, 2)), zorder=2)
    ax.scatter([peak_idx], [peak_val], s=28, color=color_main,
               edgecolors="white", linewidth=0.7, zorder=5)

    panel = chr(ord("a") + i)
    ax.text(0.03, 0.92, f"{panel}. {LABELS[cname]}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.3, fontweight="bold", color=color_main)
    x_text = peak_idx - 0.9 if peak_idx > num_layers - 4 else peak_idx + 0.6
    y_text = peak_val - 0.08 if peak_val > 0.82 else peak_val + 0.08
    ax.text(x_text, y_text, f"L{peak_idx}",
            ha="right" if peak_idx > num_layers - 4 else "left",
            va="top" if peak_val > 0.82 else "bottom",
            fontsize=6.8, color="#4D4D4D")

    ax.set_xlim(0, num_layers - 1)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([0, 7, 14, 21, 28])
    ax.set_yticks([0, 0.50, 1.00])
    ax.axhline(0.5, color="#BDBDBD", linewidth=0.5, linestyle=(0, (3, 3)), zorder=0)
    ax.grid(axis="y", color="#D9D9D9", alpha=0.45, linewidth=0.35, linestyle="--")
    ax.grid(axis="x", color="#E9E9E9", alpha=0.35, linewidth=0.30, linestyle="--")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.5, length=3)

    if i % 2 == 1:
        ax.tick_params(labelleft=False)

fig.supxlabel("Layer", fontsize=8.2, y=0.038)
fig.supylabel("Token probability", fontsize=8.2, x=0.010)
plt.subplots_adjust(left=0.078, right=0.995, bottom=0.15, top=0.985, wspace=0.14, hspace=0.26)

os.makedirs(EVIDENCE_FIG_DIR, exist_ok=True)
save_figure(fig, "Fig5_LogitLens", EVIDENCE_FIG_DIR, formats=["png", "jpg", "tiff", "svg", "pdf"])
# PAPER_FIG_DIR write removed for release isolation
print("Done.")
