"""Generate Fig3: Wait Frequency vs Accuracy scatter plot (3 models).

Data sources:
  - 7B: DeepSeek-R1-Distill-Qwen-7B (backup archive: accuracy from summary CSV,
    wait frequency from prediction JSON sample)
  - 8B: DeepSeek-R1-Distill-Llama-8B (same pipeline)
  - 1.5B: DeepSeek-R1-Distill-Qwen-1.5B (vLLM experiment, n=64, max_tokens=4096)

Output: evidence/02-state-divergence/figures/Fig3_Reflection_Scatter.{png,pdf,svg}
"""

import csv
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path (robust against any cwd)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.plot_config import setup_style, save_figure, OKABE_ITO as OI

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]  # evidence/02-state-divergence/
FIG_DIR = ROOT / "figures"
PAPER_FIG_DIR = Path(__file__).resolve().parents[3] / "paper" / "figures"

ARCHIVE_7B = (
    Path(__file__).resolve().parents[3]
    / "archive/local-noise/local_ignored_dirs_20260603/backups"
    / "aime2024_7b_pass1_n64_raw_20260528"
    / "aime2024_7b_pass1_n64_raw_20260528.tar.gz"
)
ARCHIVE_8B = (
    Path(__file__).resolve().parents[3]
    / "archive/local-noise/local_ignored_dirs_20260603/backups"
    / "aime2024_8b_pass1_n64_raw_20260528"
    / "aime2024_8b_pass1_n64_raw_20260528.tar.gz"
)

PROMPT_INDEX_MAP = {
    1: "Standard",
    2: "Think.",
    3: "Wait, think.",
    4: "Wait, wait. Think.",
    5: "Think step by step.",
    6: "Think step-by-step.",
    7: "Let's think step by step.",
    8: "Let's think step-by-step.",
    9: "Step by step, let's solve this.",
    10: "Let's address this step by step.",
    11: "Think this through step by step.",
    12: "Let's approach this step by step.",
    13: "Step by step, let's work through this.",
    14: "Let's reason through this step by step.",
    15: "Let's go through the process step by step.",
}

# ---- Style ----
setup_style({"font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
             "axes.linewidth": 0.6})

MODEL_COLORS = {"1.5B": OI["orange"], "7B": OI["sky_blue"], "8B": OI["bluish_green"]}
SIMULATED_COLOR = OI["vermillion"]


# ---- Reflection keyword set ----
# Based on R1's "aha moment" example and common LLM reflection patterns.
# Case-insensitive regex with word boundaries.
REFLECTION_PATTERNS = [
    r'\bwait\b',           # hesitation marker (R1's primary example)
    r'\bhowever\b',        # contrast/alternative path
    r'\bhmm\b',            # uncertainty
    r'\blet\s+me\b',       # "let me check/think/verify/reconsider"
    r'\bre[-]?evaluat',    # re-evaluate / reevaluate
    r'\breconsider',       # reconsider
    r'\balternatively\b',  # alternative approach
    r'\bhold\s+on\b',      # pause for re-evaluation
    r"\blet'?s\s+re",     # "let's reconsider/reevaluate/review"
]
REFLECTION_RE = re.compile('|'.join(REFLECTION_PATTERNS), re.IGNORECASE)

def count_reflection_markers(text):
    """Count total reflection-related keyword occurrences in text."""
    if not text:
        return 0
    return len(REFLECTION_RE.findall(text))

def pearson_r(x, y):
    xc, yc = x - np.mean(x), y - np.mean(y)
    denom = np.sqrt(np.sum(xc**2) * np.sum(yc**2))
    return float(np.sum(xc * yc) / denom) if denom > 0 else 0.0


def rankdata(values):
    """Average ranks for ties, using 1-based ranks."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_r(x, y):
    return pearson_r(rankdata(x), rankdata(y))


def permutation_p(x, y, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    r_obs = abs(spearman_r(x, y))
    count = 1
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if abs(spearman_r(x, yp)) >= r_obs:
            count += 1
    return count / (n_perm + 1)


def extract_real_data(archive_path, model_label):
    """Extract per-prompt (mean_reflect, accuracy) for 7B/8B.

    Prefers the materialized data in evidence/02-state-divergence/data/
    (reflection_{7b,8b}_real.json), which were extracted from the archived
    raw runs on 2026-08-13; falls back to the archive tar.gz otherwise.
    """
    materialized = ROOT / "data" / f"reflection_{model_label.lower()}_real.json"
    if materialized.exists():
        with open(materialized) as fh:
            raw = json.load(fh)
        return [(d["mean_reflect"], d["acc"], d["prompt"], d["p_idx"]) for d in raw]

    # Fallback: read accuracy from summary CSV in archive tar.gz
    summary_re = re.compile(r".*/summary/summary_.*\.csv$")
    accuracy_map = {}

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not summary_re.search(member.name):
                continue
            # Skip checkpoints
            if ".ipynb_checkpoints" in member.name:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            rows = list(csv.DictReader(io.StringIO(f.read().decode())))
            for row in rows:
                if row["dataset"] != "aime2024":
                    continue
                for col, val in row.items():
                    if col in ("dataset", "version", "metric", "mode"):
                        continue
                    if not val or val == "-":
                        continue
                    # Extract prompt index from column name
                    match = re.search(r"_(\d+)_models", col)
                    if match:
                        p_idx = int(match.group(1))
                        if p_idx not in accuracy_map:
                            accuracy_map[p_idx] = []
                        accuracy_map[p_idx].append(float(val))
            # 遍历所有 summary 合并各 run 的准确率（同一提示词跨 run 取均值）

    # 合并：同一提示词跨 run 取均值
    for p_idx in list(accuracy_map.keys()):
        vals = accuracy_map[p_idx]
        accuracy_map[p_idx] = float(np.mean(vals)) if vals else 0.0

    # 2. Sample prediction JSONs for wait frequency
    pred_re = re.compile(r".*/predictions/.*_(\d+)_models.*/aime2024_\d+\.json$")
    wait_samples = {p: [] for p in PROMPT_INDEX_MAP}

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            match = pred_re.search(member.name)
            if not match:
                continue
            p_idx = int(match.group(1))
            if p_idx not in PROMPT_INDEX_MAP:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                data = json.loads(f.read().decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            # data is dict with int keys -> {origin_prompt, prediction, gold}
            for item in data.values():
                if not isinstance(item, dict):
                    continue
                pred = str(item.get("prediction", ""))
                reflect_count = count_reflection_markers(pred)
                wait_samples[p_idx].append(reflect_count)

    # 3. Compute per-prompt means
    points = []
    for p_idx in sorted(accuracy_map.keys()):
        if p_idx not in PROMPT_INDEX_MAP:
            continue
        mean_reflect = float(np.mean(wait_samples[p_idx])) if wait_samples[p_idx] else 0.0
        acc = accuracy_map[p_idx]
        points.append((mean_reflect, acc, PROMPT_INDEX_MAP[p_idx], p_idx))

    return points


def load_1p5b_real_data() -> list[tuple[float, float, str, int]]:
    """Load real 1.5B data from vLLM experiment (uses mean_reflection as primary metric)."""
    data_path = ROOT / "data" / "reflection_1p5b_real.json"
    with open(data_path) as f:
        raw = json.load(f)
    points = []
    for item in raw:
        name, mean_wait, mean_reflect, acc, p_idx = item
        points.append((mean_reflect, acc, name, p_idx))  # use reflection count as x-axis
    return points


def draw():
    out_paths = []

    # ---- Real data: 7B ----
    points_7b = extract_real_data(ARCHIVE_7B, "7B")
    print(f"7B: {len(points_7b)} prompts extracted")
    for w, a, name, idx in points_7b:
        print(f"  prompt {idx:2d} ({name:45s}): wait={w:.1f}, acc={a:.1f}%")

    # ---- Real data: 8B ----
    points_8b = extract_real_data(ARCHIVE_8B, "8B")
    print(f"8B: {len(points_8b)} prompts extracted")
    for w, a, name, idx in points_8b:
        print(f"  prompt {idx:2d} ({name:45s}): wait={w:.1f}, acc={a:.1f}%")

    # ---- Simulated: 1.5B ----
    points_1p5b = load_1p5b_real_data()
    print(f"1.5B: {len(points_1p5b)} prompts (real data)")

    # ================================================================
    # Draw 3-panel figure
    # ================================================================
    models = [
        ("R1-Distill-Qwen-1.5B", points_1p5b, MODEL_COLORS["1.5B"]),
        ("R1-Distill-Qwen-7B",   points_7b,   MODEL_COLORS["7B"]),
        ("R1-Distill-Llama-8B",  points_8b,   MODEL_COLORS["8B"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55),
                             sharex=False, sharey=False,
                             gridspec_kw={"wspace": 0.28})

    for i, (label, points, color) in enumerate(models):
        ax = axes[i]

        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)

        # Remove points with extreme wait (outliers)
        # keep all for now

        if len(xs) >= 3:
            slope, intercept = np.polyfit(xs, ys, deg=1)
            x_grid = np.linspace(xs.min() - 1, xs.max() + 1, 100)
            y_hat = slope * x_grid + intercept

            r_val = spearman_r(xs, ys)
            p_val = permutation_p(xs, ys, seed=200 + i)
            # Fisher z 95% CI for Spearman rho
            n = len(xs)
            z = np.arctanh(float(np.clip(r_val, -0.9999, 0.9999)))
            se = 1.0 / np.sqrt(n - 3)
            ci_lo, ci_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

            ax.plot(x_grid, y_hat, color=OI["gray"], linewidth=0.85,
                    linestyle=(0, (3, 2)), alpha=0.68, zorder=2)

            # Stats box
            # Stats box — white bbox background + high zorder so text stays readable
            # when scatter points sit on top of the label region (e.g. panel b's
            # point at (~120.5, 51.5) overlapping "p = 0.137"). zorder=10 is
            # above scatter zorder=5.
            ax.text(0.97, 0.91, f"$\\rho$ = {r_val:+.2f}\n"
                                f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]\n"
                                f"$p$ = {p_val:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.8, color="#1F1F1F", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                              edgecolor="none", alpha=0.92))

        ax.set_facecolor("#FFFFFF")
        ax.scatter(xs, ys, s=27, color=color, alpha=0.88,
                   edgecolors="#FFFFFF", linewidths=0.45, zorder=5)

        panel = chr(ord("a") + i)
        # Panel label: above axes top-left (was inside axes at 0.02,0.97 — overlapped
        # data points at the y-axis top edge). set_title(loc='left') puts it above
        # the axes with proper padding, keeping the same font/color/style.
        ax.set_title(f"{panel}. {label}", loc="left",
                     fontsize=8.0, fontweight="bold", color=color, pad=4)
        # Sample size: bottom-left
        ax.text(0.02, 0.06, f"n = {len(xs)}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.5, color="#666666")

        # Per-panel axis labels (self-explanatory)
        ax.set_xlabel("Avg. reflection markers / response", fontsize=7.5, color="#333333")
        ax.set_ylabel("Pass@1 accuracy (%)", fontsize=7.5, color="#333333")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        ax.grid(axis="both", linestyle="--", color="#D9D9D9", alpha=0.36, linewidth=0.32)
        ax.tick_params(width=0.5)

    # Remove shared labels — each panel now self-contained
    fig.subplots_adjust(left=0.090, right=0.995, bottom=0.24, top=0.94, wspace=0.32)

    # ---- Save ----
    out_paths = save_figure(
        fig, "Fig3_Reflection_Scatter", FIG_DIR,
        formats=["png", "jpg", "tiff", "svg", "pdf"],
        close=False,
    )

    # Copy to paper/figures/
    save_figure(
        fig, "Fig3_Reflection_Scatter", PAPER_FIG_DIR,
        formats=["png", "jpg", "tiff", "svg", "pdf"],
    )


if __name__ == "__main__":
    draw()
