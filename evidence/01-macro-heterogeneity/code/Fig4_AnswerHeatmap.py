"""
Figure 4: Answer Fragility Heatmap Generator.

Shows how minor prompt perturbations flip correctness on a per-question level.
Primary data: extracted_matrix_1000.npy (15 prompts x 1319 GSM8K questions).
If raw OpenCompass JSON results are present, they are used instead.

Output: PNG + JPG + TIFF + SVG → evidence/01-macro-heterogeneity/figures/
"""

import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# Ensure project root is on sys.path (robust against any cwd)
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from scripts.plot_config import setup_style, save_figure, OKABE_ITO as OI

# ---- Style ----
setup_style({"font.size": 7})

COLOR_PROMINENT = OI["blue"]   # #0072B2
COLOR_BACKGROUND = '#F7F7F7'

INSTRUCTIONS = [
    "Standard", "Think.", "Wait, think.", "Wait, wait. think.",
    "Think step by step.", "Think step-by-step.",
    "Let's think step by step.", "Let's think step-by-step.",
    "Step by step, let's solve this.", "Let's address this step by step.",
    "Think this through step by step.", "Let's approach this step by step.",
    "Step by step, let's work through this.",
    "Let's reason through this step by step.",
    "Let's go through the process step by step.",
]

# ---- Paths ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, "..", "data"))
_FIG = os.path.normpath(os.path.join(_HERE, "..", "figures"))
JSON_DIR = os.path.join(_DATA, "data_fig4", "20260317_141654", "20260317_141654", "results")
NPY_PATH = os.path.join(_DATA, "extracted_matrix_1000.npy")


def load_from_json(results_dir):
    """Try to parse OpenCompass JSON results. Returns (matrix, prompts, questions) or None."""
    if not os.path.isdir(results_dir):
        return None
    entries = os.listdir(results_dir)
    if not entries:
        return None

    all_results = defaultdict(dict)
    prompt_mapping = {}

    for folder in entries:
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder == "opencompass.models.CoTDeepSeek_R1_Distill_Qwen_1_5B_models_CoTDeepSeek_R1_Distill_Qwen_1_5B.py":
            prompt_num = 0
        elif "CoTDeepSeek_R1_Distill_Qwen_1_5B_" in folder:
            m = re.search(r'CoTDeepSeek_R1_Distill_Qwen_1_5B_(\d+)_models', folder)
            if not m:
                continue
            prompt_num = int(m.group(1))
        else:
            continue
        prompt_mapping[folder] = prompt_num

        for json_file in os.listdir(folder_path):
            if not json_file.endswith(".json"):
                continue
            json_path = os.path.join(folder_path, json_file)
            try:
                with open(json_path) as f:
                    data = json.load(f)
                dataset_name = json_file.replace(".json", "")
                if "details" in data:
                    for q_id, q_data in data["details"].items():
                        if q_id == "type":
                            continue
                        key = f"{dataset_name}_{q_id}"
                        pred = q_data.get("predictions", "")
                        ref = q_data.get("references", "")
                        all_results[key][prompt_num] = 1 if (pred and pred == ref) else 0
            except Exception as e:
                print(f"  Error {json_path}: {e}")

    if not all_results:
        return None

    questions = sorted(all_results.keys())
    prompts = sorted(prompt_mapping.values())
    matrix = np.full((len(prompts), len(questions)), np.nan)
    for i, pn in enumerate(prompts):
        for j, qk in enumerate(questions):
            matrix[i, j] = all_results[qk].get(pn, 0)

    print(f"Loaded from JSON: {len(prompts)} prompts x {len(questions)} questions")
    return matrix, prompts, questions


def load_from_npy(path):
    """Load pre-extracted binary matrix."""
    if not os.path.exists(path):
        return None
    matrix = np.load(path)
    prompts = list(range(matrix.shape[0]))
    questions = list(range(matrix.shape[1]))
    print(f"Loaded from NPY: {matrix.shape[0]} prompts x {matrix.shape[1]} questions")
    return matrix, prompts, questions


def draw():
    # --- Load data ---
    result = load_from_json(JSON_DIR)
    if result is None:
        print(f"No JSON results at {JSON_DIR}, falling back to NPY: {NPY_PATH}")
        result = load_from_npy(NPY_PATH)
    if result is None:
        raise FileNotFoundError(
            f"Cannot load heatmap data.\n"
            f"  JSON dir: {JSON_DIR}\n"
            f"  NPY file: {NPY_PATH}\n"
            f"  Run extract_matrix_1000.py first to generate the npy from the existing PNG."
        )
    total_matrix, all_prompts, all_questions = result

    # --- Color strategy ---
    n_correct = int(np.sum(total_matrix == 1))
    n_incorrect = int(np.sum(total_matrix == 0))
    n_total = n_correct + n_incorrect
    print(f"Correct: {n_correct} ({n_correct/max(n_total,1):.1%})  Incorrect: {n_incorrect}")

    if n_incorrect < n_correct:
        colors = [COLOR_PROMINENT, COLOR_BACKGROUND]
        legend_labels = {0: 'Incorrect', 1: 'Correct'}
    else:
        colors = [COLOR_BACKGROUND, COLOR_PROMINENT]
        legend_labels = {0: 'Incorrect', 1: 'Correct'}

    cmap = sns.color_palette(colors)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y_labels = [INSTRUCTIONS[i] if i < len(INSTRUCTIONS) else f"P{i}"
                for i in all_prompts]

    sns.heatmap(total_matrix, yticklabels=y_labels, xticklabels=False,
                cmap=cmap, cbar=False, vmin=0, vmax=1, ax=ax)

    # x-axis ticks
    n_q = len(all_questions)
    tick_positions = list(range(0, n_q, 200))
    if tick_positions[-1] != n_q - 1 and (n_q - 1 - tick_positions[-1]) > 50:
        tick_positions.append(n_q - 1)
    elif tick_positions[-1] != n_q - 1:
        tick_positions[-1] = n_q - 1
    ax.set_xticks([p + 0.5 for p in tick_positions])
    ax.set_xticklabels([str(p + 1) for p in tick_positions])
    ax.tick_params(axis='x', rotation=0, labelsize=7, bottom=True, length=3)

    # Legend
    lh, lw, lx, ly0 = 0.03, 0.02, 0.02, 0.88
    for val in [0, 1]:
        y = ly0 - val * (lh + 0.01)
        rect = plt.Rectangle((lx, y), lw, lh, facecolor=colors[val],
                             edgecolor='black', linewidth=0.5, transform=fig.transFigure)
        fig.patches.append(rect)
        fig.text(lx + lw + 0.01, y + lh / 2, legend_labels[val],
                 ha='left', va='center', fontsize=8, transform=fig.transFigure)

    ax.set_xlabel("Question Index", fontsize=8)
    ax.set_ylabel("Prompt Variants", fontsize=8)
    plt.tight_layout()

    base = os.path.join(_FIG, "Fig4_Heatmap_AnswerFragility")
    save_figure(fig, "Fig4_Heatmap_AnswerFragility", _FIG)
    print(f"Saved {base}.png/.jpg/.tiff/.svg")


if __name__ == "__main__":
    draw()
