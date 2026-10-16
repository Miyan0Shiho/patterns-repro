"""Generate Fig. 2 (paper numbering): task heterogeneity with AIME 2024 panel.

Data provenance:
- GSM8K/MMLU/HumanEval values are copied from the v4.5-era `Fig1_TaskHeterogeneity.py`
  script (since removed from this repository as it relied on the earlier
  3-panel layout that included CMMLU).
- GSM8K/MMLU/HumanEval term-first values are from the strict group48 n=4 runs
  (the same data underlying the original 3-panel figure).
- AIME 2024 uses the included summary CSVs in
  `data/aime2024_qwen25_3b_n64_16k_scores/`:
  - `standard_baselineB_aggregate_summary.csv` (Standard, n=64)
  - `termfirst_repeatCDEF_aggregate_summary.csv` (Term-first, pooled n=256
    across 4 batches of 64)
  - `prompts15_aggregate_summary.csv` (the 14 CoT variants, n=64 each)
- AIME 2024 fields used: `mean_accuracy`, `sample_std_accuracy`, `runs`,
  `total_correct_over_1920`, and `pass_at_64_accuracy`.
- The term-first prompt is inserted into all four panels. AIME 2024 uses the
  pooled experimental summary: Standard B (n=64) and term-first pooled across
  the four repeated C-F batches (4 x 64, total n=256). The other three panels
  read the strict group48 n=4 runs.
- Error variants:
  - default: repeat-run standard deviation, matching the original script style.
  - SEM: `sample_std_accuracy / sqrt(runs)` for AIME 2024, and repeat standard
    error for the original four-repeat datasets.
  - mean-only: no error bars.

The original Fig. 2 prompt axis contains Standard + 14 CoT variants. The AIME
panel has one additional prompt, so this script keeps the common 14 variants in
the four-panel plot.
"""

from __future__ import annotations

import csv
import io
import re
import statistics
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path (robust against any cwd)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.plot_config import setup_style, save_figure, OKABE_ITO as OI


ERROR_STD = "std"
ERROR_SEM = "sem"
ERROR_NONE = "none"
TERM_FIRST_PROMPT = "What is the query? Term first, then reasoning."
TERM_FIRST_DISPLAY_LABEL = "What is the query?\nTerm first, then reasoning."

ROOT = Path(__file__).resolve().parents[1]  # evidence/01-macro-heterogeneity/
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
AIME_DIR = ROOT / "data" / "aime2024_qwen25_3b_n64_16k_scores"
BASELINE_CSV = AIME_DIR / "baseline_termfirst_aggregate_summary.csv"
STANDARD_B_CSV = AIME_DIR / "standard_baselineB_aggregate_summary.csv"
TERM_REPEAT_CDEF_CSV = AIME_DIR / "termfirst_repeatCDEF_aggregate_summary.csv"
PROMPTS_CSV = AIME_DIR / "prompts15_aggregate_summary.csv"
AIME_OFFICIAL_BASELINE_MODEL = "standard_baselineB"
# Pool the four Term-first repeat batches to avoid selecting a single batch.
AIME_OFFICIAL_TERM_MODEL = "termfirst_repeatCDEF_pooled"
GROUP48_ARCHIVE = (
    Path(__file__).resolve().parents[3]  # project root
    / "archive"
    / "local-noise"
    / "local_ignored_dirs_20260603"
    / "backups"
    / "qwen25_3b_group48_full4_n4_20260528"
    / "qwen25_3b_group48_full4_n4_20260528.tar.gz"
)
GROUP48_ARCHIVE_FALLBACKS = [
    GROUP48_ARCHIVE,
]
GROUP48_SUMMARY_RE = re.compile(
    r".*/opencompass/outputs/"
    r"(strict_term_group48_full4_run\d+_workers16)/"
    r"(\d+_\d+)/summary/summary_[0-9_]+\.csv$"
)


# ---- Style ----
setup_style({"font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})

# Paper-specific dataset colors (from Okabe-Ito)
COLORS = {
    "GSM8K": OI["orange"],
    "MMLU": OI["sky_blue"],
    "AIME2024": OI["blue"],
    "HumanEval": OI["reddish_purple"],
}


INSTRUCTIONS = [
    "Standard",
    "Think.",
    "Wait, think.",
    "Wait, wait. Think.",
    "Think step by step.",
    "Think step-by-step.",
    "Let's think step by step.",
    "Let's think step-by-step.",
    "Step by step, let's solve this.",
    "Let's address this step by step.",
    "Think this through step by step.",
    "Let's approach this step by step.",
    "Step by step, let's work through this.",
    "Let's reason through this step by step.",
    "Let's go through the process step by step.",
]

DISPLAY_INSTRUCTIONS = ["Standard", TERM_FIRST_DISPLAY_LABEL] + INSTRUCTIONS[1:]


@dataclass
class AimeRow:
    prompt: str
    mean_accuracy: float
    sample_std_accuracy: float
    runs: int
    total_correct: int
    pass_at_64_accuracy: float
    source: str
    selection_role: str = ""


@dataclass
class TermFirstSummary:
    task: str
    run_scores: list[float]
    source_paths: list[str]
    archive_path: str

    @property
    def runs(self) -> int:
        return len(self.run_scores)

    @property
    def mean_accuracy(self) -> float:
        return float(statistics.fmean(self.run_scores))

    @property
    def std_accuracy(self) -> float:
        return float(np.std(np.array(self.run_scores, dtype=float)))

    @property
    def sem_accuracy(self) -> float:
        return self.std_accuracy / np.sqrt(self.runs)


def process_dataset(
    scores_list: list[list[float]], error_mode: str
) -> tuple[np.ndarray, np.ndarray | None]:
    scores_array = np.array(scores_list, dtype=float)
    mean_scores = np.mean(scores_array, axis=0)
    std_scores = np.std(scores_array, axis=0)
    base_score = mean_scores[0]
    delta_pp = mean_scores - base_score
    if error_mode == ERROR_STD:
        delta_err = std_scores
    elif error_mode == ERROR_SEM:
        delta_err = std_scores / np.sqrt(scores_array.shape[0])
    elif error_mode == ERROR_NONE:
        delta_err = None
    else:
        raise ValueError(f"Unknown error mode: {error_mode}")
    return delta_pp, delta_err


def normalize_prompt(prompt: str) -> str:
    return prompt.strip()


def load_aime_rows() -> tuple[AimeRow, dict[str, AimeRow], list[AimeRow]]:
    with STANDARD_B_CSV.open(newline="") as f:
        standard_rows = list(csv.DictReader(f))
    with TERM_REPEAT_CDEF_CSV.open(newline="") as f:
        term_repeat_rows = list(csv.DictReader(f))

    standard = next(
        row for row in standard_rows if row["model"] == AIME_OFFICIAL_BASELINE_MODEL
    )
    termfirst = next(
        row for row in term_repeat_rows if row["model"] == AIME_OFFICIAL_TERM_MODEL
    )
    baseline = AimeRow(
        prompt="Standard",
        mean_accuracy=float(standard["mean_accuracy"]),
        sample_std_accuracy=float(standard["sample_std_accuracy"]),
        runs=int(standard["runs"]),
        total_correct=int(standard["total_correct_over_1920"]),
        pass_at_64_accuracy=float(standard["pass_at_64_accuracy"]),
        source=STANDARD_B_CSV.relative_to(ROOT).as_posix(),
        selection_role="official_batch",
    )
    termfirst_row = AimeRow(
        prompt=TERM_FIRST_PROMPT,
        mean_accuracy=float(termfirst["mean_accuracy"]),
        sample_std_accuracy=float(termfirst["sample_std_accuracy"]),
        runs=int(termfirst["runs"]),
        total_correct=int(termfirst["total_correct_over_1920"]),
        pass_at_64_accuracy=(
            float(termfirst["pass_at_64_accuracy"])
            if termfirst.get("pass_at_64_accuracy")
            else int(termfirst["total_correct_over_1920"]) / int(termfirst["pred_count"]) * 100
        ),
        source=TERM_REPEAT_CDEF_CSV.relative_to(ROOT).as_posix(),
        selection_role="official_batch",
    )

    by_prompt: dict[str, AimeRow] = {TERM_FIRST_DISPLAY_LABEL: termfirst_row}
    all_rows = [baseline, termfirst_row]
    with PROMPTS_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            item = AimeRow(
                prompt=normalize_prompt(row["system_prompt"]),
                mean_accuracy=float(row["mean_accuracy"]),
                sample_std_accuracy=float(row["sample_std_accuracy"]),
                runs=int(row["runs"]),
                total_correct=int(row["total_correct_over_1920"]),
                pass_at_64_accuracy=float(row["pass_at_64_accuracy"]),
                source=PROMPTS_CSV.relative_to(ROOT).as_posix(),
            )
            by_prompt[item.prompt] = item
            all_rows.append(item)
    return baseline, by_prompt, all_rows


def load_group48_termfirst_summaries() -> dict[str, TermFirstSummary]:
    archive_path = next(
        (candidate for candidate in GROUP48_ARCHIVE_FALLBACKS if candidate.exists()),
        None,
    )
    if archive_path is None:
        checked = "\n".join(str(candidate) for candidate in GROUP48_ARCHIVE_FALLBACKS)
        raise FileNotFoundError(f"Missing group48 backup archive. Checked:\n{checked}")

    task_scores: dict[str, list[float]] = {
        "GSM8K": [],
        "MMLU": [],
        "CMMLU": [],
        "HumanEval": [],
    }
    source_paths: list[str] = []

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in sorted(tar.getmembers(), key=lambda item: item.name):
            if not GROUP48_SUMMARY_RE.search(member.name):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            rows = list(csv.DictReader(io.StringIO(extracted.read().decode())))
            values = {row["dataset"]: float(row["CoTqwen2x5_3b_48"]) for row in rows}
            mmlu_values = [
                value
                for key, value in values.items()
                if key.startswith("lukaemon_mmlu_")
            ]
            cmmlu_values = [
                value for key, value in values.items() if key.startswith("cmmlu-")
            ]

            task_scores["GSM8K"].append(values["gsm8k"])
            task_scores["MMLU"].append(float(statistics.fmean(mmlu_values)))
            task_scores["CMMLU"].append(float(statistics.fmean(cmmlu_values)))
            task_scores["HumanEval"].append(values["openai_humaneval"])
            source_paths.append(member.name)

    missing = [task for task, scores in task_scores.items() if len(scores) != 4]
    if missing:
        raise RuntimeError(
            "Expected exactly four group48 runs for each task; missing/incomplete: "
            + ", ".join(missing)
        )

    return {
        task: TermFirstSummary(
            task=task,
            run_scores=scores,
            source_paths=source_paths,
            archive_path=str(archive_path),
        )
        for task, scores in task_scores.items()
    }


def write_source_table(
    original_scores: dict[str, list[list[float]]],
    term_summaries: dict[str, TermFirstSummary],
    all_aime_rows: list[AimeRow],
    selected_aime_rows: list[AimeRow],
) -> None:
    data_out_path = DATA_DIR / "Fig2_TaskHeterogeneity_data.csv"
    # (data CSV goes to DATA_DIR, not FIG_DIR — no separate figure data file)
    selected_aime_prompts = {row.prompt for row in selected_aime_rows}
    with data_out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "prompt",
                "included_in_four_panel_plot",
                "mean_accuracy",
                "std_accuracy",
                "std_definition",
                "sem_accuracy",
                "relative_delta_vs_standard_percent",
                "relative_sem_vs_standard_percent",
                "absolute_delta_vs_standard_pp",
                "runs",
                "total_correct_over_1920",
                "pass_at_64_accuracy",
                "source",
                "selection_role",
            ],
        )
        writer.writeheader()

        def write_row(
            *,
            dataset: str,
            prompt: str,
            included: bool,
            mean_accuracy: float,
            std_accuracy: float,
            std_definition: str,
            sem_accuracy: float,
            baseline_accuracy: float,
            runs: int,
            source: str,
            total_correct: int | str = "",
            pass_at_64_accuracy: float | str = "",
            selection_role: str = "",
        ) -> None:
            writer.writerow(
                {
                    "dataset": dataset,
                    "prompt": prompt,
                    "included_in_four_panel_plot": included,
                    "mean_accuracy": mean_accuracy,
                    "std_accuracy": std_accuracy,
                    "std_definition": std_definition,
                    "sem_accuracy": sem_accuracy,
                    "relative_delta_vs_standard_percent": (
                        (mean_accuracy - baseline_accuracy) / baseline_accuracy * 100
                    ),
                    "relative_sem_vs_standard_percent": (
                        sem_accuracy / baseline_accuracy * 100
                    ),
                    "absolute_delta_vs_standard_pp": mean_accuracy - baseline_accuracy,
                    "runs": runs,
                    "total_correct_over_1920": total_correct,
                    "pass_at_64_accuracy": pass_at_64_accuracy,
                    "source": source,
                    "selection_role": selection_role,
                }
            )

        for dataset, scores in original_scores.items():
            score_array = np.array(scores, dtype=float)
            baseline_accuracy = float(np.mean(score_array[:, 0]))
            source = "code/plotting/01_macro/Fig1_TaskHeterogeneity.py"

            standard_values = score_array[:, 0]
            standard_std = float(np.std(standard_values))
            write_row(
                dataset=dataset,
                prompt="Standard",
                included=True,
                mean_accuracy=baseline_accuracy,
                std_accuracy=standard_std,
                std_definition="original_fig1_repeat_population_std_n4",
                sem_accuracy=standard_std / np.sqrt(score_array.shape[0]),
                baseline_accuracy=baseline_accuracy,
                runs=score_array.shape[0],
                source=source,
            )

            term_summary = term_summaries[dataset]
            write_row(
                dataset=dataset,
                prompt=TERM_FIRST_PROMPT,
                included=True,
                mean_accuracy=term_summary.mean_accuracy,
                std_accuracy=term_summary.std_accuracy,
                std_definition="group48_repeat_population_std_n4",
                sem_accuracy=term_summary.sem_accuracy,
                baseline_accuracy=baseline_accuracy,
                runs=term_summary.runs,
                source=f"{term_summary.archive_path};"
                + ";".join(term_summary.source_paths),
            )

            for idx, prompt in enumerate(INSTRUCTIONS[1:], start=1):
                values = score_array[:, idx]
                std_accuracy = float(np.std(values))
                write_row(
                    dataset=dataset,
                    prompt=prompt,
                    included=True,
                    mean_accuracy=float(np.mean(values)),
                    std_accuracy=std_accuracy,
                    std_definition="original_fig1_repeat_population_std_n4",
                    sem_accuracy=std_accuracy / np.sqrt(score_array.shape[0]),
                    baseline_accuracy=baseline_accuracy,
                    runs=score_array.shape[0],
                    source=source,
                )

        aime_baseline = all_aime_rows[0]
        ordered_aime_rows = selected_aime_rows + [
            row for row in all_aime_rows if row.prompt not in selected_aime_prompts
        ]
        for row in ordered_aime_rows:
            write_row(
                dataset="AIME2024",
                prompt=row.prompt,
                included=row.prompt in selected_aime_prompts,
                mean_accuracy=row.mean_accuracy,
                std_accuracy=row.sample_std_accuracy,
                std_definition="aggregate_csv_sample_std_n64",
                sem_accuracy=row.sample_std_accuracy / np.sqrt(row.runs),
                baseline_accuracy=aime_baseline.mean_accuracy,
                runs=row.runs,
                total_correct=row.total_correct,
                pass_at_64_accuracy=row.pass_at_64_accuracy,
                source=row.source,
                selection_role=row.selection_role,
            )


def process_aime_dataset(
    baseline: AimeRow, by_prompt: dict[str, AimeRow], error_mode: str
) -> tuple[np.ndarray, np.ndarray | None, list[AimeRow]]:
    selected = [baseline]
    for prompt in DISPLAY_INSTRUCTIONS[1:]:
        key = normalize_prompt(prompt)
        if key not in by_prompt:
            raise KeyError(f"Missing AIME prompt in aggregate CSV: {prompt!r}")
        selected.append(by_prompt[key])

    means = np.array([row.mean_accuracy for row in selected], dtype=float)
    stds = np.array([row.sample_std_accuracy for row in selected], dtype=float)
    runs = np.array([row.runs for row in selected], dtype=float)
    base_score = means[0]
    delta_pp = means - base_score
    if error_mode == ERROR_STD:
        delta_err = stds
    elif error_mode == ERROR_SEM:
        delta_err = stds / np.sqrt(runs)
    elif error_mode == ERROR_NONE:
        delta_err = None
    else:
        raise ValueError(f"Unknown error mode: {error_mode}")
    return delta_pp, delta_err, selected



def load_termfirst_summaries_from_csv() -> dict[str, TermFirstSummary]:
    """Fallback: read term-first summaries from a pre-computed CSV.

    Used when the group48 backup archive is not present (typical for the
    public release). The CSV aggregates 4 pseudo-runs per task from the
    mean and std reported in
    data/Fig1_BarChart_TaskHeterogeneity_AIME2024_data.csv.
    """
    _HERE = Path(__file__).resolve().parent
    csv_path = _HERE.parent / "data" / "termfirst_4panel_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Term-first fallback CSV not found at {csv_path}. "
            "Either place the CSV or restore the group48 backup archive."
        )

    task_scores: dict[str, list[float]] = {}
    source_paths: list[str] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["task"]
            task_scores.setdefault(task, []).append(float(row["accuracy"]))
            source_paths.append(str(csv_path))

    return {
        task: TermFirstSummary(
            task=task,
            run_scores=scores,
            source_paths=source_paths,
            archive_path=str(csv_path),
        )
        for task, scores in task_scores.items()
        if len(scores) == 4
    }


def draw_variant(error_mode: str, output_suffix: str, aime_xlim: list[float]) -> None:
    # Original hardcoded datasets from Fig1_TaskHeterogeneity.py.
    scores_gsm8k = [79.38, 81.27, 79.08, 79.61, 79.45, 78.70, 78.62, 80.21, 79.30, 78.62, 76.57, 79.83, 77.03, 79.91, 78.39]
    scores_gsm8k_2 = [81.05, 80.06, 80.44, 79.30, 77.26, 80.06, 79.45, 80.29, 80.06, 78.32, 77.10, 80.14, 77.79, 78.54, 80.21]
    scores_gsm8k_3 = [80.89, 79.98, 79.15, 78.77, 77.26, 78.77, 79.98, 78.62, 78.92, 78.77, 78.09, 79.91, 78.39, 79.53, 78.09]
    scores_gsm8k_4 = [81.43, 78.47, 80.29, 78.92, 77.94, 78.32, 77.56, 78.62, 78.24, 78.92, 77.79, 78.92, 80.06, 79.00, 80.59]

    scores_mmlu = [65.08, 65.26, 65.60, 65.70, 64.18, 64.41, 63.87, 64.71, 64.36, 65.02, 63.38, 65.67, 64.34, 64.37, 65.33]
    scores_mmlu_2 = [65.23, 65.38, 65.42, 65.78, 63.48, 64.53, 64.91, 64.81, 63.90, 65.36, 63.66, 65.67, 64.62, 63.93, 64.84]
    scores_mmlu_3 = [64.73, 65.19, 65.50, 64.99, 64.33, 63.73, 64.35, 64.67, 64.60, 64.62, 63.45, 65.19, 64.48, 64.28, 65.02]
    scores_mmlu_4 = [65.25, 65.52, 64.94, 65.39, 64.20, 64.15, 64.23, 64.14, 64.41, 64.87, 63.41, 64.87, 64.48, 64.43, 65.00]

    scores_humaneval = [59.76, 64.63, 59.15, 65.24, 59.76, 56.71, 61.59, 60.98, 62.20, 60.98, 62.20, 60.98, 64.02, 62.20, 60.98]
    scores_humaneval_2 = [60.37, 62.80, 65.24, 59.15, 58.54, 54.88, 53.66, 58.54, 59.15, 61.59, 56.10, 64.63, 63.41, 62.20, 58.54]
    scores_humaneval_3 = [62.20, 60.98, 65.85, 62.80, 62.80, 59.76, 60.98, 59.76, 60.37, 64.63, 60.98, 65.24, 68.29, 60.37, 57.93]
    scores_humaneval_4 = [62.20, 65.85, 65.85, 61.59, 56.71, 57.93, 56.71, 56.71, 59.15, 64.63, 62.80, 59.15, 60.98, 65.24, 60.98]

    gsm8k_scores = [scores_gsm8k, scores_gsm8k_2, scores_gsm8k_3, scores_gsm8k_4]
    mmlu_scores = [scores_mmlu, scores_mmlu_2, scores_mmlu_3, scores_mmlu_4]
    humaneval_scores = [
        scores_humaneval,
        scores_humaneval_2,
        scores_humaneval_3,
        scores_humaneval_4,
    ]

    gsm8k_pc, gsm8k_err = process_dataset(
        gsm8k_scores,
        error_mode,
    )
    mmlu_pc, mmlu_err = process_dataset(
        mmlu_scores,
        error_mode,
    )
    humaneval_pc, humaneval_err = process_dataset(
        humaneval_scores,
        error_mode,
    )

    term_summaries = load_termfirst_summaries_from_csv() or load_group48_termfirst_summaries()
    baseline, by_prompt, all_aime_rows = load_aime_rows()
    aime_pc, aime_err, selected_aime_rows = process_aime_dataset(
        baseline, by_prompt, error_mode
    )
    write_source_table(
        {
            "GSM8K": gsm8k_scores,
            "MMLU": mmlu_scores,
            "HumanEval": humaneval_scores,
        },
        term_summaries,
        all_aime_rows,
        selected_aime_rows,
    )

    def trim_error(err: np.ndarray | None) -> np.ndarray | None:
        return None if err is None else err[1:]

    def insert_termfirst(
        pc: np.ndarray,
        err: np.ndarray | None,
        scores: list[list[float]],
        term_summary: TermFirstSummary,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        score_array = np.array(scores, dtype=float)
        base_score = float(np.mean(score_array[:, 0]))
        term_pp = term_summary.mean_accuracy - base_score
        out_pc = np.insert(pc[1:], 0, term_pp)
        if err is None:
            return out_pc, None
        if error_mode == ERROR_STD:
            term_err = term_summary.std_accuracy
        elif error_mode == ERROR_SEM:
            term_err = term_summary.sem_accuracy
        else:
            raise ValueError(f"Unknown error mode for term-first row: {error_mode}")
        trimmed = trim_error(err)
        if trimmed is None:
            return out_pc, None
        return out_pc, np.insert(trimmed, 0, term_err)

    gsm8k_plot_pc, gsm8k_plot_err = insert_termfirst(
        gsm8k_pc, gsm8k_err, gsm8k_scores, term_summaries["GSM8K"]
    )
    mmlu_plot_pc, mmlu_plot_err = insert_termfirst(
        mmlu_pc, mmlu_err, mmlu_scores, term_summaries["MMLU"]
    )
    humaneval_plot_pc, humaneval_plot_err = insert_termfirst(
        humaneval_pc, humaneval_err, humaneval_scores, term_summaries["HumanEval"]
    )

    datasets = [
        ("GSM8K", gsm8k_plot_pc, gsm8k_plot_err, COLORS["GSM8K"], [-5, 5]),
        ("MMLU", mmlu_plot_pc, mmlu_plot_err, COLORS["MMLU"], [-3, 3]),
        (
            "HumanEval",
            humaneval_plot_pc,
            humaneval_plot_err,
            COLORS["HumanEval"],
            [-6, 12],
        ),
        ("AIME 2024", aime_pc[1:], trim_error(aime_err), COLORS["AIME2024"], aime_xlim),
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.4, 4.25),
        gridspec_kw={"width_ratios": [1, 1, 1, 1], "wspace": 0.22},
    )
    y_pos = np.arange(len(DISPLAY_INSTRUCTIONS[1:]))[::-1]

    for i, (name, pc, err, color, xlim) in enumerate(datasets):
        ax = axes[i]
        ax.barh(y_pos, pc, color=color, height=0.7, edgecolor="none", alpha=0.9)
        if err is not None:
            valid = ~(np.isnan(pc) | np.isnan(err))
            ax.errorbar(
                pc[valid],
                y_pos[valid],
                xerr=err[valid],
                fmt="none",
                color=OI["gray"],
                alpha=0.7,
                linewidth=0.5,
                capsize=1.5,
            )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.xaxis.set_tick_params(width=0.5)
        ax.yaxis.set_visible(False)

        ax.set_title(name, fontsize=8, fontweight="bold", pad=10, color=color)
        ax.set_xlabel(r"$\Delta$ACC (pp)", fontsize=7)
        ax.set_xlim(xlim)

        if i == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(DISPLAY_INSTRUCTIONS[1:], fontsize=7, ha="right")
            ax.yaxis.set_visible(True)
            ax.yaxis.set_tick_params(width=0)

        ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.3)

    output_base = f"Fig2_TaskHeterogeneity{output_suffix}"
    save_figure(fig, output_base, FIG_DIR)


def draw() -> None:
    # Render only the default SEM variant, which matches the paper Figure 2
    # caption ("Error bars indicate the standard error of the mean"). The
    # _std and _mean_only variants are available via direct calls to
    # draw_variant() but not generated by default to keep the release lean.
    draw_variant(ERROR_SEM, "", [-1, 1])


if __name__ == "__main__":
    draw()
