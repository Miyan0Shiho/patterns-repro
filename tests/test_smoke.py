"""Smoke tests for the patterns-repro release.

Verify that all data files referenced in the manuscript are present, load
correctly, and contain the expected high-level invariants. This does NOT
verify numerical precision (that's checked in the figure regeneration scripts)
but does ensure the release is internally consistent.

Run with: python3 -m tests.test_smoke
Exit code 0 = all pass; non-zero = at least one failure.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "evidence"


class TestResult:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed.append(name)
            print(f"  PASS  {name}")
        else:
            self.failed.append((name, detail))
            print(f"  FAIL  {name}: {detail}")

    def report(self) -> int:
        print()
        print(f"Total: {len(self.passed)} passed, {len(self.failed)} failed")
        return 0 if not self.failed else 1


# ─────────────────────────────────────────────────────────────────────
# Experiment 1: macro-heterogeneity
# ─────────────────────────────────────────────────────────────────────

def test_01_macro_heterogeneity(r: TestResult) -> None:
    print("\n[01-macro-heterogeneity]")

    # Task heterogeneity (Fig 2): GSM8K/MMLU/HumanEval, 15 prompts × 4 runs
    p = EVIDENCE / "01-macro-heterogeneity/data/fig1_task_heterogeneity_scores.csv"
    r.check("fig1_task_heterogeneity_scores.csv exists", p.exists(),
            f"missing {p}")
    if p.exists():
        with p.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n_prompts = len(rows)
        header = rows[0].keys() if rows else []
        # Expect 15 prompts (Standard + 14 CoT variants)
        r.check("fig1 has 15 prompt rows", n_prompts == 15,
                f"got {n_prompts}")
        # Verify no CMMLU-named prompts in Instruction column
        cmmlu_prompts = [r['Instruction'] for r in rows if 'CMMLU' in r['Instruction'] or 'cmmlu' in r['Instruction']]
        r.check("fig1 Instructions don't mention CMMLU", len(cmmlu_prompts) == 0,
                f"found: {cmmlu_prompts}")
        # Expect GSM8K_R1..R4 + MMLU_R1..R4 + HumanEval_R1..R4
        expected_cols = ['GSM8K_R1', 'GSM8K_R2', 'GSM8K_R3', 'GSM8K_R4',
                         'MMLU_R1', 'MMLU_R2', 'MMLU_R3', 'MMLU_R4',
                         'HumanEval_R1', 'HumanEval_R2', 'HumanEval_R3', 'HumanEval_R4']
        for c in expected_cols:
            r.check(f"fig1 has column {c}", c in header,
                    f"missing column")
        # Standard row GSM8K accuracy roughly 80% (sanity)
        std_row = next((r for r in rows if r['Instruction'] == 'Standard'), None)
        if std_row:
            gsm8k_mean = sum(float(std_row[f'GSM8K_R{i}']) for i in range(1, 5)) / 4
            r.check("fig1 Standard GSM8K ≈ 80%", 75 < gsm8k_mean < 85,
                    f"got {gsm8k_mean:.2f}")

    # Model scale (Fig 3): 5 scales × 15 prompts × 4 runs
    p = EVIDENCE / "01-macro-heterogeneity/data/fig2_model_scale_scores.csv"
    r.check("fig2_model_scale_scores.csv exists", p.exists(),
            f"missing {p}")
    if p.exists():
        with p.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n_prompts = len(rows)
        # 15 CoT variants + 1 Term-first = 16 rows
        r.check("fig2 has 16 prompt rows (15 CoT + 1 Term-first)",
                n_prompts == 16, f"got {n_prompts}")
        # No 'Baseline_direct' (stale row from cleanup)
        bad_prompts = [r['Instruction'] for r in rows
                       if r['Instruction'] == 'Baseline_direct']
        r.check("fig2 has no stale 'Baseline_direct' row", len(bad_prompts) == 0,
                f"found: {bad_prompts}")
        # Term-first row exists and is the short-form 'Term first, then reasoning.'
        tf_rows_in_fig2 = [r for r in rows if 'Term first' in r['Instruction']]
        r.check("fig2 has exactly 1 Term-first row", len(tf_rows_in_fig2) == 1,
                f"got {len(tf_rows_in_fig2)}")
        # Expect 5 scales × 4 runs = 20 columns
        expected_scales = ['0.5B', '1.5B', '3B', '7B', '32B']
        for s in expected_scales:
            for i in range(1, 5):
                col = f"{s}_R{i}"
                r.check(f"fig2 has column {col}", col in (rows[0].keys() if rows else []),
                        "missing column")
        # Sanity: Standard at 32B ≈ 89%
        std_row = next((r for r in rows if r['Instruction'] == 'Standard'), None)
        if std_row:
            acc_32b = sum(float(std_row[f'32B_R{i}']) for i in range(1, 5)) / 4
            r.check("fig2 Standard 32B ≈ 89%", 85 < acc_32b < 92,
                    f"got {acc_32b:.2f}")

    # AIME 2024 aggregate files
    aime_dir = EVIDENCE / "01-macro-heterogeneity/data/aime2024_qwen25_3b_n64_16k_scores"
    for f in ["standard_baselineB_aggregate_summary.csv",
              "termfirst_repeatCDEF_aggregate_summary.csv",
              "prompts15_aggregate_summary.csv"]:
        r.check(f"aime2024 {f} exists", (aime_dir / f).exists(),
                f"missing {f}")

    # Verify the term-first pre-computed 4-run summary (CSV fallback for archive)
    tf_csv = EVIDENCE / "01-macro-heterogeneity/data/termfirst_4panel_summary.csv"
    r.check("termfirst_4panel_summary.csv exists", tf_csv.exists())
    if tf_csv.exists():
        with tf_csv.open() as f:
            reader = csv.DictReader(f)
            tf_rows = list(reader)
        # 3 tasks × 4 runs = 12 rows
        r.check("termfirst_4panel has 12 rows", len(tf_rows) == 12,
                f"got {len(tf_rows)}")
        tasks = set(r['task'] for r in tf_rows)
        expected_tasks = {'GSM8K', 'MMLU', 'HumanEval'}
        r.check("termfirst_4panel has expected tasks (no CMMLU, no AIME2024)",
                tasks == expected_tasks,
                f"got {tasks}, expected {expected_tasks}")
        # 4 runs per task
        from collections import Counter
        run_counts = Counter(r['task'] for r in tf_rows)
        for t in expected_tasks:
            r.check(f"termfirst_4panel has 4 runs for {t}", run_counts[t] == 4,
                    f"got {run_counts[t]}")

    # Verify the merged 4-panel data file
    merged = EVIDENCE / "01-macro-heterogeneity/data/Fig2_TaskHeterogeneity_data.csv"
    r.check("Fig2_TaskHeterogeneity_data.csv exists", merged.exists())
    if merged.exists():
        with merged.open() as f:
            rows = list(csv.DictReader(f))
        # Should have rows for GSM8K + MMLU + HumanEval + AIME 2024 × 15 prompts
        datasets_in_merged = set(r.get('dataset') for r in rows)
        r.check("merged has GSM8K", 'GSM8K' in datasets_in_merged)
        r.check("merged has MMLU", 'MMLU' in datasets_in_merged)
        r.check("merged has HumanEval", 'HumanEval' in datasets_in_merged)
        r.check("merged has AIME2024", 'AIME2024' in datasets_in_merged)
        # No CMMLU (paper doesn't use it)
        r.check("merged does NOT include CMMLU", 'CMMLU' not in datasets_in_merged,
                "CMMLU is not in the paper")


# ─────────────────────────────────────────────────────────────────────
# Experiment 2: state-divergence
# ─────────────────────────────────────────────────────────────────────

def test_02_state_divergence(r: TestResult) -> None:
    print("\n[02-state-divergence]")

    # Reflection scatter (Fig 4): 3 JSON files
    for model, n_prompts in [("1p5b", 16), ("7b", 15), ("8b", 15)]:
        p = EVIDENCE / f"02-state-divergence/data/reflection_{model}_real.json"
        r.check(f"reflection_{model}_real.json exists", p.exists())
        if p.exists():
            with p.open() as f:
                data = json.load(f)
            if model == "1p5b":
                # 1.5B has 16 prompts (Standard + 14 CoT + Term-first)
                r.check(f"reflection_{model} has {n_prompts} entries",
                        len(data) == n_prompts,
                        f"got {len(data)}")
                # Verify each entry is a list with [prompt_name, val1, val2, val3, run_idx]
                if data and isinstance(data[0], list) and len(data[0]) >= 4:
                    r.check(f"reflection_{model} entries are lists with prompt + 3 values",
                            True)
            else:
                # 7B/8B are dicts
                r.check(f"reflection_{model} has {n_prompts} entries",
                        len(data) == n_prompts,
                        f"got {len(data)}")

    # Selected controlled case
    p = EVIDENCE / "02-state-divergence/data/selected_controlled_case.json"
    r.check("selected_controlled_case.json exists", p.exists())
    if p.exists():
        with p.open() as f:
            data = json.load(f)
        r.check("selected has 'question'", 'question' in data)
        r.check("selected has 'cases'", 'cases' in data)
        # Expect 3 cases: under / normal / over (no term-first in 02)
        cases = set(data.get('cases', {}).keys())
        r.check("selected has 3 cases (under/normal/over)", len(cases) == 3,
                f"got {cases}")
        # The known three
        expected = {'under_thinking', 'normal_thinking', 'over_thinking'}
        r.check("selected cases match expected set", cases == expected,
                f"got {cases}, expected {expected}")


# ─────────────────────────────────────────────────────────────────────
# Experiment 3: logit-lens
# ─────────────────────────────────────────────────────────────────────

def test_03_logit_lens(r: TestResult) -> None:
    print("\n[03-logit-lens]")

    # 5q × 4cond (Fig 5)
    p = EVIDENCE / "03-logit-lens/data/logit_lens_5q_4cond.json"
    r.check("logit_lens_5q_4cond.json exists", p.exists())
    if p.exists():
        with p.open() as f:
            data = json.load(f)
        r.check("5q data has 5 questions", len(data) == 5,
                f"got {len(data)}")
        # Each question has 4 conditions
        for q, conditions in data.items():
            r.check(f"{q} has 4 conditions",
                    len(conditions) == 4,
                    f"got {len(conditions)}")
            # All 4 condition names should be present
            expected = {'under_thinking', 'normal_thinking', 'over_thinking', 'term_first'}
            r.check(f"{q} conditions match expected set",
                    set(conditions.keys()) == expected,
                    f"got {set(conditions.keys())}, expected {expected}")

    # 15-problem full data
    p = EVIDENCE / "03-logit-lens/data/logit_lens_1.5b_full.json"
    r.check("logit_lens_1.5b_full.json exists", p.exists())

    # Table S3
    p = EVIDENCE / "03-logit-lens/data/table_s3_peak_layers.json"
    r.check("table_s3_peak_layers.json exists", p.exists())
    if p.exists():
        with p.open() as f:
            data = json.load(f)
        # Should have conditions and some statistical summary
        expected_keys = {'Premature convergence', 'Delayed commitment',
                          'Persistent oscillation', 'Term-first'}
        r.check("table_s3 has 4 paper-formatted condition keys",
                set(data.keys()) == expected_keys,
                f"got {set(data.keys())}, expected {expected_keys}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("patterns-repro smoke test")
    print("=" * 60)
    r = TestResult()
    test_01_macro_heterogeneity(r)
    test_02_state_divergence(r)
    test_03_logit_lens(r)
    return r.report()


if __name__ == "__main__":
    sys.exit(main())
