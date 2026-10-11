#!/usr/bin/env python3
"""Aggregate repeated AIME result JSON files into pass@1-style statistics.

Usage example:
  python3 code/analysis/compute_aime_pass1_from_repeats.py \
    --group baseline='tmp/repeats/baseline/run*/results/aime2024*.json' \
    --group boundary_wait_0p6='tmp/repeats/boundary_wait_0p6/run*/results/aime2024*.json' \
    --output-json docs/.../AIME2024_PASS1_REPEAT_AGG_2026-04-13.json \
    --output-md docs/.../AIME2024_PASS1_REPEAT_AGG_2026-04-13.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute repeated-run pass@1-style aggregates for AIME result JSON files."
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Group spec: label=glob_pattern",
    )
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    parser.add_argument("--output-md", help="Optional markdown summary path.")
    return parser.parse_args()


def parse_group_specs(raw_specs: list[str]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --group spec: {spec!r}. Expected label=glob_pattern")
        label, pattern = spec.split("=", 1)
        files = sorted(Path(p) for p in glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matched group {label!r}: {pattern}")
        groups[label] = files
    if not groups:
        raise ValueError("No --group provided.")
    return groups


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected result format in {path}")
    return payload


def detail_correct(detail_item: dict[str, Any]) -> bool | None:
    correct = detail_item.get("correct")
    if isinstance(correct, list) and correct:
        return bool(correct[0])
    if isinstance(correct, bool):
        return correct
    return None


def detail_key(detail_item: dict[str, Any], idx: int) -> str:
    example_abbr = detail_item.get("example_abbr")
    if example_abbr:
        return str(example_abbr)
    return f"idx_{idx}"


def run_accuracy_from_details(details: list[dict[str, Any]]) -> float:
    vals = [detail_correct(d) for d in details]
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0
    return 100.0 * sum(vals) / len(vals)


def confidence_interval_95(avg: float, std: float, n: int) -> tuple[float, float]:
    if n <= 1:
        return (avg, avg)
    half = 1.96 * std / math.sqrt(n)
    return (avg - half, avg + half)


def aggregate_group(label: str, files: list[Path]) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    question_hits: dict[str, list[int]] = {}

    for run_id, path in enumerate(files):
        payload = load_result(path)
        details = payload.get("details", [])
        if not isinstance(details, list):
            details = []

        if details:
            run_acc = run_accuracy_from_details(details)
            for idx, detail in enumerate(details):
                if not isinstance(detail, dict):
                    continue
                key = detail_key(detail, idx)
                hit = 1 if detail_correct(detail) else 0
                question_hits.setdefault(key, []).append(hit)
        else:
            run_acc = float(payload.get("accuracy", 0.0))

        run_rows.append(
            {
                "run_id": run_id,
                "path": str(path),
                "accuracy": round(run_acc, 4),
            }
        )

    run_accs = [r["accuracy"] for r in run_rows]
    mean_acc = mean(run_accs) if run_accs else 0.0
    std_acc = pstdev(run_accs) if len(run_accs) > 1 else 0.0
    ci_low, ci_high = confidence_interval_95(mean_acc, std_acc, len(run_accs))

    question_pass_rates: dict[str, float] = {}
    for key, hits in question_hits.items():
        if hits:
            question_pass_rates[key] = sum(hits) / len(hits)

    dataset_pass1 = 100.0 * mean(question_pass_rates.values()) if question_pass_rates else mean_acc

    return {
        "label": label,
        "num_runs": len(run_rows),
        "run_accuracies": run_rows,
        "summary": {
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "min_accuracy": round(min(run_accs), 4) if run_accs else 0.0,
            "max_accuracy": round(max(run_accs), 4) if run_accs else 0.0,
            "ci95": [round(ci_low, 4), round(ci_high, 4)],
            "dataset_pass_at_1_from_repeats": round(dataset_pass1, 4),
            "num_questions": len(question_pass_rates),
        },
        "question_pass_rates": question_pass_rates,
    }


def render_md(aggregates: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# AIME Repeated-Run pass@1 Aggregate")
    lines.append("")
    lines.append("`dataset_pass_at_1_from_repeats` is computed as per-question mean hit rate across repeated runs.")
    lines.append("")
    for label, payload in aggregates.items():
        s = payload["summary"]
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- runs: `{payload['num_runs']}`")
        lines.append(f"- mean accuracy: `{s['mean_accuracy']:.4f}%`")
        lines.append(f"- std accuracy: `{s['std_accuracy']:.4f}`")
        lines.append(f"- min/max accuracy: `{s['min_accuracy']:.4f}% / {s['max_accuracy']:.4f}%`")
        lines.append(f"- 95% CI (mean): `[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]`")
        lines.append(f"- dataset pass@1 from repeats: `{s['dataset_pass_at_1_from_repeats']:.4f}%`")
        lines.append(f"- questions counted: `{s['num_questions']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    groups = parse_group_specs(args.group)

    aggregates: dict[str, Any] = {}
    for label, files in groups.items():
        aggregates[label] = aggregate_group(label, files)

    output = {
        "groups": aggregates,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_md(aggregates), encoding="utf-8")
        print(f"Wrote MD: {output_md}")


if __name__ == "__main__":
    main()
