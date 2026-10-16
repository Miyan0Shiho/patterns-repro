#!/usr/bin/env python3
"""Compute Table S3: first-token peak-layer distribution and token-conditional
cross-condition agreement for 15 AIME 2024 problems
(DeepSeek-R1-Distill-Qwen-1.5B).

Data source: evidence/03-logit-lens/data/logit_lens_1.5b_full.json
(The raw 15-problem logit-lens extraction is the data file itself; no external script needed for table generation.)

Output: evidence/03-logit-lens/data/table_s3_peak_layers.json
Mirrors the values reported in paper Table S3.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # evidence/03-logit-lens/
DATA = ROOT / "data" / "logit_lens_1.5b_full.json"
OUT = ROOT / "data" / "table_s3_peak_layers.json"

CONDITIONS = {
    "Premature convergence": "under_thinking",
    "Delayed commitment": "normal_thinking",
    "Persistent oscillation": "over_thinking",
    "Term-first": "term_first",
}


def peak_layer(cond_key, q):
    """Layer (0-indexed) at which the first-token probability peaks."""
    lps = q[cond_key]["first_token_layer_probs"]
    return int(np.argmax(lps))


def first_token(cond_key, q):
    """Decoded identity of the sampled first generated token."""
    return str(q[cond_key]["first_gen_token"])


def main():
    data = json.load(open(DATA))
    res = data["results"]

    rows = {}
    for label, key in CONDITIONS.items():
        peaks = [peak_layer(key, q) for q in res.values()]
        tokens = [first_token(key, q) for q in res.values()]
        dist = dict(sorted(Counter(peaks).items()))
        token_counts = dict(sorted(Counter(tokens).items()))
        token_peak_distribution = {}
        for token in sorted(set(tokens)):
            token_peaks = [
                peak_layer(key, q)
                for q in res.values()
                if first_token(key, q) == token
            ]
            token_peak_distribution[token] = dict(
                sorted(Counter(token_peaks).items())
            )
        correct = sum(1 for q in res.values() if q[key].get("correct"))
        rows[label] = {
            "first_token_counts": token_counts,
            "peak_layer_by_first_token": token_peak_distribution,
            "peak_layer_distribution": dist,
            "accuracy_n_correct": correct,
            "accuracy_pct": round(correct / len(res) * 100, 1),
            "n_problems": len(res),
        }

    tf_peaks = [peak_layer("term_first", q) for q in res.values()]
    tf_tokens = [first_token("term_first", q) for q in res.values()]
    for label, key in CONDITIONS.items():
        if label == "Term-first":
            continue
        peaks = [peak_layer(key, q) for q in res.values()]
        tokens = [first_token(key, q) for q in res.values()]
        agree = sum(1 for a, b in zip(peaks, tf_peaks) if a == b)
        same_token = [
            (a, b)
            for a, b, x, y in zip(peaks, tf_peaks, tokens, tf_tokens)
            if x == y
        ]
        different_token = [
            (a, b)
            for a, b, x, y in zip(peaks, tf_peaks, tokens, tf_tokens)
            if x != y
        ]
        rows[label]["agreement_with_termfirst"] = {
            "n_agree": agree,
            "n_total": len(res),
            "pct": round(agree / len(res) * 100, 1),
            "same_first_token": {
                "n_agree": sum(a == b for a, b in same_token),
                "n_total": len(same_token),
            },
            "different_first_token": {
                "n_agree": sum(a == b for a, b in different_token),
                "n_total": len(different_token),
            },
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"saved -> {OUT}")
    for label, r in rows.items():
        agree = r.get("agreement_with_termfirst", {})
        if agree:
            same = agree["same_first_token"]
            diff = agree["different_first_token"]
            a = (
                f", agree={agree['n_agree']}/{agree['n_total']} "
                f"({agree['pct']}%; same={same['n_agree']}/{same['n_total']}, "
                f"different={diff['n_agree']}/{diff['n_total']})"
            )
        else:
            a = ""
        print(
            f"  {label:24s} tokens={r['first_token_counts']} "
            f"dist={r['peak_layer_distribution']} acc={r['accuracy_pct']}%{a}"
        )


if __name__ == "__main__":
    main()
