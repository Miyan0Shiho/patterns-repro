"""End-to-end fig-script tests: actually run each Fig script and verify
the regenerated files are present with the correct 4-format output.

Complements tests/test_smoke.py: that one checks data file invariants
without running anything; this one runs the figure scripts and checks
the output files appear. Together they catch two failure modes:
- smoke test fails → data files broken (catch at load time)
- fig test fails → matplotlib / path / runtime issues (catch at draw time)

Each test is independent; failures in one script don't block the others.
Run with: python3 -m tests.test_fig_scripts
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# Mapping: (script_path, expected_output_basename, [4 formats])
FIG_SCRIPTS = [
    (REPO_ROOT / "evidence" / "00-fig1-conceptual" / "code" / "build_concept_image.py",
     "Fig1_ConceptualFramework",
     ["png", "jpg", "tiff", "svg"]),
    (REPO_ROOT / "evidence" / "01-macro-heterogeneity" / "code" / "Fig2_TaskHeterogeneity.py",
     "Fig2_TaskHeterogeneity",
     ["png", "jpg", "tiff", "svg"]),
    (REPO_ROOT / "evidence" / "01-macro-heterogeneity" / "code" / "Fig3_ScalingLaws.py",
     "Fig3_ScalingLaws",
     ["png", "jpg", "tiff", "svg"]),
    (REPO_ROOT / "evidence" / "02-state-divergence" / "code" / "Fig4_ReflectionScatter.py",
     "Fig4_ReflectionScatter",
     ["png", "jpg", "tiff", "svg"]),
    (REPO_ROOT / "evidence" / "03-logit-lens" / "code" / "Fig5_LogitLens.py",
     "Fig5_LogitLens",
     ["png", "jpg", "tiff", "svg"]),
]


def run_one(script: Path, basename: str, formats: list[str]) -> tuple[bool, str]:
    """Run a fig script. Returns (ok, summary_message).

    The script is expected to write outputs to its sibling figures/ dir.
    """
    if not script.exists():
        return False, f"script not found: {script}"

    figures_dir = script.parent.parent / "figures"
    if not figures_dir.exists():
        return False, f"figures dir missing: {figures_dir}"

    # Snapshot the figures dir to detect newly written files
    before = set(figures_dir.iterdir())

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout (>120s)"
    except Exception as e:
        return False, f"exception: {e}"
    elapsed = time.time() - start

    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-5:])
        return False, f"exit={result.returncode} in {elapsed:.1f}s; stderr-tail: {tail[:200]}"

    # Check that all 4 formats were written (or overwritten)
    expected = [figures_dir / f"{basename}.{fmt}" for fmt in formats]
    missing = [str(p.name) for p in expected if not p.exists()]

    if missing:
        return False, f"missing outputs after run: {missing}; stdout-tail: {result.stdout.splitlines()[-3:]}"

    # Sanity: file sizes > 1 KB (otherwise rendering failed silently)
    tiny = [p.name for p in expected if p.stat().st_size < 1024]
    if tiny:
        return False, f"output files suspiciously small: {tiny}"

    return True, f"OK in {elapsed:.1f}s; 4 formats x {len(expected)} files"


def main() -> int:
    print("=" * 70)
    print("End-to-end fig-script tests")
    print("=" * 70)
    print()

    failed = []
    for script, basename, formats in FIG_SCRIPTS:
        print(f"  {basename:40s}  ({script.relative_to(REPO_ROOT)})")
        ok, msg = run_one(script, basename, formats)
        marker = "PASS" if ok else "FAIL"
        print(f"    {marker}  {msg}")
        if not ok:
            failed.append(basename)
        print()

    print("=" * 70)
    if failed:
        print(f"FAIL: {len(failed)} of {len(FIG_SCRIPTS)} scripts failed:")
        for f in failed:
            print(f"  - {f}")
        return 1
    else:
        print(f"PASS: all {len(FIG_SCRIPTS)} fig scripts regenerated successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())
