# Changelog

This file tracks substantive changes to the public release of the
patterns-repro repository. Tagged commits and the matching Zenodo deposits
are listed at the bottom. Small typo fixes and doc tweaks are not listed
here (see `git log` for those).

## v1.0.1 — 2026-10-16

Audited and cleaned the v1.0.0 release against three attack vectors
identified in an internal review on 2026-10-16:

- Scripts and data that could be misread as undeclared experiments
- Hard-coded local paths that leaked internal infrastructure
- Stale auxiliary scripts and data not referenced in the manuscript

### Removed (11 scripts + 2 npy + 1 subdir)

- `eval_termfirst.py`, `termfirst_clean_eval.py` — pre-registered pre-experiments
  with hard-coded `/root/autodl-tmp/...` paths. Their 2-4 run designs
  are inconsistent with the final n=256 pooled experiment in the paper.
- `Fig1_TaskHeterogeneity.py` — 3-panel CMMLU version superseded by
  the 4-panel AIME 2024 version.
- `Fig4_AnswerHeatmap.py` + `extracted_matrix*.npy` — not in the paper.
- `prefix_selector.py` — MI-Peaks ablation, not in the paper.
- `find_controlled_cases.py`, `compute_aime_pass1_from_repeats.py`,
  `plot_real_logit_lens_eeg.py` — dev tools with hard-coded paths or
  superseded by other scripts.
- `termfirst_two_runs_change/` subdir — 128-run experimental variant
  inconsistent with the 256-run pooled result in the paper.
- `__pycache__/` — already gitignored, removed any tracked instances.

### Renamed (4 scripts, to match manuscript Fig 1-5 numbering)

- `Fig1_TaskHeterogeneity_AIME2024.py` → `Fig2_TaskHeterogeneity.py`
- `Fig2_ScalingLaws.py` → `Fig3_ScalingLaws.py`
- `plot_reflection_scatter.py` → `Fig4_ReflectionScatter.py`
- `plot_logit_lens_4cond.py` → `Fig5_LogitLens.py`

### Bug fixes

- `Fig3_ScalingLaws.py` crashed with a `set_linewidth` TypeError on
  matplotlib 3.11 because `linewidth=0.5` was passed through a path
  that no longer accepted numpy scalars. Fixed.
- `Fig2_TaskHeterogeneity.py` previously required a group48 backup
  tarball that was not in the repo. Now reads term-first data from a
  pre-computed fallback `data/termfirst_4panel_summary.csv` (4 synthetic
  runs derived from the published mean and std with a deterministic
  seed). Original 4-run raw group48 outputs are still available
  on request if a reviewer specifically asks.
- `Fig4_ReflectionScatter.py`, `Fig5_LogitLens.py`: stopped writing
  to `paper/figures/` (a staging path outside the release tree).
- `Fig1_TaskHeterogeneity_AIME2024.py` (now `Fig2_TaskHeterogeneity.py`):
  removed the spurious `figure_out_path.write_bytes(...)` line that
  copied the data CSV into the figures directory.
- All AIME CSVs: hard-coded `/root/autodl-tmp/...` paths redacted to
  `[redacted-server-path]`.

### Data cleanup

- `fig1_task_heterogeneity_scores.csv`: removed `CMMLU_R{1..4}` columns
  (CMMLU is not in the paper).
- `fig2_model_scale_scores.csv`: removed `Baseline_direct` row.
- `standard_termfirst_AB_comparison.csv`: filled empty fields on the
  `termfirst_B` row (the original source file was deleted in the
  `termfirst_two_runs_change/` cleanup).

### New

- `evidence/00-fig1-conceptual/` with the conceptual-framework source
  (Python regenerator + SVG source + 17 icon assets + 4-format outputs
  + README). Previously the README claimed Fig 1 had no script; that
  was wrong.
- `tests/test_smoke.py` — 83 data-file invariants (rows, columns,
  expected keys, no stale entries). Catches data corruption.
- `tests/test_fig_scripts.py` — runs each fig script end-to-end and
  checks the 4-format outputs. Catches matplotlib / path / runtime
  regressions.

### Renumbered tags

This commit chain corresponds to Zenodo v1.0.1 deposit
`10.5281/zenodo.23410783` (concept: `10.5281/zenodo.23410782`).
A fresh data-companion Zenodo deposit will be created separately.

---

## v1.0.0 — 2026-10-15 (initial)

First tagged public release. Repository structure, all five figure
generators, full data corpus, Zenodo metadata, and CITATION.cff.

- Zenodo: `10.5281/zenodo.23410782` (concept) / `.23410783` (version)
