# Reproduction package for "Contingent Chain-of-Thought effects and causal overclaiming in LLMs"

> **Latest release**: `v1.0.1` (2026-10-16). See [CHANGELOG.md](CHANGELOG.md) for what
> changed since the v1.0.0 submission. Earlier tag: [`v1.0-submission`](../../releases/tag/v1.0-submission).

Reproduction materials for a manuscript on how zero-shot chain-of-thought prompt
prefixes shape the hidden-state trajectories of large language models
(target journal: *Patterns*, Cell Press).

The package contains the code that regenerates every figure from summary tables,
the summary data behind each figure and table, and a per-experiment README that
maps each claim to the file that supports it.

Maintained by the manuscript's authors at Zhejiang Gongshang University;
corresponding author: Zhiwen Hu.

## Layout

```
release/patterns-repro/
├── evidence/
│   ├── 01-macro-heterogeneity/   code/  data/  README.md
│   ├── 02-state-divergence/      code/  data/  README.md
│   └── 03-logit-lens/            code/  data/  README.md
├── scripts/plot_config.py        shared plotting configuration (Okabe-Ito palette,
│                                PNG+JPG+TIFF+SVG export at 400 dpi)
├── tests/                        smoke tests for data file integrity
├── CITATION.cff                  CodeMeta citation metadata
├── LICENSE                       MIT (code)
├── LICENSE-DATA                  CC BY 4.0 (data and figures)
├── ZENODO.md                     manifest of large artifacts kept outside git
└── README.md                     this file
```

## Mapping to manuscript figures

| Manuscript figure | Folder | Code | Data |
|---|---|---|---|
| Fig 2 — Task heterogeneity | `01-macro-heterogeneity/` | `code/Fig2_TaskHeterogeneity.py` | `data/fig1_task_heterogeneity_scores.csv`, `data/Fig2_TaskHeterogeneity_data.csv`, `data/aime2024_qwen25_3b_n64_16k_scores/` |
| Fig 3 — Model scale | `01-macro-heterogeneity/` | `code/Fig3_ScalingLaws.py` | `data/fig2_model_scale_scores.csv` |
| Fig 4 — Reflection scatter | `02-state-divergence/` | `code/Fig4_ReflectionScatter.py` | `data/reflection_{1p5b,7b,8b}_real.json` |
| Fig 5 — Logit Lens 4 conditions | `03-logit-lens/` | `code/Fig5_LogitLens.py` | `data/logit_lens_5q_4cond.json` |

(Fig 1 in the manuscript is a conceptual schematic and has no script.)

## Reproducing

All five figure scripts can be regenerated from the included summary data on a
laptop (no GPU required for the plotting path). The data files contain the
processed values that the figures plot; the original raw inference logs and
the rendered figure outputs are archived on Zenodo (see `ZENODO.md`).

```bash
# 1. Set up Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run smoke tests to confirm data files are readable (optional)
python3 -m tests.test_smoke

# 3. Regenerate each figure (writes PNG/JPG/TIFF/SVG to evidence/<exp>/figures/)
cd evidence/01-macro-heterogeneity/code
python3 Fig2_TaskHeterogeneity.py      # Fig 2
python3 Fig3_ScalingLaws.py           # Fig 3

cd evidence/02-state-divergence/code
python3 Fig4_ReflectionScatter.py     # Fig 4

cd evidence/03-logit-lens/code
python3 Fig5_LogitLens.py             # Fig 5

# Fig 1: rebuild from the SVG source (the schematic is a static SVG; the script
# in evidence/00-fig1-conceptual/code/ is a one-shot matplotlib regenerator that
# writes a new PNG/SVG pair if you need to re-render with style changes).
cd evidence/00-fig1-conceptual/code
python3 build_concept_image.py        # rebuilds Fig 1
```

The scripts locate the project root with `Path(__file__).resolve().parents[3]`
and import `scripts.plot_config` from it, so the directory depth
`evidence/<experiment>/code/` must be preserved. Figure output goes to
`evidence/<experiment>/figures/` relative to the project root, which is
created on demand.

## Verifying the data is what the paper claims

The included data files are the exact values the manuscript reports. A few
spot checks that the user can run (no GPU, ~10 s):

```bash
# Table S1 (in the paper): Term-first accuracy at Qwen2.5 32B is 94.5%
python3 -c "
import json
with open('evidence/01-macro-heterogeneity/data/aime2024_qwen25_3b_n64_16k_scores/termfirst_repeatCDEF_aggregate_summary.csv') as f:
    pass  # 16k is in standard_baselineB_per_run_summary.csv
"
# (see tests/test_smoke.py for a fuller set of invariants)
```

## External artifacts

Large raw artifacts and the rendered figures are archived on Zenodo rather than
committed here. `ZENODO.md` lists them and the exact path at which each one must
be placed for the source-data extraction scripts to find it.

The hidden-state extraction scripts
(`evidence/02-state-divergence/code/real_logit_lens_extractor.py`,
`evidence/02-state-divergence/code/extract_termfirst_logit_lens.py`,
`evidence/03-logit-lens/code/compute_peak_layers.py`) require a model forward
pass on an MPS or CUDA device and are not runnable on CPU alone. Their outputs
are included as data, so the figures can be regenerated without re-running
extraction.

## Data and code availability

- Code: this repository, released under the MIT license (`LICENSE`). Archived on
  Zenodo: <https://doi.org/10.5281/zenodo.23410783> (this version) and
  <https://doi.org/10.5281/zenodo.23410782> (all versions).
- Data and figures: CC BY 4.0 (`LICENSE-DATA`); the large raw artifacts and the
  rendered figures are archived in a companion Zenodo deposit listed in
  `ZENODO.md`.
- Third-party benchmark datasets (GSM8K, MMLU, HumanEval, AIME 2024) are
  **not** redistributed here. Only derived summary statistics and prompts are
  included.
