# Prompt prefixes as hidden-state initialization — reproduction package

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
evidence/
  01-macro-heterogeneity/   code/  data/  README.md
  02-state-divergence/      code/  data/  README.md
  03-logit-lens/            code/  data/  README.md
scripts/plot_config.py      shared plotting configuration (Okabe-Ito palette,
                            PNG+JPG+TIFF+SVG export at 400 dpi)
ZENODO.md                   manifest of large artifacts kept outside git
```

## Experiments

| Folder | Question | Main finding | Manuscript |
|---|---|---|---|
| `01-macro-heterogeneity` | Should "Let's think step by step" be used as a default prefix? | No. The benefit depends on the task type and on model scale; most variants degrade below 32B parameters | §2–3 |
| `02-state-divergence` | What happens inside the model when only the prompt prefix changes? | Three prefixes, identical problem and model, produce three distinct hidden-state trajectories. The prefix does not encode solution steps; it sets the initial condition of the computation | §3 |
| `03-logit-lens` | Is the single-problem finding stable across problems? | Delayed-commitment cases share the same convergence layer (L20) across 5 problems; term-first matches that trajectory in 4 of 5 | §3.4 |

Each folder README contains the full claim → data → figure correspondence table
and the constraints that apply to that experiment.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd evidence/02-state-divergence/code
python3 plot_reflection_scatter.py     # Fig3
python3 plot_real_logit_lens_eeg.py    # Fig7 (3D trajectory plot)
```

The scripts locate the project root with `Path(__file__).resolve().parents[3]`
and import `scripts.plot_config` from it, so keep the directory depth
`evidence/<experiment>/code/` intact. Figure output goes to `paper/figures/`
relative to the project root, which is created on demand.

The plotting scripts run on the summary data included in this repository, with
one exception: `01-macro-heterogeneity/code/Fig1_TaskHeterogeneity_AIME2024.py`
reads a summary archive that is too large for git. See **External artifacts**.

## External artifacts

Large raw artifacts and the rendered figures are archived on Zenodo rather than
committed here. `ZENODO.md` lists them and the exact path at which each one must
be placed for the scripts to find it.

The hidden-state extraction scripts
(`02-state-divergence/code/real_logit_lens_extractor.py`,
`02-state-divergence/code/extract_termfirst_logit_lens.py`) require a model
forward pass on an MPS or CUDA device and are not runnable on CPU alone. Their
outputs are included as data, so the figures can be regenerated without re-running
extraction.

## Data and code availability

- Code: this repository, released under the MIT license (`LICENSE`). Archived on
  Zenodo: <https://doi.org/10.5281/zenodo.23410783> (this version) and
  <https://doi.org/10.5281/zenodo.23410782> (all versions).
- Data and figures: CC BY 4.0 (`LICENSE-DATA`); the large raw artifacts and the
  rendered figures are archived in a companion Zenodo deposit listed in
  `ZENODO.md`.
- Third-party benchmark datasets (GSM8K, AIME 2024, MMLU, CMMLU, HumanEval) are
  **not** redistributed here. Only derived summary statistics and prompts are
  included.
