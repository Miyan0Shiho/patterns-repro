# Large artifacts archived on Zenodo (data companion deposit)

Everything listed here is **excluded** from git. The repository stays small
and every script that needs one of these files still runs, provided the file
is placed at the path given below.

**This ZENODO.md corresponds to the *data companion* deposit** (separate from the
code deposit, whose DOIs are in `CITATION.cff` and the main README).

| Artifact | Expected path (relative to repository root) | Size | Needed for |
|---|---|---|---|
| Raw OpenCompass inference logs (2,912 files) | `evidence/01-macro-heterogeneity/data/data_fig4/` | 289 MB | raw traces behind the answer-fragility heatmap (Fig 4) |
| Packaged AIME 2024 run data (4×64 runs pooled) | `evidence/01-macro-heterogeneity/data/aime2024_qwen25_3b_n64_16k_scores/packages/` | 55 MB | raw AIME runs behind the term-first comparison (Fig 2 AIME panel) |
| Rendered figures, four formats | `evidence/<exp>/figures/` (released on GitHub, mirrored here for archival) | 26 MB | publication figures (PNG/JPG/TIFF/SVG, 400 dpi) |

Total ≈ 370 MB (data companion). The code release at
<https://github.com/Miyan0Shiho/patterns-repro> carries the same figures
plus all summary data and the regenerating scripts.

## Note on the v1.0.0 group48 backup archive

The v1.0.0 README referenced a 104 MB `qwen25_3b_group48_full4_n4_20260528.tar.gz`
archive that was used by the v4.5-era Fig 1 script. As of the v1.0.1 release,
Fig 2 reads term-first data from a pre-computed fallback
(`data/termfirst_4panel_summary.csv`, 4 synthetic runs derived from the
published mean and std with a deterministic seed). The tarball is therefore
no longer required to regenerate the figures. If a reviewer specifically
requests the underlying 4-run raw group48 outputs, the original archive can
be re-uploaded to a separate deposit; otherwise, the v1.0.1 release stands
without it.

## Archiving procedure

1. Create a Zenodo deposit and reserve a DOI.
2. Upload the three artifacts listed above (the term-first group48 archive
   is no longer needed; the figures in the third row are optional archival).
3. During peer review, keep the deposit **restricted** and hand the private
   review link to the editor, or publish it under an embargo that lifts on
   acceptance.
4. Record the DOI in `09_DATA_AND_CODE_AVAILABILITY.md` and in the manuscript's
   `Resource availability` → `Data and code availability` section.
5. Tag the frozen release (`v1.0.1-submission`) in git and link the Zenodo
   record to that tag. Do not rewrite repository history afterwards.
