# Large artifacts archived on Zenodo

Everything listed here is deliberately **excluded** from git. The repository
stays small and every script that needs one of these files still runs, provided
the file is placed at the path given below.

| Artifact | Expected path (relative to repository root) | Size | Needed for |
|---|---|---|---|
| Raw OpenCompass inference logs (2,912 files) | `evidence/01-macro-heterogeneity/data/data_fig4/` | 289 MB | raw traces behind the answer-fragility heatmap (Fig4) |
| term-first group48 summary archive | `archive/local-noise/local_ignored_dirs_20260603/backups/qwen25_3b_group48_full4_n4_20260528/qwen25_3b_group48_full4_n4_20260528.tar.gz` | 104 MB | **required to run** `Fig1_TaskHeterogeneity_AIME2024.py` |
| Packaged AIME 2024 run data | `evidence/01-macro-heterogeneity/data/aime2024_qwen25_3b_n64_16k_scores/packages/` | 55 MB | raw AIME runs behind the term-first comparison |
| Rendered figures, four formats | `evidence/<experiment>/figures/` | 26 MB | publication figures (PNG/JPG/TIFF/SVG, 400 dpi) |

Total ≈ 474 MB.

## Why the group48 archive matters

`Fig1_TaskHeterogeneity_AIME2024.py` resolves its input archive as

```
<repository root>/archive/local-noise/local_ignored_dirs_20260603/backups/
    qwen25_3b_group48_full4_n4_20260528/qwen25_3b_group48_full4_n4_20260528.tar.gz
```

so the file must be restored at exactly that path (or the path must be adjusted
in the script). The archive is read with `tarfile` and streamed directly; nothing
is extracted to disk.

## Archiving procedure

1. Create a Zenodo deposit and reserve a DOI.
2. Upload the four artifacts listed above.
3. During peer review, keep the deposit **restricted** and hand the private
   review link to the editor, or publish it under an embargo that lifts on
   acceptance.
4. Record the DOI in `CITATION.cff`, in the repository README, and in the
   manuscript's `Resource availability` → `Data and code availability` section.
5. Tag the frozen release (`v1.0-submission`) in git and link the Zenodo record
   to that tag. Do not rewrite repository history afterwards.
