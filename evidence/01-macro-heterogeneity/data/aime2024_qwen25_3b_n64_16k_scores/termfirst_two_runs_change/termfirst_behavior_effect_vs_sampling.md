# Term-first Behavior Effect vs Sampling Variation

Scope:

- Standard A: original Standard outputs from `aime2024_qwen25_3b_standard_vs_termfirst_n64_16k_20260601_211000`
- Term-first A: original old-config term-first outputs from the same run batch
- Term-first B: old-config term-first rerun `aime2024_qwen25_3b_termfirst_oldconfig_rerun_n64_16k_20260602_155734`

Term-first system prompt:

`What is the query? Term first, then reasoning.`

## Correct Reading

The useful question is not whether the output literally prints `term-first`.

The useful question is whether the system prompt changes the model's behavior relative to baseline in a way that is larger than ordinary sampling variation.

## Corpus-Level Metrics

| group | predictions | correct | accuracy | boxed | mean chars | first-sentence domain cue | first-sentence target cue |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard A | 1920 | 125 | 6.5104% | 1855 | 3435.5 | 1632 (85.00%) | 955 (49.74%) |
| Term-first A | 1920 | 111 | 5.7813% | 1888 | 2865.3 | 1541 (80.26%) | 1116 (58.13%) |
| Term-first B | 1920 | 107 | 5.5729% | 1882 | 2948.2 | 1528 (79.58%) | 1122 (58.44%) |

Interpretation:

- Term-first outputs are systematically shorter than Standard.
- Term-first has slightly more target-oriented openings (`find/determine/calculate`) but fewer first-sentence domain cues than Standard.
- This does not look like a clean shift toward naming the mathematical term first; it looks like a mild style/trajectory perturbation.

## Pairwise Comparisons

| comparison | same final answer | same correctness | both correct | left only | right only | both wrong | net right-left |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard A vs Term-first A | 214/1920 (11.15%) | 1792/1920 (93.33%) | 54 | 71 | 57 | 1738 | -14 |
| Term-first A vs Term-first B | 203/1920 (10.57%) | 1790/1920 (93.23%) | 44 | 67 | 63 | 1746 | -4 |
| Standard A vs Term-first B | 178/1920 (9.27%) | 1778/1920 (92.60%) | 45 | 80 | 62 | 1733 | -18 |

Interpretation:

- Standard-vs-term answer changes are only slightly larger than term-vs-term resampling changes.
- The term-first prompt does change outputs, but the effect is not clearly separable from ordinary stochastic variation at this model/dataset difficulty level.
- Correctness changes are mostly flips among rare correct samples, not a stable coverage expansion.

## Answer Distribution Distance

For each AIME problem, answer distributions over 64 samples were compared with total variation distance.

| comparison | mean TV | median TV | same top answer problems | max TV |
|---|---:|---:|---:|---:|
| Standard A vs Term-first A | 0.4547 | 0.5078 | 16/30 | 0.7656 |
| Term-first A vs Term-first B | 0.4286 | 0.3906 | 18/30 | 0.7813 |
| Standard A vs Term-first B | 0.4443 | 0.4844 | 17/30 | 0.7969 |

Interpretation:

The distributional shift from Standard to term-first is close in magnitude to the shift between two independent term-first runs. This is evidence for a weak prompt effect relative to sampling noise, not for a strong controlled mechanism.

## Current Conclusion

The old term-first prompt is present in the system message and does perturb generation: outputs get shorter, answer distributions shift, and target-oriented openings increase.

However, the effect is weak and not aligned with the intended mechanism. It does not reliably make the model identify the mathematical term/concept first, and it does not improve AIME2024 accuracy. On Qwen2.5-3B-Instruct AIME2024, the old term-first prompt is best described as a weak stochastic trajectory perturbation rather than an effective term-first reasoning intervention.
