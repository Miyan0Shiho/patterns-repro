# Term-first Output Protocol Check

Scope: two old-config term-first AIME2024 n=64 runs, total 128 runs and 3840 predictions.

System prompt:

`What is the query? Term first, then reasoning.`

## Summary

The term-first instruction did not visibly take effect as an output protocol.

- Total predictions checked: `3840`
- Strict term-first outputs: `0`
- Outputs containing `What is the query?`: `0`
- Outputs containing `term first`: `0`
- Outputs containing `query`: `1`
- Outputs containing `term` as a standalone word: `260`
- Outputs containing `concept`: `18`
- Outputs whose first 300 chars contain `query`, `term`, `concept`, `problem type`, or `key mathematical`: `24`
- Outputs starting with `To solve...`: `3095`
- Outputs with `\boxed{}`: `3770`

Most occurrences of `term` or `concept` are ordinary mathematical prose, not evidence that the model followed the instruction. The dominant output pattern is normal chain-of-thought style, such as `To solve this problem...` and `To solve the problem...`.

## Per Run Group

| group | predictions | starts with `To solve` | strict term-first | first-300 query/term/concept | contains query | contains `term first` | contains term | boxed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A original | 1920 | 1551 | 0 | 11 | 0 | 0 | 120 | 1888 |
| B rerun | 1920 | 1544 | 0 | 13 | 1 | 0 | 140 | 1882 |

## Top Output Prefixes

| count | first words |
|---:|---|
| 1462 | `To solve this problem,` |
| 853 | `To solve the problem,` |
| 288 | `To solve for the` |
| 276 | `To solve for \(` |
| 186 | `To find the number` |
| 161 | `To solve the given` |
| 155 | `To determine the number` |
| 124 | `To find the largest` |

## Interpretation

The old term-first prompt may still perturb sampling trajectories because it is present in the system message, but the output text does not show the intended behavior of naming a term or query first. The evidence supports a weak hidden prompt perturbation, not an executed term-first reasoning protocol.
