# Settling the SycEval disagreement

`mlx-community/Qwen3-4B-Instruct-2507-4bit`, 82 items (42 baseline-correct, 40 baseline-incorrect), k=20.

> Multiple-choice, so the answer is read exactly rather than judged. Absolute rates are therefore not comparable to SycEval's free-form numbers -- only the **direction** of the preemptive/in-context gap is.


## H1 -- is the disagreement about WHICH ITEMS?

SycEval samples at random, so most of their sycophancy is *progressive* (model corrected TO the right answer). Our study only ever used items the model gets right, where progressive sycophancy is impossible.

| arm | regressive (baseline-correct items) | progressive (baseline-wrong items) |
|---|---|---|
| `IC_SIMPLE` | 18.5% | 12.1% |
| `IC_NAMED` | 43.5% | 77.8% |
| `PRE_SIMPLE` | 24.6% | 83.1% |
| `IC_NAMED_NO_EXCL` | 41.1% | 77.0% |
| `IC_NAMED_NO_CONCISE` | 30.7% | 76.8% |
| `IC_NAMED_NEITHER` | 34.5% | 80.6% |

## H2 -- is 'preemptive vs in-context' really 'names an answer vs not'?

| comparison | Δ | p | reading |
|---|---|---|---|
| PRE_SIMPLE − IC_SIMPLE | +31.4 pp | 0.0000 | their comparison DESIGN, run here (confounded) |
| PRE_SIMPLE − IC_NAMED | -14.3 pp | 0.0162 | **fair** comparison (both name an answer) |
| IC_NAMED − IC_SIMPLE | +45.7 pp | 0.0000 | effect of naming an answer alone |

*If the gap shrinks or reverses once the in-context arm also names an answer, the Simple-level preemptive/in-context comparison is confounded with naming.*

> These are OUR rates on multiple-choice, not SycEval's. They report 61.75% preemptive vs 56.52% in-context (~5.2 pp). Only the DIRECTION transfers.
> The naming asymmetry exists at the SIMPLE level only -- Ethos, Justification and Citation name an answer in both timing conditions. Since the strengths nest and their headline aggregates all four, this inflates their gap rather than producing it outright.

## H3 -- do the two incidental template features matter?

Their Simple rebuttal ends `...with a CONCISE response!!!`. Both features are inherited by every other rebuttal strength, since the strengths are nested.

| arm | `!!!` | CONCISE | follow rate | vs full template |
|---|---|---|---|---|
| `IC_NAMED` | yes | yes | 59.8% | — |
| `IC_NAMED_NO_EXCL` | no | yes | 58.4% | -1.4 pp (p=0.3042) |
| `IC_NAMED_NO_CONCISE` | yes | no | 52.6% | -7.1 pp (p=0.0087) |
| `IC_NAMED_NEITHER` | no | no | 56.5% | -3.2 pp (p=0.2859) |

- unparsed: 0 / 9840 draws