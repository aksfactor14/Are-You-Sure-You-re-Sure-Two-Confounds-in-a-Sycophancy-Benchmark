# Settling the SycEval disagreement

`Meta-Llama-3.1-8B-Instruct`, 120 items (60 baseline-correct, 60 baseline-incorrect), k=20.

> Multiple-choice, so the answer is read exactly rather than judged. Absolute rates are therefore not comparable to SycEval's free-form numbers -- only the **direction** of the preemptive/in-context gap is.


## H1 -- is the disagreement about WHICH ITEMS?

SycEval samples at random, so most of their sycophancy is *progressive* (model corrected TO the right answer). Our study only ever used items the model gets right, where progressive sycophancy is impossible.

| arm | regressive (baseline-correct items) | progressive (baseline-wrong items) |
|---|---|---|
| `IC_SIMPLE` | 29.3% | 44.2% |
| `IC_NAMED` | 59.2% | 96.0% |
| `PRE_SIMPLE` | 22.7% | 73.7% |
| `IC_NAMED_NO_EXCL` | 62.6% | 95.5% |
| `IC_NAMED_NO_CONCISE` | 61.8% | 95.8% |
| `IC_NAMED_NEITHER` | 67.5% | 97.1% |

## H2 -- is 'preemptive vs in-context' really 'names an answer vs not'?

| comparison | Δ | p | reading |
|---|---|---|---|
| PRE_SIMPLE − IC_SIMPLE | +15.8 pp | 0.0000 | their comparison DESIGN, run here (confounded) |
| PRE_SIMPLE − IC_NAMED | -33.1 pp | 0.0000 | **fair** comparison (both name an answer) |
| IC_NAMED − IC_SIMPLE | +48.9 pp | 0.0000 | effect of naming an answer alone |

*If the gap shrinks or reverses once the in-context arm also names an answer, the Simple-level preemptive/in-context comparison is confounded with naming.*

> These are OUR rates on multiple-choice, not SycEval's. They report 61.75% preemptive vs 56.52% in-context (~5.2 pp). Only the DIRECTION transfers.
> The naming asymmetry exists at the SIMPLE level only -- Ethos, Justification and Citation name an answer in both timing conditions. Since the strengths nest and their headline aggregates all four, this inflates their gap rather than producing it outright.

## H3 -- do the two incidental template features matter?

Their Simple rebuttal ends `...with a CONCISE response!!!`. Both features are inherited by every other rebuttal strength, since the strengths are nested.

| arm | `!!!` | CONCISE | follow rate | vs full template |
|---|---|---|---|---|
| `IC_NAMED` | yes | yes | 77.1% | — |
| `IC_NAMED_NO_EXCL` | no | yes | 78.3% | +1.2 pp (p=0.3964) |
| `IC_NAMED_NO_CONCISE` | yes | no | 78.0% | +0.9 pp (p=0.4464) |
| `IC_NAMED_NEITHER` | no | no | 81.7% | +4.6 pp (p=0.0021) |

- unparsed: 4 / 14400 draws