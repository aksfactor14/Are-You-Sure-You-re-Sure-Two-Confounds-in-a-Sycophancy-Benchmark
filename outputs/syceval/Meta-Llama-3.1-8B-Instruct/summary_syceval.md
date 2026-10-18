# SycEval — results

Implementation of arXiv:2502.08177 on local models.

| role | model |
|---|---|
| subject_model | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| rebuttal_model | `Qwen/Qwen2.5-7B-Instruct` |
| judge_model | `Qwen/Qwen2.5-7B-Instruct` |
| subject_temp | `0.7` |
| seed | `0` |

2252 analysable cells over 191 questions, 3 contexts x 4 strengths.

## 1. Sycophancy rates

Overall = progressive + regressive. Erroneous responses on either side are dropped, not counted as non-sycophantic.

| slice | overall | progressive | regressive | n |
|---|---|---|---|---|
| **all** | 70.6% [68.7%, 72.4%]  n=2252 | 32.86% | 37.74% | 2252 |
| dataset: `amps` | 64.7% [61.9%, 67.3%]  n=1180 | 45.00% | 19.66% | 1180 |
| dataset: `medquad` | 77.1% [74.5%, 79.6%]  n=1072 | 19.50% | 57.65% | 1072 |
| context: `in_context` | 57.5% [53.9%, 61.0%]  n=744 | 22.98% | 34.54% | 744 |
| context: `in_context_named` | 84.2% [81.5%, 86.7%]  n=755 | 43.31% | 40.93% | 755 |
| context: `preemptive` | 69.9% [66.5%, 73.0%]  n=753 | 32.14% | 37.72% | 753 |
| strength: `simple` | 58.9% [54.8%, 62.9%]  n=565 | 25.84% | 33.10% | 565 |
| strength: `ethos` | 61.6% [57.5%, 65.6%]  n=558 | 25.81% | 35.84% | 558 |
| strength: `justification` | 78.4% [74.9%, 81.6%]  n=566 | 38.16% | 40.28% | 566 |
| strength: `citation` | 83.3% [80.0%, 86.2%]  n=563 | 41.56% | 41.74% | 563 |

*Paper reference (3 frontier models, 500 items/dataset): overall 58.19%, progressive 43.52%, regressive 14.66%.*

## 2. Preemptive vs in-context

**Pooled over responses — the paper's test.**

| context | overall | progressive | regressive | n |
|---|---|---|---|---|
| preemptive | 69.9% [66.5%, 73.0%]  n=753 | 32.14% | 37.72% | 753 |
| in_context | 57.5% [53.9%, 61.0%]  n=744 | 22.98% | 34.54% | 744 |

Two-proportion z = +4.96, p = 7.1e-07

**Per-item paired — this repo's test.**

- mean per-question difference (preemptive − in_context): **+0.124**
- paired Wilcoxon p = 4.8e-06 over **190 questions** (higher preemptive on 91, lower on 43, tied on 56)

*Paper: preemptive 61.75% vs in-context 56.52% on AMPS, Z = 5.87, p < 0.001; no significant difference on MedQuAD.*

## 2b. Naming vs timing — the confound, in free form

SycEval's in-context Simple rebuttal names no answer; their preemptive one does. `in_context_named` adds the naming to the in-context arm and changes nothing else, so the two effects separate.

| contrast | isolates | mean Δ (per item) | Wilcoxon p | n |
|---|---|---|---|---|
| preemptive − in_context | their design — both vary | **+0.124** | 4.8e-06 | 190 |
| in_context_named − in_context | **naming alone** | **+0.268** | 2.5e-21 | 190 |
| preemptive − in_context_named | **timing alone** | **-0.144** | 1.1e-07 | 190 |

**Per strength — the naming contrast should be largest at `simple` and decay as the ladder converges.**

| strength | naming Δ (named − unnamed) | p | n |
|---|---|---|---|
| `simple` | +0.495 | 9.4e-18 | 186 |
| `ethos` | +0.475 | 8.3e-16 | 181 |
| `justification` | +0.059 | 0.1818 | 186 |
| `citation` | +0.043 | 0.3069 | 185 |

_Above `simple` all three contexts name the answer — the justification clause is hard-coded to include it and the citation is generated about it, exactly as in the paper. The contrast is therefore read at `simple`; its decay across strengths is the mechanism by which aggregating over the nested ladder dilutes the confound._

**Reading.** Naming is worth +0.268 and timing -0.144; the paper's confounded contrast is +0.124.
The naming effect is LARGER than the timing effect, so the paper's Simple-level comparison is measuring naming more than timing.
WARNING: **The timing effect REVERSES sign once naming is held constant** — as it did on both models in the multiple-choice replication (−33.1 pp Llama, −14.3 pp Qwen2.5).

*Multiple-choice replication, for comparison: naming +48.9 pp (Llama) / +45.7 pp (Qwen2.5); timing −33.1 pp / −14.3 pp; confounded contrast +15.8 pp / +31.4 pp.*

## 3. Rebuttal strength

| strength | overall | progressive | regressive | n |
|---|---|---|---|---|
| `simple` | 58.94% | 25.84% | 33.10% | 565 |
| `ethos` | 61.65% | 25.81% | 35.84% | 558 |
| `justification` | 78.45% | 38.16% | 40.28% | 566 |
| `citation` | 83.30% | 41.56% | 41.74% | 563 |

Chi-square, strength × outcome: χ² = 125.97, dof = 6, p = 9.1e-25

**Per-item contrasts vs `simple`, Holm-corrected across the three.**

| strength vs `simple` | Δ per question | raw p | Holm p | items |
|---|---|---|---|---|
| `ethos` | +0.030 | 0.1767 | 0.1767 | 191 |
| `justification` | +0.196 | 8.9e-17 | 1.8e-16 | 190 |
| `citation` | +0.245 | 1.1e-18 | 3.3e-18 | 190 |

*Paper: simple rebuttals maximise progressive sycophancy (Z = 6.59, p < 0.001); citation rebuttals produce the most regressive and the least progressive.*

## 4. Persistence of sycophantic chains

A chain is one (question, context) walked across the four escalating strengths. Persistent = the sycophantic state changes **at most once**.

| slice | persistent | rate | binomial p vs 50% |
|---|---|---|---|
| **all** | 481/571 | 84.2% [81.0%, 87.0%] | 1.6e-65 |
| `in_context` | 159/191 | 83.2% [77.3%, 87.9%] | 1.9e-21 |
| `in_context_named` | 165/190 | 86.8% [81.3%, 90.9%] | 1.7e-26 |
| `preemptive` | 157/190 | 82.6% [76.6%, 87.4%] | 1.5e-20 |

*Paper: 78.5% overall, 95% CI [77.2%, 79.8%], p < 0.001; no significant difference across contexts, models or datasets.*

## 5. Instrument health

- analysable cells: **2252** of 2292 collected
  - dropped, erroneous either side: 40
- initial-answer classes: {'incorrect': 96, 'correct': 95, 'erroneous': 9}

**Judge accuracy — Beta(matches+1, mismatches+1)**

_No human audit yet._ Run `python experiments/syceval_judge.py --audit`, fill in `human_class`, and rerun this analysis. Without it the judge's error rate is unmeasured and every rate above inherits an unknown bias.

---

**Reading note.** Pooled proportions treat every response as independent; eight come from each question and questions differ greatly in movability. The per-item columns are the ones that support inference. Where they disagree, believe the per-item column — see README §13, failure #7.