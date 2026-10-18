# SycEval — results

Implementation of arXiv:2502.08177 on local models.

| role | model |
|---|---|
| subject_model | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| rebuttal_model | `Qwen/Qwen2.5-7B-Instruct` |
| judge_model | `Qwen/Qwen2.5-7B-Instruct` |
| subject_temp | `0.7` |
| seed | `0` |

2261 analysable cells over 196 questions, 3 contexts x 4 strengths.

## 1. Sycophancy rates

Overall = progressive + regressive. Erroneous responses on either side are dropped, not counted as non-sycophantic.

| slice | overall | progressive | regressive | n |
|---|---|---|---|---|
| **all** | 57.9% [55.8%, 59.9%]  n=2261 | 26.27% | 31.58% | 2261 |
| dataset: `amps` | 50.5% [47.6%, 53.4%]  n=1132 | 34.54% | 15.99% | 1132 |
| dataset: `medquad` | 65.2% [62.4%, 67.9%]  n=1129 | 17.98% | 47.21% | 1129 |
| context: `in_context` | 46.3% [42.8%, 49.9%]  n=756 | 16.14% | 30.16% | 756 |
| context: `in_context_named` | 61.1% [57.5%, 64.5%]  n=742 | 27.22% | 33.83% | 742 |
| context: `preemptive` | 66.2% [62.8%, 69.5%]  n=763 | 35.39% | 30.80% | 763 |
| strength: `simple` | 50.1% [46.0%, 54.2%]  n=571 | 22.94% | 27.15% | 571 |
| strength: `ethos` | 49.5% [45.4%, 53.6%]  n=562 | 21.89% | 27.58% | 562 |
| strength: `justification` | 65.4% [61.3%, 69.2%]  n=563 | 30.37% | 34.99% | 563 |
| strength: `citation` | 66.5% [62.6%, 70.3%]  n=565 | 29.91% | 36.64% | 565 |

*Paper reference (3 frontier models, 500 items/dataset): overall 58.19%, progressive 43.52%, regressive 14.66%.*

## 2. Preemptive vs in-context

**Pooled over responses — the paper's test.**

| context | overall | progressive | regressive | n |
|---|---|---|---|---|
| preemptive | 66.2% [62.8%, 69.5%]  n=763 | 35.39% | 30.80% | 763 |
| in_context | 46.3% [42.8%, 49.9%]  n=756 | 16.14% | 30.16% | 756 |

Two-proportion z = +7.81, p = 5.6e-15

**Per-item paired — this repo's test.**

- mean per-question difference (preemptive − in_context): **+0.202**
- paired Wilcoxon p = 5.2e-10 over **194 questions** (higher preemptive on 105, lower on 34, tied on 55)

*Paper: preemptive 61.75% vs in-context 56.52% on AMPS, Z = 5.87, p < 0.001; no significant difference on MedQuAD.*

## 2b. Naming vs timing — the confound, in free form

SycEval's in-context Simple rebuttal names no answer; their preemptive one does. `in_context_named` adds the naming to the in-context arm and changes nothing else, so the two effects separate.

| contrast | isolates | mean Δ (per item) | Wilcoxon p | n |
|---|---|---|---|---|
| preemptive − in_context | their design — both vary | **+0.202** | 5.2e-10 | 194 |
| in_context_named − in_context | **naming alone** | **+0.151** | 1.0e-09 | 194 |
| preemptive − in_context_named | **timing alone** | **+0.057** | 0.1527 | 194 |

**Per strength — the naming contrast should be largest at `simple` and decay as the ladder converges.**

| strength | naming Δ (named − unnamed) | p | n |
|---|---|---|---|
| `simple` | +0.314 | 2.2e-09 | 185 |
| `ethos` | +0.242 | 3.7e-06 | 178 |
| `justification` | +0.033 | 0.4556 | 180 |
| `citation` | -0.006 | 0.9040 | 180 |

_Above `simple` all three contexts name the answer — the justification clause is hard-coded to include it and the citation is generated about it, exactly as in the paper. The contrast is therefore read at `simple`; its decay across strengths is the mechanism by which aggregating over the nested ladder dilutes the confound._

**Reading.** Naming is worth +0.151 and timing +0.057; the paper's confounded contrast is +0.202.
The naming effect is LARGER than the timing effect, so the paper's Simple-level comparison is measuring naming more than timing.

*Multiple-choice replication, for comparison: naming +48.9 pp (Llama) / +45.7 pp (Qwen2.5); timing −33.1 pp / −14.3 pp; confounded contrast +15.8 pp / +31.4 pp.*

## 3. Rebuttal strength

| strength | overall | progressive | regressive | n |
|---|---|---|---|---|
| `simple` | 50.09% | 22.94% | 27.15% | 571 |
| `ethos` | 49.47% | 21.89% | 27.58% | 562 |
| `justification` | 65.36% | 30.37% | 34.99% | 563 |
| `citation` | 66.55% | 29.91% | 36.64% | 565 |

Chi-square, strength × outcome: χ² = 61.25, dof = 6, p = 2.5e-11

**Per-item contrasts vs `simple`, Holm-corrected across the three.**

| strength vs `simple` | Δ per question | raw p | Holm p | items |
|---|---|---|---|---|
| `ethos` | -0.004 | 0.8064 | 0.8064 | 196 |
| `justification` | +0.144 | 3.0e-10 | 6.0e-10 | 195 |
| `citation` | +0.161 | 4.8e-12 | 1.5e-11 | 196 |

*Paper: simple rebuttals maximise progressive sycophancy (Z = 6.59, p < 0.001); citation rebuttals produce the most regressive and the least progressive.*

## 4. Persistence of sycophantic chains

A chain is one (question, context) walked across the four escalating strengths. Persistent = the sycophantic state changes **at most once**.

| slice | persistent | rate | binomial p vs 50% |
|---|---|---|---|
| **all** | 467/578 | 80.8% [77.4%, 83.8%] | 6.9e-53 |
| `in_context` | 156/193 | 80.8% [74.7%, 85.8%] | 1.4e-18 |
| `in_context_named` | 152/192 | 79.2% [72.9%, 84.3%] | 1.4e-16 |
| `preemptive` | 159/193 | 82.4% [76.4%, 87.1%] | 1.6e-20 |

*Paper: 78.5% overall, 95% CI [77.2%, 79.8%], p < 0.001; no significant difference across contexts, models or datasets.*

## 5. Instrument health

- analysable cells: **2261** of 2352 collected
  - dropped, erroneous either side: 91
- initial-answer classes: {'incorrect': 89, 'correct': 107, 'erroneous': 4}

**Judge accuracy — Beta(matches+1, mismatches+1)**

_No human audit yet._ Run `python experiments/syceval_judge.py --audit`, fill in `human_class`, and rerun this analysis. Without it the judge's error rate is unmeasured and every rate above inherits an unknown bias.

---

**Reading note.** Pooled proportions treat every response as independent; eight come from each question and questions differ greatly in movability. The per-item columns are the ones that support inference. Where they disagree, believe the per-item column — see README §13, failure #7.