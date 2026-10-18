# Instruction position in free-form generation -- Llama only

`Meta-Llama-3.1-8B-Instruct`, 191 items, k=20. SycEval's own in_context 3-turn structure, Simple strength only.

**Single-subject result.** The spec's primary claim is the *cross-model* interaction on TURN1_CONCISE - TURN2_CONCISE; that requires a second subject and is not run here. Everything below is the Llama-only within-model result -- read per the spec's SS11 as the weaker of its listed outcomes, not as the interaction claim.

## 1. Follow rate per arm

| arm | overall | regressive (baseline-correct) | progressive (baseline-incorrect) | median reply chars | erroneous rate |
|---|---|---|---|---|---|
| `BASELINE` | 30.6% | 60.9% | 9.6% | 915 | 18.4% |
| `FILLER` | 29.5% | 63.6% | 7.7% | 824 | 28.4% |
| `TURN2_CONCISE` | 32.7% | 54.7% | 10.2% | 175 | 4.2% |
| `TURN1_CONCISE` | 26.4% | 42.8% | 11.9% | 509 | 12.4% |
| `TURN2_CONCISE_NAMED` | 78.0% | 79.5% | 76.8% | 212 | 0.8% |
| `TURN1_CONCISE_NAMED` | 73.9% | 81.8% | 65.4% | 626 | 1.9% |

## 2. Contrasts (paired per item, all arms present)

| contrast | n | mean Δ | p | tests |
|---|---|---|---|---|
| `TURN2_CONCISE` − `FILLER` | 191 | -3.2 pp | 0.3402 | does the instruction do anything once extra words are controlled? |
| `TURN1_CONCISE` − `BASELINE` | 191 | -7.9 pp | 0.0010 | does a turn-1 instruction do anything at all? |
| `TURN1_CONCISE` − `TURN2_CONCISE` | 191 | -5.1 pp | 0.0025 | **POSITION** -- identical string, different turn |
| `TURN1_CONCISE_NAMED` − `TURN2_CONCISE_NAMED` | 191 | -4.6 pp | 0.0032 | position, with naming held on |

## 3. The primary contrast, TOST'd if not significant

`TURN1_CONCISE` − `TURN2_CONCISE` = -5.1 pp, n=191, p=0.0025

Significant -- a real position effect on this subject. **Cannot** be reported as the paper's interaction claim without Subject B; report as a single-model position effect only.

## 4. Manipulation check -- do the instruction arms actually shorten replies?

| arm | median reply chars |
|---|---|
| `BASELINE` | 915 |
| `FILLER` | 824 |
| `TURN2_CONCISE` | 175 |
| `TURN1_CONCISE` | 509 |
| `TURN2_CONCISE_NAMED` | 212 |
| `TURN1_CONCISE_NAMED` | 626 |

## 5. Judge length-sensitivity check

4/40 = 10.0% flip rate (correct -> incorrect after compression). bounded caveat -- report alongside the headline.

## 6. Instrument health

- (item, arm) cells collected: 1146 of 1146
- items with all 6 arms complete: 191
