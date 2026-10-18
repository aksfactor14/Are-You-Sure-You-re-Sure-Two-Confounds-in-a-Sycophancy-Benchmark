# Instruction position in free-form generation -- Llama only

`Qwen3-4B-Instruct-2507`, 196 items, k=20. SycEval's own in_context 3-turn structure, Simple strength only.

**Single-subject result.** The spec's primary claim is the *cross-model* interaction on TURN1_CONCISE - TURN2_CONCISE; that requires a second subject and is not run here. Everything below is the Llama-only within-model result -- read per the spec's SS11 as the weaker of its listed outcomes, not as the interaction claim.

## 1. Follow rate per arm

| arm | overall | regressive (baseline-correct) | progressive (baseline-incorrect) | median reply chars | erroneous rate |
|---|---|---|---|---|---|
| `BASELINE` | 37.4% | 59.6% | 6.7% | 1489 | 8.8% |
| `FILLER` | 40.0% | 64.5% | 6.8% | 1504 | 7.3% |
| `TURN2_CONCISE` | 26.0% | 32.8% | 16.8% | 276 | 2.4% |
| `TURN1_CONCISE` | 27.0% | 28.8% | 23.1% | 796 | 3.1% |
| `TURN2_CONCISE_NAMED` | 59.3% | 57.9% | 57.7% | 332 | 3.2% |
| `TURN1_CONCISE_NAMED` | 54.6% | 63.3% | 40.7% | 718 | 5.9% |

## 2. Contrasts (paired per item, all arms present)

| contrast | n | mean Δ | p | tests |
|---|---|---|---|---|
| `TURN2_CONCISE` − `FILLER` | 196 | -12.8 pp | 0.0002 | does the instruction do anything once extra words are controlled? |
| `TURN1_CONCISE` − `BASELINE` | 196 | -9.4 pp | 0.0012 | does a turn-1 instruction do anything at all? |
| `TURN1_CONCISE` − `TURN2_CONCISE` | 196 | +0.7 pp | 0.4853 | **POSITION** -- identical string, different turn |
| `TURN1_CONCISE_NAMED` − `TURN2_CONCISE_NAMED` | 196 | -4.8 pp | 0.2593 | position, with naming held on |

## 3. The primary contrast, TOST'd if not significant

`TURN1_CONCISE` − `TURN2_CONCISE` = +0.7 pp, n=196, p=0.4853

Not significant at p<0.05. TOST against ±4.5pp: p=0.0415, BF01=11.89 -> TOST-confirmed equivalent

## 4. Manipulation check -- do the instruction arms actually shorten replies?

| arm | median reply chars |
|---|---|
| `BASELINE` | 1489 |
| `FILLER` | 1504 |
| `TURN2_CONCISE` | 276 |
| `TURN1_CONCISE` | 796 |
| `TURN2_CONCISE_NAMED` | 332 |
| `TURN1_CONCISE_NAMED` | 718 |

## 5. Judge length-sensitivity check

4/40 = 10.0% flip rate (correct -> incorrect after compression). bounded caveat -- report alongside the headline.

## 6. Instrument health

- (item, arm) cells collected: 1176 of 1176
- items with all 6 arms complete: 196
