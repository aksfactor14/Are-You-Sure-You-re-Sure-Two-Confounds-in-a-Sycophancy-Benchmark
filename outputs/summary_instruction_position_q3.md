# Does a brevity instruction suppress caving -- and does POSITION matter?

`Qwen3-4B-Instruct-2507-4bit`, 36 items, k=20. Two-turn throughout; the same fabricated argument in every arm.

| arm | instruction | where | cave rate | median reply |
|---|---|---|---|---|
| `CONTROL` | none | -- | **5.0%** | 492 chars |
| `FILLER` | none (length-matched) | -- | **5.1%** | 494 chars |
| `CONCISE` | be concise | rebuttal | **0.7%** | 249 chars |
| `LETTER_ONLY` | letter only | rebuttal | **7.8%** | 1 chars |
| `TURN1_LOCK` | letter only | **turn 1** | **0.0%** | 1 chars |

## The comparisons

| contrast | Δ | p | what it tests |
|---|---|---|---|
| CONCISE − FILLER | -4.4 pp | 0.0260 | brevity instruction, extra words controlled |
| LETTER_ONLY − FILLER | +2.6 pp | 0.6858 | a stronger brevity instruction |
| TURN1_LOCK − CONTROL | -5.0 pp | 0.0169 | does the section-3 result replicate here? |
| TURN1_LOCK − LETTER_ONLY | -7.8 pp | 0.1025 | **POSITION** -- same instruction, turn 1 vs rebuttal |

## Verdict

**UNRESOLVED** (-7.8 pp; difference test p=0.1025, TOST vs ±4.5 pp p=0.7701). Neither significant nor equivalent -- n=36 only bounds the effect below ±15.2 pp, which is larger than the SESOI. Do not report this as a null.

- section-3 replication here: TURN1_LOCK − CONTROL = -5.0 pp
- unparsed: 0 / 3600 draws