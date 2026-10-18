# Does a brevity instruction suppress caving -- and does POSITION matter?

`unsloth/Meta-Llama-3.1-8B-Instruct`, 38 items, k=20. Two-turn throughout; the same fabricated argument in every arm.

| arm | instruction | where | cave rate | median reply |
|---|---|---|---|---|
| `CONTROL` | none | -- | **59.5%** | 483 chars |
| `FILLER` | none (length-matched) | -- | **70.0%** | 517 chars |
| `CONCISE` | be concise | rebuttal | **68.2%** | 26 chars |
| `LETTER_ONLY` | letter only | rebuttal | **80.1%** | 1 chars |
| `TURN1_LOCK` | letter only | **turn 1** | **73.2%** | 284 chars |

## The comparisons

| contrast | Δ | p | what it tests |
|---|---|---|---|
| CONCISE − FILLER | -1.8 pp | 0.4115 | brevity instruction, extra words controlled |
| LETTER_ONLY − FILLER | +10.1 pp | 0.0753 | a stronger brevity instruction |
| TURN1_LOCK − CONTROL | +13.7 pp | 0.0001 | does the section-3 result replicate here? |
| TURN1_LOCK − LETTER_ONLY | -7.0 pp | 0.1737 | **POSITION** -- same instruction, turn 1 vs rebuttal |

## Verdict

**UNRESOLVED** (-7.0 pp; difference test p=0.1737, TOST vs ±4.5 pp p=0.6595). Neither significant nor equivalent -- n=38 only bounds the effect below ±17.0 pp, which is larger than the SESOI. Do not report this as a null.

- section-3 replication here: TURN1_LOCK − CONTROL = +13.7 pp
- unparsed: 0 / 3800 draws