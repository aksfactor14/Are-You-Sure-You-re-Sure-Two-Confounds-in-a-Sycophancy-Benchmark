# Does a brevity instruction suppress caving -- and does POSITION matter?

`Qwen2.5-3B-Instruct-4bit`, 42 items, k=20. Two-turn throughout; the same fabricated argument in every arm.

| arm | instruction | where | cave rate | median reply |
|---|---|---|---|---|
| `CONTROL` | none | -- | **43.3%** | 478 chars |
| `FILLER` | none (length-matched) | -- | **44.5%** | 526 chars |
| `CONCISE` | be concise | rebuttal | **37.0%** | 48 chars |
| `LETTER_ONLY` | letter only | rebuttal | **33.2%** | 1 chars |
| `TURN1_LOCK` | letter only | **turn 1** | **47.5%** | 9 chars |

## The comparisons

| contrast | Δ | p | what it tests |
|---|---|---|---|
| CONCISE − FILLER | -7.5 pp | 0.1459 | brevity instruction, extra words controlled |
| LETTER_ONLY − FILLER | -11.3 pp | 0.0986 | a stronger brevity instruction |
| TURN1_LOCK − CONTROL | +4.2 pp | 0.5552 | does the section-3 result replicate here? |
| TURN1_LOCK − LETTER_ONLY | +14.3 pp | 0.0848 | **POSITION** -- same instruction, turn 1 vs rebuttal |

## Verdict

**UNRESOLVED** (+14.3 pp; difference test p=0.0848, TOST vs ±4.5 pp p=0.8613). Neither significant nor equivalent -- n=42 only bounds the effect below ±29.2 pp, which is larger than the SESOI. Do not report this as a null.

- section-3 replication here: TURN1_LOCK − CONTROL = +4.2 pp
- unparsed: 0 / 4200 draws