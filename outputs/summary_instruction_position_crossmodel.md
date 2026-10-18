# Does the format-lock effect belong to the model or the items?

Models compared: Qwen2.5-3B, Qwen3-4B, Llama-3.1-8B. Each pair is restricted to the items BOTH models in that pair answer correctly unaided, same fabricated arguments, k=20 per cell. Two-turn throughout.

| arm | instruction | where | **Qwen2.5-3B** | **Qwen3-4B** | **Llama-3.1-8B** |
|---|---|---|---|---|---|
| `CONTROL` | none | -- | **43.3%** | **5.0%** | **59.5%** |
| `FILLER` | none (length-matched) | -- | **44.5%** | **5.1%** | **70.0%** |
| `CONCISE` | be concise | rebuttal | **37.0%** | **0.7%** | **68.2%** |
| `LETTER_ONLY` | letter only | rebuttal | **33.2%** | **7.8%** | **80.1%** |
| `TURN1_LOCK` | letter only | **turn 1** | **47.5%** | **0.0%** | **73.2%** |

## Contrasts, and the cross-model interaction (every pair present)

### Qwen2.5-3B vs Qwen3-4B (36 shared items)

| contrast | Qwen2.5-3B | Qwen3-4B | interaction | p |
|---|---|---|---|---|
| section-3 lock (TURN1_LOCK − CONTROL) | +7.4 pp | -5.0 pp | -12.4 pp | 0.1947 |
| **POSITION** (TURN1_LOCK − LETTER_ONLY) | +19.0 pp | -7.8 pp | -26.8 pp | 0.0059 |
| brevity (CONCISE − FILLER) | -7.5 pp | -4.4 pp | +3.1 pp | 0.6265 |

### Qwen2.5-3B vs Llama-3.1-8B (38 shared items)

| contrast | Qwen2.5-3B | Llama-3.1-8B | interaction | p |
|---|---|---|---|---|
| section-3 lock (TURN1_LOCK − CONTROL) | +2.4 pp | +13.7 pp | +11.3 pp | 0.2203 |
| **POSITION** (TURN1_LOCK − LETTER_ONLY) | +14.6 pp | -7.0 pp | -21.6 pp | 0.0611 |
| brevity (CONCISE − FILLER) | -10.0 pp | -1.8 pp | +8.2 pp | 0.5556 |

### Qwen3-4B vs Llama-3.1-8B (33 shared items)

| contrast | Qwen3-4B | Llama-3.1-8B | interaction | p |
|---|---|---|---|---|
| section-3 lock (TURN1_LOCK − CONTROL) | -5.5 pp | +16.2 pp | +21.7 pp | 0.0000 |
| **POSITION** (TURN1_LOCK − LETTER_ONLY) | -6.1 pp | -4.7 pp | +1.4 pp | 0.9544 |
| brevity (CONCISE − FILLER) | -4.7 pp | -2.0 pp | +2.7 pp | 0.9696 |

## Can each model's own POSITION contrast be called a null?

A non-significant difference is not evidence of no difference. TOST against the ±4.5 pp SESOI, using each model's own full item set (not restricted to any particular pair):

| model | n items | Δ | difference p | TOST p | verdict |
|---|---|---|---|---|---|
| Qwen2.5-3B | 42 | +14.3 pp | 0.0848 | 0.8613 | **unresolved** (only bounded below ±29.2 pp) |
| Qwen3-4B | 36 | -7.8 pp | 0.1025 | 0.7701 | **unresolved** (only bounded below ±15.2 pp) |
| Llama-3.1-8B | 38 | -7.0 pp | 0.1737 | 0.6595 | **unresolved** (only bounded below ±17.0 pp) |

## What this establishes

**Qwen2.5-3B vs Qwen3-4B: POSITION has opposite signs** (+19.0 pp vs -7.8 pp, interaction p=0.0059). The same instruction behaves differently depending on where it sits, in opposite directions between these two models.

- models compared: Qwen2.5-3B, Qwen3-4B, Llama-3.1-8B