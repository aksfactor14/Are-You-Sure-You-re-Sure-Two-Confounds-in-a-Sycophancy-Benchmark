"""Free-form instruction-position results for both models (Table 9, the shared-item comparison,
and the Qwen3-4B equivalence test). Needs no model."""
import sys as _sys
import pathlib as _pl

_ROOT = _pl.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT)]

import json
import os

import pandas as pd
from scipy import stats

from core import config
from analysis.equivalence_tests import tost, SESOI_CAVE

SUBJECTS = [("Llama-3.1-8B", "Meta-Llama-3.1-8B-Instruct"),
            ("Qwen3-4B", "Qwen3-4B-Instruct-2507")]
ARMS = ["BASELINE", "FILLER", "TURN2_CONCISE", "TURN1_CONCISE",
        "TURN2_CONCISE_NAMED", "TURN1_CONCISE_NAMED"]
OUT = os.path.join(config.OUT_DIR, "summary_freeform_position_crossmodel.md")


def load(tag):
    path = os.path.join(config.OUT_DIR, "syceval", tag, "instruction_position_freeform.jsonl")
    with open(path, encoding="utf8") as f:
        d = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    d["rate"] = d.follows / (d.k - d.erroneous).clip(lower=1)
    full = d.groupby("qid").arm.nunique()
    d = d[d.qid.isin(full[full == len(ARMS)].index)]
    return d, d.pivot_table(index="qid", columns="arm", values="rate")


def wilcoxon(diff):
    diff = diff.dropna()
    p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
    return len(diff), diff.mean() * 100, p


def main():
    data = {name: load(tag) for name, tag in SUBJECTS}
    out = ["# Free-form instruction position: cross-model summary\n"]

    out.append("## 1. Per-arm rates (Table 9)\n")
    out.append("| arm | " + " | ".join(f"{n} follow | {n} regressive | {n} chars" for n, _ in SUBJECTS) + " |")
    out.append("|---|" + "---|---|---|" * len(SUBJECTS))
    for arm in ARMS:
        cells = []
        for name, _ in SUBJECTS:
            d, pv = data[name]
            correct_ids = d[d.baseline_correct].qid.unique()
            chars = d[d.arm == arm].median_reply_chars.median()
            cells.append(f"{pv[arm].mean() * 100:.1f} | {pv.loc[pv.index.isin(correct_ids), arm].mean() * 100:.1f} | {chars:.0f}")
        out.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    out.append("\n## 2. Within-model position contrast (turn 1 minus turn 2)\n")
    out.append("| model | version | items | change (pp) | Wilcoxon p |")
    out.append("|---|---|---|---|---|")
    for name, _ in SUBJECTS:
        _, pv = data[name]
        for label, a, b in [("instruction only", "TURN2_CONCISE", "TURN1_CONCISE"),
                            ("with named answer", "TURN2_CONCISE_NAMED", "TURN1_CONCISE_NAMED")]:
            n, mean, p = wilcoxon(pv[b] - pv[a])
            out.append(f"| {name} | {label} | {n} | {mean:+.1f} | {p:.4f} |")

    out.append("\n## 3. Cross-model interaction on shared items (Figure 2)\n")
    (_, pv_a), (_, pv_b) = data[SUBJECTS[0][0]], data[SUBJECTS[1][0]]
    shared = pv_a.index.intersection(pv_b.index)
    da = pv_a.loc[shared, "TURN1_CONCISE"] - pv_a.loc[shared, "TURN2_CONCISE"]
    db = pv_b.loc[shared, "TURN1_CONCISE"] - pv_b.loc[shared, "TURN2_CONCISE"]
    n, mean, p = wilcoxon(db - da)
    out.append(f"- shared items: {len(shared)}")
    out.append(f"- {SUBJECTS[0][0]}: {da.mean() * 100:+.1f} pp")
    out.append(f"- {SUBJECTS[1][0]}: {db.mean() * 100:+.1f} pp")
    out.append(f"- difference ({SUBJECTS[0][0]} minus {SUBJECTS[1][0]}, per item): {-mean:+.1f} pp, Wilcoxon p = {p:.4f}")

    out.append("\n## 4. Is the Qwen3-4B position effect equivalent to zero?\n")
    _, pv = data["Qwen3-4B"]
    diff = (pv["TURN1_CONCISE"] - pv["TURN2_CONCISE"]).dropna()
    r = tost(diff.values, SESOI_CAVE)
    out.append(f"- change {r['mean'] * 100:+.1f} pp on {r['n']} items, bound +/-{SESOI_CAVE * 100:.1f} pp")
    out.append(f"- equivalence test p = {r['p_tost']:.4f}, Bayes factor for no effect = {r['bf01']:.2f}")

    text = "\n".join(out) + "\n"
    with open(OUT, "w", encoding="utf8") as f:
        f.write(text)
    print(text)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
