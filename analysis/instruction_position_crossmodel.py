"""Compares the multiple-choice instruction-position results across models and tests whether
the effect differs between them. Needs no model."""
import itertools
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT)]

import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from core import config

ARMS = ["CONTROL", "FILLER", "CONCISE", "LETTER_ONLY", "TURN1_LOCK"]
SESOI = 0.045
WHERE = {"CONTROL": ("none", "--"), "FILLER": ("none (length-matched)", "--"),
         "CONCISE": ("be concise", "rebuttal"), "LETTER_ONLY": ("letter only", "rebuttal"),
         "TURN1_LOCK": ("letter only", "**turn 1**")}

MODEL_FILES = [
    ("Qwen2.5-3B",   config.OUT_DIR + "/instruction_position.jsonl"),
    ("Qwen3-4B",     config.OUT_DIR + "/instruction_position_q3.jsonl"),
    ("Llama-3.1-8B", config.OUT_DIR + "/instruction_position__Meta-Llama-3.1-8B-Instruct.jsonl"),
]

CONTRASTS = [("CONTROL", "TURN1_LOCK", "section-3 lock (TURN1_LOCK − CONTROL)"),
             ("LETTER_ONLY", "TURN1_LOCK", "**POSITION** (TURN1_LOCK − LETTER_ONLY)"),
             ("FILLER", "CONCISE", "brevity (CONCISE − FILLER)")]


def load(path):
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    d["rate"] = d.caves / (d.k - d.unparsed).clip(lower=1)
    return d.pivot_table(index="qid", columns="arm", values="rate")


def tost_verdict(delta, sesoi=SESOI):
    """A non-significant difference is not evidence of no difference. TOST against the
    SESOI is what licenses calling something equivalent rather than merely unresolved."""
    n = len(delta); se = delta.std(ddof=1) / np.sqrt(n); nu = n - 1
    pdiff = stats.wilcoxon(delta).pvalue if (delta != 0).any() else 1.0
    ptost = max(stats.t.sf((delta.mean() + sesoi) / se, nu),
                stats.t.cdf((delta.mean() - sesoi) / se, nu))
    bound = abs(delta.mean()) + stats.t.ppf(0.95, nu) * se
    verdict = ("differs" if pdiff < 0.05 else
               "equivalent" if ptost < 0.05 else
               f"**unresolved** (only bounded below ±{bound*100:.1f} pp)")
    return pdiff, ptost, bound, verdict


def main():
    models = {}
    for label, path in MODEL_FILES:
        if os.path.exists(path):
            models[label] = load(path)
    if len(models) < 2:
        _sys.exit(f"Need at least 2 model collections to compare; found "
                   f"{list(models)}. Run experiments/instruction_position.py for "
                   f"another model first.")

    out = ["# Does the format-lock effect belong to the model or the items?\n",
           f"Models compared: {', '.join(models)}. Each pair is restricted to the items "
           "BOTH models in that pair answer correctly unaided, same fabricated arguments, "
           "k=20 per cell. Two-turn throughout.\n",
           "| arm | instruction | where | " + " | ".join(f"**{m}**" for m in models) + " |",
           "|---|---|---|" + "---|" * len(models)]
    for arm in ARMS:
        cells = []
        for m in models:
            v = models[m][arm].mean() if arm in models[m] else float("nan")
            cells.append(f"**{v:.1%}**" if v == v else "--")
        out.append(f"| `{arm}` | {WHERE[arm][0]} | {WHERE[arm][1]} | " + " | ".join(cells) + " |")

    pairs = list(itertools.combinations(models, 2))
    out += ["\n## Contrasts, and the cross-model interaction (every pair present)\n"]
    for m1, m2 in pairs:
        a, b = models[m1], models[m2]
        shared = sorted(set(a.index) & set(b.index))
        a_s, b_s = a.loc[shared], b.loc[shared]
        out.append(f"### {m1} vs {m2} ({len(shared)} shared items)\n")
        out.append("| contrast | " + f"{m1} | {m2} | interaction | p |")
        out.append("|---|---|---|---|---|")
        for x, y, lab in CONTRASTS:
            if x not in a_s or y not in a_s or x not in b_s or y not in b_s:
                continue
            da, db = a_s[y] - a_s[x], b_s[y] - b_s[x]
            dd = db - da
            p = stats.wilcoxon(dd).pvalue if (dd != 0).any() else 1.0
            out.append(f"| {lab} | {da.mean()*100:+.1f} pp | {db.mean()*100:+.1f} pp | "
                       f"{dd.mean()*100:+.1f} pp | {p:.4f} |")
        out.append("")

    out += ["## Can each model's own POSITION contrast be called a null?\n",
            "A non-significant difference is not evidence of no difference. TOST against "
            f"the ±{SESOI*100:.1f} pp SESOI, using each model's own full item set (not "
            "restricted to any particular pair):\n",
            "| model | n items | Δ | difference p | TOST p | verdict |",
            "|---|---|---|---|---|---|"]
    for m, f in models.items():
        if "TURN1_LOCK" not in f or "LETTER_ONLY" not in f:
            continue
        d = (f["TURN1_LOCK"] - f["LETTER_ONLY"]).dropna()
        pdiff, ptost, bound, verdict = tost_verdict(d)
        out.append(f"| {m} | {len(d)} | {d.mean()*100:+.1f} pp | {pdiff:.4f} | "
                   f"{ptost:.4f} | {verdict} |")

    out += ["\n## What this establishes\n"]
    reversals = []
    for m1, m2 in pairs:
        a, b = models[m1], models[m2]
        shared = sorted(set(a.index) & set(b.index))
        if "TURN1_LOCK" not in a or "LETTER_ONLY" not in a: continue
        if "TURN1_LOCK" not in b or "LETTER_ONLY" not in b: continue
        a_s, b_s = a.loc[shared], b.loc[shared]
        da = (a_s["TURN1_LOCK"] - a_s["LETTER_ONLY"]).mean()
        db = (b_s["TURN1_LOCK"] - b_s["LETTER_ONLY"]).mean()
        dd = (b_s["TURN1_LOCK"] - b_s["LETTER_ONLY"]) - (a_s["TURN1_LOCK"] - a_s["LETTER_ONLY"])
        p = stats.wilcoxon(dd).pvalue if (dd != 0).any() else 1.0
        if p < 0.05 and da * db < 0:
            reversals.append((m1, m2, da, db, p))
    if reversals:
        for m1, m2, da, db, p in reversals:
            out.append(f"**{m1} vs {m2}: POSITION has opposite signs** "
                       f"({da*100:+.1f} pp vs {db*100:+.1f} pp, interaction p={p:.4f}). "
                       "The same instruction behaves differently depending on where it "
                       "sits, in opposite directions between these two models.")
    else:
        out.append("No pair currently shows a significant sign-reversal in the POSITION "
                   "contrast; see the per-pair table above for magnitude-only "
                   "interactions and the TOST table for which single-model contrasts "
                   "are actually resolved vs merely unresolved.")
    out.append(f"\n- models compared: {', '.join(models)}")

    p_out = config.OUT_DIR + "/summary_instruction_position_crossmodel.md"
    open(p_out, "w", encoding="utf8").write("\n".join(out))
    print("\n".join(out)); print("\nwrote", p_out)


if __name__ == "__main__":
    main()
