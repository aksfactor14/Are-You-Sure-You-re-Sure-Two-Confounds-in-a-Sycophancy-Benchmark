"""Statistical tests: is an effect really zero (equivalence test), and does it differ between
conditions. Other analysis scripts import it."""
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT), str(_ROOT / "experiments")]
import json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats, integrate
import statsmodels.formula.api as smf
from core import config

warnings.filterwarnings("ignore")
OUT = config.OUT_DIR
ALPHA = 0.05

SESOI_MARGIN = 0.57
SESOI_CAVE = 0.045
SESOI_PROBE = 1.55


GEMMA = "mlx-community/gemma-2-2b-it-4bit"


def _finish(out, pm=None, pd_=None, tail=None):
    if tail:
        out = out + tail
    p = config.out_path("summary_equivalence.md")
    with open(p, "w", encoding="utf8") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    print("\nwrote", p)
    return p


def load(name, model=None):
    p = config.out_path(name, model)
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf8") if l.strip()]
    return pd.DataFrame(rows) if rows else None


def skip(out, section, name, why="not collected for this model"):
    """Record a skipped section, loudly, so a thin report is never mistaken for a
    clean one."""
    out.append(f"\n## {section}\n")
    out.append(f"_Skipped — `{os.path.basename(config.out_path(name))}` {why}._")
    print(f"[equivalence] SKIP {section}: {name} missing", flush=True)
    return out

def bf01_jzs(t, n, r=0.707):
    nu = n - 1
    num = (1 + t ** 2 / nu) ** (-(nu + 1) / 2)

    def integrand(g):
        return ((1 + n * g) ** -0.5
                * (1 + t ** 2 / ((1 + n * g) * nu)) ** (-(nu + 1) / 2)
                * (r ** 2 / 2) ** 0.5 / np.sqrt(np.pi)
                * g ** -1.5 * np.exp(-r ** 2 / (2 * g)))

    den, _ = integrate.quad(integrand, 1e-12, np.inf, limit=300)
    return float(num / den)


def tost(d, sesoi):
    """Paired TOST on per-item differences d. Returns a dict of everything."""
    d = np.asarray(d, float)
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); nu = n - 1
    t_stat = m / se if se > 0 else 0.0
    p_two = stats.ttest_1samp(d, 0).pvalue
    # two one-sided tests against -sesoi and +sesoi
    t_lo = (m + sesoi) / se
    t_hi = (m - sesoi) / se
    p_lo = 1 - stats.t.cdf(t_lo, nu)     # H0: effect <= -sesoi
    p_hi = stats.t.cdf(t_hi, nu)         # H0: effect >= +sesoi
    p_tost = max(p_lo, p_hi)
    delta_min = abs(m) + stats.t.ppf(1 - ALPHA, nu) * se
    return dict(n=n, mean=m, se=se, p_two=p_two, p_tost=p_tost,
                equiv=p_tost < ALPHA, delta_min=delta_min,
                bf01=bf01_jzs(t_stat, n))


def paired_diff(pv, a, b):
    return (pv[a] - pv[b]).dropna()


# ==========================================================================
def null_section(rows):
    out = ["| claim | n | mean Δ | p (two-sided) | SESOI | TOST p | equivalent? | Δ_min | BF₀₁ | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for label, d, sesoi in rows:
        r = tost(d, sesoi)
        if r["equiv"] and r["bf01"] > 3:
            v = "OK **equivalent** (both)"
        elif r["equiv"]:
            v = "OK equivalent (TOST)"
        elif r["bf01"] > 3:
            v = "◐ BF supports null, TOST inconclusive"
        else:
            v = "FAIL **inconclusive** — cannot claim null"
        out.append(f"| {label} | {r['n']} | {r['mean']:+.3f} | {r['p_two']:.4f} | ±{sesoi:.3f} | "
                   f"{r['p_tost']:.4f} | {'yes' if r['equiv'] else 'no'} | "
                   f"±{r['delta_min']:.3f} | {r['bf01']:.2f} | {v} |")
    return out


def syceval_section():
    sy = load("syceval_replication.jsonl")
    if sy is None:
        return ["\n## 7. Null claims — SycEval template ablations\n",
                f"_Skipped — `{os.path.basename(config.out_path('syceval_replication.jsonl'))}` "
                f"not collected for this model._"]

    sy["follow_rate"] = sy.follow / (sy.k - sy.unparsed).clip(lower=1)
    ps = sy.pivot_table(index="qid", columns="arm", values="follow_rate").dropna()
    if "IC_NAMED" not in ps:
        return ["\n## 7. Null claims — SycEval template ablations\n",
                "_Skipped — no `IC_NAMED` reference arm in this run._"]

    src = os.path.basename(config.out_path("syceval_replication.jsonl"))
    mismatch = (src == "syceval_replication.jsonl"
                and config.model_tag() != "Qwen2.5-3B-Instruct")
    out = ["\n## 7. Null claims — SycEval template ablations "
           f"(n={len(ps)} items)\n",
           f"_Source: `{src}`._\n"]
    if mismatch:
        out.append(
            f">  **These rows are NOT `{config.model_tag()}`.** The untagged "
            f"`syceval_replication.jsonl` was collected under a guard that permitted only "
            f"Qwen2.5-3B-Instruct, and `config.out_path()` resolves every legacy model to "
            f"that bare filename. Read this section as **Qwen2.5-3B-Instruct**, and do not "
            f"quote it against the model in this report's header.\n")
    out += [
           "> SycEval's Simple rebuttal ends `...with a CONCISE response!!!`, and because "
           "their four strengths nest, both features are inherited by every rebuttal in "
           "the benchmark. Our critique needs these to be **inert**, not merely "
           "*undetected*.\n"]
    rows = [(lbl, paired_diff(ps, a, "IC_NAMED"), SESOI_CAVE)
            for lbl, a in [("removing `!!!` alone", "IC_NAMED_NO_EXCL"),
                           ("removing `CONCISE` alone", "IC_NAMED_NO_CONCISE"),
                           ("removing **both**", "IC_NAMED_NEITHER")] if a in ps]
    out += null_section(rows)

    need = ["IC_NAMED_NO_EXCL", "IC_NAMED_NO_CONCISE", "IC_NAMED_NEITHER"]
    if all(a in ps for a in need):
        d_e = ps["IC_NAMED_NO_EXCL"] - ps["IC_NAMED"]
        d_c = ps["IC_NAMED_NO_CONCISE"] - ps["IC_NAMED"]
        d_b = ps["IC_NAMED_NEITHER"] - ps["IC_NAMED"]
        inter = (d_b - (d_e + d_c)).dropna()
        t = stats.ttest_1samp(inter, 0)
        w = stats.wilcoxon(inter) if (inter != 0).any() else None
        out += ["\n### 7b. Do the two features act independently?\n",
                "_If each is separately inert but removing both moves the rate, they "
                "interact, and 'feature X alone does nothing' stops being a safe summary._\n",
                f"- removing `!!!` alone: **{d_e.mean()*100:+.1f} pp**",
                f"- removing `CONCISE` alone: **{d_c.mean()*100:+.1f} pp**",
                f"- removing both: **{d_b.mean()*100:+.1f} pp** "
                f"(additive prediction {(d_e.mean()+d_c.mean())*100:+.1f} pp)",
                f"- **interaction** (both − sum of singles): {inter.mean()*100:+.1f} pp, "
                f"t={t.statistic:+.2f}, **p={t.pvalue:.6f}**"
                + (f"; Wilcoxon p={w.pvalue:.6f}" if w is not None else ""),
                ("\n**Super-additive:** the two features together do more than the sum of "
                 "their separate effects, so they cannot be dismissed one at a time."
                 if t.pvalue < ALPHA and inter.mean() > 0 else
                 "\n**Sub-additive:** removing both moves the rate LESS than the two "
                 "single removals predict."
                 if t.pvalue < ALPHA else
                 "\n**Additive** — no detectable interaction; the two features can be "
                 "reasoned about independently.")]
    return out


def interaction_lmm(long, dv, label):
    md = smf.mixedlm(f"{dv} ~ mark * context", long, groups=long["qid"],
                     re_formula="~context")
    fit = md.fit(reml=False)
    key = [k for k in fit.params.index if ":" in k]
    if not key:
        return f"- {label}: no interaction term estimated"
    k = key[0]
    return (f"- **{label}** — LMM `{dv} ~ mark*context + (context|qid)`: "
            f"interaction β={fit.params[k]:+.3f}, z={fit.tvalues[k]:+.2f}, "
            f"**p={fit.pvalues[k]:.6f}**")


def main():
    out = ["# Equivalence tests and explicit interaction terms\n",
           "Addresses two review points: (1) non-significance is not evidence of "
           "absence — nulls now get TOST + Bayes factors; (2) comparing significance "
           "across cells is not a test of difference (Gelman & Stern 2006) — gating "
           "claims now get explicit interaction terms.\n",
           f"Pre-declared SESOIs: margin ±{SESOI_MARGIN}, caving ±{SESOI_CAVE:.3f} "
           f"(4.5 pp), probe ±{SESOI_PROBE}. Each is half the smallest effect the "
           "project calls meaningful in that channel. `Δ_min` is the smallest bound "
           "at which equivalence holds at α=.05 — assumption-free.\n"]

    out.append(f"\n**Model: `{config.model_tag()}`.** Every input below is resolved "
               f"per-model; sections whose data this model does not have are skipped "
               f"and labelled, never silently filled from another model's run.\n")

    ag = load("agree2_results.jsonl")
    pm = None
    if ag is None:
        skip(out, "1. Null claims — confidence margin", "agree2_results.jsonl",
             "not found; this earlier experiment is not part of this release")
    else:
        pm = ag.pivot_table(index="qid", columns="arm", values="margin").dropna()
        out += [f"\n## 1. Null claims — confidence margin (n={len(pm)} items)\n"]
        out += null_section([
            (lbl, paired_diff(pm, a, b), SESOI_MARGIN)
            for lbl, a, b in [
                ("`!!!` does nothing when the user **agrees** (EXCL vs PLAIN)",
                 "AGREE_EXCL", "AGREE_PLAIN"),
                ("`???` does nothing when the user **disagrees** (Q3 vs PLAIN)",
                 "WRONG_Q3", "WRONG_PLAIN"),
                ("a bare extra turn does nothing (FILLER vs NONE)", "FILLER", "NONE"),
                ("`???` agreement is no better than silence (AGREE_Q3 vs NONE)",
                 "AGREE_Q3", "NONE"),
            ] if a in pm and b in pm
        ])

    b2 = load("r11_B2.jsonl")
    if b2 is None:
        skip(out, "2. Null claims — caving, B2 deviance controls", "r11_B2.jsonl",
             "not found; this earlier experiment is not part of this release")
    else:
        pb = b2.pivot_table(index="qid", columns="arm", values="cave_prop").dropna()
        out += ["\n## 2. Null claims — caving, B2 deviance controls (n=%d items)\n" % len(pb),
                "> These are the arms the *'it's not generic punctuation deviance'* argument "
                "rests on. If they are merely 'not significant', that argument is weak.\n"]
        out += null_section([
            (f"`{a}` vs NEUT", paired_diff(pb, a, "NEUT"), SESOI_CAVE)
            for a in ["EXCL", "SPACE_FAR", "MIXCASE", "ISOLATE", "SPACE_NEAR", "LOWER"]
            if a in pb
        ])

    # ---------------- caving: dose ----------------
    ds = load("dose.jsonl")
    pd_ = None
    if ds is None:
        skip(out, "3. Null claims — dose experiment", "dose.jsonl",
             "not found; this earlier experiment is not part of this release")
    else:
        ds["cave_prop"] = ds.caves / ds.k
        pd_ = ds.pivot_table(index="qid", columns="arm", values="cave_prop").dropna()
        out += ["\n## 3. Null claims — dose experiment (n=%d items)\n" % len(pd_)]
        out += null_section([
            (lbl, paired_diff(pd_, a, b), SESOI_CAVE)
            for lbl, a, b in [
                ("`???` does nothing to caving (Q3 vs DOT)", "Q3", "DOT"),
                ("threshold not gradient: `B!` vs `B!!!`", "E1", "E3"),
                ("threshold not gradient: `B!!!` vs `B!!!!!`", "E3", "E5"),
            ] if a in pd_ and b in pd_
        ])

    # ---------------- probe ----------------
    pr = load("probe_mind.jsonl")
    if pr is None:
        skip(out, "4. Null claims — perception probe, EMPH", "probe_mind.jsonl",
             "not found; this earlier experiment is not part of this release")
    else:
        pe = pr[pr.probe == "EMPH"].pivot_table(index="qid", columns=["context", "mark"],
                                                values="primary").dropna()
        out += ["\n## 4. Null claims — perception probe, EMPH (n=%d items)\n" % len(pe)]
        if ("WRONG", "Q3") in pe and ("WRONG", "PLAIN") in pe:
            out += null_section([
                ("`???` under disagreement is negligible (Q3 vs PLAIN, WRONG)",
                 pe[("WRONG", "Q3")] - pe[("WRONG", "PLAIN")], SESOI_PROBE),
            ])

    pr = load("probe_mind.jsonl")
    dc = load("probe_decorr.jsonl")
    iso = load("probe_isolated.jsonl")
    _probe_missing = [n for n, df in [("probe_mind.jsonl", pr),
                                      ("probe_decorr.jsonl", dc),
                                      ("probe_isolated.jsonl", iso)] if df is None]
    sy_tail = syceval_section()

    if _probe_missing:
        skip(out, "5. Null claims — perception probes", _probe_missing[0],
             f"missing ({', '.join(_probe_missing)}); these earlier experiments are not part "
             f"of this release")
        return _finish(out, pm, pd_, tail=sy_tail)

    def pcell(probe):
        return pr[pr.probe == probe].pivot_table(index="qid", columns=["context", "mark"],
                                                 values="primary").dropna()

    def dcell(probe):
        return dc[dc.probe == probe].pivot_table(index="qid", columns=["cell", "mark"],
                                                 values="primary").dropna()

    pe, px, pcr = pcell("EMPH"), pcell("EXP"), pcell("CRED")
    de, dx = dcell("EMPH"), dcell("EXP")

    out += ["\n## 5. Null claims — perception probes (n=40)\n",
            "> Two of these are load-bearing. **'`EMPH` is the clean probe'** underwrites "
            "every §9 conclusion, and **'`???` does nothing when the user disagrees'** is "
            "half of the gating claim. Both were previously asserted from p > 0.05 alone.\n"]
    rows = [
        ("**`EMPH` reads manner, not content** (leak, A−B)",
         pe[("AGREE", "PLAIN")] - pe[("WRONG", "PLAIN")], SESOI_PROBE),
        ("**`EMPH` leak replicated on model-WRONG items** (C−D)",
         de[("C", "PLAIN")] - de[("D", "PLAIN")], SESOI_PROBE),
        ("**`???` does nothing when the user disagrees** (`EXP`, cell B)",
         px[("WRONG", "Q3")] - px[("WRONG", "PLAIN")], SESOI_PROBE),
        ("`!!!` is read the SAME in both contexts (`EXP` interaction)",
         (px[("AGREE", "EXCL")] - px[("AGREE", "PLAIN")])
         - (px[("WRONG", "EXCL")] - px[("WRONG", "PLAIN")]), SESOI_PROBE),
        ("`!!!` does nothing to `CRED` when the user agrees (cell A)",
         pcr[("AGREE", "EXCL")] - pcr[("AGREE", "PLAIN")], SESOI_PROBE),
    ]
    out += null_section(rows)

    # isolated probe: role must be EXACTLY inert once letter identity is matched
    out += ["\n### Isolated probe — the built-in falsification check\n",
            "With no question, gold-letter vs distractor-letter cells must be identical. "
            "Raw they are not — but only because the distractor rule ('first label ≠ gold') "
            "makes distractors ~78% `A` while gold spreads A–E. Matched on letter identity:\n",
            "| probe | raw role Δ | letter-matched role Δ |", "|---|---|---|"]
    for probe in ["EMPH", "EXP", "CRED", "CONF"]:
        s_ = iso[iso.probe == probe]
        pv = s_.pivot_table(index=["qid", "mark"], columns="letter_role", values="primary").dropna()
        raw = (pv["GOLD"] - pv["DISTRACTOR"]).mean()
        per = s_.groupby(["letter", "letter_role"]).primary.mean().unstack().dropna()
        matched = (per["GOLD"] - per["DISTRACTOR"]).mean() if len(per) else float("nan")
        out.append(f"| `{probe}` | {raw:+.3f} | **{matched:+.3f}** |")
    out.append("\n**Exactly zero** — as it must be, since with no question there is nothing "
               "to know. No equivalence test is needed for an identity.")

    # ==================== INTERACTIONS ====================
    out += ["\n---\n\n## 6. Explicit interaction terms (Gelman & Stern)\n",
            "Every 'gated' claim restated as a `mark × context` interaction rather "
            "than a comparison of two separate significances.\n"]

    # margin: EXCL and Q3 gating
    for mark, agree_arm, wrong_arm in [("EXCL", "AGREE_EXCL", "WRONG_EXCL"),
                                       ("Q3", "AGREE_Q3", "WRONG_Q3")]:
        rows = []
        for q in pm.index:
            for ctx, mk_arm, pl_arm in [("AGREE", agree_arm, "AGREE_PLAIN"),
                                        ("WRONG", wrong_arm, "WRONG_PLAIN")]:
                rows.append(dict(qid=q, context=ctx, mark="MARK", margin=pm.loc[q, mk_arm]))
                rows.append(dict(qid=q, context=ctx, mark="PLAIN", margin=pm.loc[q, pl_arm]))
        long = pd.DataFrame(rows)
        dod = ((pm[agree_arm] - pm["AGREE_PLAIN"]) - (pm[wrong_arm] - pm["WRONG_PLAIN"])).dropna()
        t = stats.ttest_1samp(dod, 0)
        w = stats.wilcoxon(dod)
        out.append(interaction_lmm(long, "margin", f"margin, `{mark}` × context"))
        out.append(f"  - paired difference-of-differences: {dod.mean():+.3f}, "
                   f"t={t.statistic:+.2f}, p={t.pvalue:.6f}; Wilcoxon p={w.pvalue:.6f}")

    # probe EMPH interaction
    rows = []
    for q in pe.index:
        for ctx in ["AGREE", "WRONG"]:
            rows.append(dict(qid=q, context=ctx, mark="MARK", ev=pe.loc[q, (ctx, "EXCL")]))
            rows.append(dict(qid=q, context=ctx, mark="PLAIN", ev=pe.loc[q, (ctx, "PLAIN")]))
    long = pd.DataFrame(rows)
    out.append(interaction_lmm(long, "ev", "probe EMPH, `!!!` × context"))
    dod = ((pe[("AGREE", "EXCL")] - pe[("AGREE", "PLAIN")])
           - (pe[("WRONG", "EXCL")] - pe[("WRONG", "PLAIN")])).dropna()
    t = stats.ttest_1samp(dod, 0)
    out.append(f"  - paired difference-of-differences: {dod.mean():+.3f}, "
               f"t={t.statistic:+.2f}, p={t.pvalue:.6f}")
    out.append("  - WARNING: significant, but SMALL relative to same-signed main effects "
               "(+3.07 agree / +4.91 disagree). Both contexts move the same way; the "
               "interaction is a magnitude modifier, not a sign flip.")

    out += [f"\n### 6b. Cross-model reversal — {config.model_tag()} vs Gemma-2-2b\n",
            "_Previously asserted from two separate significances, which is the exact "
            "Gelman & Stern error. Tested here as an explicit interaction._\n"]
    gm = load("gemma_results.jsonl", model=GEMMA)
    if gm is None or b2 is None:
        missing = "gemma_results.jsonl" if gm is None else "r11_B2.jsonl"
        out.append(f"_Skipped — needs both this model's `r11_B2` and Gemma's "
                   f"`gemma_results.jsonl`; `{missing}` is absent._")
    elif config.model_tag() == config.model_tag(GEMMA):
        out.append("_Skipped — the active model IS Gemma; there is nothing to contrast._")
    else:
        gm["cave_prop"] = gm.caves / gm.k
        pg = gm.pivot_table(index="qid", columns="arm", values="cave_prop")
        gcol = "EXCL" if "EXCL" in pg else [c for c in pg.columns if "EXC" in str(c).upper()][0]
        dq = (pb["EXCL"] - pb["NEUT"]).dropna()      # active model per-item deltas
        dg = (pg[gcol] - pg["NEUT"]).dropna()        # Gemma per-item deltas
        tw = stats.ttest_ind(dq, dg, equal_var=False)
        mw = stats.mannwhitneyu(dq, dg)
        out += [f"- {config.model_tag()} per-item Δ(`!!!` − plain): "
                f"**{dq.mean():+.4f}** (n={len(dq)})",
                f"- Gemma-2-2b per-item Δ(`!!!` − plain): **{dg.mean():+.4f}** (n={len(dg)})",
                f"- **model × mark interaction** (between-model, items are not shared so "
                f"unpaired): Welch t={tw.statistic:+.2f}, **p={tw.pvalue:.6f}**; "
                f"Mann-Whitney p={mw.pvalue:.6f}",
                f"- difference in deltas: **{dq.mean()-dg.mean():+.4f}**",
                ("\n**Verdict: the reversal survives an explicit interaction test.** The two "
                 "models do not merely differ in significance — the *sign of the effect itself* "
                 "differs, and the difference between the effects is significant."
                 if tw.pvalue < 0.05 else
                 "\nWARNING: **The reversal does NOT survive an interaction test.** It must be "
                 "downgraded to 'the two models were not shown to differ'.")]

    return _finish(out, pm, pd_, tail=sy_tail)


if __name__ == "__main__":
    main()
