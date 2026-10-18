"""Free-form step: a judge model grades each answer as correct, incorrect or erroneous.
Run it twice, before and after the pushbacks."""
import sys as _sys
import pathlib as _pl

_ROOT = _pl.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT), str(_ROOT / "experiments")]

import json  # noqa: E402
import random  # noqa: E402

from syceval import config, judge, runner  # noqa: E402


def collect_pending():
    """Everything on disk that lacks a classification, as
    (qid, cell, question, ground_truth, ai_answer) tuples."""
    items = runner.index_by(runner.read_jsonl(config.ITEMS), ["qid"])
    runner.require(items, f"no items found — run syceval_select.py first ({config.ITEMS})")

    done = runner.done_keys(config.CLASSIFICATIONS, ["qid", "cell"])
    pending = []

    for r in runner.read_jsonl(config.INITIAL):
        key = (r["qid"], "initial")
        if key in done or r["qid"] not in items:
            continue
        it = items[r["qid"]]
        pending.append((r["qid"], "initial", it["question"], it["answer"], r["response"]))

    for r in runner.read_jsonl(config.RESPONSES):
        cell = f"{r['context']}:{r['strength']}"
        key = (r["qid"], cell)
        if key in done or r["qid"] not in items:
            continue
        it = items[r["qid"]]
        pending.append((r["qid"], cell, it["question"], it["answer"], r["response"]))

    return pending, items


def audit_sample():
    rows = runner.read_jsonl(config.CLASSIFICATIONS)
    runner.require(rows, "nothing classified yet")
    items = runner.index_by(runner.read_jsonl(config.ITEMS), ["qid"])
    by_source = {}
    for r in rows:
        it = items.get(r["qid"])
        if it:
            by_source.setdefault(it["source"], []).append(r)

    rng = random.Random(config.SEED)
    out = []
    for src, rs in by_source.items():
        for r in rng.sample(rs, min(config.N_AUDIT_PER_SOURCE, len(rs))):
            it = items[r["qid"]]
            out.append({"qid": r["qid"], "cell": r["cell"], "source": src,
                        "question": it["question"], "ground_truth": it["answer"],
                        "ai_answer": r.get("ai_answer_preview", ""),
                        "llm_class": r["classification"], "human_class": ""})
    with open(config.HUMAN_AUDIT, "w", encoding="utf8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} rows to {config.HUMAN_AUDIT}")
    print("fill in `human_class` (correct|incorrect|erroneous) and rerun the analysis.")


def main():
    if "--audit" in _sys.argv:
        return audit_sample()

    self_judging = config.JUDGE_MODEL == config.SUBJECT_MODEL
    runner.require(
        not self_judging or config.ALLOW_SELF_JUDGE,
        f"judge and subject are both {config.JUDGE_MODEL}. The benchmark would grade "
        f"its own homework.\n  Set SYCEVAL_JUDGE_MODEL to a different model, or "
        f"SYCEVAL_ALLOW_SELF_JUDGE=1 to proceed anyway.")

    runner.stage_header(
        "SYCEVAL — LLM-as-a-Judge classification",
        judge_model=config.JUDGE_MODEL,
        subject_model=config.SUBJECT_MODEL,
        self_judging=self_judging,
        out=config.CLASSIFICATIONS,
    )

    pending, _ = collect_pending()
    print(f"{len(pending)} responses awaiting classification")
    if not pending:
        print("nothing to do.")
        return

    m = runner.load_model(config.JUDGE_MODEL)
    stamp = {"judge_model": config.JUDGE_MODEL, "self_judged": self_judging}
    prog = runner.Progress(len(pending), "judge")

    # Coarse chunks; runner.chat() batches by config.SUB inside. See syceval_main.py --
    # slicing by SUB here only added a torch.cuda.empty_cache() per batch.
    CHUNK = config.SUB * 16
    with runner.Appender(config.CLASSIFICATIONS, stamp=stamp) as w:
        for i in range(0, len(pending), CHUNK):
            chunk = pending[i:i + CHUNK]
            classes = judge.classify_batch(
                m, [(q, gt, ai) for _, _, q, gt, ai in chunk])
            for (qid, cell, _, _, ai), cls in zip(chunk, classes):
                w.write({"qid": qid, "cell": cell, "classification": cls,
                         "ai_answer_preview": ai[:280]})
                prog.tick(f"{qid} {cell} -> {cls}")
            runner.release()

    from collections import Counter
    dist = Counter(r["classification"] for r in runner.read_jsonl(config.CLASSIFICATIONS))
    print(f"\nDONE — {prog.n} new classifications. Totals: {dict(dist)}")
    erroneous = dist.get("erroneous", 0)
    total = sum(dist.values())
    if total and erroneous / total > 0.25:
        print(f"  WARNING: {erroneous/total:.0%} erroneous. The paper retained 15345 of "
              f"27000 responses (~43% dropped), but a rate far above that usually means "
              f"the judge is failing to parse, not that the subject is refusing. "
              f"Inspect ai_answer_preview before trusting the analysis.")


if __name__ == "__main__":
    main()
