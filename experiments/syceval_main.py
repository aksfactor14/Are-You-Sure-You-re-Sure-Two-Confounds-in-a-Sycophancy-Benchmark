"""Free-form step: shows every pushback to the model being tested and saves its replies."""
import sys as _sys
import pathlib as _pl

_ROOT = _pl.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT), str(_ROOT / "experiments")]

from syceval import config, prompts, runner  # noqa: E402


def build_plan(limit=None):
    items = runner.index_by(runner.read_jsonl(config.ITEMS), ["qid"])
    runner.require(items, "no items — run syceval_select.py first")

    initial = {r["qid"]: r["response"] for r in runner.read_jsonl(config.INITIAL)}
    rebuttals = runner.read_jsonl(config.REBUTTALS)
    runner.require(rebuttals,
                   "no rebuttals — run syceval_rebuttals.py before this stage")

    done = runner.done_keys(config.RESPONSES, ["qid", "context", "strength"])
    plan = []
    for r in rebuttals:
        key = (r["qid"], r["context"], r["strength"])
        if key in done or r["qid"] not in items or r["qid"] not in initial:
            continue
        plan.append({
            "qid": r["qid"], "context": r["context"], "strength": r["strength"],
            "question": items[r["qid"]]["question"],
            "source": items[r["qid"]]["source"],
            "initial": initial[r["qid"]],
            "rebuttal": r["rebuttal"],
        })
    if limit:
        keep = sorted({p["qid"] for p in plan})[:limit]
        plan = [p for p in plan if p["qid"] in keep]
    return plan


def main():
    limit = None
    if "--limit" in _sys.argv:
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1])

    runner.stage_header(
        "SYCEVAL STAGE 3 — rebuttal responses",
        subject_model=config.SUBJECT_MODEL,
        temperature=config.SUBJECT_TEMP,
        cells_per_item=len(config.CONTEXTS) * len(config.STRENGTHS),
        limit=limit or "none",
        out=config.RESPONSES,
    )

    plan = build_plan(limit)
    print(f"{len(plan)} cells to collect")
    if not plan:
        print("nothing to do.")
        return

    m = runner.load_model(config.SUBJECT_MODEL)
    stamp = {"subject_model": config.SUBJECT_MODEL,
             "subject_temp": config.SUBJECT_TEMP}
    prog = runner.Progress(len(plan), "responses")

    CHUNK = config.SUB * 16
    with runner.Appender(config.RESPONSES, stamp=stamp) as w:
        for i in range(0, len(plan), CHUNK):
            chunk = plan[i:i + CHUNK]
            msgs = [prompts.rebuttal_messages(
                p["context"], p["question"], p["initial"], p["rebuttal"])
                for p in chunk]
            outs = runner.chat(m, msgs, max_tokens=config.SUBJECT_MAX_TOKENS,
                               temp=config.SUBJECT_TEMP)
            for p, out in zip(chunk, outs):
                w.write({"qid": p["qid"], "source": p["source"],
                         "context": p["context"], "strength": p["strength"],
                         "response": out, "n_chars": len(out)})
                prog.tick(f"{p['qid']} {p['context']}/{p['strength']}")
            runner.release()

    print(f"\nDONE — wrote {prog.n} responses to {config.RESPONSES}")
    print("next: python experiments/syceval_judge.py   (classifies these)")
    print("then: python analysis/analyze_syceval.py")


if __name__ == "__main__":
    main()
