"""Free-form step: samples the questions and saves the model's first answer to each."""
import sys as _sys
import pathlib as _pl
import os as _os

_ROOT = _pl.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT), str(_ROOT / "experiments")]

from syceval import config, data, prompts, runner  # noqa: E402


def sample_items():
    """Freeze the item sample to disk. Idempotent -- existing qids are kept as-is."""
    existing = runner.read_jsonl(config.ITEMS)
    if existing:
        print(f"reusing {len(existing)} frozen items from {config.ITEMS}")
        return existing

    print(f"sampling {config.N_PER_SOURCE} items per source (seed={config.SEED}) ...")
    items = data.load_all()

    shard = int(_os.environ.get("SYCEVAL_SHARD", -1))
    n_shards = int(_os.environ.get("SYCEVAL_N_SHARDS", 0))
    if n_shards > 1 and shard >= 0:
        total = len(items)
        items = items[shard::n_shards]
        print(f"shard {shard}/{n_shards}: {len(items)} of {total} items", flush=True)

    with runner.Appender(config.ITEMS) as w:
        for it in items:
            w.write(it)
    from collections import Counter
    print(f"froze {len(items)} items  {dict(Counter(i['source'] for i in items))}")
    return items


def main():
    runner.stage_header(
        "SYCEVAL STAGE 1 — items + initial inquiry responses",
        subject_model=config.SUBJECT_MODEL,
        temperature=config.SUBJECT_TEMP,
        max_tokens=config.SUBJECT_MAX_TOKENS,
        out=config.INITIAL,
    )
    items = sample_items()
    if "--items-only" in _sys.argv:
        print("--items-only: stopping before the model is loaded.")
        return

    done = runner.done_keys(config.INITIAL, ["qid"])
    todo = [it for it in items if (it["qid"],) not in done]
    print(f"{len(done)} initial responses already collected, {len(todo)} to go")
    if not todo:
        print("nothing to do.")
        return

    m = runner.load_model(config.SUBJECT_MODEL)
    stamp = {"subject_model": config.SUBJECT_MODEL,
             "subject_temp": config.SUBJECT_TEMP}
    prog = runner.Progress(len(todo), "initial")

    # Coarse chunks; runner.chat() batches by config.SUB inside. See syceval_main.py.
    CHUNK = config.SUB * 16
    with runner.Appender(config.INITIAL, stamp=stamp) as w:
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            outs = runner.chat(
                m, [prompts.initial_messages(it["question"]) for it in chunk],
                max_tokens=config.SUBJECT_MAX_TOKENS, temp=config.SUBJECT_TEMP)
            for it, out in zip(chunk, outs):
                w.write({"qid": it["qid"], "source": it["source"],
                         "response": out, "n_chars": len(out)})
                prog.tick(f"{it['qid']} {len(out)}ch")
            runner.release()

    print(f"\nDONE — wrote {prog.n} initial responses to {config.INITIAL}")
    print("next: python experiments/syceval_judge.py")


if __name__ == "__main__":
    main()
