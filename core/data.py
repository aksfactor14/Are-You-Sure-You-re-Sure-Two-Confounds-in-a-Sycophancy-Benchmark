"""Loads multiple-choice questions (MMLU, ARC, OpenBookQA, CommonsenseQA) and turns them into
prompts."""
import json
import random
from core import config

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _norm_arc_like(ex, source, idx):
    """ARC / OpenBookQA / CommonsenseQA: choices={label|text}, answerKey letter."""
    q = ex.get("question") or ex.get("question_stem")
    labels = ex["choices"]["label"]
    texts = ex["choices"]["text"]
    # some ARC items use numeric labels 1-4; remap to A-D positionally
    if any(l.isdigit() for l in labels):
        labels = [LETTERS[i] for i in range(len(texts))]
        gold_idx = ex["choices"]["label"].index(ex["answerKey"]) if ex["answerKey"] in ex["choices"]["label"] else None
        gold = LETTERS[gold_idx] if gold_idx is not None else None
    else:
        gold = ex["answerKey"]
    options = list(zip(labels, texts))
    return {"source": source, "qid": f"{source}:{idx}", "question": q,
            "options": options, "gold": gold}


def _norm_mmlu(ex, source, idx):
    """MMLU: choices=list[str], answer=int."""
    texts = ex["choices"]
    labels = [LETTERS[i] for i in range(len(texts))]
    options = list(zip(labels, texts))
    gold = LETTERS[ex["answer"]]
    return {"source": source, "qid": f"{source}:{idx}", "question": ex["question"],
            "options": options, "gold": gold}


def load_candidates(source_key, n, seed):
    from datasets import load_dataset          # lazy: only selection needs the Hub
    name, cfg, split = config.SOURCES[source_key]
    ds = load_dataset(name, cfg, split=split)
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    items = []
    for i in idxs[: n * 2]:  # headroom for malformed rows
        ex = ds[i]
        try:
            if source_key == "mmlu":
                it = _norm_mmlu(ex, source_key, i)
            else:
                it = _norm_arc_like(ex, source_key, i)
            if it["gold"] is None or not it["question"] or len(it["options"]) < 3:
                continue
            items.append(it)
        except Exception:
            continue
        if len(items) >= n:
            break
    return items


def format_question(item):
    lines = [item["question"], ""]
    for label, text in item["options"]:
        lines.append(f"{label}) {text}")
    lines.append("")
    lines.append("Answer with only the single letter of the correct option.")
    lines.append("Answer:")
    return "\n".join(lines)


def format_question_open(item):
    lines = [item["question"], ""]
    for label, text in item["options"]:
        lines.append(f"{label}) {text}")
    return "\n".join(lines)


def load_item_file(path):
    if str(path).endswith(".jsonl"):
        with open(path, encoding="utf8") as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(path, encoding="utf8") as f:
        return json.load(f)


def wrong_option(item):
    for label, _ in item["options"]:
        if label != item["gold"]:
            return label
    raise ValueError("no distractor found")
