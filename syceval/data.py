"""Loads the free-form questions (AMPS maths and MedQuAD medical) and samples them reproducibly."""
import random

from syceval import config

_LOADED = {}


def _load_split(source_key):
    """Fetch and cache one HF split. Imported lazily so prompt/stat tests run without
    `datasets` installed."""
    if source_key in _LOADED:
        return _LOADED[source_key]
    from datasets import load_dataset
    name, cfg, split, _, _ = config.SOURCES[source_key]
    ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
    _LOADED[source_key] = ds
    return ds


def _subtopic(source_key, ex):
    if source_key == "amps":
        main, sub = ex.get("main_topic"), ex.get("sub_topic")
        return f"{main}/{sub}" if main and sub else (sub or main or "unknown")
    return ex.get("question_type") or ex.get("qtype") or "unknown"


def _clean(s):
    return " ".join(str(s or "").split())


def _usable(source_key, question, answer):
    min_a, max_q, max_a = config.limits(source_key)
    if not question or not answer:
        return False
    if len(answer) < min_a:
        return False
    if len(question) > max_q:
        return False
    if len(answer) > max_a:
        return False
    return True


def load_items(source_key, n=None, seed=None):
    """Sample `n` usable question/answer pairs. A fixed seed makes the sample repeatable."""
    n = config.N_PER_SOURCE if n is None else n
    seed = config.SEED if seed is None else seed
    _, _, _, qfield, afield = config.SOURCES[source_key]
    ds = _load_split(source_key)

    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)

    max_scan = min(len(idxs), max(n * config.MAX_SCAN_MULTIPLIER, 2000))
    items, seen, scanned = [], set(), 0
    for i in idxs[:max_scan]:
        scanned += 1
        ex = ds[i]
        question = _clean(ex.get(qfield))
        answer = _clean(ex.get(afield))
        if not _usable(source_key, question, answer):
            continue
        if question in seen:          # MedQuAD repeats questions across source documents
            continue
        seen.add(question)
        items.append({
            "source": source_key,
            "qid": f"{source_key}:{i}",
            "subtopic": _subtopic(source_key, ex),
            "question": question,
            "answer": answer,
        })
        if len(items) >= n:
            break

    if len(items) < n:
        # Stop with an error rather than quietly continue with too few items.
        rate = len(items) / scanned if scanned else 0.0
        raise RuntimeError(
            f"{source_key}: asked for {n} items but only {len(items)} survived after "
            f"scanning {scanned} of {len(idxs)} rows ({rate:.0%} survival). "
            f"Check config.limits({source_key!r}) = {config.limits(source_key)} against "
            f"this dataset's real answer lengths, or raise MAX_SCAN_MULTIPLIER "
            f"(currently {config.MAX_SCAN_MULTIPLIER}) if the source is simply sparse.")
    print(f"[data] {source_key}: {len(items)} items from {scanned} rows scanned "
          f"({len(items)/scanned:.0%} survival)", flush=True)
    return items


def load_all(n=None, seed=None):
    """Both datasets, concatenated. AMPS first, then MedQuAD."""
    out = []
    for key in config.SOURCES:
        out += load_items(key, n=n, seed=seed)
    return out
