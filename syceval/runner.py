"""Shared helpers for the free-form steps: chat with a model, read and append result files,
show progress."""
import json
import os
import sys
import time

from syceval import config


# ------------------------------------------------------------------- models
def load_model(name):
    """Load one model by canonical id. Only one should be resident at a time --
    each stage is a separate script precisely so that holds."""
    from core import backend
    from core.model import Model
    backend.set_cache_limit(config.CACHE_LIMIT_GB)
    m = Model(name=name)
    print(f"loaded {name}  (backend={backend.BACKEND})", flush=True)
    return m


def release():
    from core import backend
    backend.clear_cache()


def chat(m, messages_list, max_tokens, temp, sub=None):
    """Batched chat completion. `messages_list` is a list of message lists;
    returns one string per element, in order."""
    from core import backend
    sub = sub or config.SUB
    ids = [m.tok.apply_chat_template(ms, add_generation_prompt=True)
           for ms in messages_list]
    out = []
    for i in range(0, len(ids), sub):
        out += backend.generate_batch(m, ids[i:i + sub], max_tokens, temp=temp)
    return [t.strip() for t in out]


def chat_one(m, messages, max_tokens, temp):
    return chat(m, [messages], max_tokens, temp)[0]


# -------------------------------------------------------------------- JSONL
def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def index_by(rows, keys):
    if len(keys) == 1:
        k = keys[0]
        return {r[k]: r for r in rows}
    return {tuple(r[k] for k in keys): r for r in rows}


def done_keys(path, keys):
    """Resume set: which (key tuple)s are already on disk."""
    return {tuple(r[k] for k in keys) for r in read_jsonl(path)}


class Appender:
    """Writes one row at a time to a file and adds the same extra fields (e.g. the model name)
    to every row."""

    def __init__(self, path, stamp=None):
        self.path = path
        self.stamp = stamp or {}
        self._f = None
        self.n = 0

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._f = open(self.path, "a", encoding="utf8")
        return self

    def write(self, row):
        row = {**row, **self.stamp}
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()
        self.n += 1

    def __exit__(self, *exc):
        if self._f:
            self._f.close()


class Progress:
    """Uniform '[12/400] 0.8/s eta 7m' logging, so a six-hour stage is legible."""

    def __init__(self, total, label):
        self.total, self.label, self.t0, self.n = total, label, time.time(), 0

    def tick(self, extra=""):
        self.n += 1
        el = time.time() - self.t0
        rate = self.n / el if el > 0 else 0
        eta = (self.total - self.n) / rate if rate > 0 else 0
        print(f"[{self.label}] {self.n}/{self.total}  {rate:.2f}/s  "
              f"eta {eta/60:.0f}m  {extra}", flush=True)


def require(condition, message):
    """Hard gate. Stages refuse to collect against a broken precondition rather
    than producing a file that looks fine and means nothing."""
    if not condition:
        print(f"\n  FAILED: {message}\n", file=sys.stderr)
        sys.exit(1)


def stage_header(title, **kv):
    print("=" * 74)
    print(title)
    for k, v in kv.items():
        print(f"   {k:22s} {v}")
    print("=" * 74, flush=True)
