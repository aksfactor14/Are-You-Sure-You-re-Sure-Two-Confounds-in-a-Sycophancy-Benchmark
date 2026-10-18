"""Wraps a loaded model so scripts can generate text and read which answer letter it picks."""
import re
from core import backend
import numpy as np
from core import config

def parse_option_letter(out, labels):
    if not out:
        return None
    s = out.strip()
    up = {l.upper() for l in labels}

    # 1. the whole reply is essentially one letter
    bare = re.fullmatch(r"[^A-Za-z0-9]*([A-Za-z])[^A-Za-z0-9]*", s)
    if bare and bare.group(1).upper() in up:
        return bare.group(1).upper()

    # 2. explicit "answer is X" / "answer: X" / "option X" patterns
    for pat in (r"answer\s*(?:is|:)\s*\**\(?\s*([A-Za-z])\b",
                r"\boption\s*\**\(?\s*([A-Za-z])\b",
                r"^\s*\**\(?([A-Za-z])[\)\.\:]"):
        mt = re.search(pat, s, flags=re.IGNORECASE | re.MULTILINE)
        if mt and mt.group(1).upper() in up:
            return mt.group(1).upper()

    # 3. first STANDALONE single-letter token (not a letter inside a word)
    for tok in re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", s):
        if tok.upper() in up:
            return tok.upper()
    return None


class Model(backend.Backend):
    def __init__(self, name=config.MODEL_NAME, seed=config.SEED):
        backend.set_seed(seed)
        super().__init__(name)

    def _letter_ids(self, labels):
        return {l: self.tok.encode(l, add_special_tokens=False)[0] for l in labels}

    def answer_mcq(self, messages, labels):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True)
        logprobs = backend.last_logprobs(self, ids)
        lid = self._letter_ids(labels)
        letter_lp = np.array([logprobs[lid[l]] for l in labels])
        choice = int(letter_lp.argmax())
        p = np.exp(letter_lp - letter_lp.max()); p = p / p.sum()
        return labels[choice], float(p[choice]), float(letter_lp[choice])

    def gen(self, messages, max_tokens=12):
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True)
        return backend.generate_batch(self, [prompt], max_tokens, temp=0.0)[0]

    def gen_sample(self, messages, max_tokens=140, temp=0.7):
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True)
        return backend.generate_batch(self, [prompt], max_tokens, temp=temp)[0]

    def answer_letter_gen(self, messages, labels, max_tokens=6):
        out = self.gen(messages, max_tokens=max_tokens).strip()
        return parse_option_letter(out, labels), out

    def rate_1_10(self, messages, max_tokens=8):
        out = self.gen(messages, max_tokens=max_tokens)
        for tok in re.findall(r"\d+", out):
            v = int(tok)
            if 1 <= v <= 10:
                return v
        return None
