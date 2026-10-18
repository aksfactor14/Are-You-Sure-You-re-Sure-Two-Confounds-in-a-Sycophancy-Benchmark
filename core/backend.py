"""Lets the same code run on Apple Silicon (MLX) or PyTorch: load a model, read next-word
scores, generate text."""
import os
import numpy as np

BACKEND = os.environ.get("BACKEND", "auto")

if BACKEND == "auto":
    try:
        import mlx.core  # noqa: F401
        BACKEND = "mlx"
    except ImportError:
        BACKEND = "torch"

# MLX 4-bit conversions -> upstream repos that transformers can load
_TORCH_EQUIV = {
    "mlx-community/Qwen3-4B-Instruct-2507-4bit": "Qwen/Qwen3-4B-Instruct-2507",
    "mlx-community/Qwen2.5-3B-Instruct-4bit":    "Qwen/Qwen2.5-3B-Instruct",
    "mlx-community/gemma-2-2b-it-4bit":          "google/gemma-2-2b-it",
    "mlx-community/Llama-3.2-3B-Instruct-4bit":  "meta-llama/Llama-3.2-3B-Instruct",
}


def resolve_model_name(name):
    """Map an MLX repo id to its torch equivalent when running the torch backend."""
    if BACKEND == "torch":
        return _TORCH_EQUIV.get(name, name.replace("mlx-community/", "").replace("-4bit", ""))
    return name


if BACKEND == "mlx":
    import mlx.core as mx
    from mlx_lm import load as _load, generate as _gen, batch_generate as _bgen
    from mlx_lm.sample_utils import make_sampler

    class Backend:
        def __init__(self, name):
            self.model, self.tok = _load(name)

        def apply_template(self, messages, add_generation_prompt=True):
            return self.tok.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt)

        def encode(self, text):
            return self.tok.encode(text, add_special_tokens=False)

        def decode(self, ids):
            return self.tok.decode(ids)

    def last_logits(m, ids):
        """Next-token logits at the final position, as float64 numpy."""
        lg = m.model(mx.array(ids)[None])[:, -1, :].astype(mx.float32)
        return np.array(lg.tolist(), dtype=np.float64)[0]

    def generate_batch(m, prompts, max_tokens, temp=0.0, sub=None):
        """Completions for a list of tokenised prompts. `sub` caps the batch size."""
        sampler = make_sampler(temp=temp)
        sub = sub or len(prompts)
        out = []
        for i in range(0, len(prompts), sub):
            chunk = prompts[i:i + sub]
            try:
                rr = _bgen(m.model, m.tok, prompts=chunk, max_tokens=max_tokens,
                           sampler=sampler, verbose=False)
                out += list(rr.texts if hasattr(rr, "texts") else rr)
            except Exception:
                # Gemma-2's sliding-window mask cannot broadcast over a batch.
                out += [_gen(m.model, m.tok, prompt=p, max_tokens=max_tokens,
                             sampler=sampler, verbose=False) for p in chunk]
        return out

    def last_logprobs(m, ids):
        """log softmax over the full vocabulary at the final position."""
        lg = m.model(mx.array(ids)[None])[:, -1, :].astype(mx.float32)
        return np.array(mx.log(mx.softmax(lg, axis=-1))[0].tolist(), dtype=np.float64)

    def clear_cache():
        mx.clear_cache()

    def set_cache_limit(gb):
        mx.set_cache_limit(int(gb * 2 ** 30))

    def peak_memory_gb():
        return mx.get_peak_memory() / 2 ** 30

    def set_seed(s):
        mx.random.seed(s)


# ====================================================================== TORCH
else:
    import torch

    def _as_ids(x):
        if hasattr(x, "ids"):                       # tokenizers.Encoding
            return list(x.ids)
        if hasattr(x, "input_ids"):                 # BatchEncoding
            v = x.input_ids
            if hasattr(v, "tolist"):
                v = v.tolist()
            return list(v[0]) if v and isinstance(v[0], (list, tuple)) else list(v)
        if hasattr(x, "tolist"):                    # tensor / ndarray
            return list(x.tolist())
        return list(x)

    _DEVICE = ("cuda" if torch.cuda.is_available()
               else "mps" if getattr(torch.backends, "mps", None)
               and torch.backends.mps.is_available() else "cpu")

    def _pick_dtype():
        want = os.environ.get("SYC_DTYPE", "auto").lower()
        table = {"float16": torch.float16, "fp16": torch.float16,
                 "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                 "float32": torch.float32, "fp32": torch.float32}
        if want in table:
            return table[want]
        if _DEVICE == "cpu":
            return torch.float32
        if _DEVICE == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    class Backend:
        def __init__(self, name):
            name = resolve_model_name(name)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tok = AutoTokenizer.from_pretrained(name)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            if getattr(self.tok, "padding_side", None) != "left":
                self.tok.padding_side = "left"
            dtype = _pick_dtype()
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                dtype=dtype,
                device_map=_DEVICE if _DEVICE != "mps" else None,
            )
            if _DEVICE == "mps":
                self.model = self.model.to("mps")
            self.model.eval()

        def apply_template(self, messages, add_generation_prompt=True):
            return self.tok.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, tokenize=True)

        def encode(self, text):
            return self.tok.encode(text, add_special_tokens=False)

        def decode(self, ids):
            return self.tok.decode(ids)

    @torch.no_grad()
    def last_logits(m, ids):
        t = torch.tensor([_as_ids(ids)], device=m.model.device)
        return m.model(t).logits[0, -1, :].float().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def generate_batch(m, prompts, max_tokens, temp=0.0, sub=None):
        prompts = [_as_ids(p) for p in prompts]
        sub = sub or len(prompts)
        out = []
        for i in range(0, len(prompts), sub):
            chunk = prompts[i:i + sub]
            width = max(len(p) for p in chunk)
            pad = m.tok.pad_token_id
            # LEFT-pad so every sequence ends at the same position; generation
            # continues from the final token, which must be real for all rows.
            ids = torch.tensor([[pad] * (width - len(p)) + list(p) for p in chunk],
                               device=m.model.device)
            att = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in chunk],
                               device=m.model.device)
            pos = att.cumsum(dim=-1) - 1
            pos.masked_fill_(att == 0, 0)
            gen = m.model.generate(
                input_ids=ids, attention_mask=att, position_ids=pos,
                max_new_tokens=max_tokens,
                do_sample=temp > 0, temperature=temp if temp > 0 else None,
                top_p=1.0 if temp > 0 else None, pad_token_id=pad)
            out += [m.tok.decode(g[width:], skip_special_tokens=True) for g in gen]
        return out

    @torch.no_grad()
    def last_logprobs(m, ids):
        """log softmax over the full vocabulary at the final position."""
        t = torch.tensor([_as_ids(ids)], device=m.model.device)
        lg = m.model(t).logits[0, -1, :].float()
        return torch.log_softmax(lg, dim=-1).cpu().numpy().astype(np.float64)

    def clear_cache():
        if _DEVICE == "cuda":
            torch.cuda.empty_cache()

    def set_cache_limit(gb):
        pass  # torch manages its own allocator

    def peak_memory_gb():
        return (torch.cuda.max_memory_allocated() / 2 ** 30
                if _DEVICE == "cuda" else float("nan"))

    def set_seed(s):
        torch.manual_seed(s)
