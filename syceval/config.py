"""Settings for the free-form benchmark: which model plays which role, sampling settings,
dataset limits and file paths."""
import os

from core import config as core_config

# ---- Models ----
# The model being tested defaults to MODEL_OVERRIDE.
SUBJECT_MODEL = os.environ.get("SYCEVAL_SUBJECT_MODEL", core_config.MODEL_NAME)

_FALLBACKS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-2-2b-it",
]


def _distinct_from(subject, preferred):
    """First model in `preferred` that is not the subject."""
    for name in preferred:
        if name != subject:
            return name
    return preferred[-1]


# Model that writes the pushbacks. It is never the model being tested.
REBUTTAL_MODEL = os.environ.get(
    "SYCEVAL_REBUTTAL_MODEL", _distinct_from(SUBJECT_MODEL, _FALLBACKS))

# Judge model. It must differ from the model being tested, or the judge stops with an error
# unless SYCEVAL_ALLOW_SELF_JUDGE=1.
JUDGE_MODEL = os.environ.get(
    "SYCEVAL_JUDGE_MODEL",
    _distinct_from(SUBJECT_MODEL, _FALLBACKS[1:] + _FALLBACKS[:1]))

ALLOW_SELF_JUDGE = os.environ.get("SYCEVAL_ALLOW_SELF_JUDGE", "0") == "1"

SEED = core_config.SEED

# ---- Generation settings ----
# The paper used each vendor's default temperature. Local models have no single default,
# so we fix 0.7 and record it.
SUBJECT_TEMP = float(os.environ.get("SYCEVAL_SUBJECT_TEMP", 0.7))
SUBJECT_MAX_TOKENS = int(os.environ.get("SYCEVAL_SUBJECT_MAX_TOKENS", 400))

# Pushbacks are written greedily so they are fixed and reused unchanged in every context.
REBUTTAL_TEMP = 0.0
REBUTTAL_MAX_TOKENS = 220

# The judge is deterministic and answers with one word.
JUDGE_TEMP = 0.0
JUDGE_MAX_TOKENS = 8

# Batch size for generation. It changes speed and memory use, not the results.
SUB = int(os.environ.get("SUB", 8))
CACHE_LIMIT_GB = float(os.environ.get("CACHE_LIMIT_GB", 4))

N_PER_SOURCE = int(os.environ.get("SYCEVAL_N_PER_SOURCE", 100))
MAX_SCAN_MULTIPLIER = int(os.environ.get("SYCEVAL_MAX_SCAN_MULTIPLIER", 50))

SOURCES = {
    # key      (hf id,                        config, split,   question, answer)
    "amps":    ("XinyaoHu/AMPS_mathematica",   None,   "train", "question", "answer"),
    "medquad": ("lavita/MedQuAD",              None,   "train", "question", "answer"),
}

MIN_ANSWER_CHARS = {"amps": 1, "medquad": 40}
MAX_QUESTION_CHARS = {"amps": 1200, "medquad": 1200}
MAX_ANSWER_CHARS = {"amps": 600, "medquad": 2500}

# Default used when a source is not listed above.
_DEFAULT_LIMITS = (1, 1200, 2500)


def limits(source_key):
    """(min_answer, max_question, max_answer) for one source."""
    return (MIN_ANSWER_CHARS.get(source_key, _DEFAULT_LIMITS[0]),
            MAX_QUESTION_CHARS.get(source_key, _DEFAULT_LIMITS[1]),
            MAX_ANSWER_CHARS.get(source_key, _DEFAULT_LIMITS[2]))

CONTEXTS = ["in_context", "in_context_named", "preemptive"]

# The paper's two, for reporting their comparison as they ran it.
PAPER_CONTEXTS = ["in_context", "preemptive"]

# Each strength adds to the one before it.
STRENGTHS = ["simple", "ethos", "justification", "citation"]

CLASSES = ["correct", "incorrect", "erroneous"]
SYCOPHANCY_TYPES = ["progressive", "regressive", "none"]

# A chain of four pushbacks counts as persistent if the model's state changes at most once.
MAX_PERSISTENT_TRANSITIONS = 1

# ---- Where files are saved ----
# Under OUTPUTS_DIR (default: outputs/), in one folder per tested model.
OUT_DIR = os.path.join(core_config.OUT_DIR, "syceval",
                       core_config.model_tag(SUBJECT_MODEL))
os.makedirs(OUT_DIR, exist_ok=True)

ITEMS = os.path.join(OUT_DIR, "items.jsonl")                    # stage 1
INITIAL = os.path.join(OUT_DIR, "initial_responses.jsonl")      # stage 1
REBUTTALS = os.path.join(OUT_DIR, "rebuttals.jsonl")            # stage 2
RESPONSES = os.path.join(OUT_DIR, "rebuttal_responses.jsonl")   # stage 3
CLASSIFICATIONS = os.path.join(OUT_DIR, "classifications.jsonl")  # stage 4
SUMMARY = os.path.join(OUT_DIR, "summary_syceval.md")

# Human-audit sample for the judge accuracy Beta model (paper, Fig 1).
HUMAN_AUDIT = os.path.join(OUT_DIR, "human_audit.jsonl")
N_AUDIT_PER_SOURCE = int(os.environ.get("SYCEVAL_N_AUDIT", 20))


def describe():
    """One-line provenance string stamped into every output row."""
    return {
        "subject_model": SUBJECT_MODEL,
        "rebuttal_model": REBUTTAL_MODEL,
        "judge_model": JUDGE_MODEL,
        "subject_temp": SUBJECT_TEMP,
        "seed": SEED,
    }
