# Are You Sure You're Sure? Two Confounds in a Sycophancy Benchmark

Code and data for the paper:

> **[Are You Sure You're Sure? Two Confounds in a Sycophancy Benchmark](https://arxiv.org/abs/2609.32867)**  
> *NeurIPS 2026 Workshop JUDGe (Oral)*

**Paper:** [arXiv:2609.32867](https://arxiv.org/abs/2609.32867)

This README explains how to set the repo up and how to reproduce every table in it.

## What the paper is about

A language model is *sycophantic* when it drops a correct answer because a user pushes
back. The benchmark SycEval measures this by scripting a pushback and counting how often
the model gives in. We show that two details of SycEval's prompt templates change the
result, so the benchmark measures them along with sycophancy:

1. **Naming.** SycEval's "preemptive" pushback names an answer ("I think the answer is
   B"). Its "in-context" pushback does not. So the benchmark's timing comparison is
   really also a naming comparison. We add the missing in-context pushback that names an
   answer and re-measure.
2. **Instruction position.** Where the "reply concisely" style instruction sits (in the
   first message or in the pushback) changes how often the model gives in, and the change
   goes in opposite directions for different models.

Terms used below:

- **Follow rate**: the share of the model's replies that end on the answer the user
  pushed. When the user pushes a wrong answer this is the "gave in" rate.
- **MCQ**: multiple-choice questions, scored by reading the chosen letter.
- **Free-form**: open answers (maths and medical questions) scored by a second model
  acting as judge, exactly as SycEval does.
- **pp**: percentage points.

## What is in the repo

| Folder or file | What it holds |
|---|---|
| `core/` | Shared code: loads models, builds MCQ items, shared settings |
| `syceval/` | Our re-implementation of the SycEval free-form benchmark (prompts, judge, statistics) |
| `experiments/` | Scripts that collect data (they run models) |
| `analysis/` | Scripts that turn collected data into the paper's tables (no model needed) |
| `outputs/` | All collected data and the summary tables made from it |
| `docs/` | Written description of the free-form pipeline and every prompt |

Some scripts in `experiments/` (probes, credentials, punctuation, `mechanistic/`) belong
to earlier work on typographic sycophancy and are **not used by this paper**. The
scripts the paper uses are listed in the tables below.

## Setup

You need Python 3.10 or newer.

```bash
git clone <this repository>
cd <repository folder>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then pick one of the two model back ends. You only need this to **collect new data**.
Re-making the tables from the data already in `outputs/` needs neither.

| You have | Install | Can run |
|---|---|---|
| Apple Silicon Mac | `pip install -r requirements-mlx.txt` | The 4-bit runs (Qwen2.5-3B, Qwen3-4B in the MCQ position test) |
| NVIDIA GPU with 24 GB (we used an A10G) | `pip install -r requirements-torch.txt` (install the CUDA build of PyTorch first, see pytorch.org) | The full-precision (bf16) runs: Llama-3.1-8B, Qwen3-4B at full precision, and all free-form runs |

The back end is chosen automatically. To force one, set `BACKEND=mlx` or `BACKEND=torch`.
4-bit runs only work with the Apple back end. With PyTorch the same model name loads the
full-precision version instead.

Models and datasets download from Hugging Face the first time they are needed. Nothing
needs a login: Llama is loaded from the open `unsloth/` copy.

Setting an environment variable (used in the commands below):

```bash
MODEL_OVERRIDE=some/model python script.py        # Mac / Linux / Git Bash
```
```powershell
$env:MODEL_OVERRIDE="some/model"; python script.py   # Windows PowerShell
```

The analysis scripts were checked with Python 3.10, numpy 2.2, pandas 2.3, scipy 1.15 and
statsmodels 0.14.

### Models used in the paper

| Short name | Hugging Face name (`MODEL_OVERRIDE` value) | Used for |
|---|---|---|
| Qwen2.5-3B, 4-bit | `mlx-community/Qwen2.5-3B-Instruct-4bit` | MCQ naming, MCQ position |
| Qwen3-4B, 4-bit | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | MCQ position |
| Qwen3-4B, full precision | `Qwen/Qwen3-4B-Instruct-2507` | MCQ naming and position, free-form |
| Llama-3.1-8B, full precision | `unsloth/Meta-Llama-3.1-8B-Instruct` | MCQ naming and position, free-form |
| Qwen2.5-7B (writes the pushbacks and judges the free-form answers) | `Qwen/Qwen2.5-7B-Instruct` | Free-form only |

## Reproducing the tables

### A. Re-make the tables from the data in `outputs/` 

Every collected reply is saved in `outputs/`, so all tables can be rebuilt without
running any model. The commands below **overwrite the summary files in `outputs/`**.
To keep the shipped copies, work on a copy of the folder and point the scripts at it:

```bash
cp -r outputs outputs_copy
export OUTPUTS_DIR=$PWD/outputs_copy       # Windows PowerShell: $env:OUTPUTS_DIR="$PWD\outputs_copy"
```

**Naming, multiple choice**

```bash
MODEL_OVERRIDE=mlx-community/Qwen2.5-3B-Instruct-4bit python experiments/syceval_replication.py --analyze
MODEL_OVERRIDE=unsloth/Meta-Llama-3.1-8B-Instruct     python experiments/syceval_replication.py --analyze
```

Results are written to `summary_syceval_replication.md` (Qwen2.5-3B) and
`summary_syceval_replication__Meta-Llama-3.1-8B-Instruct.md` (Llama).

**Naming, free-form** (Llama and Qwen3-4B)

```bash
export SYCEVAL_JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct
SYCEVAL_SUBJECT_MODEL=Meta-Llama-3.1-8B-Instruct python analysis/analyze_syceval.py
SYCEVAL_SUBJECT_MODEL=Qwen3-4B-Instruct-2507     python analysis/analyze_syceval.py
```

Results are written to `syceval/<model>/summary_syceval.md` inside the outputs folder.

**Instruction position, multiple choice**

```bash
MODEL_OVERRIDE=mlx-community/Qwen2.5-3B-Instruct-4bit python experiments/instruction_position.py --analyze
MODEL_OVERRIDE=unsloth/Meta-Llama-3.1-8B-Instruct     python experiments/instruction_position.py --analyze
python analysis/instruction_position_crossmodel.py
```

The Qwen3-4B (4-bit) table is already saved as `summary_instruction_position_q3.md`.
Its data file was renamed (`instruction_position_q3.jsonl`) so that it does not collide
with Qwen2.5-3B's, which means it is read only by the cross-model script, not by a
single-model command.

**Instruction position, free-form**

```bash
python analysis/freeform_position_crossmodel.py
SYCEVAL_SUBJECT_MODEL=Meta-Llama-3.1-8B-Instruct python experiments/instruction_position_freeform_analysis.py
SYCEVAL_SUBJECT_MODEL=Qwen3-4B-Instruct-2507     python experiments/instruction_position_freeform_analysis.py
```

**Which command makes which table or figure**

| Paper item | Command above | Where to look in the output |
|---|---|---|
| Table 1, rows Qwen2.5-3B and Llama MCQ | Naming, multiple choice | Section "H2" of each summary (naming = `IC_NAMED − IC_SIMPLE`, timing = `PRE_SIMPLE − IC_NAMED`), and the two ablation rows in Section "H3" |
| Table 1, rows Llama and Qwen3-4B free-form | Naming, free-form | Section 2b of `summary_syceval.md` |
| Table 4 (Qwen2.5-3B per-arm rates) | Naming, multiple choice (Qwen2.5-3B) | Section "H1" |
| Table 6 (Llama per-arm rates) | Naming, multiple choice (Llama) | Section "H1" |
| Table 7 (strength ladder) | Naming, free-form (Llama) | Section 3 of `summary_syceval.md` |
| Table 8 and the three contrasts | Naming, free-form (Llama) | Section 2b |
| Figure 3 (naming effect by strength) | Naming, free-form (Llama) | Per-strength table in Section 2b |
| Table 9 (free-form position) | `analysis/freeform_position_crossmodel.py` | Section 1 |
| Figure 2, free-form panel, and its cross-model difference | `analysis/freeform_position_crossmodel.py` | Section 3 |
| Figure 2, multiple-choice panel, and the p = 0.0059 result | `analysis/instruction_position_crossmodel.py` | Table "Qwen2.5-3B vs Qwen3-4B (36 shared items)", row POSITION |
| Table 10 (all five MCQ position arms) | `analysis/instruction_position_crossmodel.py` | First table |
| Table 11 (within-model contrasts) | Single-model position summaries | Contrast table of each `summary_instruction_position*.md` |
| Tables 12 and 13 (Llama reply length and contrasts) | Instruction position, multiple choice (Llama) | `summary_instruction_position__Meta-Llama-3.1-8B-Instruct.md` |
| Table 17 (equivalence tests, position) | `analysis/freeform_position_crossmodel.py` (Qwen3-4B free-form), `analysis/instruction_position_crossmodel.py` (the two MCQ rows) | Section 4, and the section on whether each contrast is a null |
| Table 15 (every p-value) | All of the above | The p column of each table |
| Table 2 and Appendix A (every prompt) | Nothing to run | `docs/SYCEVAL_PROMPTS.md`, `syceval/prompts.py`, `experiments/syceval_replication.py` (list `ARMS`), `experiments/instruction_position.py` (dictionary `TAIL`) |


**Small differences you may see.** The Qwen3-4B median reply lengths in Table 9 differ
from what the script prints by a few characters (the script gives 1489, 1504, 796, 332
and 718 for five of them). The conclusions do not change.

### B. Collect the data again

Replies are sampled at temperature 0.7, so a new run gives numbers close to, but not
identical to, the paper. Every script saves after each item and picks up where it
stopped if you restart it.

Run the three kinds of experiment separately:

**1. Naming, multiple choice.** First find items the model answers right and wrong, then
collect.

```bash
export MODEL_OVERRIDE=unsloth/Meta-Llama-3.1-8B-Instruct
python experiments/select_items.py              # writes items_screen__… and items_wrong__…
python experiments/syceval_replication.py       # collects, then prints the tables
```

The item lists for Qwen2.5-3B are already in `outputs/` (`q25_items.json`,
`q25_wrongitems.json`).

**2. Instruction position, multiple choice.** Uses the same fixed items and arguments for
every model (`outputs/q25_items.json`, `outputs/q25_arguments.json`).

```bash
MODEL_OVERRIDE=<model name> python experiments/instruction_position.py
```

**3. Free-form (naming, then position).** Run these in order. Details are in
`docs/SYCEVAL.md`.

```bash
export SYCEVAL_SUBJECT_MODEL=Meta-Llama-3.1-8B-Instruct      # or Qwen3-4B-Instruct-2507
export SYCEVAL_REBUTTAL_MODEL=Qwen/Qwen2.5-7B-Instruct
export SYCEVAL_JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct

python experiments/syceval_preflight.py          # checks the setup, stops on a problem
python experiments/syceval_select.py             # subject model answers 200 questions
python experiments/syceval_judge.py              # judge grades those answers
python experiments/syceval_rebuttals.py          # writes the pushbacks
python experiments/syceval_main.py               # subject model replies to every pushback
python experiments/syceval_judge.py              # judge grades the replies
python analysis/analyze_syceval.py               # tables
python experiments/instruction_position_freeform.py   # the position test, 20 replies per cell
```

Set the judge explicitly as shown. If you leave it out, a smaller default judge is used
and the results will not match the paper.

**Two things that go wrong.**

- Qwen2.5-3B and Qwen3-4B share output file names. Collecting for both in the same
  `outputs/` folder mixes them. Give each its own copy with `OUTPUTS_DIR`, as in part A.
- Each run writes to `outputs/`. A run stops with a message if the item lists for that
  model are missing; follow the message.


## Citation

If you use this code or find our work helpful, please cite the following paper:

```bibtex
@article{gupta2026you,
  title={Are You Sure You're Sure? Two Confounds in a Sycophancy Benchmark},
  author={Gupta, Atharv and Jindal, Akshat and Nigam, Lavanya and Sood, Aryan},
  journal={arXiv preprint arXiv:2609.32867},
  year={2026}
}
