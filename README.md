# JRC-MoE

Clean Python project for the paper-faithful REFLEX component replication in the supplied notebook, targeting the pinned LLaDA-MoE checkpoint and 50-prompt GSM8K subset.

## What it reproduces

- The notebook's `rrb_only`, `fger_only`, and `reflex` modes, including its frozen RRB/FPS-FGER settings and variable-K MoE forward patch.
- The frozen canonical numeric grader and deterministic 5-shot GSM8K contexts.
- Per-forward traces, resumable per-prompt results, and summary diagnostics stored on the machine running the experiment.
- The paper's stated fixed8 comparison is an input artifact. Supply the validated CSV; this project does not invent or download a substitute.

The notebook also contains Google Drive authentication, Colab setup, and direct Drive persistence. Those are notebook-host conveniences. This project uses local paths so it runs on the GPU server and keeps checkpoints in the project output directory.

## Install on the GPU server

```bash
source /home/new_drive/rashidi/venvs/reflex-llada/bin/activate
cd /home/new_drive/rashidi/JRC-MoE
python -m pip install -e .
```

The model is pinned to `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` revision `783d3467f108d28ac0a78d3e41af16ab05cabd8d`. Model access may require Hugging Face credentials and a populated server cache.

## Run

Place the validated Fixed8 diagnostic CSV at a local path, then run:

```bash
jrc-moe run --fixed8-csv /path/to/validated_fixed8.csv --output-dir outputs/reflex_paper_replication50
```

The run checks input schema and prompt-token counts, model architecture, routing allocation invariants, and per-forward traces. It resumes successful `(test_idx, mode)` rows from `results.jsonl`; traces and final summary files are written beside it. Use `--limit N` for an explicitly partial pilot. A partial run is not the 50×3 replication.

## Git workflow

The local repository and server checkout use `https://github.com/KiarashRashidi/JRC-MoE.git` as `origin`. Push and pull through the configured GitHub credentials on the machine doing the operation.

