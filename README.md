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

## Gated JRC-MoE v4 experiments

The new `jrc-moe jrc-v4` namespace keeps the REFLEX replication path separate. Begin with the offline audit; it records the environment, cache contents, historical Fixed8 input hash, and architecture details when the pinned model is already cached. It never downloads checkpoint weights. Smoke and pilot stages start no branches until the audit manifest records a passed G0.

```bash
jrc-moe jrc-v4 audit --config configs/jrc_v4/pilot.yaml --output-dir outputs/jrc_v4/audit
python -m unittest discover -s tests -p 'test_jrc_v4_*.py' -v
jrc-moe jrc-v4 smoke --config configs/jrc_v4/pilot.yaml --max-prompts 3 --max-branches 12 --output-dir outputs/jrc_v4/smoke
jrc-moe jrc-v4 pilot --config configs/jrc_v4/pilot.yaml --max-prompts 50 --max-branches 200 --max-gpu-hours 24 --max-wall-hours 30 --max-output-gb 10 --resume --output-dir outputs/jrc_v4/pilot
```

Run `audit`, then the bounded `smoke`, before Stage 1 `pilot`; pilot checks that the smoke used the same config, source commit, and audit manifest. The first pilot invocation may use `--resume` with an empty output directory; later invocations must match its recorded run identity and caps. See [docs/jrc_v4/runbook.md](docs/jrc_v4/runbook.md) for artifact handling. Community, allocation, and systems stages remain conditional on earlier measured evidence.

## Git workflow

The local repository and server checkout use `https://github.com/KiarashRashidi/JRC-MoE.git` as `origin`. Push and pull through the configured GitHub credentials on the machine doing the operation.

