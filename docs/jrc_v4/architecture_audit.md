# JRC-MoE v4 architecture audit

## Status

Static source review is complete; the required live checkpoint/remote-code audit is **blocked**. The Ubuntu server was reachable for an initial environment inventory, but subsequent SSH inspection timed out. No checkpoint weights, server cache contents, or fixed8 input path were confirmed in the server checkout. The current GitHub `main` commit matched the clean local source at `c6be22116250ee0d709437a8bbf3feededff3eb0` when checked. The server checkout was at `3c05ee32659b1f26f2a653259232fee203725409` with four modified source files; that working tree was left untouched.

## Static project findings

- `src/jrc_moe/config.py` pins `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` at revision `783d3467f108d28ac0a78d3e41af16ab05cabd8d`; the replication constants specify 16 MoE layers, 64 experts, and native top-8.
- `src/jrc_moe/model.py` loads remote code with `trust_remote_code=True`, FP16 weights, and `device_map="auto"`. It checks 16 layers, 64 experts, top-8, softmax scores, and no top-K probability normalization.
- `src/jrc_moe/routing.py` patches every module whose class name is `LLaDAMoESparseMoeBlock`. Its current allocation is a single positionwise K vector shared across layers. It computes softmax scores, adds `expert_bias` if present, takes top-k of those post-addition scores, and multiplies selected scores by expert outputs without renormalizing when `norm_topk_prob=False`. It has no explicit layer-indexed override or replay API. Runtime bias behavior, scaling, residual/shared paths, and exact equality to the pinned remote code remain unverified.
- Existing REFLEX K values through 14 are preserved. The new intervention protocol separately restricts K to `[2,8]` and begins from the proposal's operational K=6 schedule.
- Historical Fixed8 diagnostic input was found locally at `data/diagnostic_50_results.csv`: 50 rows, SHA-256 `177942409f8dbf8ea0d54fe81c86c2021118a0d335d655adec6531410cb4cd26`. The committed prior REFLEX summary records Fixed8 39/50, RRB 36/50, FGER 42/50, and REFLEX 39/50. These remain prior development evidence; neither the server-side CSV copy nor a fresh same-hardware baseline was confirmed.
- Ordered live MoE paths, gate tensor shapes, direct expert output comparisons, exact attention Q/K paths and head dimensions, positional transforms, mask implementation, attention backend, cache behavior, and remote-code hashes: **not observed yet**. The new audit CLI records them when the pinned checkpoint is available offline.

## Server environment observed on 2026-10-10

- Host `ubuntu24-server`; configured repository path `/home/new_drive/rashidi/JRC-MoE`; original checkout branch `main`, commit `3c05ee32659b1f26f2a653259232fee203725409`.
- Original server checkout had local modifications in `analysis.py`, `data.py`, `model.py`, and `runner.py`; no files were changed there.
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, 49,140 MiB total and 48,510 MiB free at observation; 0% utilization and no process reported by `nvidia-smi`.
- Driver 595.71.05; `nvidia-smi` reported CUDA 13.2; server PyTorch `2.7.1+cu128` with CUDA runtime 12.8; `nvcc` toolkit 12.0; Transformers 4.53.2; Python 3.12.3; huggingface-hub 0.33.4.
- Memory: 108 GiB total, 105 GiB available. Root filesystem: 97 GiB total, 1.4 GiB free. `/home/new_drive`: 492 GiB total, 439 GiB free.
- The configured venv exists at `/home/new_drive/rashidi/venvs/reflex-llada`; the hub default reported by that Python environment is `/home/rashidi/.cache/huggingface/hub`. `HF_HOME` and `TRANSFORMERS_CACHE` were unset. Cached checkpoint files were not confirmed.

## G0 consequence

G0 is **blocked** until a read-only server session confirms the cached model and runs the pinned remote-code inspection, direct K=8 parity, legal K forwards, and replay tests. Do not use the current internet-facing model `main` metadata as a substitute for the pinned server revision.
