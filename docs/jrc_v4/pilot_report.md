# JRC-MoE v4 GPU pilot execution report

## Scope and disposition

The bounded Stage 1 pilot completed on the configured Ubuntu GPU server. Stage 0 (G0), the three-prompt smoke test, and the pilot all passed their execution and replay gates. The pilot produced 100 paired exchanges on 50 historical GSM8K development prompts. Every paired correctness effect was zero. This is an exploratory pilot with no confirmatory prompts; it does not establish H1/H2, a community benefit, or a speedup.

The first smoke attempt is preserved at `outputs/jrc_v4/smoke_attempt_01_negative_gpu_duration`. It detected a reversed CUDA event elapsed-time call and started no intervention branches. The timing direction was corrected and committed as `a62e164df870a6480b4c86325660716f17b31cc2`; the rerun below passed. No prior REFLEX results or files in the original server checkout were overwritten.

## Environment and provenance

- Server: configured SSH target `research-gpu`, Ubuntu 24.04.
- GPU: NVIDIA RTX 4090, 49,140 MiB reported VRAM; CUDA 12.8.
- Environment: `/home/new_drive/rashidi/venvs/reflex-llada`; Python 3.12.3, PyTorch 2.7.1+cu128, Transformers 4.53.2.
- Checkpoint: `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`, revision `783d3467f108d28ac0a78d3e41af16ab05cabd8d`, loaded offline from the existing cache.
- Experimental source commit: `a62e164df870a6480b4c86325660716f17b31cc2`, branch `feat/jrc-v4-intervention-pilot`.
- Follow-up CLI/reporting commit: `fb53e9d5a3ac485a4b7a49b8896760fcbf795c5b`; it changed analysis output routing and exit propagation, not the intervention or measured pilot.
- Pilot configuration SHA-256: `40925d46079175c0cb0ec27b74bf9e0becd92ce258e5c3e746fba8c3baa6a561`.
- Fixed8 diagnostic CSV SHA-256: `177942409f8dbf8ea0d54fe81c86c2021118a0d335d655adec6531410cb4cd26`.
- Original `/home/new_drive/rashidi/JRC-MoE` checkout remained on its existing `main` commit with its four pre-existing modified source files; work ran in `/home/new_drive/rashidi/JRC-MoE-v4`.

The LLaDA implementation warns that its bidirectional model sets attention masks to `None`. Attention features therefore remained disabled: no verified post-QK-normalization/post-RoPE hook or short-sequence fidelity result was available.

## Stage results

### Stage 0: audit and validation

G0 passed on the pinned checkpoint. The audit verified checkpoint/cache availability, architecture and router semantics, native-versus-patched behavior, legal-K forwards, legacy reproduction, fresh Fixed8 profiling, replay, and synchronized profiling. Before the pilot, the focused JRC-v4 suite ran 23 tests with 1 opt-in GPU integration test skipped. After the follow-up CLI/reporting commit, 24 core tests passed and the separate native patch/multiphase logits/router/legal-K GPU integration test passed in 31.15 seconds.

Audit artifacts: `outputs/jrc_v4/audit/manifest.json` and `outputs/jrc_v4/audit/console.log`.

On the follow-up commit, 24 core test cases passed in 5.70 seconds, followed by 1 pinned-checkpoint GPU integration test in 31.15 seconds. Logs: `outputs/jrc_v4/validation_fb53e9d_gpu.log` and `outputs/jrc_v4/validation_fb53e9d_gpu_integration.log`. The first follow-up discovery launch selected no tests due to a malformed shell pattern; its log is preserved as `outputs/jrc_v4/validation_fb53e9d.log`, and the corrected core and GPU test-file runs above completed successfully. The Windows-local full-suite attempt was not a valid environment check because the local Python lacks `lm_eval` and its default temporary directory is sandbox-protected. The new CLI subprocess test was moved to a workspace-local temporary directory and both new CLI tests then passed locally; target-environment suite results are from the configured Ubuntu server.

### Smoke test

The corrected smoke test passed with 3 prompts, 12 branches, and 6 paired exchanges. Both arms had 100% correctness on this tiny development-only sample; all six correctness effects were zero. It consumed 426.01 GPU seconds (0.11834 GPU hours). This checks execution and replay, not scientific utility.

Artifacts: `outputs/jrc_v4/smoke/manifest.json` and `outputs/jrc_v4/smoke/console.log`.

### Bounded Stage 1 pilot

The pilot passed all requested caps and completed 50/50 prompts, 50/50 captured states, 200/200 branches, and 100/100 paired exchanges, with zero recorded failures. It used 7,250.55 GPU seconds (2.01404 GPU hours), 7,270.62 wall seconds, and 5,836,890 bytes in the progress output counter, within the 24 GPU-hour, 30 wall-hour, 10 GB, 200-branch, and 50-prompt limits.

| Measurement | Control | Treatment |
|---|---:|---:|
| Correct rate | 0.80 | 0.80 |
| Mean NFE | 94.02 | 93.97 |
| Mean full-rollout executed expert-token pairs | 15,620,199.68 | 15,613,570.56 |

Across paired exchanges, effects were 0 negative, 100 zero, and 0 positive. The prompt-clustered treatment-minus-control mean was 0.0 with 95% bootstrap CI [0.0, 0.0] across 50 prompt clusters (2,000 resamples; seed 20261009). Branch runtime pooled across arms was median 33.8749 seconds / p95 54.5644 CUDA-event seconds and median 33.8764 / p95 54.5664 wall seconds.

Each exchange preserved the target layer's local pair budget. Full-trajectory pair totals were not assumed to match: some recorded deltas were nonzero. The pair-count averages above are descriptive only; no end-to-end speedup is claimed. This pilot did not compare a matched executed-pair quality frontier or a full system wall-clock frontier.

Pilot artifacts are under `outputs/jrc_v4/pilot/`, including `manifest.json`, `pilot_summary.json`, `paired_exchanges.csv`, `paired_exchange_records.jsonl`, `branch_outcomes.jsonl`, `features.jsonl`, `cost_ledger.csv`, `failures.jsonl`, `progress.json`, and `traces/`.

### Exploratory signal analysis and gates

The analysis used 35 development prompts (70 exchange rows) and 15 prompt-disjoint validation prompts (30 rows); there were no confirmatory prompts. Token-only and router-only ridge controls and a seeded random ranking were evaluated. All 30 held-out labels were zero, so all three predictors had zero nonzero-label coverage; their selected-effect gains were 0.0 with 95% CI [0.0, 0.0]. Kendall tau and sign concordance are undefined for these all-tied labels. These values do not demonstrate useful prediction.

G1 remains **blocked** because attention features and the attention-only/joint attention-router comparators are unavailable. G2 (communities), G3 (matched executed-pair utility), and G4 (measured end-to-end wall-clock frontier) were not run. No community or allocation improvement is claimed.

## Commands used

Run from the feature worktree after activating the existing server environment. The execution remained offline for model and dataset loading:

```bash
source /home/new_drive/rashidi/venvs/reflex-llada/bin/activate
cd /home/new_drive/rashidi/JRC-MoE-v4
export PYTHONPATH="$PWD/src"
export HF_HOME=/home/new_drive/rashidi/hf_cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JRC_MOE_GSM8K_DIR=/home/new_drive/rashidi/JRC-MoE/data/gsm8k
python -m jrc_moe.cli jrc-v4 audit --config configs/jrc_v4/pilot.yaml --output-dir outputs/jrc_v4/audit
python -m unittest discover -s tests -p 'test_jrc_v4_*.py' -v
JRC_MOE_RUN_GPU_INTEGRATION=1 python -m unittest discover -s tests -p=test_jrc_v4_gpu.py -v
python -m jrc_moe.cli jrc-v4 smoke --config configs/jrc_v4/pilot.yaml --max-prompts 3 --max-branches 12 --max-gpu-hours 2 --output-dir outputs/jrc_v4/smoke
python -m jrc_moe.cli jrc-v4 pilot --config configs/jrc_v4/pilot.yaml --max-prompts 50 --max-branches 200 --max-gpu-hours 24 --max-wall-hours 30 --max-output-gb 10 --resume --output-dir outputs/jrc_v4/pilot
python -m jrc_moe.cli jrc-v4 analyze --input-dir outputs/jrc_v4/pilot
python -m jrc_moe.cli jrc-v4 report
```

The initial `analyze` invocation wrote its intact outputs under `outputs/jrc_v4/analyze/`. The follow-up CLI fix makes `analyze` and `signal` share the canonical `outputs/jrc_v4/signal/` gate directory and propagate blocked-gate exit status. The analysis was rerun there; the original `analyze/` outputs and all pilot data remain preserved. The `signal/manifest.json` records exploratory analysis completion with G1 blocked.

The consolidated `outputs/jrc_v4/report/report.json` is **incomplete** and records `scientific_claims: none_established`; audit, smoke, and pilot passed, exploratory signal controls completed with G1 blocked, and communities/allocation/systems remain not run.

## Next steps

1. Verify the actual post-RoPE attention hook against the checkpoint source and a short-sequence numerical fidelity test before enabling attention features.
2. If that passes, preregister the analysis and collect a new pilot with non-degenerate labels; the historical GSM8K-50 prompts are development evidence only.
3. Re-evaluate G1 on prompt-disjoint data. Run G2 only if G1 passes; run G3/G4 only after their own community/allocation and matched-resource prerequisites pass.
