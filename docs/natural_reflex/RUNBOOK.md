# Natural REFLEX observational study runbook

This runbook executes the preregistered observational study in `configs/natural_reflex/pilot.yaml`. It does not change native REFLEX decoding or expert budgets. Treat the attached v4 proposal as the scientific authority and the follow-up agent instructions as the operational protocol.

## 1. Freeze the execution identity

On `research-gpu`, enter the checked-out feature branch and activate the existing environment:

```bash
cd /path/to/JRC-MoE
git status --short --branch
git rev-parse HEAD
source /home/new_drive/rashidi/venvs/reflex-llada/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -V
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
nvidia-smi
```

Run only while the configured GPU is idle. The CLI refuses to start a GPU stage if an NVIDIA compute process is already active. Do not kill another user's process. Keep the original checkpoint and baseline outputs read-only.

## 2. Stage 0: audit, parity smoke, and 10-prompt pilot

From the repository root:

```bash
bash scripts/run_natural_reflex_bounded.sh
```

The wrapper runs the Q/K and SDPA attention audit, a three-prompt paired no-observer/observer instrumentation parity smoke in both modes, the first 10-prompt native REFLEX/Fixed8 pilot, prompt-disjoint CPU analysis, and the review bundle. Every stage writes a separate log under `outputs/natural_reflex/logs/` and the persistent cumulative resource ledger stays under the same output root.

An attention-audit exit code 2 means attention did not pass its runtime parity gate. In that case the wrapper proceeds only with the router-only observational path, and the report records attention as unavailable. Other audit failures stop the wrapper. Any failed token, commit-trace, fallback, K-allocation, or fixed-K equivalence parity stops collection.

The wrapper never resets the ledger. It stops on a failed stage; fix the documented cause, then rerun with the existing output directory so completed trajectories are validated and resumed. Do not delete or replace prior output directories to make a gate pass.

## 3. Stage 1 continuation gate

Review `analysis/natural_report.md`, `analysis/analysis.json`, `smoke/smoke_manifest.json`, `campaign_ledger.json`, and the per-prompt output summaries before expanding. Continue only if parity passed, trajectory validation passed, output and GPU-time limits remain, and collection did not expose a systematic instrumentation issue.

If those conditions hold, extend the same run to 50 prompts and reanalyze:

```bash
python -m jrc_moe.cli natural collect --config configs/natural_reflex/pilot.yaml --max-prompts 50 --resume
python -m jrc_moe.cli natural analyze --config configs/natural_reflex/pilot.yaml
python -m jrc_moe.cli natural package --config configs/natural_reflex/pilot.yaml
```

The 50-prompt historical GSM8K set is exploratory. It does not support confirmatory claims or an unbiased generalization estimate. Prompt identities, not positions or tokens, define the development/validation split and bootstrap clusters.

## 4. Analysis and interpretation gates

- D1–D14 are descriptive or observational. Report denominators, prompt clusters, and missing feature coverage.
- M0–M5 predict first commitment within 1, 2, or 4 future local forwards. Predictors must use features available at `t-1`; same-forward confidence and current commit decisions are outcomes only.
- Compare M3/M4/M5 to M2 on the same eligible held-out positions. Preserve missing-view status rather than treating absent attention/router data as zero.
- M6 and all temporal-community claims remain closed unless M5 first passes the development-only gate and the available budget supports the preregistered community analysis. The current collector does not construct online communities.
- Do not interpret sparse scaled QK scores as attention probabilities. Do not infer inference acceleration from fewer active tokens or router pairs. No treatment/counterfactual budget intervention is part of this study.
- Pair- or token-level rows are not independent replicates. Uncertainty uses prompt-cluster resampling (2,000 draws by default).

## 5. Resume and independent review

Collection transactions are written atomically and indexed with hashes. Re-run the same command with `--resume`; completed `(run, prompt, mode, seed)` trajectories are skipped only after their record hashes validate. `package` creates a full archive and, when needed, a deterministic review-lite archive capped at 100 MiB. Check `MANIFEST.txt` and `SHA256SUMS` before sharing. Never include model weights, credentials, or access tokens.

## 6. Exact commands

The individual commands are:

```bash
python -m jrc_moe.cli natural attention-audit --config configs/natural_reflex/pilot.yaml
python -m jrc_moe.cli natural smoke --config configs/natural_reflex/pilot.yaml --max-prompts 3 --resume
python -m jrc_moe.cli natural collect --config configs/natural_reflex/pilot.yaml --max-prompts 10 --resume
python -m jrc_moe.cli natural analyze --config configs/natural_reflex/pilot.yaml
python -m jrc_moe.cli natural package --config configs/natural_reflex/pilot.yaml
```

Set `PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"` in the active shell before issuing these commands, as the wrapper does automatically. This ensures the selected worktree's implementation is imported even if the environment has another checkout installed in editable mode.

Override `--out` only to intentionally create a separate campaign. Use the same config, input CSV, checkpoint revision, and source commit for every resumed stage.
