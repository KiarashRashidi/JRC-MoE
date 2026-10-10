# JRC-MoE v4 runbook

Use only the existing `reflex-llada` environment and cached pinned checkpoint/GSM8K data. The audit and rollout commands force Hugging Face model and dataset loading offline, so missing local inputs fail before a download. The existing `/home/new_drive/rashidi/JRC-MoE` checkout was observed with four modified source files; preserve it. From an SSH session, inspect its current status, then create a separate worktree on the pushed feature branch under `/home/new_drive`:

```bash
git -C /home/new_drive/rashidi/JRC-MoE status --short
git -C /home/new_drive/rashidi/JRC-MoE fetch origin feat/jrc-v4-intervention-pilot
git -C /home/new_drive/rashidi/JRC-MoE worktree add --detach /home/new_drive/rashidi/JRC-MoE-v4 origin/feat/jrc-v4-intervention-pilot
git -C /home/new_drive/rashidi/JRC-MoE-v4 switch -c feat/jrc-v4-intervention-pilot --track origin/feat/jrc-v4-intervention-pilot
```

If that worktree path already exists, inspect its branch/status and reuse it only if it points at the intended commit. Keep all outputs under the configured stage roots.

```bash
source /home/new_drive/rashidi/venvs/reflex-llada/bin/activate
cd /home/new_drive/rashidi/JRC-MoE-v4
set -e
export PYTHONPATH="$PWD/src"
python -m jrc_moe.cli jrc-v4 audit --config configs/jrc_v4/pilot.yaml --output-dir outputs/jrc_v4/audit
python -m unittest discover -s tests -p 'test_jrc_v4_*.py' -v
python -m jrc_moe.cli jrc-v4 smoke --config configs/jrc_v4/pilot.yaml --max-prompts 3 --max-branches 12 --max-gpu-hours 2 --output-dir outputs/jrc_v4/smoke
python -m jrc_moe.cli jrc-v4 pilot --config configs/jrc_v4/pilot.yaml --max-prompts 50 --max-branches 200 --max-gpu-hours 24 --max-wall-hours 30 --max-output-gb 10 --resume --output-dir outputs/jrc_v4/pilot
python -m jrc_moe.cli jrc-v4 analyze --input-dir outputs/jrc_v4/pilot
python -m jrc_moe.cli jrc-v4 preregister --output-dir docs/jrc_v4/preregistration_draft.yaml
```

`audit` records environment/cache/source fingerprints and must pass the pinned model, router, Fixed8, legal-K, and replay gates. `smoke` requires G0; Stage 1 `pilot` additionally requires a completed smoke from the same config, commit, and audit manifest. Both write blocker manifests and start no branches when prerequisites fail. `signal` requires pilot outcomes; `communities` additionally requires G1; `allocate` requires G1; `systems` requires G3; resource caps apply to every stage. `--resume` can initialize an empty output directory, then requires the same recorded run identity and caps for continuation while accumulating prior branch/GPU-time usage.

`analyze` is an alias for `signal` and writes to the same canonical `outputs/jrc_v4/signal/` gate directory. A blocked scientific gate is recorded in its manifest and returns a nonzero process exit status.

Never use the historical 50-prompt set for confirmatory claims. Do not enable attention claims until a verified post-QK-normalization/post-RoPE hook passes short-sequence fidelity. A per-forward budget-neutral exchange is not a full-trajectory matched-budget result.
