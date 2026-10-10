# JRC-MoE v4 runbook

Use only the existing `reflex-llada` environment and cached pinned checkpoint. The CLI forces offline model loading during audit so it cannot silently download weights. Work from an isolated worktree under `/home/new_drive`, not the dirty server checkout. Keep all outputs under the configured stage roots.

```bash
source /home/new_drive/rashidi/venvs/reflex-llada/bin/activate
cd /home/new_drive/rashidi/JRC-MoE-v4
PYTHONPATH="$PWD/src" python -m jrc_moe.cli jrc-v4 audit --config configs/jrc_v4/pilot.yaml --output-dir outputs/jrc_v4/audit
python -m unittest discover -s tests -p 'test_jrc_v4_*.py' -v
python -m jrc_moe.cli jrc-v4 smoke --config configs/jrc_v4/pilot.yaml --max-prompts 3 --max-branches 12 --max-gpu-hours 2 --output-dir outputs/jrc_v4/smoke
python -m jrc_moe.cli jrc-v4 pilot --config configs/jrc_v4/pilot.yaml --max-prompts 50 --max-branches 200 --max-gpu-hours 24 --max-wall-hours 30 --max-output-gb 10 --resume --output-dir outputs/jrc_v4/pilot
python -m jrc_moe.cli jrc-v4 analyze --input-dir outputs/jrc_v4/pilot
python -m jrc_moe.cli jrc-v4 preregister --input-dir outputs/jrc_v4/pilot --output docs/jrc_v4/preregistration_draft.yaml
```

`audit` records environment/cache/source fingerprints and must pass the pinned model, router, Fixed8, legal-K, and replay gates. `smoke` and `pilot` write a blocker manifest and start no branches unless the audit manifest contains explicit G0 evidence. `signal` requires pilot outcomes; `communities` additionally requires G1; `allocate` requires G1; `systems` requires G3; resource caps apply to every stage. `--resume` requires the same recorded cap values and accumulates prior branch/GPU-time usage.

Never use the historical 50-prompt set for confirmatory claims. Do not enable attention claims until a verified post-QK-normalization/post-RoPE hook passes short-sequence fidelity. A per-forward budget-neutral exchange is not a full-trajectory matched-budget result.
