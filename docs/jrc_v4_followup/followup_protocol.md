# JRC-MoE v4 bounded follow-up protocol

## Scientific question and fixed rollout

The prior 100 paired v4 exchanges retained unchanged final correctness while showing selected-position logit changes, some aggregate refinement-trace divergence, and three decoded-string changes. The follow-up measures whether those downstream effects vary with intervention dose, candidate context size, and the number of changed positions. It is development-only and does not test confirmatory accuracy, community superiority, or inference acceleration.

Every state is captured immediately before one decoder forward under uniform baseline K=5. A treatment changes only the selected target-layer position budgets, with integer deltas summing to zero and no clipping. The saved control and treatment states share the same RNG snapshot. After the target decision, both arms continue with fixed K=8 and the unchanged REFLEX commit/fallback procedure through the final 256 generated tokens. The target layer's executed expert-token-pair count must match exactly; whole-trajectory pair totals are measured independently and may differ.

## Factors and predeclared conditions

The action slate records three separate quantities:

- `m`: candidate tokens considered, with nested candidate groups of 2, 5, 8, or 16 unresolved positions.
- `c`: positions whose K value changes.
- `A`: total K transferred to recipient positions (the absolute donor changes also sum to A).

The uniform development slate contains `m=2, c=2, A=1/2/3`; `m=5, c=2, A=2`; `m=5, c=4, A=2/4`; and `m=8/16, c=4, A=2`. At `m=2, A=2`, the exact action is `5+5 -> 7+3`. At fixed `m=5, A=2`, `c=2` is compared with `c=4`; at fixed `A=2, c=4`, candidate context sizes 5, 8, and 16 are compared. The magnitude ladder at `m=2` and the `m=5,c=4` dose comparison are kept separate from those context comparisons.

Two additional `m=16,c=4,A=4` selection arms use only pre-decision token confidence (token-only) or pre-decision router admission margin. Missing features or infeasible K values are recorded as explicit infeasible conditions. Uniform action probabilities and nested candidate-set probabilities are saved; deterministic predictor-selected arms do not receive an invented propensity.

The bounded sensitivity run uses 10 development prompts, with target global NFE values 1, 32, and 64 assigned cyclically (early, middle, late), and chooses target layer 4 or 12 through an indexed, reproducible stream. The plan is at most 100 paired actions / 200 branch attempts. The smoke is limited to one prompt and 20 branch attempts. No confirmatory prompts are used.

## Measurements and interpretation

For each action the raw bundle retains the complete control/treatment refinement traces, final generated token IDs and hashes, text/grading, exact K vectors and sparse deltas, action propensity where defined, target-layer pair counts, whole-trajectory pair deltas, and runtime ledger rows. A bounded Tier-1 comparator records target and two downstream NFE logit/JSD/top-k summaries only for candidate positions whose control and treatment input token IDs still match. Dense vocabulary logits are not persisted. Aggregate refinement-counter differences are not represented as exact committed-token identities.

Exploratory model controls use prompt-disjoint development/validation splits and compare token-only, router-only, ordinary ungrouped-joint, and random scores. Outcomes retain all zero labels. A constant binary correctness-effect vector makes prediction and selection performance non-estimable. Attention/router joint models are not run unless the actual loaded attention backend, post-QK-normalization/post-RoPE scores, and short-sequence full-bidirectional reference fidelity pass; passing that runtime probe alone does not imply attention features were collected in this D1 run.

G2 is conditional on a G1 signal beyond the strong ungrouped controls. G3 requires a separately locked whole-trajectory matched-cost utility comparison. G4 requires direct synchronized end-to-end wall-clock measurements with a quality/latency frontier. None is inferred from lower expert-token-pair counts.

## Resource limits

The campaign wrapper enforces a one-hour combined audit/smoke wall-time cap, a three-hour sensitivity wall/GPU cap with at most 250 attempted branches, a two-hour attention-probe cap, a 12-hour total wall limit, at most 400 intervention branches, and a 30 GiB total output limit. Per-stage caps are lower where required. All GPU jobs are serial and the wrapper refuses to start if another GPU compute process is present. The 10-prompt slate is sized to stay below the sensitivity branch cap with duplicated paired controls.

## Exact commands

From the clean follow-up worktree on `research-gpu`, after activating `/home/new_drive/rashidi/venvs/reflex-llada` and confirming the GPU is idle:

The worktree needs the local GSM8K parquet mirror at `data/gsm8k/main/`; the runner exports `JRC_MOE_GSM8K_DIR` and records both source-file hashes. This keeps prompt reconstruction offline and reproducible.

```bash
bash scripts/run_jrc_v4_followup_bounded.sh
```

The script runs audit, smoke, and sensitivity in order; it stops at failed gates, then runs the attention runtime probe and prompt-disjoint control analysis. It writes all stage logs and the cumulative ledger under `outputs/jrc_v4_followup/`.

For a partial sensitivity run with a valid audit and smoke, resume without resetting its ledger:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH="$PWD/src" \
/home/new_drive/rashidi/venvs/reflex-llada/bin/python -m jrc_moe.cli jrc-v4 pilot \
  --config configs/jrc_v4/followup_sensitivity.yaml \
  --audit-dir outputs/jrc_v4_followup/audit \
  --smoke-dir outputs/jrc_v4_followup/smoke \
  --output-dir outputs/jrc_v4_followup/sensitivity --resume \
  --max-gpu-hours 3 --max-wall-hours 3 --max-branches 250 \
  --max-prompts 10 --max-output-gb 2
```

The resume identity binds source commit, config, checkpoint, input CSV, audit, caps, and prompt limit. Increasing any cap creates a different run identity and is rejected.
