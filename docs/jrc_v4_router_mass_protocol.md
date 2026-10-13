# JRC-MoE confidence-adaptive router-mass pilot

## Frozen comparison

The control requests cumulative router gate mass **P = 0.6 for every token in
every MoE layer and decoder forward**. The adaptive arm starts from the same
P = 0.6 allocation. At unresolved generated positions with a previous REFLEX
confidence score, a separate deterministic Bernoulli draw is made for each
token/layer/forward:

| Latest REFLEX confidence | P = 0.1 chance | P = 0.9 chance | Otherwise |
|---|---:|---:|---|
| <= 0.50 | 50% | 2% | P = 0.6 |
| 0.70 | 26% | 26% | P = 0.6 |
| >= 0.90 | 2% | 50% | P = 0.6 |

The two non-baseline choices are mutually exclusive. Probabilities interpolate
linearly between the confidence boundaries. Tokens without a previous REFLEX
score, prompt tokens, resolved generated tokens, and all other positions retain
P = 0.6. Counter-based draws make the policy reproducible and independent of
arm execution order.

For a token, the router probabilities are the checkpoint's full softmax over
64 experts. The dispatcher sorts those probabilities and activates the
shortest top-ranked prefix whose **original, unnormalized weights sum to at
least P**. For example, P = 0.1 may already be reached by the top expert;
P = 0.9 can require many experts. Selected expert outputs keep the checkpoint's
original weights and are not renormalized. Consequently this is a routing-mass
experiment, not a promised compute reduction. Reports must include achieved
mass, experts per token, expert-token pairs, wall time, and CUDA-event time.

## Running on the GPU server

From the isolated `codex/reflex-routing-mass` checkout, activate the existing
`reflex-llada` environment and set the existing model cache and GSM8K mirror:

```bash
source /home/new_drive/rashidi/venvs/reflex-llada/bin/activate
export PYTHONPATH=src
export HF_HOME=/home/new_drive/rashidi/hf_cache
export HF_HUB_CACHE=/home/new_drive/rashidi/hf_cache/hub
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JRC_MOE_GSM8K_DIR=/home/new_drive/rashidi/JRC-MoE-router-mass/data/gsm8k
python -m jrc_moe.jrc_v4.mass_experiment --stage audit \
  --config configs/jrc_v4/router_mass_adaptive.yaml \
  --output-dir outputs/jrc_v4_router_mass/audit
python -m jrc_moe.jrc_v4.mass_experiment --stage pilot \
  --config configs/jrc_v4/router_mass_adaptive.yaml \
  --audit-dir outputs/jrc_v4_router_mass/audit \
  --output-dir outputs/jrc_v4_router_mass/pilot
```

The audit must pass the pinned checkpoint/router semantics and native Fixed8
parity checks, an actual P = 0.6 GPU forward with selected-mass accounting, and
deterministic allocation replay before the paired pilot can start. The pilot is
bounded to 10 prompts and 20 full rollouts by the checked-in resource caps. It
is exploratory: no claims about predictive benefit, community superiority, or
inference acceleration follow from this run alone.

## Output files

The audit writes its manifest under `outputs/jrc_v4_router_mass/audit/`. The
pilot writes its paired branch outcomes, per-forward policy summaries,
non-baseline mass assignment events, decoder traces, resource ledger, summary,
and manifest under `outputs/jrc_v4_router_mass/pilot/`. Use a new output path
for a different source revision or protocol; resumption requires the same
commit, config, and audit identity.
