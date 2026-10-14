# Prompt-level router P and probability pilot

## Question

Do prompts respond differently to a lower cumulative gate-mass target, or to
the probability of assigning low/high mass based on REFLEX confidence? The
pilot measures prompt-by-arm effects; it does not fit a custom routing rule to
each prompt and then score that rule on the same rollout.

## Four matched arms

Each of the same 10 preselected GSM8K prompts receives four complete REFLEX
rollouts (40 branches total):

1. **Fixed P=.6:** the control, applied to every token in each MoE layer and
   decoder forward.
2. **Fixed P=.1:** tests the effect of changing the mass threshold while
   holding it constant for that prompt's whole sequence.
3. **Adaptive base probability:** for unresolved generated tokens with a
   previous confidence, assign P=.1 or P=.9 using the approved schedule: at
   confidence <=.50 the chances are 50% and 2%; at >=.90 they reverse; values
   between interpolate linearly. Otherwise use P=.6.
4. **Adaptive half probability:** exactly the same targets, confidence rule,
   and random draws, but halve both non-.6 probabilities to 25% and 1% at the
   endpoints. This isolates assignment frequency from the P values.

The adaptive arms use the same counter-based draw for each token/layer/forward.
For each prompt the report will show correctness and sequence changes, expert
counts and achieved mass by target, expert-token pairs, decoder forwards, and
measured times. The core prompt-level contrasts are fixed P=.1 versus P=.6,
and the base-probability policy versus its half-probability version. The
approved adaptive policy is also compared with fixed P=.6.

## Interpretation limits

Ten prompts provide a bounded development sample. Per-prompt outcome
differences are descriptive and can identify heterogeneity worth testing; they
cannot establish that a prompt-specific setting is reliably better. Selecting
a setting separately for each prompt requires a later held-out evaluation.
No inference speedup is claimed unless end-to-end measured time improves.

## Commands

```bash
source /home/new_drive/rashidi/venvs/reflex-llada/bin/activate
export PYTHONPATH=src
export HF_HOME=/home/new_drive/rashidi/hf_cache
export HF_HUB_CACHE=/home/new_drive/rashidi/hf_cache/hub
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JRC_MOE_GSM8K_DIR=/home/new_drive/rashidi/JRC-MoE-followup/data/gsm8k
python -m jrc_moe.jrc_v4.mass_experiment --stage audit \
  --config configs/jrc_v4/router_mass_prompt_heterogeneity.yaml
python -m jrc_moe.jrc_v4.mass_experiment --stage pilot \
  --config configs/jrc_v4/router_mass_prompt_heterogeneity.yaml \
  --audit-dir outputs/jrc_v4_prompt_heterogeneity/audit \
  --output-dir outputs/jrc_v4_prompt_heterogeneity/pilot
```

The audit runs one mixed-target forward with P=.1, .6, and .9, validates
achieved mass and expert-count accounting, then verifies deterministic policy
allocation. The pilot starts only if that audit passes and the 3 GPU-hour,
3 wall-hour, 40-branch, 10-prompt, and 2 GB output limits remain available.
