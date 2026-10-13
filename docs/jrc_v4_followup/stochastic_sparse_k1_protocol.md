# Sparse stochastic K=1 experiment

## Hypothesis

For an unresolved token in the active 32-position decoding block, reduce its
expert count from K=8 to K=1 only occasionally. Use the confidence value REFLEX
already calculated on the previous forward: lower-confidence tokens receive a
higher chance of K=1, while higher-confidence tokens receive a lower chance.
The treatment may reduce executed expert-token work, but could also change the
next-token distributions, commitment path, final answer, or runtime.

## Treatment definition

The probability is a pre-decision, piecewise-linear function of the most recent
max-softmax confidence for that still-unresolved token:

| Previous confidence | Probability of K=1 at a token/layer/forward |
|---|---:|
| `<= 0.50` | 12% |
| `0.70` | 7% |
| `>= 0.90` | 2% |

Probabilities between those points are linearly interpolated and clipped to the
endpoint values. The 0.90 value is REFLEX's commit threshold. On the first
forward, prior confidence does not exist, so every position retains K=8. A
separate Bernoulli decision is made for each eligible token x MoE-layer pair on
each forward. Only unresolved generated positions inside the current 32-token
block are eligible. The policy runs on every decoder forward and all 16 MoE
layers. A selected pair uses the router's highest-ranked expert only.

The intervention is intentionally compute-reducing: it does not transfer the
seven removed expert slots to another token. K=1 assignments are not budget
neutral. As verified for this checkpoint, selected router weights are not
renormalized; a K=1 token retains its original top-expert softmax gate mass.

## Comparators

1. **Fixed8 control:** K=8 for all tokens and layers.
2. **Confidence-weighted sparse K=1:** the probability rule above.
3. **Matched-rate random sparse K=1:** independently sample each eligible pair
   using the mean confidence-policy probability among eligible positions for
   that forward and block. It has the same expected number of K=1 assignments
   at that state but does not prefer low-confidence tokens. Both sparse arms
   use the same deterministic uniform draw for each token/layer/forward pair.

Ten historical development prompts are selected with five correct and five
incorrect Fixed8 references, then rerun in all three arms. The actual fresh
Fixed8 result is used as each prompt's paired control outcome. Arm order is
randomized within prompt. The selection uses prior confidence only; no current
or future logits are used to choose a K=1 assignment.

## Measurements

- Exact final token IDs, decoded text, extracted answer, and canonical correctness.
- Number of refinement forwards (NFE), total executed expert-token pairs, and
  synchronized CUDA-event and wall-clock duration.
- For each eligible confidence band (`<=0.50`, `0.50..0.90`, `>=0.90`), the
  number of token/layer/forward opportunities and observed K=1 assignments.
- K=1 assignment counts by token/layer, block and forward; target confidence
  and the assigned probability are retained for each sampled event.
- Paired prompt-level uncertainty for correctness, NFE, expert-token pairs,
  and timing. Results remain exploratory and no confirmatory claim is made.

## Gates

Before the pilot: verify checkpoint/router semantics; require exact native versus
patched K=8 parity; run a masked forward with sparse K=1 assignments at each
layer and confirm finite logits, expected pair counts, and unnormalized top-1
gate weights; then require exact no-op replay of a full rollout using the
stochastic policy and at least one realized K=1 assignment.

The bounded pilot is capped at 10 prompts, 40 branch attempts, 3 GPU hours,
3 wall-clock hours, and 2 GiB of outputs. The source experiment is run on the
development sample only. A performance or quality claim requires a separate,
locked evaluation and direct measurement; reduced pair counts alone are not a
speedup.

## Command

Run from the repository root after activating the existing `reflex-llada`
environment and setting offline dataset/model cache variables:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m jrc_moe.jrc_v4.stochastic_k1_experiment \
  --stage audit --config configs/jrc_v4/stochastic_k1_sparse.yaml

PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m jrc_moe.jrc_v4.stochastic_k1_experiment \
  --stage pilot --config configs/jrc_v4/stochastic_k1_sparse.yaml \
  --audit-dir outputs/jrc_v4_stochastic_k1/audit
```
