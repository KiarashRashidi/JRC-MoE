# Sparse stochastic K=1 follow-up: 50% low-confidence chance

This is a separate, pre-registered sensitivity run of the sparse K=1 pilot.
It preserves the original 12% configuration and output tree. The sole policy
change is the K=1 assignment chance at or below confidence 0.50: it is raised
from 12% to 50%. The high-confidence endpoint remains 2% at confidence 0.90,
and values between the boundaries are linearly interpolated (so confidence
0.70 maps to 26%). Above 0.90 the chance stays 2%.

The underlying event remains an independent Bernoulli decision for each
eligible unresolved-token x MoE-layer x decoder-forward pair. A selected pair
uses K=1 instead of K=8, without transferring expert slots to another token.
The first forward for a token has no prior confidence and therefore retains
K=8. The same ten prompt IDs, model revision, decode settings, randomization,
and three arms are used as the 12% pilot. The matched-random arm uses the mean
confidence-policy rate at each forward/block. Consequently, this run isolates
the effect of increasing the low-confidence assignment probability while
keeping the original experimental design.

The run remains a bounded development pilot (10 prompts, 40 branches, 3 GPU
hours, 3 wall hours, 2 GiB output maximum). It cannot establish a general
quality or speed claim. Changed outputs, refinement forwards, measured runtime,
and exact assignment rates must be reported alongside correctness.

Commands, from the repository root after activating `reflex-llada`:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m jrc_moe.jrc_v4.stochastic_k1_experiment \
  --stage audit --config configs/jrc_v4/stochastic_k1_sparse_50pct.yaml

PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m jrc_moe.jrc_v4.stochastic_k1_experiment \
  --stage pilot --config configs/jrc_v4/stochastic_k1_sparse_50pct.yaml \
  --audit-dir outputs/jrc_v4_stochastic_k1_50pct/audit
```
