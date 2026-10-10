# JRC-MoE v4 gate decisions

| Gate | Status | Evidence / decision |
|---|---|---|
| G0: model, routing, replay, profiling | **blocked** | Server identity/GPU/PyTorch were inspected. Exact remote code, cached weights, native/patched parity, replay, and synchronized baseline profile were not verified. SSH follow-up timed out. |
| G1: joint feature decision value | **not run** | No full-rollout exchange labels or held-out predictor comparison. |
| G2: community necessity | **not run** | G1 missing; community claims remain gated. |
| G3: matched-pair allocation | **not run** | No full-trajectory budget-matched evaluation. |
| G4: wall-clock frontier | **not run** | No end-to-end timing comparison including overhead. |

No scientific hypothesis is accepted or rejected. The next action is to resume server access, inspect the preserved dirty checkout and cache, and run `jrc-moe jrc-v4 audit` offline before permitting smoke branches.
