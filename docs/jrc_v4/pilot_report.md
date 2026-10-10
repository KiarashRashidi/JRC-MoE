# JRC-MoE v4 pilot report

## Status

No GPU pilot has run. The required live checkpoint/cache and remote-code audit could not be completed after the configured SSH connection timed out. G0 is blocked, so Stage 1 branches were not started and no causal outcomes, Tier-1 sensitivity measures, NFE/pair costs, or latency values exist.

The local repository contains the historical `data/diagnostic_50_results.csv` (50 rows, SHA-256 `177942409f8dbf8ea0d54fe81c86c2021118a0d335d655adec6531410cb4cd26`); it is prior development evidence, not an unseen test set. Its server-side presence and checksum remain unverified. The historical 50-prompt results are not reused as confirmatory evidence.

| Quantity | Observed value |
|---|---:|
| Prompts / states / paired exchanges | 0 / 0 / 0 |
| Full-rollout branches | 0 |
| Zero / positive / negative effects | not measured |
| NFE / executed expert-token pairs | not measured |
| GPU hours / branch runtime | 0 / not measured |
| Tier-1 to Tier-2 agreement | not measured |

This is an execution blocker report, not a negative scientific result.

The CLI has not been run against the server. Local verification was limited to 21 passing CPU synthetic tests, one skipped model integration test, compilation, and CLI help. The audit and smoke/pilot runners are wired, but remain unexecuted and unverified against the pinned checkpoint. The signal analyzer is ready to compare token and router controls, but cannot test the joint attention/router hypothesis until the attention hook passes its source and fidelity checks. Community/allocation/systems execution remains gated.
