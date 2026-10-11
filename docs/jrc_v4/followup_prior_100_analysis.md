# Prior 100 paired exchanges: follow-up baseline analysis

## Scope and provenance

This is a read-only reanalysis of the actual GPU pilot copied from the preserved server run at `outputs/jrc_v4_server_run_c854042/pilot`. The source pilot manifest records 50 prompts, 100 paired exchanges, 200 generated branches, and source commit `a62e164df870a6480b4c86325660716f17b31cc2`. It pins `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` to checkpoint revision `783d3467f108d28ac0a78d3e41af16ab05cabd8d`.

All 200 branch records, 100 paired records, 100 unique exchange IDs, and 50 prompt IDs were validated. Each treatment arm transfers one K unit at one layer and one recipient/donor position pair: the two selected positions move from K=6/K=6 to K=7/K=5 for the target forward. The target layer's expert-token-pair count is equal in both arms. Future rollout costs are allowed to differ.

The analysis joins trace rows only when `(block_idx, global_nfe)` matches across both arms. Confidence intervals below use a prompt-cluster bootstrap that resamples prompt IDs with all of each selected prompt's exchange rows (20,000 resamples, seed 20261011). These are exploratory intervals for this pilot, not confirmatory inference.

## Findings

| Outcome | Observed result |
|---|---:|
| Correctness changed | 0/100 pairs |
| Control and treatment correct | 80/100 each |
| Canonical extracted answer changed | 0/100 pairs |
| Generated sequence changed | 3/100 pairs, across 2/50 prompts |
| Prompt-cluster bootstrap interval for sequence change rate | [0.00, 0.08] |
| Selected-position next-forward JSD nonzero | 100/100 pairs |
| Mean / median / maximum JSD | 0.0015375 / 0.0008558 / 0.0091896 |
| Prompt-cluster bootstrap interval for mean JSD | [0.0011231, 0.0019894] |
| Maximum top-token probability change nonzero | 100/100 pairs |
| Mean / median / maximum top-token probability change | 0.0157786 / 0.0106106 / 0.0917193 |
| Target-forward aggregate refinement counters changed | 3/100 pairs |
| At least one common later forward had changed counters | 7/100 pairs |
| Prompt-cluster bootstrap interval for later-counter divergence rate | [0.02, 0.14] |
| Common later trace rows | 8,848; 402 had at least one counter difference |
| Later-row counter-field differences | 311 `num_above_threshold`, 236 `num_committed`, 393 `masked_after`, 215 `fallback_used` |
| Exchange pairs with different trace-key sets | 7/100 |

The three final-sequence changes were:

| Exchange | Control answer | Treatment answer | Correct in both? | NFE difference (treatment − control) |
|---|---:|---:|---:|---:|
| `p215-n4-l4-r935-d949` | 44 | 44 | Yes | −5 |
| `p215-n4-l4-r941-d953` | 44 | 44 | Yes | −5 |
| `p537-n1-l12-r1071-d1058` | 188 | 188 | Yes | +3 |

The pilot's Tier 1 calculation compares control and treatment next-forward vocabulary distributions at only the selected recipient and donor positions. It computes the mean JSD across those positions and the largest absolute top-token probability change. The saved artifacts do not retain every downstream logit tensor. Trace CSVs contain aggregate refinement counters, not token identities, so counter divergence does not identify exactly which token was committed.

## Interpretation and gates

The saved intervention changed measured next-forward distributions for all 100 pairs, aggregate refinement counters for a subset of pairs, and final generated sequences in three pairs. It changed no answer or correctness label in this sample. That demonstrates sensitivity, not a beneficial effect.

Treatment NFE was lower/equal/higher than control in 3/93/4 pairs (mean difference −0.05). Whole-trajectory expert-token-pair counts were lower/equal/higher in 3/93/4 pairs (mean difference −6,629.12; range −754,560 to +501,888). These pair counts are compute accounting, not measured latency. No inference acceleration is supported.

The prior audit recorded G0 as passed for checkpoint/router semantics, fidelity, legal-K forwards, replay, and runtime profiling. The attention feature extractor remained disabled because a post-RoPE Q/K hook and its short-sequence fidelity had not been verified. The prior signal analysis therefore blocked G1; G2, G3, and G4 were not run. This follow-up analysis does not change those gates.

This pilot used a single K-unit transfer between two positions. It does not test 5+5 → 7+3, intervention magnitude as an independent factor, or candidate groups of 2, 5, 8, and 16 tokens.

## Recorded resources

The smoke and pilot manifests record 426.01 GPU seconds (0.11834 hours) and 7,250.55 GPU seconds (2.01404 hours), respectively: 7,676.56 ledgered GPU seconds (2.13238 hours) for those two stages. They record 12 smoke branches and 200 pilot branches. The failed first smoke attempt records 0 GPU seconds. Stage 0 also performed GPU validation, replay, fidelity, legal-K forwards, and synchronized fixed-K profiling, but its manifest does not record one cumulative GPU-seconds total. Consequently 2.13238 hours is the sum of the smoke and pilot ledgers, not a complete total for all prior GPU activity.

The follow-up instructions package was not present in the available Downloads or repository files at the time of this analysis. No follow-up run was started. The configured SSH connection resolved to `172.20.27.4`, but an authorized connection attempt timed out, so current server state and additional remote ledgers could not be refreshed.

## Reproduction

From the repository root, with Python and Matplotlib available:

```powershell
python scripts/analyze_jrc_v4_pilot_pairs.py `
  --pilot-dir outputs/jrc_v4_server_run_c854042/pilot `
  --output-dir outputs/jrc_v4_followup_baseline_analysis
```

The command writes `baseline_analysis.json`, `exchange_level_analysis.csv`, `trace_decision_comparison.csv`, `previous_100_exchange_report.md`, and `next_forward_sensitivity.png`. The source script uses only the Python standard library for statistics; Matplotlib is optional and is used only for the plot.
