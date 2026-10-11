# JRC-MoE v4 follow-up: completed evidence and gates

## Historical 100-pair reconciliation

The prior pilot's 100 paired exchanges all had zero final-correctness effect (both arms were correct on 80/100). Its selected-position distribution changed in all 100 pairs (mean JSD 0.0015375), aggregate refinement-trace counters differed in 7/100, and decoded output strings differed in 3/100 across two prompts. The available prior artifact did not retain final token IDs or exact commit-token identities, so it cannot establish token edit distance or identify which individual downstream commit decisions changed. Target-layer executed-pair deltas were zero for all 100 pairs. The full reconciliation is in `outputs/jrc_v4_followup/forensics/` and `docs/jrc_v4_followup/previous_pilot_forensics.md`.

## Follow-up run identity

- Experiment source commit: `596b325c679227a30aa4415f7aedd39d3ea1814b` on `codex/jrc-v4-followup-experiments`.
- Checkpoint: `inclusionAI/LLaDA-MoE-7B-A1B-Instruct@783d3467f108d28ac0a78d3e41af16ab05cabd8d`.
- Fixed8 CSV SHA-256: `177942409f8dbf8ea0d54fe81c86c2021118a0d335d655adec6531410cb4cd26`.
- Dataset mirror parquet hashes are recorded in the run root as `gsm8k_mirror.sha256`.
- Server: one RTX 4090, 49,140 MiB total; the campaign is complete and the GPU is idle.

## Stage and gate decisions

| Gate/stage | Result | Evidence and interpretation |
|---|---|---|
| G0 runtime audit | **Passed** | Actual loaded architecture: 16 MoE layers, 64 experts, softmax router, native K=8, `norm_topk_prob=false`, expert bias disabled, routed scaling 1.0. Native and patched forward parity had exact zero maximum logit, block-output, weight, and confidence differences; argmax and selected-rank agreement were 1.0. |
| Legal K and legacy reproduction | **Passed** | Actual per-layer K forwards covered every integer K=2..8. Historical REFLEX modes were replayed for three rows with matching correctness, NFE, pair count, and stable trace fields. Fixed8 direct/expected pairs both equaled 170,624. |
| Replay | **Passed** | Full no-op Fixed8 replay matched final token IDs and trace exactly; both reference and replay had 115 NFEs and 19,621,760 executed pairs. |
| GPU smoke | **Passed** | One prompt; 9 completed paired actions / 18 branches and one infeasible/no-site action. All nine correctness effects were zero. There were no generated-token edits; two pairs had aggregate refinement-counter changes. |
| Bounded sensitivity pilot | **Passed, development only** | 10 planned prompts; 9 states captured, 77 completed paired exchanges / 154 branch continuations, and 23 explicit infeasible/no-site actions. Prompt 215 had no unresolved token at its scheduled target NFE and all 10 actions were recorded as `no_site`. No confirmatory prompts were used. |
| Attention runtime fidelity | **Passed** | 16 SDPA calls over a short length-7 sequence. Runtime had no attention mask and `is_causal=false`; post-QK-norm/post-RoPE Q and K exactly matched the tensors passed to SDPA. Landmark score proxy error was zero. Explicit full-bidirectional reference output max absolute error was 0.007603, below the 0.03 tolerance. Activations were not persisted. A static source-string heuristic did not independently classify the bidirectional path; the runtime flags and numerical reference comparison passed. |
| G1 predictive signal | **Not estimable** | All 77 correctness labels were zero. Token-only, router-only, ungrouped-joint, and random controls completed on a 7/3 prompt split, but nonzero-label coverage was zero and ranking/sign metrics were undefined. The D1 sensitivity run kept attention values disabled, so attention-only and attention-plus-router comparisons are unavailable. |
| G2 communities | **Not run** | G1 has no estimable utility signal; no community superiority claim is supported. |
| G3 matched-cost utility | **Not run** | The sensitivity pilot is not a matched whole-trajectory final-utility experiment. |
| G4 inference latency | **Not run** | No optimized treatment-vs-baseline end-to-end frontier was measured. Pair-count changes alone do not establish speedup. |

## Follow-up sensitivity findings

All 77 completed pairs had unchanged final correctness and identical final generated token IDs. Aggregate refinement-trace counters differed in 3/77 pairs; these aggregates do not identify exact per-token commit decisions. The target-layer pair budget was exactly neutral for every pair. Two trajectories had a whole-run executed-pair delta of +170,624; all other trajectory deltas were zero.

The target and two aligned downstream logit diagnostics changed without translating into final sequence changes. Across the 77 action comparisons, mean target JSD was 0.003730 and mean absolute logit delta was 0.135846; there were 61 candidate-position top-1 flips across those target comparisons. At the first downstream lookahead, mean JSD was 0.002660 and there were no top-1 flips; at the second lookahead, mean JSD was 0.000591 and there were no top-1 flips. These are local sensitivity measurements, not utility or predictive labels.

| Controlled comparison | Completed pairs | Mean target JSD | Final token edits | Aggregate trace changes |
|---|---:|---:|---:|---:|
| Magnitude `m=2,c=2,A=1` | 9 | 0.003085 | 0 | 1 |
| Canonical `5+5 → 7+3` (`m=2,c=2,A=2`) | 9 | 0.005705 | 0 | 0 |
| Magnitude `m=2,c=2,A=3` | 9 | 0.006053 | 0 | 1 |
| Changed-token count `m=5,A=2,c=2` | 9 | 0.002353 | 0 | 0 |
| Changed-token count `m=5,A=2,c=4` | 9 | 0.002343 | 0 | 0 |
| Candidate context `m=5,A=2,c=4` | 9 | 0.002343 | 0 | 0 |
| Candidate context `m=8,A=2,c=4` | 9 | 0.003716 | 0 | 0 |
| Candidate context `m=16,A=2,c=4` | 6 | 0.003107 | 0 | 1 |

The dose and context values are descriptive for this small development sample. The m=16 condition has fewer completed pairs, and the constant correctness outcome prevents G1 selection or prediction inference. The token-only and router-margin selected arms had only 2 and 6 completed pairs, respectively, after explicit infeasibility checks.

## Resources and artifacts

The successful audit, smoke, sensitivity, and attention stages took 461, 865, 4,566, and 16 wall seconds, respectively. Two earlier audit attempts (3 and 28 seconds) were blocked by missing local inputs and are preserved in the run tree. Including all attempts, stage wall time was 5,939 seconds (1.65 hours), below the 12-hour cap; audit plus smoke was 1,357 seconds (22.6 minutes), below the one-hour cap. The sensitivity run used 4,541 GPU-seconds (1.26 hours), 154 branches, and about 5.5 MB at pilot completion, below its three-hour, 250-branch, and 2 GiB caps. The complete raw run tree, including reports, plots, manifests, and failed preflight evidence, is archived separately with a SHA-256 manifest.

The synchronized Fixed8 baseline profile used one warm-up and three measured complete 256-token continuations on a 1,077-token prompt; wall median was 43.086 seconds, p95 43.096 seconds, and throughput 5.94 generated tokens/s. This is a baseline profile only and does not demonstrate an intervention speedup.
