# Previous v4 pilot forensic audit

## Evidence and provenance

This is a read-only reanalysis of the server run at `/home/new_drive/rashidi/JRC-MoE-v4/outputs/jrc_v4/pilot`, copied to `outputs/jrc_v4_server_run_c854042/pilot` for analysis. The server copy and local copy have matching SHA-256 values for all 216 files, including 200 per-arm trace CSVs. The checksums are saved in `outputs/jrc_v4_followup/forensics/remote_pilot_sha256.txt`; the run-level manifest is `outputs/jrc_v4_followup/forensics/forensic_manifest.json`.

The old run records 50 prompts, 50 captured states, 100 paired exchanges, 200 branches, 500 start/completion ledger rows, no failures, 7,250.551 CUDA-event seconds (2.0140 hours), 7,270.623 process wall seconds, and 5,836,890 output bytes. Its pinned checkpoint was `inclusionAI/LLaDA-MoE-7B-A1B-Instruct@783d3467f108d28ac0a78d3e41af16ab05cabd8d`; the run used source commit `a62e164df870a6480b4c86325660716f17b31cc2` and config SHA-256 `40925d46079175c0cb0ec27b74bf9e0becd92ce258e5c3e746fba8c3baa6a561`.

All cross-file checks passed: the audit-manifest hash matches, prompt/state/action/branch references join, each exchange has one control and one treatment, all 100 local target-layer pair deltas are zero, and all zero-valued correctness effects remain in the paired records. The raw output records per-arm whole-trajectory pair totals and does not assume those totals match.

## Did the exchanges change downstream behavior?

Yes, in the observables that the old files retained. The intervention was a one-unit `K=6 + K=6 -> K=7 + K=5` transfer.

- **Selected-position next-forward logits:** pair-average JSD was nonzero in 100/100 exchanges (mean 0.00153750, median 0.000855833, p90 0.00400192, p95 0.00531715, max 0.00918957). Maximum absolute top-token probability change was nonzero in 100/100 (mean 0.0157786, median 0.0106106, p90 0.0378439, p95 0.0577155, max 0.0917193). The prompt-cluster 95% bootstrap interval for mean JSD is [0.00112308, 0.00198936]. These are diagnostics over the recipient and donor at the intervention forward, averaged across the two positions; they are not full downstream-logit comparisons.
- **Refinement traces:** the selected-forward aggregate counters differed in 3/100 pairs. At least one later common aligned forward differed in 7/100 pairs. Across 8,848 common post-target trace rows, 402 rows differed in one or more of `num_above_threshold`, `num_committed`, `masked_after`, or `fallback_used`; the trace-key sets also differed in 7/100 pairs. The traces do not retain exact committed token identities, so this is evidence of aggregate decision divergence, not a count of changed token decisions.
- **Final strings and extracted answers:** decoded generated strings changed in 3/100 exchanges, involving 2/50 prompt clusters. The two changed exchanges on prompt 215 retained canonical answer `44`; the changed exchange on prompt 537 retained canonical answer `188`. Canonical answer changed in 0/100, correctness changed in 0/100, and both arms were correct in 80/100 pairs. The prompt-cluster bootstrap interval for the exchange-level string-change rate is [0, 0.08]. Final token IDs were not saved; edit distance is unavailable.
- **Trajectory accounting:** treatment-minus-control NFE and whole-trajectory expert-token-pair deltas were negative/zero/positive in 3/93/4 pairs; mean NFE delta was -0.05 and mean pair delta was -6,629.12, with pair deltas from -754,560 to +501,888. This does not establish a latency improvement. The target-layer pair budget was conserved in all 100 pairs.

The old run retained no dense downstream logits and no final token IDs. Therefore it cannot answer whether later logits shifted when the trace counter did not, identify exact committed-token changes, calculate token edit distance, or reconstruct the first differing token. `outputs/jrc_v4_followup/forensics/prior_intervention_diagnostics.csv` provides the exchange/position feature rows; the JSD is explicitly repeated per selected position and is not represented as position-specific.

## Heterogeneity and missingness

The descriptive exchange-level JSD mean was 0.00285443 for layer 4 (46 exchanges) and 0.000415676 for layer 12 (54 exchanges). By target NFE, counts and JSD summaries are in `outputs/jrc_v4_followup/forensics/tier1_distribution_summary.json`; this is exploratory and has low counts in some strata. The diagnostics include candidate role/position, router entropy, admission margin, top-K gate mass, next-expert gate, latest saved confidence, mask ratio, relative position, and active/unresolved/provisional flags. These are strata of exchange-level pair averages, not causal predictor evaluations.

All 100 paired JSD and top-token-delta values were present and finite; none were exactly zero or below `1e-6`. The 200 feature rows contain router summaries and confidence history. Attention scores are null in all 200 rows and the pilot manifest says attention was disabled. Final token IDs and exact per-token commit identities are absent for every branch.

## Uncertainty and interpretation

There were no correctness-changing pairs across 50 independent prompt clusters. The empirical prompt-cluster bootstrap is degenerate at zero; a two-sided exact 95% binomial interval for the prompt-level incidence of any correctness transition has an upper endpoint of about 7.1% (one-sided 95% upper endpoint about 5.8%). This does not establish a zero population effect. The two changed-string prompts produce much wider uncertainty than a branch-level analysis would imply.

No predictor, community, attention, accuracy, or inference-speed benefit is established. The previous `signal` output has zero nonzero binary labels, undefined rank statistics, and no attention comparator. G1 is not estimable from that constant-label pilot.

## Reproduction

```powershell
python scripts/forensic_jrc_v4_pilot.py `
  --server-run-root outputs/jrc_v4_server_run_c854042 `
  --output-dir outputs/jrc_v4_followup/forensics `
  --remote-checksums outputs/jrc_v4_followup/forensics/remote_pilot_sha256.txt
python scripts/analyze_jrc_v4_pilot_pairs.py `
  --pilot-dir outputs/jrc_v4_server_run_c854042/pilot `
  --output-dir outputs/jrc_v4_followup/forensics/exchange_analysis
```
