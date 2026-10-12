# Natural REFLEX data dictionary

All raw trajectory records are gzip-compressed JSON Lines, committed atomically only after trajectory validation. Every record carries run, prompt, mode, seed, checkpoint/config identity, block, local NFE, and source-time provenance where applicable. `manifest.json` records byte lengths, hashes, counts, and validation status.

## Files per trajectory

| File | Unit | Meaning |
|---|---|---|
| `events.jsonl.gz` | One decoder forward | Exact active positions, pre-forward eligible set, threshold commits, fallback use, committed positions, mask counts, K-allocation hash, active K counts, prompt/block metadata, and wall/CUDA forward timings. |
| `positions.jsonl.gz` | One unresolved position at one forward | Past-only prediction covariates, confidence history through `t-1`, active-block and risk-set indicators, feature availability/timestamps, and attention/router summaries. Current-forward decision information is not a predictor covariate. |
| `relations.jsonl.gz` | One sampled attention or router relation | Sparse direction-aware attention scores or router logits/selection/top-k relations, with layer/head, query/key role, local-NFE source, score semantics, sampling weight, and feature availability. |
| `prompt_output.json` | One completed rollout | Generated token IDs and strings, decoded text, grade, generation summary, and peak allocated memory. |
| `manifest.json` | One completed rollout | SHA-256 and size for the above files, counts, replay/config identity, and validation result. |

## Timing and feature semantics

- `local_nfe` restarts for each active block; `global_nfe` is monotonic within the rollout. A relation observed at local forward `s` is only eligible to predict at a later forward `t` when its availability timestamp is at most `t - 1`.
- `candidate_at_risk=true` requires the token's block to be active and the token to remain unresolved at that forward. Future blocks are not treated as negative labels.
- Horizon labels are first commitment within the next 1, 2, or 4 exposed local forwards. Near-terminal exposures are recorded explicitly.
- `score_semantics=post_qk_norm_post_rope_scaled_dot_product_score` identifies a sampled scaled QK dot product before softmax. It is not a probability or a complete attention distribution.
- Attention rows retain direction (`unresolved_to_key`, `committed_to_unresolved`, or `prompt_to_unresolved`), key role, causal-mask and `is_causal` metadata. Prompt, committed, and unresolved key categories are separate.
- Router rows distinguish raw router logits, exact post-bias selection scores, and selected expert IDs/weights. Expert IDs are layer-local. Popularity correction is estimated only from development prompts.
- Similarity-based joint features rank-transform attention and router channels within the same candidate set before combining them.
- Pair labels include deterministic sampling weights if a forward contains more than the configured pair cap. Bootstrap units remain prompts.
- Before a forward is written, its canonical uncompressed event/position/relation records are bounded by `resources.max_forward_record_bytes_mib` (128 MiB by default). The collector reserves a further two-forward margin against the cumulative 10 GiB output cap, flushes compressed streams after each forward, and rechecks projected headroom before continuing.

## Campaign outputs

`campaign_ledger.json` is cumulative across attempts and records occupied GPU seconds, output bytes, active reservation/deadline, and exit details. `smoke/` stores observer parity evidence. `attention_audit/` stores source identity and runtime Q/K/SDPA parity. `analysis/` contains derived horizon labels, paired-position labels, predictor metrics, D1–D14 summaries, plots, and the narrative report. `MANIFEST.txt` and `SHA256SUMS` identify the review archive inputs.
