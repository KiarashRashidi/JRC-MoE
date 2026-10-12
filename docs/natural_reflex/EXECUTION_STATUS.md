# Natural REFLEX execution status

Date: 2026-10-12
Feature branch: `codex/natural-reflex-unmasking-characterization`

## Scope and run identity

The natural REFLEX study ran on `research-gpu` using the existing
`reflex-llada` environment and NVIDIA RTX 4090. The isolated server checkout is
`/home/new_drive/rashidi/JRC-MoE-natural-reflex`; the pre-existing main,
follow-up, and v4 worktrees were left in place. Collection run ID:
`ac7472d40f49c9173038`.

The checkpoint was `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` at revision
`783d3467f108d28ac0a78d3e41af16ab05cabd8`. The collection configuration SHA256
is `1be56c49ef6f13542e117f47a58bb931bfca70eb789605329d73288e9ffb05c0` and
the source commit used to collect data is
`4bded4f1a6223ab0381e95e1963e02ee60760330`.

## Completed validation and collection

- The attention runtime audit passed for all 16 attention modules, including
  the saved Q/K parity checks.
- Fixed8 observer instrumentation passed its three-prompt, six-trajectory
  smoke gate. Baseline and observed generated IDs and commits matched;
  `argmax_match_fraction=1.0`, `direct_pairs=expected_pairs=170624`, maximum
  absolute logit difference `0.0`, and maximum confidence difference `0.0`.
- The resumed collection completed 50 paired prompts and 100 trajectories
  (50 per mode), with zero duplicate trajectories and zero invalid trajectory
  validations. The final 24 trajectories were resumed after a client pipe
  interruption; the saved records passed validation.
- Fixed8 patch equivalence remained exact on the completed collection:
  `argmax_match_fraction=1.0`, `direct_pairs=expected_pairs=170624`, and
  maximum logit difference `0.0`.
- The earlier 10-prompt analysis from run
  `48c9cabff6e7196bf17c` was preserved byte-for-byte under
  `outputs/natural_reflex/analysis_pre_50prompt_run_48c9cabff6e7196bf17c/`.

## Analysis and findings

The streaming CPU analysis completed over all 100 trajectories. Prompt IDs
were hash-split into 33 development and 17 validation clusters; every token
row from a prompt stayed in one split. The archived report contains D1–D14,
predictor metrics, per-prompt summaries, horizon labels, and plots.

Paired final-answer correctness was 39/50 for REFLEX and 40/50 for Fixed8:
37 prompts were correct in both modes, 8 incorrect in both, 3 REFLEX-only
incorrect, and 2 Fixed8-only incorrect. Generated token ID hashes differed
for all 50 pairs. These are decoding-mode comparisons, not causal
interventions. REFLEX-minus-Fixed8 prompt means were `-0.2458` for mean wave
size (95% prompt-cluster bootstrap CI `[-0.3412, -0.1405]`), `+0.0380` for
fallback fraction (`[0.0233, 0.0541]`), `-0.0425` for multi-position wave
fraction (`[-0.0569, -0.0276]`), and `+9.04` for total NFE (`[4.76, 13.32]`).
NFE is not wall-clock latency.

The D1–D14 estimates are observational. Raw scaled post-RoPE QK scores are not
probabilities, router IDs are layer-local, and the lexical measure is exact
tokenizer-piece identity only. The historical GSM8K-50 sample and its
17-prompt validation partition are exploratory, not confirmatory. M5 feature
coverage was 84.45%; its same-overlap AUCPR deltas versus M2 were `+0.00746`,
`+0.00319`, and `+0.00047` at horizons 1, 2, and 4, while the corresponding
M2-minus-M5 log-loss deltas were `-0.01519`, `+0.00070`, and `+0.01137`.
These single-split comparisons do not establish predictive benefit.

The temporal-community gate D9 and M6 gate remain closed: development-only M5
cross-validation and adequate community-feature coverage were not established.
No causal expert-budget intervention, community-superiority claim, or
inference-speedup claim was tested or is supported.

## Resource use and analysis recovery

The cumulative campaign ledger records 13,825.9012 GPU seconds out of the
14,400-second cap (3 h 50 min 25.9 s of 4 h) and 4,101,710,606 output bytes
out of the 10 GiB cap. The run manifest records 3,717,622,037 bytes for the
collection itself. The first eager CPU analysis was interrupted after memory
use approached 100 GiB; it did not alter the preserved prior analysis. The
analyzer was changed to stream trajectories one at a time. The successful
streaming analysis used about 2,395 seconds and peaked near 8.9 GiB sampled
RSS, with the GPU idle.

## Tests and artifacts

On the server, the regression command
`python -m unittest tests.test_generation_trace tests.test_observational_cpu tests.test_natural_analysis_streaming`
passed 26 tests after the package selection fix.

Run commands, environment, hashes, campaign ledgers, raw trajectories, plots,
and review archive are under
`/home/new_drive/rashidi/JRC-MoE-natural-reflex/outputs/natural_reflex/`.
The full archive is
`/home/new_drive/rashidi/JRC-MoE-natural-reflex/outputs/natural_reflex/jrc_natural_reflex_ac7472d40f49c9173038_full.zip`.
See `RUNBOOK.md` and `50_PROMPT_SCIENTIFIC_REPORT.md` in the output directory
for reproduction steps and the full measured result summary.
