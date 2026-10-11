# Live evidence gates

Status below distinguishes historical evidence from gates that must be re-run for the new follow-up source/config.

| Gate | Status | Evidence / reason |
|---|---|---|
| S0 evidence reconciliation | **Passed** | Server source pilot and local copy match all 216 SHA-256 checks; 100 pairs, 200 branches, 50 states, joins, zero labels, costs, and traces reconcile. See `outputs/jrc_v4_followup/forensics/forensic_manifest.json`. |
| Historical G0 | **Passed for prior commit only** | The prior audit manifest records native K=8 parity, exact replay, and runtime router semantics under source `a62e164...`; this does not validate the follow-up branch. |
| Follow-up G0 / S1 mechanism | **Not run** | The new generalized API and new configuration require a new pinned-checkpoint audit, K-sweep, deliberate multi-token perturbation, and replay checks bound to the follow-up commit. |
| S2 intervention sensitivity | **Not run for the new actions** | Historical K=6 one-unit transfer perturbed selected-position logits and sometimes later traces/string output, but did not test K=5, A dose, m=2/5/8/16, or independently varied c. |
| S3 / attention | **Blocked pending fidelity probe** | Prior features have null attention values and the old run explicitly disabled attention. The pinned source uses per-head Q/K normalization followed by RoPE and full bidirectional attention; post-transform numerical fidelity still needs an independent runtime check. |
| G1 predictive value | **Not estimable from historical pilot** | All 100 binary correctness effects are zero; validation rank statistics are undefined. Attention features are absent. No surrogate labels will be created. |
| G2 explicit communities | **Not run** | Conditional on G1 and matched ungrouped joint evidence. |
| G3 matched whole-trajectory allocation | **Not authorized in this campaign** | Requires a separately locked protocol and measured final-utility gain at matched whole-trajectory pair cost. |
| G4 inference latency | **Not authorized as a claim** | No optimized same-server end-to-end quality/latency frontier measurement exists. |

The current server is `ubuntu24-server` with one idle RTX 4090 (49,140 MiB total, 48,510 MiB free at the latest audit, 1 MiB used, 0% utilization). Follow-up GPU work must begin only after the committed source/config passes the new audit and smoke gate. Prior October 10 reports saying SSH prevented execution are retained as historical attempt notes; the later raw pilot manifest and its server-verified hashes are the evidence for the completed prior run.
