#!/usr/bin/env python3
"""Reanalyze the saved JRC-MoE v4 paired pilot outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


DECISION_FIELDS = (
    "num_above_threshold",
    "fallback_used",
    "num_committed",
    "masked_after",
)
EXPECTED_PAIRS = 100
EXPECTED_ARMS = 200
BOOTSTRAP_SEED = 20261011
BOOTSTRAP_REPLICATES = 20_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_trace(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    indexed = {}
    for row in rows:
        key = (int(row["block_idx"]), int(row["global_nfe"]))
        if key in indexed:
            raise ValueError(f"duplicate trace key {key} in {path}")
        indexed[key] = row
    return indexed


def prompt_cluster_bootstrap(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float | int | None],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float]:
    """Return a percentile interval after resampling prompt clusters."""
    grouped: dict[str, list[float | int | None]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(value(row))

    prompts = sorted(grouped)
    if not prompts:
        raise ValueError("cannot bootstrap an empty prompt set")

    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled_prompts = [prompts[rng.randrange(len(prompts))] for _ in prompts]
        sample = [
            item
            for prompt in sampled_prompts
            for item in grouped[prompt]
            if item is not None
        ]
        if sample:
            estimates.append(statistics.fmean(sample))

    estimates.sort()
    low_index = int(0.025 * (len(estimates) - 1))
    high_index = int(0.975 * (len(estimates) - 1))
    return [estimates[low_index], estimates[high_index]]


def compare_trace_pair(
    pilot_dir: Path,
    exchange_id: str,
    pair: dict[str, Any],
    control: dict[str, Any],
    treatment: dict[str, Any],
) -> tuple[dict[str, Any], Counter[str]]:
    """Compare aggregate decision counters at common aligned forwards."""
    target = (int(pair["block_idx"]), int(pair["global_nfe"]))
    traces_dir = pilot_dir / "traces"
    control_trace = read_trace(traces_dir / Path(control["trace_path"]).name)
    treatment_trace = read_trace(traces_dir / Path(treatment["trace_path"]).name)

    common_keys = sorted(set(control_trace) & set(treatment_trace))
    later_keys = [key for key in common_keys if key > target]
    field_counts: Counter[str] = Counter()
    target_changed = False
    later_changed = False
    later_rows_changed = 0
    first_counter_divergence = None

    for key in common_keys:
        changed_fields = [
            field
            for field in DECISION_FIELDS
            if control_trace[key][field] != treatment_trace[key][field]
        ]
        if key == target and changed_fields:
            target_changed = True
        if key > target and changed_fields:
            later_changed = True
            later_rows_changed += 1
            field_counts.update(changed_fields)
        if key >= target and changed_fields and first_counter_divergence is None:
            first_counter_divergence = key

    control_only = sorted(set(control_trace) - set(treatment_trace))
    treatment_only = sorted(set(treatment_trace) - set(control_trace))

    result = {
        "exchange_id": exchange_id,
        "prompt_id": str(pair["prompt_id"]),
        "target_trace_key": f"{target[0]}:{target[1]}",
        "control_trace_rows": len(control_trace),
        "treatment_trace_rows": len(treatment_trace),
        "common_trace_rows": len(common_keys),
        "common_post_target_rows": len(later_keys),
        "post_target_rows_with_any_counter_difference": later_rows_changed,
        "trace_key_sets_differ": set(control_trace) != set(treatment_trace),
        "target_decision_counters_changed": target_changed,
        "post_target_decision_counters_changed": later_changed,
        "first_common_counter_divergence_key": (
            f"{first_counter_divergence[0]}:{first_counter_divergence[1]}"
            if first_counter_divergence
            else None
        ),
        "first_control_only_trace_key": (
            f"{control_only[0][0]}:{control_only[0][1]}" if control_only else None
        ),
        "first_treatment_only_trace_key": (
            f"{treatment_only[0][0]}:{treatment_only[0][1]}" if treatment_only else None
        ),
    }
    result.update(
        {f"post_target_{field}_differing_rows": field_counts[field] for field in DECISION_FIELDS}
    )
    return result, field_counts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prompt_rollup(
    exchange_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    traces = {row["exchange_id"]: row for row in trace_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exchange_rows:
        grouped[str(row["prompt_id"])].append(row)
    result = []
    for prompt_id, rows in sorted(grouped.items()):
        trace = [traces[row["exchange_id"]] for row in rows]
        result.append({
            "prompt_id": prompt_id,
            "paired_exchanges": len(rows),
            "control_correct": sum(row["correctness_control"] for row in rows),
            "treatment_correct": sum(row["correctness_treatment"] for row in rows),
            "correctness_changed_pairs": sum(row["correctness_changed"] for row in rows),
            "canonical_answer_changed_pairs": sum(row["canonical_answer_changed"] for row in rows),
            "generated_sequence_changed_pairs": sum(row["generated_sequence_changed"] for row in rows),
            "target_counter_changed_pairs": sum(row["target_decision_counters_changed"] for row in trace),
            "post_target_counter_changed_pairs": sum(row["post_target_decision_counters_changed"] for row in trace),
            "mean_pair_average_tier1_jsd": statistics.fmean(row["tier1_mean_jsd"] for row in rows),
            "mean_pair_max_top_probability_delta": statistics.fmean(
                row["tier1_max_top_probability_delta"] for row in rows
            ),
            "sum_nfe_delta_treatment_minus_control": sum(
                row["nfe_delta_treatment_minus_control"] for row in rows
            ),
            "sum_trajectory_pair_delta_treatment_minus_control": sum(
                row["trajectory_pair_delta_treatment_minus_control"] for row in rows
            ),
        })
    return result


def count_signs(values: list[int]) -> dict[str, int]:
    return {
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "positive": sum(value > 0 for value in values),
    }


def analyze(pilot_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outcomes = read_jsonl(pilot_dir / "branch_outcomes.jsonl")
    paired = read_jsonl(pilot_dir / "paired_exchange_records.jsonl")

    arms_by_exchange: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for outcome in outcomes:
        arms_by_exchange[outcome["exchange_id"]][outcome["arm"]] = outcome
    pairs_by_exchange = {row["exchange_id"]: row for row in paired}

    complete = all(set(arms) == {"control", "treatment"} for arms in arms_by_exchange.values())
    if (
        len(outcomes) != EXPECTED_ARMS
        or len(paired) != EXPECTED_PAIRS
        or len(arms_by_exchange) != EXPECTED_PAIRS
        or set(arms_by_exchange) != set(pairs_by_exchange)
        or not complete
    ):
        raise ValueError("Expected 200 branch records forming 100 complete paired exchanges")

    exchange_rows = []
    trace_rows = []
    field_counts: Counter[str] = Counter()
    all_common_rows = 0
    common_post_target_rows = 0
    post_target_rows_changed = 0
    target_changed = 0
    post_target_changed = 0
    trace_key_sets_differ = 0

    for exchange_id in sorted(pairs_by_exchange):
        pair = pairs_by_exchange[exchange_id]
        arms = arms_by_exchange[exchange_id]
        control, treatment = arms["control"], arms["treatment"]
        trace_result, local_field_counts = compare_trace_pair(
            pilot_dir, exchange_id, pair, control, treatment
        )
        trace_rows.append(trace_result)
        field_counts.update(local_field_counts)
        all_common_rows += trace_result["common_trace_rows"]
        common_post_target_rows += trace_result["common_post_target_rows"]
        post_target_rows_changed += trace_result["post_target_rows_with_any_counter_difference"]
        target_changed += int(trace_result["target_decision_counters_changed"])
        post_target_changed += int(trace_result["post_target_decision_counters_changed"])
        trace_key_sets_differ += int(trace_result["trace_key_sets_differ"])

        target_pair_delta = int(pair["target_layer_pair_delta"])
        exchange_rows.append(
            {
                "exchange_id": exchange_id,
                "prompt_id": str(pair["prompt_id"]),
                "layer_idx": pair["layer_idx"],
                "global_nfe": pair["global_nfe"],
                "recipient_position": pair["recipient_position"],
                "donor_position": pair["donor_position"],
                "canonical_prediction_control": control["canonical_prediction"],
                "canonical_prediction_treatment": treatment["canonical_prediction"],
                "canonical_answer_changed": (
                    control["canonical_prediction"] != treatment["canonical_prediction"]
                ),
                "correctness_control": int(control["correctness"]),
                "correctness_treatment": int(treatment["correctness"]),
                "correctness_changed": (
                    int(control["correctness"]) != int(treatment["correctness"])
                ),
                "generated_sequence_changed": (
                    control["generated_text"] != treatment["generated_text"]
                ),
                "raw_generated_sequence_changed": (
                    control["raw_generated_text"] != treatment["raw_generated_text"]
                ),
                "tier1_mean_jsd": float(pair["tier1_mean_jsd"]),
                "tier1_max_top_probability_delta": float(
                    pair["tier1_max_top_probability_delta"]
                ),
                "nfe_delta_treatment_minus_control": (
                    int(pair["treatment_nfe"]) - int(pair["control_nfe"])
                ),
                "trajectory_pair_delta_treatment_minus_control": int(
                    pair["trajectory_pair_delta"]
                ),
                "target_layer_pair_delta": target_pair_delta,
                "target_layer_pair_budget_neutral": target_pair_delta == 0,
                "trace_key_sets_differ": trace_result["trace_key_sets_differ"],
                "target_decision_counters_changed": trace_result[
                    "target_decision_counters_changed"
                ],
                "post_target_decision_counters_changed": trace_result[
                    "post_target_decision_counters_changed"
                ],
                "common_post_target_rows": trace_result["common_post_target_rows"],
            }
        )

    return exchange_rows, {
        "all_common_trace_rows": all_common_rows,
        "common_post_target_trace_rows": common_post_target_rows,
        "post_target_rows_with_any_counter_difference": post_target_rows_changed,
        "target_forward_any_counter_change_exchanges": target_changed,
        "post_target_any_counter_change_exchanges": post_target_changed,
        "exchanges_with_different_trace_key_sets": trace_key_sets_differ,
        "different_counter_row_counts_on_common_post_target_rows": dict(field_counts),
        "trace_rows": trace_rows,
    }


def summarize(
    pilot_dir: Path,
    rows: list[dict[str, Any]],
    trace_summary: dict[str, Any],
) -> dict[str, Any]:
    jsd = [row["tier1_mean_jsd"] for row in rows]
    probability_delta = [row["tier1_max_top_probability_delta"] for row in rows]
    nfe_delta = [row["nfe_delta_treatment_minus_control"] for row in rows]
    pair_delta = [row["trajectory_pair_delta_treatment_minus_control"] for row in rows]
    changed_sequences = [row for row in rows if row["generated_sequence_changed"]]
    changed_prompts = sorted({row["prompt_id"] for row in changed_sequences})

    correct_control = sum(row["correctness_control"] for row in rows)
    correct_treatment = sum(row["correctness_treatment"] for row in rows)
    return {
        "source": {
            "pilot_directory": str(pilot_dir),
            "outcome_records": EXPECTED_ARMS,
            "paired_exchange_records": EXPECTED_PAIRS,
            "unique_prompts": len({row["prompt_id"] for row in rows}),
            "checkpoint_revision": (
                "inclusionAI/LLaDA-MoE-7B-A1B-Instruct@"
                "783d3467f108d28ac0a78d3e41af16ab05cabd8d"
            ),
        },
        "final_outcomes": {
            "generated_sequences_changed": len(changed_sequences),
            "generated_sequence_change_rate": len(changed_sequences) / len(rows),
            "changed_exchange_ids": sorted(row["exchange_id"] for row in changed_sequences),
            "raw_generated_sequences_changed": sum(
                row["raw_generated_sequence_changed"] for row in rows
            ),
            "canonical_answers_changed": sum(row["canonical_answer_changed"] for row in rows),
            "correctness_changed": sum(row["correctness_changed"] for row in rows),
            "control_correct": correct_control,
            "treatment_correct": correct_treatment,
            "prompts_with_any_changed_sequence": len(changed_prompts),
            "changed_prompt_ids": changed_prompts,
            "sequence_change_rate_prompt_cluster_bootstrap_95": prompt_cluster_bootstrap(
                rows, lambda row: int(row["generated_sequence_changed"])
            ),
        },
        "next_forward_logits_selected_positions": {
            "quantity": (
                "JSD between control/treatment vocabulary distributions at the selected "
                "recipient and donor positions, averaged across positions; maximum absolute "
                "change in top-token probability across those positions"
            ),
            "nonzero_jsd_exchanges": sum(value > 0 for value in jsd),
            "mean_jsd": statistics.fmean(jsd),
            "median_jsd": statistics.median(jsd),
            "max_jsd": max(jsd),
            "mean_jsd_prompt_cluster_bootstrap_95": prompt_cluster_bootstrap(
                rows, lambda row: row["tier1_mean_jsd"]
            ),
            "nonzero_top_probability_change_exchanges": sum(
                value > 0 for value in probability_delta
            ),
            "mean_max_absolute_top_probability_delta": statistics.fmean(probability_delta),
            "median_max_absolute_top_probability_delta": statistics.median(probability_delta),
            "max_max_absolute_top_probability_delta": max(probability_delta),
            "top_probability_delta_prompt_cluster_bootstrap_95": prompt_cluster_bootstrap(
                rows, lambda row: row["tier1_max_top_probability_delta"]
            ),
        },
        "refinement_trace_counters": {
            key: value
            for key, value in trace_summary.items()
            if key != "trace_rows"
        }
        | {
            "fields": list(DECISION_FIELDS),
            "post_target_counter_change_prompt_cluster_bootstrap_95": prompt_cluster_bootstrap(
                rows, lambda row: int(row["post_target_decision_counters_changed"])
            ),
        },
        "trajectory_compute_diagnostics": {
            "nfe_delta_treatment_minus_control_counts": count_signs(nfe_delta),
            "mean_nfe_delta_treatment_minus_control": statistics.fmean(nfe_delta),
            "whole_trajectory_expert_token_pair_delta_counts": count_signs(pair_delta),
            "mean_whole_trajectory_pair_delta": statistics.fmean(pair_delta),
            "min_whole_trajectory_pair_delta": min(pair_delta),
            "max_whole_trajectory_pair_delta": max(pair_delta),
            "target_layer_budget_neutral_exchanges": sum(
                row["target_layer_pair_budget_neutral"] for row in rows
            ),
            "speedup_claim": (
                "not supported; no end-to-end latency comparison for this intervention"
            ),
        },
        "interpretation_limits": [
            "Tier 1 compares only the two selected positions at the intervention forward; "
            "it is not a complete downstream-logit audit or utility measure.",
            "Trace CSVs contain aggregate counters, not exact per-token commitment identities.",
            "Unchanged correctness in this pilot does not establish equivalence on other prompts.",
            "This pilot tested a one-unit K transfer between two positions, not 5+5 to 7+3 "
            "or candidate groups 2, 5, 8, or 16.",
            "No predictive benefit, community superiority, accuracy gain, or inference "
            "acceleration is established.",
        ],
    }


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    jsd = [row["tier1_mean_jsd"] for row in rows]
    probability_delta = [row["tier1_max_top_probability_delta"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    axes[0].hist(jsd, bins=18, color="#3676a8", edgecolor="white")
    axes[0].set(
        title="Selected-position next-forward JSD",
        xlabel="Jensen-Shannon divergence",
        ylabel="Paired exchanges",
    )
    axes[1].hist(probability_delta, bins=18, color="#d17a32", edgecolor="white")
    axes[1].set(
        title="Top-token probability change",
        xlabel="Maximum absolute probability change",
        ylabel="Paired exchanges",
    )
    figure.suptitle("Saved pilot sensitivity diagnostics (100 paired exchanges)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def make_report(
    pilot_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    logits = summary["next_forward_logits_selected_positions"]
    outcomes = summary["final_outcomes"]
    trace = summary["refinement_trace_counters"]
    compute = summary["trajectory_compute_diagnostics"]
    nfe_counts = compute["nfe_delta_treatment_minus_control_counts"]
    pair_counts = compute["whole_trajectory_expert_token_pair_delta_counts"]
    fields = trace["different_counter_row_counts_on_common_post_target_rows"]
    return f"""# Previous 100 paired exchanges: saved-output analysis

## Data and method

Read-only reanalysis of `{pilot_dir}`. All 200 branch records form 100 complete exchanges across 50 prompts. Branch traces are joined by `(block_idx, global_nfe)`; unmatched forwards are not treated as paired. Prompt-cluster bootstrap intervals resample prompt IDs while retaining each sampled prompt's exchange rows (20,000 replicates; seed 20261011).

The pilot implementation defines its Tier 1 sensitivity measurements as JSD between control/treatment next-forward vocabulary distributions at the selected recipient and donor positions, averaged across those positions, and maximum absolute change in top-token probability across the two positions.

## Findings

- Correctness changed in **{outcomes['correctness_changed']}/100** pairs; control and treatment each had **{outcomes['control_correct']}/100** correct.
- Canonical extracted answers changed in **{outcomes['canonical_answers_changed']}/100** pairs.
- Generated sequences changed in **{outcomes['generated_sequences_changed']}/100** pairs ({outcomes['prompts_with_any_changed_sequence']}/50 prompts); raw generated strings changed in **{outcomes['raw_generated_sequences_changed']}/100**. Changed exchange IDs: `{', '.join(outcomes['changed_exchange_ids'])}`. Prompt-cluster bootstrap 95% interval for the pair change rate: **{outcomes['sequence_change_rate_prompt_cluster_bootstrap_95']}**.
- Selected-position next-forward JSD was nonzero in **{logits['nonzero_jsd_exchanges']}/100** pairs (mean **{logits['mean_jsd']:.7g}**, median **{logits['median_jsd']:.7g}**, maximum **{logits['max_jsd']:.7g}**; prompt-cluster bootstrap 95% interval for the mean **{logits['mean_jsd_prompt_cluster_bootstrap_95']}**). Maximum top-token probability change was nonzero in **{logits['nonzero_top_probability_change_exchanges']}/100** (mean **{logits['mean_max_absolute_top_probability_delta']:.7g}**, median **{logits['median_max_absolute_top_probability_delta']:.7g}**, maximum **{logits['max_max_absolute_top_probability_delta']:.7g}**; interval **{logits['top_probability_delta_prompt_cluster_bootstrap_95']}**).
- Aggregate refinement counters changed on the target forward in **{trace['target_forward_any_counter_change_exchanges']}/100** pairs and on at least one common later forward in **{trace['post_target_any_counter_change_exchanges']}/100** (prompt-cluster bootstrap 95% interval **{trace['post_target_counter_change_prompt_cluster_bootstrap_95']}**). Of **{trace['common_post_target_trace_rows']}** common later rows, **{trace['post_target_rows_with_any_counter_difference']}** differed in at least one counter. Differing-row counts by field: `{json.dumps(fields, sort_keys=True)}`. Trace-key sets differed in **{trace['exchanges_with_different_trace_key_sets']}/100** pairs.
- Treatment-minus-control NFE was lower/equal/higher in **{nfe_counts['negative']}/{nfe_counts['zero']}/{nfe_counts['positive']}** pairs (mean {compute['mean_nfe_delta_treatment_minus_control']:.3f}). Whole-trajectory expert-token-pair differences were lower/equal/higher in **{pair_counts['negative']}/{pair_counts['zero']}/{pair_counts['positive']}** pairs. No speedup is inferred.

## Conclusion

The saved one-unit K transfer changed the measured selected-position next-forward distributions in all 100 pairs, aggregate refinement counters for some pairs, and final generated sequences in three pairs, while changing no canonical answer or correctness label. This establishes sensitivity in the saved pilot, not a beneficial effect. The traces do not identify exact committed token positions, and the saved artifacts do not include every downstream logit tensor.

This pilot did not evaluate 5+5 → 7+3, intervention magnitude independently, or candidate groups of 2, 5, 8, and 16. Those experiments remain gated on the follow-up instructions and cumulative resource rules.

## Reproduction

```powershell
python scripts/analyze_jrc_v4_pilot_pairs.py `
  --pilot-dir outputs/jrc_v4_server_run_c854042/pilot `
  --output-dir outputs/jrc_v4_followup_baseline_analysis
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, trace_summary = analyze(pilot_dir)
    summary = summarize(pilot_dir, rows, trace_summary)
    write_csv(output_dir / "exchange_level_analysis.csv", rows)
    write_csv(output_dir / "trace_decision_comparison.csv", trace_summary["trace_rows"])
    write_csv(output_dir / "prompt_level_analysis.csv", prompt_rollup(rows, trace_summary["trace_rows"]))
    (output_dir / "baseline_analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "previous_100_exchange_report.md").write_text(
        make_report(pilot_dir, rows, summary), encoding="utf-8"
    )
    write_plot(output_dir / "next_forward_sensitivity.png", rows)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "exchanges": len(rows),
                "sequences_changed": summary["final_outcomes"]["generated_sequences_changed"],
                "target_counter_changed": trace_summary[
                    "target_forward_any_counter_change_exchanges"
                ],
                "post_target_counter_changed": trace_summary[
                    "post_target_any_counter_change_exchanges"
                ],
                "common_post_target_rows": trace_summary[
                    "common_post_target_trace_rows"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
