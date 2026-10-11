"""Summarize bounded JRC-v4 D1 follow-up outcomes by prompt-clustered condition."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap_by_prompt(rows: list[dict], value, *, seed: int, draws: int = 5000) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        number = value(row)
        if number is not None and np.isfinite(float(number)):
            grouped[str(row["prompt_id"])].append(float(number))
    if not grouped:
        return {"prompts": 0, "mean": None, "ci95": [None, None]}
    prompt_values = np.asarray([np.mean(values) for values in grouped.values()], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(prompt_values, size=(draws, len(prompt_values)), replace=True), axis=1)
    return {"prompts": int(len(prompt_values)), "mean": float(prompt_values.mean()),
            "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "bootstrap_unit": "prompt"}


def summarize(input_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(input_dir / "paired_exchange_records.jsonl")
    actions = read_jsonl(input_dir / "actions.jsonl")
    branches = read_jsonl(input_dir / "branch_outcomes.jsonl")
    features = read_jsonl(input_dir / "features.jsonl")
    attention_path = input_dir.parent / "attention_probe.json"
    attention_probe = json.loads(attention_path.read_text(encoding="utf-8")) if attention_path.exists() else None
    tier_rows: list[dict] = []
    tier_path = input_dir / "tier1_metrics.csv"
    if tier_path.exists():
        with tier_path.open("r", encoding="utf-8", newline="") as stream:
            tier_rows = list(csv.DictReader(stream))

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition") or "unspecified")].append(row)
    condition_summaries = []
    for condition, group in sorted(by_condition.items()):
        effects = [int(row["effect"]) for row in group]
        jsd = [float(row["tier1_mean_jsd"]) for row in group
               if row.get("tier1_mean_jsd") not in (None, "")]
        changed = [int(row.get("final_token_edit_count") or 0) for row in group]
        trace_changed = [int(row.get("trace_changed_nfe_count") or 0) for row in group]
        correctness = [int(row["treatment_correct"]) - int(row["control_correct"]) for row in group]
        condition_summaries.append({
            "condition": condition, "m": group[0].get("m"), "c": group[0].get("c"),
            "A": group[0].get("A"), "selection_strategy": group[0].get("selection_strategy"),
            "pairs": len(group), "prompts": len({str(row["prompt_id"]) for row in group}),
            "effect_negative": sum(value < 0 for value in effects),
            "effect_zero": sum(value == 0 for value in effects),
            "effect_positive": sum(value > 0 for value in effects),
            "mean_treatment_minus_control_correctness": float(np.mean(correctness)),
            "correctness_prompt_bootstrap": bootstrap_by_prompt(
                group, lambda row: int(row["treatment_correct"]) - int(row["control_correct"]), seed=20261011),
            "target_jsd_mean": float(np.mean(jsd)) if jsd else None,
            "target_jsd_prompt_bootstrap": bootstrap_by_prompt(
                group, lambda row: row.get("tier1_mean_jsd"), seed=20261012),
            "pairs_with_sequence_edits": sum(value > 0 for value in changed),
            "mean_final_token_edits": float(np.mean(changed)) if changed else None,
            "pairs_with_trace_aggregate_changes": sum(value > 0 for value in trace_changed),
            "mean_trace_changed_nfes": float(np.mean(trace_changed)) if trace_changed else None,
            "mean_trajectory_pair_delta": float(np.mean([int(row["trajectory_pair_delta"]) for row in group])),
        })

    group_summaries = []
    dimensions: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("m")), str(row.get("c")), str(row.get("A")),
               str(row.get("selection_strategy")))
        dimensions[key].append(row)
    for (m, c, amount, strategy), group in sorted(dimensions.items()):
        group_summaries.append({
            "m": int(m), "c": int(c), "A": int(amount), "selection_strategy": strategy,
            "pairs": len(group), "prompts": len({str(row["prompt_id"]) for row in group}),
            "effect_nonzero": sum(int(row["effect"]) != 0 for row in group),
            "target_jsd_mean": float(np.mean([float(row["tier1_mean_jsd"]) for row in group
                                               if row.get("tier1_mean_jsd") not in (None, "")]))
            if any(row.get("tier1_mean_jsd") not in (None, "") for row in group) else None,
            "sequence_change_pairs": sum(int(row.get("final_token_edit_count") or 0) > 0 for row in group),
        })

    tier_offsets: dict[int, list[float]] = defaultdict(list)
    comparable_positions = 0
    for row in tier_rows:
        if row.get("jsd") not in (None, ""):
            tier_offsets[int(row["offset_from_intervention"])].append(float(row["jsd"]))
        if row.get("compared_positions") not in (None, ""):
            comparable_positions += int(row["compared_positions"])
    router_fields = defaultdict(int)
    for row in features:
        for name in (row.get("router") or {}):
            router_fields[name] += 1
    effect_values = [int(row["effect"]) for row in rows]
    gate = ("not_estimable_constant_binary_outcome" if effect_values and len(set(effect_values)) == 1
            else "not_run_attention_features_unverified" if rows else "not_estimable_no_completed_pairs")
    summary = {
        "stage": manifest.get("stage"), "run_status": manifest.get("status"),
        "g0": manifest.get("g0"), "g1": gate,
        "scientific_claims": "exploratory_descriptive_only; no community, predictive, utility, or speedup claim",
        "paired_outcomes": len(rows), "actions_logged": len({str(row.get("exchange_id")) for row in actions}),
        "branches_logged": len(branches),
        "distinct_prompts": len({str(row["prompt_id"]) for row in rows}),
        "binary_effect_values": sorted(set(effect_values)),
        "binary_effect_counts": {"negative": sum(value < 0 for value in effect_values),
                                 "zero": sum(value == 0 for value in effect_values),
                                 "positive": sum(value > 0 for value in effect_values)},
        "prompt_bootstrap_overall_correctness_difference": bootstrap_by_prompt(
            rows, lambda row: int(row["treatment_correct"]) - int(row["control_correct"]), seed=20261010),
        "target_jsd_prompt_bootstrap": bootstrap_by_prompt(rows, lambda row: row.get("tier1_mean_jsd"), seed=20261012),
        "aligned_logit_jsd_by_nfe_offset": {
            str(offset): {"n": len(values), "mean": float(np.mean(values)),
                          "median": float(np.median(values)),
                          "nonzero": sum(value > 0 for value in values)}
            for offset, values in sorted(tier_offsets.items())},
        "aligned_candidate_position_comparisons": comparable_positions,
        "router_feature_rows": len(features), "router_feature_field_coverage": dict(router_fields),
        "attention_feature_rows_with_scores": sum(
            bool(row.get("attention_scores")) for row in features),
        "attention_runtime_probe": ({"status": attention_probe.get("status"),
                                     "backend": attention_probe.get("loaded_backend"),
                                     "source_sha256": attention_probe.get("source_sha256"),
                                     "reference_error": attention_probe.get("maximum_backend_reference_error"),
                                     "bidirectional_mask_verified": attention_probe.get("runtime_mask_is_none_and_is_causal_false")}
                                    if attention_probe else "not_run"),
        "per_condition": condition_summaries, "per_m_c_A_strategy": group_summaries,
        "resource_accounting": {key: manifest.get(key) for key in
                                ("branches_started", "gpu_seconds", "gpu_hours", "resource_caps",
                                 "states_captured", "stop_reason", "infeasible_or_no_site")},
        "manifest_path": str(input_dir / "manifest.json"),
    }
    (output_dir / "followup_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                                         encoding="utf-8")
    with (output_dir / "condition_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(condition_summaries[0]) if condition_summaries else ["condition", "pairs"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(condition_summaries)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for axis, filter_fn, title, xlabel in (
            (axes[0], lambda r: int(r["m"]) == 2 and int(r["c"]) == 2,
             "Magnitude at m=2, c=2", "Transferred expert slots A"),
            (axes[1], lambda r: int(r["A"]) == 2 and int(r["c"]) == 4,
             "Candidate size at A=2, c=4", "Candidate group size m"),
            (axes[2], lambda r: int(r["m"]) == 5 and int(r["A"]) == 2,
             "Changed-token count at m=5, A=2", "Changed tokens c"),
        ):
            selected = [r for r in rows if filter_fn(r) and r.get("tier1_mean_jsd") not in (None, "")]
            xs = sorted({int(r["A"] if axis is axes[0] else r["m"] if axis is axes[1] else r["c"])
                         for r in selected})
            means = [float(np.mean([float(r["tier1_mean_jsd"]) for r in selected
                                    if int(r["A"] if axis is axes[0] else r["m"] if axis is axes[1]
                                           else r["c"]) == x])) for x in xs]
            axis.plot(xs, means, marker="o")
            axis.set(title=title, xlabel=xlabel, ylabel="Target-NFE mean token JSD")
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "dose_context_sensitivity.png", dpi=160)
        plt.close(fig)
    except ImportError:
        summary["plot_status"] = "matplotlib_missing"
        (output_dir / "followup_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                                             encoding="utf-8")

    report_lines = [
        "# JRC-MoE v4 bounded follow-up sensitivity report", "",
        f"Run status: **{manifest.get('status')}**; G0: **{manifest.get('g0')}**; G1: **{gate}**.",
        f"Completed paired outcomes: {len(rows)} over {summary['distinct_prompts']} prompts; "
        f"branches logged: {len(branches)}.", "",
        "All estimates are exploratory and prompt-clustered where intervals are reported. "
        "A nonzero next-token logit difference is a mechanism diagnostic, not a utility gain. "
        "The saved refinement columns are aggregate commit/fallback counts; they do not identify each committed token.", "",
        "## Condition summaries", "",
        "| Condition | m | c | A | Pairs | Prompts | Correctness effect (−/0/+) | Mean target JSD | Sequence edits | Trace changes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in condition_summaries:
        report_lines.append(
            f"| {row['condition']} | {row['m']} | {row['c']} | {row['A']} | {row['pairs']} | {row['prompts']} | "
            f"{row['effect_negative']}/{row['effect_zero']}/{row['effect_positive']} | "
            f"{row['target_jsd_mean']} | {row['pairs_with_sequence_edits']} | "
            f"{row['pairs_with_trace_aggregate_changes']} |")
    report_lines += ["", "## Interpretation", "",
                     f"Binary final-correctness effects: {summary['binary_effect_counts']}. "
                     f"G1 is {gate}; no predictive, community, inference-acceleration, or community-superiority claim is authorized.",
                     (f"The separate runtime attention probe status is {attention_probe.get('status')}; "
                      "this sensitivity run did not enable or persist attention features, so no attention predictor comparison is claimed."
                      if attention_probe else
                      "Attention features remain disabled because the actual loaded attention backend and post-QK-normalization/post-RoPE reference scores have not been verified."), ""]
    (output_dir / "followup_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input_dir, args.output_dir)
    print(json.dumps({key: result.get(key) for key in
                      ("run_status", "g0", "g1", "paired_outcomes", "distinct_prompts",
                       "binary_effect_counts", "resource_accounting")}, indent=2))
    return 0 if result["run_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
