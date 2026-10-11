#!/usr/bin/env python3
"""Validate and summarize the saved JRC-MoE v4 paired pilot artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    point = (len(ordered) - 1) * probability
    low = math.floor(point)
    high = math.ceil(point)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "exact_zero_n": sum(value == 0.0 for value in values),
        "near_zero_lt_1e-6_n": sum(0.0 < abs(value) < 1e-6 for value in values),
        "nonzero_ge_1e-6_n": sum(abs(value) >= 1e-6 for value in values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "max": max(values) if values else None,
    }


def verify_remote_checksums(pilot: Path, checksum_path: Path) -> dict[str, Any]:
    """Compare sha256sum output from the server with every copied pilot file."""
    if not checksum_path.exists():
        return {"status": "not_supplied", "files_checked": 0}
    remote: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, remote_path = line.split(maxsplit=1)
        remote[Path(remote_path.strip().lstrip("* ")).name] = digest
    local_paths = [path for path in pilot.rglob("*") if path.is_file()]
    local = {path.name: sha256(path) for path in local_paths}
    missing = sorted(set(remote) - set(local))
    extra = sorted(set(local) - set(remote))
    mismatches = sorted(name for name in set(remote) & set(local) if remote[name] != local[name])
    return {
        "status": "passed" if not missing and not extra and not mismatches else "failed",
        "files_checked": len(set(remote) & set(local)),
        "remote_files": len(remote),
        "local_files": len(local),
        "missing_local": missing,
        "extra_local": extra,
        "mismatches": mismatches,
    }


def check_artifact_consistency(root: Path) -> dict[str, Any]:
    pilot = root / "pilot"
    manifest = read_json(pilot / "manifest.json")
    identity = read_json(pilot / "run_identity.json")
    progress = read_json(pilot / "progress.json")
    pilot_summary = read_json(pilot / "pilot_summary.json")
    audit_manifest_path = root / "audit" / "manifest.json"
    state_rows = read_jsonl(pilot / "state_index.jsonl")
    intervention_rows = read_jsonl(pilot / "interventions.jsonl")
    feature_rows = read_jsonl(pilot / "features.jsonl")
    paired_rows = read_jsonl(pilot / "paired_exchange_records.jsonl")
    outcomes = read_jsonl(pilot / "branch_outcomes.jsonl")
    failures = read_jsonl(pilot / "failures.jsonl")
    cost_rows = read_csv(pilot / "cost_ledger.csv")
    paired_csv = read_csv(pilot / "paired_exchanges.csv")

    expected_artifacts = set(manifest["artifacts"])
    present_artifacts = {
        str(path.relative_to(pilot)).replace("\\", "/")
        for path in pilot.rglob("*")
        if path.is_file()
    }
    missing_artifacts = sorted(
        item for item in expected_artifacts
        if item != "traces/" and item not in present_artifacts
    )
    checks: dict[str, Any] = {
        "manifest_declared_artifacts_present": not missing_artifacts,
        "missing_artifacts": missing_artifacts,
        "manifest_identity_matches_run_identity": all(
            manifest.get(key) == identity.get(key)
            for key in (
                "audit_manifest_sha256",
                "checkpoint_revision",
                "config_sha256",
                "fixed8_input_sha256",
                "git_commit",
                "protocol_version",
            )
        ),
        "audit_manifest_sha256_matches": (
            sha256(audit_manifest_path) == manifest["audit_manifest_sha256"]
        ),
        "state_count_50_unique_prompts": (
            len(state_rows) == 50
            and len({str(row["prompt_id"]) for row in state_rows}) == 50
        ),
        "intervention_count_100_unique_ids": (
            len(intervention_rows) == 100
            and len({row["exchange_id"] for row in intervention_rows}) == 100
        ),
        "paired_records_100_unique_ids": (
            len(paired_rows) == 100
            and len({row["exchange_id"] for row in paired_rows}) == 100
        ),
        "outcomes_200_two_arms_each": (
            len(outcomes) == 200
            and len({(row["exchange_id"], row["arm"]) for row in outcomes}) == 200
            and all(
                {row["arm"] for row in outcomes if row["exchange_id"] == exchange_id}
                == {"control", "treatment"}
                for exchange_id in {row["exchange_id"] for row in outcomes}
            )
        ),
        "failures_empty": len(failures) == 0,
        "paired_csv_matches_jsonl_count": len(paired_csv) == len(paired_rows),
        "target_layer_budget_neutral_all": all(
            int(row["target_layer_pair_delta"]) == 0 for row in paired_rows
        ),
        "all_zero_correctness_effects_retained": (
            len(paired_rows) == 100
            and all(int(row["effect"]) == 0 for row in paired_rows)
            and manifest["pilot_summary"]["effect_counts"] == {
                "negative": 0,
                "positive": 0,
                "zero": 100,
            }
        ),
        "attention_disabled_and_null": (
            manifest["attention_features"].startswith("disabled")
            and all(row.get("attention_scores") is None for row in feature_rows)
            and all(not row.get("extra", {}).get("attention_enabled", True) for row in feature_rows)
        ),
    }

    intervention_by_id = {row["exchange_id"]: row for row in intervention_rows}
    paired_by_id = {row["exchange_id"]: row for row in paired_rows}
    outcome_by_branch = {row["branch_id"]: row for row in outcomes}
    state_by_prompt = {str(row["prompt_id"]): row for row in state_rows}
    join_errors: list[str] = []
    for exchange_id, intervention in intervention_by_id.items():
        pair = paired_by_id.get(exchange_id)
        if pair is None:
            join_errors.append(f"missing paired record: {exchange_id}")
            continue
        for field in ("prompt_id", "block_idx", "global_nfe", "layer_idx", "recipient_position", "donor_position"):
            if str(pair[field]) != str(intervention[field]):
                join_errors.append(f"{exchange_id}: intervention/paired mismatch {field}")
        if pair.get("state_input_sha256") and pair["state_input_sha256"] != intervention["state_input_sha256"]:
            join_errors.append(f"{exchange_id}: intervention/paired state hash mismatch")
        state = state_by_prompt.get(str(intervention["prompt_id"]))
        if state is None or state["state_input_sha256"] != intervention["state_input_sha256"]:
            join_errors.append(f"{exchange_id}: captured state hash mismatch")
        branches = pair.get("branch_ids", {})
        for arm in ("control", "treatment"):
            outcome = outcome_by_branch.get(branches.get(arm, ""))
            if outcome is None or outcome["exchange_id"] != exchange_id or outcome["arm"] != arm:
                join_errors.append(f"{exchange_id}: missing/mismatched {arm} outcome")
    checks["intervention_pair_state_branch_joins"] = not join_errors
    checks["join_errors"] = join_errors[:30]

    ledger_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cost_rows:
        ledger_groups[row["branch_id"]].append(row)
    bad_ledger = {
        branch: [item["status"] for item in rows]
        for branch, rows in ledger_groups.items()
        if len(rows) != 2 or Counter(item["status"] for item in rows) != Counter({"started": 1, "completed": 1})
    }
    completed = [row for row in cost_rows if row["status"] == "completed"]
    expected_branches = {row["branch_id"] for row in outcomes}
    expected_capture_ids = {f"p{row['prompt_id']}:state-capture" for row in state_rows}
    ledger_ids = set(ledger_groups)
    completed_gpu_seconds = sum(float(row["gpu_seconds"] or 0.0) for row in completed)
    checks["cost_ledger_exactly_started_and_completed"] = (
        not bad_ledger
        and len(cost_rows) == 500
        and ledger_ids == expected_branches | expected_capture_ids
    )
    checks["cost_ledger_bad_ids"] = bad_ledger
    checks["cost_ledger_completed_gpu_seconds_match_progress"] = math.isclose(
        completed_gpu_seconds, float(progress["gpu_seconds"]), rel_tol=0.0, abs_tol=0.05
    )
    checks["cost_ledger_gpu_seconds"] = completed_gpu_seconds
    checks["progress_gpu_seconds"] = float(progress["gpu_seconds"])
    checks["trace_files_match_outcomes"] = (
        len(list((pilot / "traces").glob("*.csv"))) == 200
        and all((pilot / "traces" / Path(row["trace_path"]).name).is_file() for row in outcomes)
    )
    checks["feature_rows_expected_two_per_exchange"] = (
        len(feature_rows) == 200
        and len({(row["extra"]["exchange_id"], row["extra"]["role"]) for row in feature_rows}) == 200
    )
    checks["status_and_gate_state"] = {
        "pilot_status": manifest["status"],
        "G0": manifest["g0"],
        "G1": manifest["g1"],
        "gates": manifest["gates"],
        "attention": manifest["attention_features"],
        "branches_started": manifest["branches_started"],
        "prompts_completed": manifest["prompts_completed"],
        "gpu_seconds": manifest["gpu_seconds"],
        "gpu_hours": manifest["gpu_hours"],
        "output_bytes": progress["output_bytes"],
        "wall_seconds": progress["wall_seconds"],
        "failure_count": len(failures),
        "pilot_summary_status": pilot_summary.get("status", "status recorded by manifest"),
    }
    checks["all_boolean_checks_pass"] = all(value for value in checks.values() if isinstance(value, bool))
    return checks


def make_diagnostics(pilot: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = read_jsonl(pilot / "paired_exchange_records.jsonl")
    features = read_jsonl(pilot / "features.jsonl")
    outcomes = read_jsonl(pilot / "branch_outcomes.jsonl")
    by_exchange_features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in features:
        by_exchange_features[feature["extra"]["exchange_id"]].append(feature)
    by_exchange_outcomes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for outcome in outcomes:
        by_exchange_outcomes[outcome["exchange_id"]][outcome["arm"]] = outcome

    diagnostics: list[dict[str, Any]] = []
    strata: dict[tuple[str, str], list[float]] = defaultdict(list)
    numeric_feature_names = (
        "router_entropy",
        "admission_margin",
        "top_k_gate_mass",
        "next_expert_gate",
        "latest_confidence",
        "mask_ratio",
        "relative_position",
        "hidden_state_norm",
        "feature_age",
    )
    for pair in pairs:
        exchange_id = pair["exchange_id"]
        control, treatment = by_exchange_outcomes[exchange_id]["control"], by_exchange_outcomes[exchange_id]["treatment"]
        trace_files_differ = False
        jsd = float(pair["tier1_mean_jsd"])
        base = {
            "exchange_id": exchange_id,
            "prompt_id": pair["prompt_id"],
            "layer_idx": pair["layer_idx"],
            "global_nfe": pair["global_nfe"],
            "block_idx": pair["block_idx"],
            "recipient_position": pair["recipient_position"],
            "donor_position": pair["donor_position"],
            "tier1_mean_jsd_pair_average": jsd,
            "tier1_max_top_probability_delta_pair_max": float(pair["tier1_max_top_probability_delta"]),
            "canonical_answer_changed": control["canonical_prediction"] != treatment["canonical_prediction"],
            "correctness_changed": int(control["correctness"]) != int(treatment["correctness"]),
            "generated_text_changed": control["generated_text"] != treatment["generated_text"],
            "raw_generated_text_changed": control["raw_generated_text"] != treatment["raw_generated_text"],
            "control_nfe": control["total_nfe"],
            "treatment_nfe": treatment["total_nfe"],
            "control_total_pairs": control["total_pairs"],
            "treatment_total_pairs": treatment["total_pairs"],
            "trajectory_pair_delta": pair["trajectory_pair_delta"],
            "target_layer_pair_delta": pair["target_layer_pair_delta"],
        }
        diagnostic_pairs = [
            ("exchange", None),
            *[(str(item["extra"]["role"]), item) for item in by_exchange_features[exchange_id]],
        ]
        for unit, feature in diagnostic_pairs:
            row = dict(base)
            row["unit"] = unit
            for key in (
                "position", "is_prompt_token", "is_target_token", "unresolved", "provisional",
                "active_block", "router_history", "confidence_history", "router",
            ):
                row[key] = feature.get(key) if feature else None
            for name in numeric_feature_names:
                if name == "latest_confidence":
                    history = feature.get("confidence_history", []) if feature else []
                    value = history[-1] if history else None
                elif name in {"router_entropy", "admission_margin", "top_k_gate_mass", "next_expert_gate"}:
                    value = feature.get("router", {}).get(name) if feature else None
                else:
                    value = feature.get(name) if feature else None
                row[name] = value
                if value is not None:
                    strata[(name, str(value))].append(jsd)
            row["attention_available"] = bool(feature and feature.get("attention_scores") is not None)
            diagnostics.append(row)

    summary: dict[str, Any] = {
        "exchange_count": len(pairs),
        "position_records": len(features),
        "tier1_pair_average_jsd": describe([float(row["tier1_mean_jsd"]) for row in pairs]),
        "tier1_max_top_probability_delta": describe([float(row["tier1_max_top_probability_delta"]) for row in pairs]),
        "strata": {},
    }
    # Quartile bins avoid making the report depend on arbitrary raw-value precision.
    all_features = diagnostics
    for name in numeric_feature_names:
        observed = [float(row[name]) for row in all_features if row[name] is not None]
        cuts = [quantile(observed, q) for q in (0.25, 0.50, 0.75)]
        groups: dict[str, list[float]] = defaultdict(list)
        for row in all_features:
            value = row[name]
            if value is None:
                groups["missing"].append(float(row["tier1_mean_jsd_pair_average"]))
                continue
            numeric = float(value)
            bucket = f"Q{1 + sum(numeric > cut for cut in cuts if cut is not None)}"
            groups[bucket].append(float(row["tier1_mean_jsd_pair_average"]))
        summary["strata"][name] = {
            "quartile_cutpoints": cuts,
            "bins": {key: describe(values) for key, values in sorted(groups.items())},
            "interpretation": "pair-averaged JSD repeated for each selected token; descriptive only",
        }

    for field in ("layer_idx", "global_nfe", "unit"):
        groups: dict[str, list[float]] = defaultdict(list)
        for row in diagnostics:
            if row["unit"] == "exchange" and field == "unit":
                groups[str(row[field])].append(float(row["tier1_mean_jsd_pair_average"]))
            elif row["unit"] != "exchange" and field != "unit":
                groups[str(row[field])].append(float(row["tier1_mean_jsd_pair_average"]))
        summary["strata"][field] = {key: describe(values) for key, values in sorted(groups.items())}

    for field in ("unit", "is_prompt_token", "is_target_token", "unresolved", "provisional", "active_block"):
        groups: dict[str, list[float]] = defaultdict(list)
        for row in diagnostics:
            if row["unit"] == "exchange":
                continue
            groups[str(row[field])].append(float(row["tier1_mean_jsd_pair_average"]))
        summary["strata"][f"category_{field}"] = {
            key: describe(values) for key, values in sorted(groups.items())
        }
    position_groups: dict[str, list[float]] = defaultdict(list)
    role_groups: dict[str, list[float]] = defaultdict(list)
    for row in diagnostics:
        if row["unit"] == "exchange":
            continue
        position_groups[str(row["position"])].append(float(row["tier1_mean_jsd_pair_average"]))
        role_groups[str(row["unit"])].append(float(row["tier1_mean_jsd_pair_average"]))
    summary["strata"]["candidate_position"] = {
        key: describe(values) for key, values in sorted(position_groups.items(), key=lambda item: int(item[0]))
    }
    summary["strata"]["candidate_role"] = {
        key: describe(values) for key, values in sorted(role_groups.items())
    }
    summary["feature_missingness"] = {
        field: {
            "missing": sum(row[field] is None for row in diagnostics if row["unit"] != "exchange"),
            "present": sum(row[field] is not None for row in diagnostics if row["unit"] != "exchange"),
        }
        for field in numeric_feature_names
    }
    summary["feature_missingness"].update({
        "attention_scores": {"missing": len(features), "present": 0},
        "final_token_ids": {"missing": len(outcomes), "present": 0},
        "per_token_commit_identities": {"missing": len(outcomes), "present": 0},
    })

    summary["limitations"] = [
        "Tier-1 values are exchange-level averages over recipient and donor; they are repeated across the two feature rows and are not token-specific logits.",
        "No complete downstream-logit tensors or committed token identities are saved; exact downstream logit/decision changes and edit distance cannot be reconstructed.",
        "Trace counters record aggregate decision quantities, not the exact token commitment chosen at each forward.",
        "Attention scores are absent and the pilot explicitly disabled attention collection.",
    ]
    return diagnostics, summary


def write_diagnostics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def write_plot(path: Path, pilot: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = read_jsonl(pilot / "paired_exchange_records.jsonl")
    jsd = [float(row["tier1_mean_jsd"]) for row in pairs]
    delta = [float(row["tier1_max_top_probability_delta"]) for row in pairs]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, values, title, xlabel in (
        (axes[0], jsd, "Pair-average selected-position JSD", "Jensen-Shannon divergence"),
        (axes[1], delta, "Max top-token probability delta", "Absolute probability change"),
    ):
        axis.hist(values, bins=20, color="#3d7195", edgecolor="white")
        axis.axvline(0.0, color="#9c332c", linewidth=1)
        axis.set(title=title, xlabel=xlabel, ylabel="Paired exchanges")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Saved 100-exchange pilot: selected-position sensitivity")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-checksums", type=Path)
    args = parser.parse_args()
    root = args.server_run_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    checks = check_artifact_consistency(root)
    if args.remote_checksums:
        checks["remote_checksum_comparison"] = verify_remote_checksums(root / "pilot", args.remote_checksums)
    diagnostics, analysis = make_diagnostics(root / "pilot")
    write_diagnostics_csv(output / "prior_intervention_diagnostics.csv", diagnostics)
    (output / "forensic_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    (output / "tier1_distribution_summary.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    write_plot(output / "prior_tier1_distribution.png", root / "pilot")
    data_dir = output.parent / "data"
    figures_dir = output.parent / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "prior_intervention_diagnostics.csv").write_bytes(
        (output / "prior_intervention_diagnostics.csv").read_bytes()
    )
    (figures_dir / "prior_tier1_distribution.png").write_bytes(
        (output / "prior_tier1_distribution.png").read_bytes()
    )
    pilot_checksums = {
        str(path.relative_to(root / "pilot")).replace("\\", "/"): sha256(path)
        for path in sorted((root / "pilot").rglob("*"))
        if path.is_file()
    }
    output_paths = [
        output / "prior_intervention_diagnostics.csv",
        output / "prior_tier1_distribution.png",
        output / "tier1_distribution_summary.json",
        output / "forensic_checks.json",
        data_dir / "prior_intervention_diagnostics.csv",
        figures_dir / "prior_tier1_distribution.png",
    ]
    forensic_manifest = {
        "schema": "jrc-v4-followup-forensics-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_root": str(root),
        "source_run_identity": read_json(root / "pilot" / "run_identity.json"),
        "source_manifest_sha256": sha256(root / "pilot" / "manifest.json"),
        "source_audit_manifest_sha256": sha256(root / "audit" / "manifest.json"),
        "source_files_sha256": pilot_checksums,
        "server_local_checksum_reconciliation": checks.get("remote_checksum_comparison"),
        "cross_file_checks": checks,
        "analysis": analysis,
        "output_files_sha256": {str(path.relative_to(output.parent)).replace("\\", "/"): sha256(path)
                                for path in output_paths},
    }
    (output / "forensic_manifest.json").write_text(
        json.dumps(forensic_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "checks_passed": checks["all_boolean_checks_pass"], "checks": checks, "tier1": analysis["tier1_pair_average_jsd"]}, indent=2))


if __name__ == "__main__":
    main()
