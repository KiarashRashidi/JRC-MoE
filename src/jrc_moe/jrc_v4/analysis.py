"""Prompt-disjoint exploratory analysis of completed pilot exchange labels."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .predictors import RidgeBenefitPredictor
from .sampling import IndexedRandom
from .statistics import kendall_tau_b, paired_prompt_bootstrap
from .io import atomic_write_json


ROUTER_KEYS = ("router_entropy", "top_k_gate_mass", "admission_margin", "next_expert_gate", "selected_k")


def _token_features(row: dict) -> np.ndarray:
    history = [float(x) for x in row.get("confidence_history", [])]
    hidden = row.get("hidden_state_norm")
    return np.asarray([
        float(row["mask_ratio"]), float(row["is_target_token"]), float(row["active_block"]),
        float(row["relative_position"]), float(row["unresolved"]),
        history[-1] if history else 0.0, float(np.mean(history)) if history else 0.0,
        history[-1] - history[0] if len(history) > 1 else 0.0,
        float(len(history)), float(hidden or 0.0), float(row["global_nfe"]), float(row["layer_idx"]),
    ], dtype=np.float64)


def _router_features(row: dict) -> np.ndarray:
    return np.asarray([float(row["router"][key]) for key in ROUTER_KEYS], dtype=np.float64)


def _role_aggregates(role_rows: dict[str, list[dict]], extractor, *, width: int, denominator: float) -> np.ndarray:
    """Return deterministic mean/min/max summaries by action role and all candidates."""
    chunks = []
    for role in ("recipient", "donor", "unchanged_candidate", "all"):
        rows = role_rows.get("all", []) if role == "all" else role_rows.get(role, [])
        values = np.stack([extractor(row) for row in rows]) if rows else np.zeros((0, width), dtype=np.float64)
        if len(values):
            chunks.extend((values.mean(axis=0), values.min(axis=0), values.max(axis=0)))
        else:
            chunks.extend((np.zeros(width), np.zeros(width), np.zeros(width)))
        chunks.append(np.asarray([len(values) / max(1.0, denominator)], dtype=np.float64))
    return np.concatenate(chunks)


def _json_list(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def _sign_concordance(predictions: list[float], labels: list[float]) -> dict:
    nonzero_labels = sum(y != 0 for y in labels)
    eligible = [(p, y) for p, y in zip(predictions, labels) if y != 0 and p != 0]
    if not eligible:
        return {"nonzero_labels": nonzero_labels, "sign_decisions": 0,
                "coverage": 0.0, "sign_concordance": None}
    correct = sum((p > 0) == (y > 0) for p, y in eligible)
    return {"nonzero_labels": nonzero_labels, "sign_decisions": len(eligible),
            "coverage": len(eligible) / max(1, nonzero_labels),
            "sign_concordance": correct / len(eligible)}


def run_pilot_analysis(*, pilot_dir: Path, output_dir: Path, seed: int = 20261009,
                       bootstrap_resamples: int = 2000) -> dict:
    """Fit low-capacity controls on development prompts and score held-out pilot prompts.

    The pilot data are historical GSM8K development data, never confirmatory. Since
    verified post-RoPE attention features are not available, the joint attention /
    router hypothesis is explicitly non-estimable and G1 cannot pass here.
    """
    pilot_manifest_path = pilot_dir / "manifest.json"
    if not pilot_manifest_path.is_file():
        raise FileNotFoundError(f"pilot manifest is missing: {pilot_manifest_path}")
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    if pilot_manifest.get("status") != "passed" or pilot_manifest.get("g0") != "passed":
        raise RuntimeError("analysis requires a completed pilot with passed G0")
    split_path = pilot_dir / "dataset_split_manifest.json"
    feature_path = pilot_dir / "features.jsonl"
    outcome_path = pilot_dir / "paired_exchange_records.jsonl"
    for path in (split_path, feature_path, outcome_path):
        if not path.is_file():
            raise FileNotFoundError(f"required pilot artifact is missing: {path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development = {str(x) for x in split.get("development", [])}
    validation = {str(x) for x in split.get("validation", [])}
    if not development or not validation or development & validation:
        raise ValueError("pilot split must have nonempty disjoint development and validation prompts")

    features: dict[str, dict[int, dict]] = {}
    for line in feature_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        exchange_id = row.get("extra", {}).get("exchange_id")
        role = row.get("extra", {}).get("role")
        if exchange_id and role in {"recipient", "donor", "unchanged_candidate"}:
            # A position has one pre-decision record per action. If a resumed
            # run appended a duplicate, retain one copy rather than changing
            # its statistical weight.
            features.setdefault(str(exchange_id), {})[int(row["position"])] = row
    outcomes = [json.loads(line) for line in outcome_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    outcomes.sort(key=lambda row: str(row["exchange_id"]))
    rows = []
    for outcome in outcomes:
        prompt = str(outcome["prompt_id"])
        candidate_rows = list(features.get(str(outcome["exchange_id"]), {}).values())
        if not candidate_rows:
            continue
        role_rows = {role: [row for row in candidate_rows if row.get("extra", {}).get("role") == role]
                     for role in ("recipient", "donor", "unchanged_candidate")}
        if not role_rows["recipient"] or not role_rows["donor"]:
            continue
        role_rows["all"] = candidate_rows
        m = int(outcome.get("m") or len(candidate_rows))
        c = int(outcome.get("c") or (len(role_rows["recipient"]) + len(role_rows["donor"])))
        amount = int(outcome.get("A") or 1)
        before = _json_list(outcome.get("K_before"))
        base_k = float(np.mean([float(k) for k in before])) if before else 6.0
        global_nfe = float(outcome.get("global_nfe", candidate_rows[0].get("global_nfe", 0)))
        action_x = np.asarray([m / 16.0, c / 16.0, amount / 8.0, base_k / 8.0,
                               float(outcome["layer_idx"]) / 15.0,
                               global_nfe / 256.0], dtype=np.float64)
        token_x = np.concatenate((_role_aggregates(role_rows, _token_features, width=12, denominator=m), action_x))
        router_x = np.concatenate((_role_aggregates(role_rows, _router_features, width=len(ROUTER_KEYS), denominator=m),
                                   action_x))
        joint_x = np.concatenate((token_x[:-len(action_x)], router_x[:-len(action_x)], action_x))
        rows.append({"exchange_id": str(outcome["exchange_id"]), "prompt_id": prompt,
                     "label": float(outcome["effect"]), "token_x": token_x,
                     "router_x": router_x, "joint_x": joint_x,
                     "role_features": role_rows, "m": m, "c": c, "A": amount})
    train = [r for r in rows if r["prompt_id"] in development]
    valid = [r for r in rows if r["prompt_id"] in validation]
    if not train or not valid:
        raise ValueError("no complete prompt-disjoint feature/outcome rows for both splits")

    models = {
        "token_only": RidgeBenefitPredictor(alpha=1.0).fit(
            np.stack([r["token_x"] for r in train]), [r["label"] for r in train],
            [r["prompt_id"] for r in train], allowed_training_prompts=development),
        "router_only": RidgeBenefitPredictor(alpha=1.0).fit(
            np.stack([r["router_x"] for r in train]), [r["label"] for r in train],
            [r["prompt_id"] for r in train], allowed_training_prompts=development),
        "ungrouped_joint": RidgeBenefitPredictor(alpha=1.0).fit(
            np.stack([r["joint_x"] for r in train]), [r["label"] for r in train],
            [r["prompt_id"] for r in train], allowed_training_prompts=development),
    }
    feature_keys = {"token_only": "token_x", "router_only": "router_x",
                    "ungrouped_joint": "joint_x"}
    predicted = {name: model.predict(np.stack([r[feature_keys[name]] for r in valid]),
                                     prompt_ids=[r["prompt_id"] for r in valid])
                 for name, model in models.items()}
    # An independently seeded random ranking provides a transparent low bar.
    randomizer = IndexedRandom(seed)
    random_scores = [randomizer.stream(prompt_id=r["prompt_id"], stream_id="analysis-random",
                                       step=0, block=0,
                                       position=int.from_bytes(hashlib.sha256(r["exchange_id"].encode()).digest()[:8], "big"),
                                       purpose="rank").random()
                     for r in valid]
    predicted["random"] = np.asarray(random_scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    metrics = {}
    valid_by_prompt: dict[str, list[int]] = {}
    for index, row in enumerate(valid):
        valid_by_prompt.setdefault(row["prompt_id"], []).append(index)
    for name, scores in predicted.items():
        decision_rows = []
        state_taus = []
        for prompt, indices in sorted(valid_by_prompt.items()):
            labels = [valid[i]["label"] for i in indices]
            local_scores = [float(scores[i]) for i in indices]
            if len(indices) >= 2:
                tau = kendall_tau_b(local_scores, labels)
                if tau is not None:
                    state_taus.append(tau)
                selected = max(range(len(indices)), key=lambda j: (local_scores[j], -j))
                selected_effect = labels[selected]
                random_baseline = float(np.mean(labels))
                decision_rows.append({"prompt_id": prompt, "selected": selected_effect,
                                      "random_expected": random_baseline})
            else:
                selected_effect = labels[0]
                decision_rows.append({"prompt_id": prompt, "selected": selected_effect,
                                      "random_expected": selected_effect})
        gain_rows = [{"prompt_id": r["prompt_id"], "treatment": r["selected"],
                      "control": r["random_expected"]} for r in decision_rows]
        gain = paired_prompt_bootstrap(gain_rows, treatment_key="treatment", comparator_key="control",
                                       resamples=bootstrap_resamples, seed=seed)
        signs = (_sign_concordance([float(x) for x in scores], [r["label"] for r in valid])
                 if name != "random" else {"nonzero_labels": sum(r["label"] != 0 for r in valid),
                                            "sign_decisions": 0, "coverage": 0.0,
                                            "sign_concordance": None})
        metrics[name] = {"validation_rows": len(valid), "validation_prompts": len(valid_by_prompt),
                         "mean_within_prompt_kendall_tau_b": float(np.mean(state_taus)) if state_taus else None,
                         "mean_selected_effect_gain_vs_random_expectation": gain,
                         **signs}
        for row, score in zip(valid, scores):
            prediction_rows.append({"model": name, "exchange_id": row["exchange_id"],
                                    "prompt_id": row["prompt_id"], "prediction": float(score),
                                    "observed_effect": row["label"]})

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "exchange_id", "prompt_id", "prediction", "observed_effect"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    label_values = sorted({float(row["label"]) for row in rows})
    g1_status = ("not_estimable_constant_binary_outcome" if len(label_values) <= 1
                 else "blocked_missing_verified_attention_features")
    report = {
        "stage": "signal", "status": "exploratory_controls_complete",
        "g1": g1_status,
        "data_scope": "historical GSM8K-50 development/pilot only; not confirmatory",
        "split": {"development_prompts": len(development), "validation_prompts": len(validation),
                  "confirmatory_prompts": len(split.get("confirmatory", [])), "sha256": split["sha256"]},
        "rows": {"all_complete": len(rows), "train": len(train), "validation": len(valid)},
        "binary_label_values": label_values,
        "predictors": metrics,
        "predictor_specs": {
            "token_only": "confidence history, mask/refinement state, position, hidden-state norm, plus m/c/A/base-K/layer/NFE action covariates; recipient/donor/unchanged/all role summaries; no router or attention values",
            "router_only": "router entropy, top-K mass, admission margin, next-expert gate, selected-K, plus action covariates; role summaries",
            "ungrouped_joint": "ordinary ridge over concatenated token-only and router-only summaries; no community structure",
            "random": "independent indexed random ranking",
        },
        "unavailable_comparators": ["attention-only", "attention-plus-router joint"],
        "reason": ("Binary outcome is constant in the available rows; predictive value is not estimable. Attention features were disabled in the sensitivity run."
                   if len(label_values) <= 1 else
                   "Token-only, router-only, ungrouped-joint, and random controls were evaluated; attention-only and attention-plus-router comparisons remain blocked until runtime feature fidelity is verified and those features are collected."),
        "later_gates": {"G2": "not_run", "G3": "not_run", "G4": "not_run"},
    }
    atomic_write_json(output_dir / "signal_report.json", report)
    (output_dir / "signal_report.md").write_text(
        "# Exploratory pilot signal analysis\n\n"
        f"Status: **{report['status']}**. G1: **{report['g1']}**.\n\n"
        f"Development/validation prompts: {len(development)}/{len(validation)}. "
        f"Complete paired rows: {len(rows)} (train {len(train)}, held-out validation {len(valid)}).\n\n"
        "Token-only, router-only, and ungrouped-joint ridge controls were fit on development prompts only, with action dose and context covariates. "
        "Attention-only and attention-plus-router comparisons are unavailable, so the joint-signal gate remains blocked. "
        "These exploratory metrics do not support H1/H2 or confirmatory claims.\n",
        encoding="utf-8")
    return report
