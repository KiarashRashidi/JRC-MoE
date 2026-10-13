"""Bounded confidence-guided sparse K=1 study on complete REFLEX rollouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _config(path: Path) -> dict:
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol_version") != "jrc-v4-stochastic-sparse-k1-1":
        raise ValueError("unexpected stochastic K=1 protocol config")
    policy = raw["policy"]
    if policy["missing_confidence_action"] != "retain_k8":
        raise ValueError("cold-start policy must preserve K=8")
    if not (0 <= policy["high_confidence_k1_probability"] <= policy["low_confidence_k1_probability"] <= 1):
        raise ValueError("K=1 probability must be higher at low confidence")
    return raw


def _choose_prompts(frame, *, seed: int, correct_count: int, incorrect_count: int,
                    grader) -> tuple[list[int], dict[int, int]]:
    tagged = frame.copy()
    tagged["_canonical_correct"] = tagged.apply(
        lambda row: int(grader(str(row["clean_generated_text"]),
                               str(row["gold_full_answer"]))["canonical_correct"]), axis=1)
    rng = random.Random(seed)
    chosen = []
    strata = {}
    for correct, count in ((1, correct_count), (0, incorrect_count)):
        available = tagged[tagged["_canonical_correct"] == correct].sort_values("diagnostic_ordinal")
        if len(available) < count:
            raise ValueError(f"need {count} baseline {'correct' if correct else 'incorrect'} prompts; found {len(available)}")
        indices = rng.sample(available["test_idx"].astype(int).tolist(), count)
        chosen.extend(indices)
        strata.update({int(index): int(correct) for index in indices})
    ordinal = tagged.set_index("test_idx")["diagnostic_ordinal"].to_dict()
    chosen.sort(key=lambda index: (int(ordinal[index]), index))
    return chosen, strata


def _seed_torch(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fixed8_provider(*, num_layers: int):
    def provider(*, prompt_len, unresolved_abs, confidence_history, global_nfe, **kwargs):
        return [[8] * (int(prompt_len) + 256) for _ in range(num_layers)], {
            "variant": "fixed8_control", "global_nfe": int(global_nfe),
            "k1_assignment_count": 0, "selected": [],
        }
    return provider


def _policy_provider(*, variant: str, num_layers: int, seed: int,
                     prompt_id: str, config: dict):
    from .stochastic_k1_policy import build_sparse_k1_plan
    policy = config["policy"]
    def provider(*, prompt_len, block_idx, unresolved_abs, confidence_history,
                 global_nfe, **kwargs):
        sequence_length = int(prompt_len) + int(config["decoder"]["generation_length"])
        return build_sparse_k1_plan(
            sequence_length=sequence_length, num_layers=num_layers,
            unresolved_positions=unresolved_abs,
            confidence_history=confidence_history, variant=variant,
            seed=seed, prompt_id=prompt_id, block_idx=block_idx,
            global_nfe=global_nfe, base_k=int(config["decoder"]["baseline_k"]),
            low_confidence=float(policy["low_confidence_boundary"]),
            high_confidence=float(policy["high_confidence_boundary"]),
            low_probability=float(policy["low_confidence_k1_probability"]),
            high_probability=float(policy["high_confidence_k1_probability"]),
        )
    return provider


def _audit_k1_forward(model, input_ids, names: tuple[str, ...], contract, *,
                      active_positions: tuple[int, ...]) -> dict:
    import torch
    import torch.nn.functional as F
    from .controller import LayerwiseKController
    from .patch import install_layerwise_patch
    controller = LayerwiseKController(
        names, min_k=1, max_k=8, monitored_layers=tuple(range(len(names))),
        monitored_positions=active_positions[:1],
    )
    install_layerwise_patch(model, controller, contract)
    total_length = int(input_ids.shape[1])
    plan = [[8] * total_length for _ in names]
    changed_position = active_positions[0]
    for layer in range(len(names)):
        plan[layer][changed_position] = 1
    modules = [(name, module) for name, module in model.named_modules()
               if module.__class__.__name__ == "LLaDAMoESparseMoeBlock"]
    first_module = modules[0][1]
    captured = {}
    pre_hook = first_module.register_forward_pre_hook(
        lambda _module, args: captured.__setitem__("hidden", args[0].detach()))
    out_hook = first_module.register_forward_hook(
        lambda _module, _args, output: captured.__setitem__(
            "output", output[0].detach() if isinstance(output, tuple) else output.detach()))
    try:
        controller.begin_forward(plan, global_nfe=1)
        with torch.inference_mode():
            logits = model(input_ids).logits.detach().float()
            hidden = captured["hidden"][0, changed_position].reshape(1, -1)
            router_logits = first_module.gate(hidden)
            probabilities = F.softmax(router_logits, dim=1, dtype=torch.float32)
            expert_id = int(probabilities.argmax(dim=1).item())
            expected_weight = probabilities[0, expert_id]
            expected_output = first_module.experts[expert_id](hidden) * expected_weight.to(hidden.dtype)
            observed_output = captured["output"][0, changed_position].reshape(1, -1)
            manual_output_error = float((observed_output.float() - expected_output.float()).abs().max().item())
        result = controller.end_forward()
        expected_per_layer = 8 * total_length - 7
        if result["layer_pair_counts"] != (expected_per_layer,) * len(names):
            raise RuntimeError("K=1 sparse pair accounting did not match the requested allocations")
        if not torch.isfinite(logits).all():
            raise RuntimeError("K=1 forward produced non-finite logits")
        records = [result["routing_records"][(layer, changed_position)] for layer in range(len(names))]
        weight_errors = [abs(float(row["weights"][0]) - float(row["router_statistics"]["top_k_gate_mass"]))
                         for row in records]
        # Router weights are cast to fp16 before dispatch; compare the recorded
        # value with the fp32 softmax mass within half-precision rounding.
        if any(row["k"] != 1 for row in records) or max(weight_errors, default=1.0) > 5e-4:
            raise RuntimeError("K=1 router weights do not match the unnormalized top-1 softmax mass")
        if any(not 0.0 < float(row["weights"][0]) < 1.0 for row in records):
            raise RuntimeError("K=1 gate mass is outside expected softmax probability range")
        if records[0]["experts"][0] != expert_id or manual_output_error > 1e-5:
            raise RuntimeError("K=1 block output differs from an independent top-1 expert reference")
        return {
            "passed": True, "tested_layers": len(names), "changed_position": changed_position,
            "sequence_length": total_length, "per_layer_pairs": expected_per_layer,
            "direct_pairs": result["direct_pairs"],
            "baseline_per_layer_pairs": 8 * total_length,
            "top1_gate_weight_mean": sum(float(row["weights"][0]) for row in records) / len(records),
            "max_weight_accounting_error": max(weight_errors, default=0.0),
            "manual_top1_block_output_max_abs_error": manual_output_error,
            "manual_top1_expert_id": expert_id,
            "finite_logits": True,
        }
    finally:
        pre_hook.remove()
        out_hook.remove()
        if controller.enabled:
            controller.enabled = False
        for _, module in model.named_modules():
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
                delattr(module, "_jrc_v4_original_forward")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def _load_runtime(root: Path, config: dict, config_path: Path):
    import torch
    from dataclasses import replace
    from ..config import GEN_LENGTH, MASK_ID
    from ..data import load_fixed8, reconstruct_gsm8k_records
    from ..model import load_model, build_model_input, input_device, fixed8_patch_equivalence
    from ..routing import VariableKController
    from .audit import audit_loaded_model
    from .fidelity import verified_contract_from_loaded_model, make_mask_phase_cases
    from .controller import LayerwiseKController
    from .patch import install_layerwise_patch

    csv_path = root / config["dataset"]["historical_fixed8_csv"]
    frame = load_fixed8(csv_path)
    from ..grading import canonical_grade_text
    ids, strata = _choose_prompts(
        frame, seed=int(config["dataset"]["prompt_selection_seed"]),
        correct_count=int(config["dataset"]["baseline_correct_prompts"]),
        incorrect_count=int(config["dataset"]["baseline_incorrect_prompts"]),
        grader=canonical_grade_text,
    )
    records = reconstruct_gsm8k_records(ids)
    model, tokenizer = load_model()
    model_audit = audit_loaded_model(model)
    if not model_audit["config_matches_frozen_expectations"] or not model_audit["moe_block_count_matches"]:
        raise RuntimeError("loaded checkpoint architecture failed the frozen audit")
    contract, semantic_evidence = verified_contract_from_loaded_model(
        model, config["model"]["revision"])
    if not contract.router_semantics_verified:
        raise RuntimeError("checkpoint router semantics gate failed")
    names = tuple(model_audit["ordered_moe_paths"])

    # Confirm the patched K=8 reference matches the model's native path first.
    _, first_ids = build_model_input(records[ids[0]]["harness_context"], tokenizer)
    first_ids = first_ids.to(input_device(model))
    legacy_controller = VariableKController(len(names))
    fixed8 = fixed8_patch_equivalence(model, first_ids, legacy_controller)
    if fixed8["max_abs_logit_diff"] != 0.0 or fixed8["argmax_match_fraction"] != 1.0:
        raise RuntimeError("native Fixed8 parity gate failed")
    contract = replace(contract, native_fixed8_equivalence_verified=True)

    # A representative masked forward places one token at K=1 in every MoE layer.
    cases = make_mask_phase_cases(first_ids, mask_id=MASK_ID,
                                  generation_length=GEN_LENGTH,
                                  block_length=int(config["decoder"]["block_length"]))
    masked_input, active_positions = cases[0]
    k1_forward = _audit_k1_forward(
        model, masked_input, names, contract, active_positions=active_positions)

    return {
        "model": model, "tokenizer": tokenizer, "names": names,
        "contract": contract, "frame": frame, "ids": ids, "strata": strata,
        "records": records, "model_audit": model_audit,
        "semantic_evidence": semantic_evidence, "fixed8_parity": fixed8,
        "k1_forward": k1_forward, "first_prompt_ids": first_ids,
        "config_digest": _sha(config_path), "fixed8_sha256": _sha(csv_path),
    }


def _k1_replay_gate(runtime: dict, config: dict, *, output_dir: Path) -> dict:
    import torch
    from ..generation import generate_with_routing_mode
    from ..jrc_v4.interventions import verify_noop_replay
    from ..jrc_v4.replay import capture_replay_state
    from ..jrc_v4.patch import install_layerwise_patch
    from ..jrc_v4.controller import LayerwiseKController

    prompt_id = str(runtime["ids"][0])
    prompt_ids = runtime["first_prompt_ids"]
    controller = LayerwiseKController(runtime["names"], min_k=1, max_k=8)
    install_layerwise_patch(runtime["model"], controller, runtime["contract"])
    provider = _policy_provider(variant="confidence_weighted", num_layers=len(runtime["names"]),
                                seed=int(config["statistics"]["bootstrap_seed"]),
                                prompt_id=prompt_id, config=config)
    captured = {}
    def capture(state):
        if "state" not in captured:
            captured["state"] = state
    _seed_torch(int(config["statistics"]["bootstrap_seed"]))
    events = []
    def observe(payload):
        events.extend(payload["k_by_layer"][layer][position] == 1
                      for layer in range(len(runtime["names"]))
                      for position in payload["unresolved_positions"])
    try:
        expected_ids, expected_trace, expected_meta = generate_with_routing_mode(
            runtime["model"], prompt_ids, "fixed8", controller,
            allocation_provider=provider, before_forward=capture,
            forward_observer=observe,
            checkpoint_revision=config["model"]["revision"],
            config_digest=runtime["config_digest"], before_forward_nfe=1,
        )
        if "state" not in captured:
            raise RuntimeError("K=1 replay gate did not capture the first pre-forward state")
        replay = verify_noop_replay(
            model=runtime["model"], prompt_ids=prompt_ids, mode="fixed8",
            controller=controller, state=captured["state"], allocation_provider=provider,
            checkpoint_revision=config["model"]["revision"],
            config_digest=runtime["config_digest"], expected_output=expected_ids,
            expected_trace=expected_trace,
        )
        if sum(events) <= 0:
            raise RuntimeError("K=1 policy replay had no sampled K=1 assignments; cannot validate sparse replay")
        return {**replay, "policy_k1_assignments_observed": int(sum(events)),
                "prompt_id": prompt_id, "total_nfe": expected_meta["total_nfe"],
                "total_pairs": expected_meta["total_pairs"]}
    finally:
        for _, module in runtime["model"].named_modules():
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
                delattr(module, "_jrc_v4_original_forward")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def run_audit(*, root: Path, config_path: Path, output_dir: Path) -> dict:
    import torch
    from .audit import collect_environment
    from .io import atomic_write_json
    raw = _config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runtime = _load_runtime(root, raw, config_path)
    try:
        replay = _k1_replay_gate(runtime, raw, output_dir=output_dir)
        manifest = {
            "protocol_version": raw["protocol_version"], "stage": "k1_audit",
            "status": "passed", "config": str(config_path),
            "config_sha256": runtime["config_digest"], "git_commit": _git(root, "rev-parse", "HEAD"),
            "git_branch": _git(root, "branch", "--show-current"),
            "checkpoint": f'{raw["model"]["id"]}@{raw["model"]["revision"]}',
            "fixed8_input_sha256": runtime["fixed8_sha256"],
            "prompt_ids": runtime["ids"], "baseline_correct_strata": runtime["strata"],
            "environment": collect_environment(root), "model_audit": runtime["model_audit"],
            "router_semantics": runtime["semantic_evidence"],
            "native_fixed8_parity": runtime["fixed8_parity"],
            "k1_forward_gate": runtime["k1_forward"], "k1_sparse_policy_replay": replay,
            "policy": raw["policy"], "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        }
        atomic_write_json(output_dir / "manifest.json", manifest)
        atomic_write_json(output_dir / "environment.json", manifest["environment"])
        return manifest
    finally:
        for _, module in runtime["model"].named_modules():
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
                delattr(module, "_jrc_v4_original_forward")
        del runtime["model"], runtime["tokenizer"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def _run_one_arm(*, model, tokenizer, prompt_ids, prompt_id: str, arm: str,
                 controller, names, contract, config, config_digest: str,
                 seed: int, output_dir: Path) -> dict:
    import torch
    import pandas as pd
    from ..generation import generate_with_routing_mode, postprocess_generated_region
    from ..grading import canonical_grade_text
    from .patch import install_layerwise_patch

    install_layerwise_patch(model, controller, contract)
    if arm == "fixed8_control":
        provider = _fixed8_provider(num_layers=len(names))
    else:
        variant = "confidence_weighted" if arm == "confidence_weighted_k1" else "matched_random"
        provider = _policy_provider(variant=variant, num_layers=len(names), seed=seed,
                                    prompt_id=prompt_id, config=config)

    policy_rows = []
    selected_events = []
    def observe(payload):
        details = payload.get("allocation_meta")
        if details is None:
            details = {"variant": arm, "k1_assignment_count": 0, "selected": []}
        row = {"prompt_id": prompt_id, "arm": arm,
               "block_idx": payload["block_idx"], "global_nfe": payload["global_nfe"],
               "unresolved_positions": len(payload["unresolved_positions"]), **details}
        policy_rows.append(row)
        selected_events.extend({**event, "prompt_id": prompt_id, "arm": arm,
                                "block_idx": payload["block_idx"],
                                "global_nfe": payload["global_nfe"]}
                               for event in details.get("selected", []))

    # Counter-based intervention draws make the arms independent of call order.
    _seed_torch(seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        event_start.record()
    else:
        event_start = event_end = None
    wall_start = time.perf_counter()
    try:
        output_ids, trace, meta = generate_with_routing_mode(
            model, prompt_ids, "fixed8", controller,
            allocation_provider=provider, forward_observer=observe,
            checkpoint_revision=config["model"]["revision"], config_digest=config_digest,
        )
        _validate_stochastic_trace(trace)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            event_end.record()
            torch.cuda.synchronize()
            gpu_seconds = float(event_start.elapsed_time(event_end) / 1000.0)
        else:
            gpu_seconds = None
        wall_seconds = time.perf_counter() - wall_start
        prompt_len = int(prompt_ids.shape[1])
        generated = output_ids[0, prompt_len:].detach().cpu().tolist()
        raw_text, clean_text = postprocess_generated_region(generated, tokenizer)
        return {
            "prompt_id": prompt_id, "arm": arm, "status": "completed",
            "generated_ids": generated, "generated_ids_sha256": hashlib.sha256(
                json.dumps(generated, separators=(",", ":")).encode()).hexdigest(),
            "raw_text": raw_text, "clean_text": clean_text,
            "extracted_answer": None,
            "total_nfe": int(meta["total_nfe"]), "total_pairs": int(meta["total_pairs"]),
            "avg_k": float(meta["avg_k"]), "wall_seconds": wall_seconds,
            "cuda_event_seconds": gpu_seconds,
            "k1_assignment_count": len(selected_events),
            "policy_forward_count": len(policy_rows),
            "policy_rows": policy_rows, "selected_events": selected_events,
            "trace_records": trace.to_dict(orient="records"),
            "seed": int(seed),
        }
    finally:
        for _, module in model.named_modules():
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
                delattr(module, "_jrc_v4_original_forward")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def _bootstrap_mean(differences: list[float], *, seed: int, repetitions: int) -> dict:
    import numpy as np
    values = np.asarray(differences, dtype=float)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n_prompts": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "n_prompts": int(len(values))}


def _validate_stochastic_trace(trace, *, num_blocks: int = 8,
                               block_length: int = 32) -> None:
    """Validate complete block refinement without assuming legacy K buckets."""
    required = {"block_idx", "local_nfe", "global_nfe", "masked_before",
                "masked_after", "num_committed", "direct_pairs", "forward_mean_k"}
    if trace.empty or not required.issubset(trace.columns):
        raise RuntimeError("policy rollout trace is empty or missing required fields")
    if set(int(x) for x in trace["block_idx"].unique()) != set(range(num_blocks)):
        raise RuntimeError("policy rollout did not visit every decoding block")
    if not trace["global_nfe"].is_monotonic_increasing:
        raise RuntimeError("global NFE is not monotonic in the policy rollout")
    if (trace["direct_pairs"] <= 0).any():
        raise RuntimeError("policy rollout has a forward with no executed expert-token pairs")
    if not trace["forward_mean_k"].between(1.0, 8.0, inclusive="both").all():
        raise RuntimeError("policy rollout allocation is outside K=1..8")
    for block_idx in range(num_blocks):
        group = trace[trace["block_idx"] == block_idx].sort_values("local_nfe")
        if int(group["num_committed"].sum()) != block_length:
            raise RuntimeError(f"block {block_idx} did not commit exactly {block_length} tokens")
        if not (group["masked_after"] < group["masked_before"]).all():
            raise RuntimeError(f"block {block_idx} had a refinement forward with no progress")
        if int(group.iloc[-1]["masked_after"]) != 0:
            raise RuntimeError(f"block {block_idx} ended with unresolved masks")


def _summarize(rows: list[dict], config: dict) -> dict:
    by_prompt = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], {})[row["arm"]] = row
    comparisons = {}
    for arm in ("confidence_weighted_k1", "matched_random_k1"):
        paired = [(prompt, arms) for prompt, arms in by_prompt.items()
                  if "fixed8_control" in arms and arm in arms]
        fields = {}
        for metric in ("total_nfe", "total_pairs", "wall_seconds", "cuda_event_seconds"):
            diffs = [float(a[arm][metric] - a["fixed8_control"][metric])
                     for _, a in paired if a[arm][metric] is not None and a["fixed8_control"][metric] is not None]
            fields[metric + "_treatment_minus_fixed8"] = _bootstrap_mean(
                diffs, seed=int(config["statistics"]["bootstrap_seed"]) + len(fields),
                repetitions=int(config["statistics"]["bootstrap_resamples"]))
        correctness_diffs = [int(a[arm]["canonical_correct"] - a["fixed8_control"]["canonical_correct"])
                             for _, a in paired]
        fields["correctness_treatment_minus_fixed8"] = _bootstrap_mean(
            correctness_diffs, seed=int(config["statistics"]["bootstrap_seed"]) + 20,
            repetitions=int(config["statistics"]["bootstrap_resamples"]))
        fields["correctness_repairs"] = sum(
            a[arm]["canonical_correct"] == 1 and a["fixed8_control"]["canonical_correct"] == 0
            for _, a in paired)
        fields["correctness_regressions"] = sum(
            a[arm]["canonical_correct"] == 0 and a["fixed8_control"]["canonical_correct"] == 1
            for _, a in paired)
        fields["changed_generated_sequences"] = sum(
            a[arm]["generated_ids_sha256"] != a["fixed8_control"]["generated_ids_sha256"]
            for _, a in paired)
        comparisons[arm] = fields

    exposure = {arm: {band: {"eligible": 0, "selected": 0}
                      for band in ("low", "middle", "high")}
                for arm in ("confidence_weighted_k1", "matched_random_k1")}
    for row in rows:
        arm = row["arm"]
        if arm not in exposure:
            continue
        for band in exposure[arm]:
            exposure[arm][band]["eligible"] += int(row.get("exposures_by_confidence_band", {}).get(band, 0))
            exposure[arm][band]["selected"] += int(row.get("k1_assignments_by_confidence_band", {}).get(band, 0))
    for arm_rows in exposure.values():
        for band, vals in arm_rows.items():
            vals["selection_rate"] = (vals["selected"] / vals["eligible"] if vals["eligible"] else None)

    all_arms_present = all(set(arms) == {"fixed8_control", "confidence_weighted_k1", "matched_random_k1"}
                           for arms in by_prompt.values())
    return {
        "status": "complete" if len(by_prompt) == int(config["dataset"]["prompts"]) and all_arms_present else "partial",
        "prompt_count": len(by_prompt), "branch_count": len(rows),
        "arms": {arm: {"n": sum(1 for row in rows if row["arm"] == arm),
                        "mean_correct": (sum(row["canonical_correct"] for row in rows if row["arm"] == arm)
                                         / max(1, sum(1 for row in rows if row["arm"] == arm))),
                        "mean_nfe": (sum(row["total_nfe"] for row in rows if row["arm"] == arm)
                                     / max(1, sum(1 for row in rows if row["arm"] == arm))),
                        "mean_pairs": (sum(row["total_pairs"] for row in rows if row["arm"] == arm)
                                       / max(1, sum(1 for row in rows if row["arm"] == arm))),
                        "mean_wall_seconds": (sum(row["wall_seconds"] for row in rows if row["arm"] == arm)
                                               / max(1, sum(1 for row in rows if row["arm"] == arm)))}
                 for arm in ("fixed8_control", "confidence_weighted_k1", "matched_random_k1")},
        "paired_vs_fixed8": comparisons,
        "k1_selection_rates_by_prior_confidence": exposure,
        "interpretation": "development pilot only; confidence weighting is not declared useful unless it improves paired full-rollout outcomes over matched-rate random K=1 and Fixed8",
    }


def _aggregate_policy_exposure(forward_rows: list[dict]) -> tuple[dict, dict]:
    """Retain the confidence-band opportunity and selection totals per branch."""
    bands = ("low", "middle", "high")
    exposures = {band: 0 for band in bands}
    selected = {band: 0 for band in bands}
    for row in forward_rows:
        for band in bands:
            exposures[band] += int(row.get("exposures_by_confidence_band", {}).get(band, 0))
            selected[band] += int(row.get("k1_assignments_by_confidence_band", {}).get(band, 0))
    return exposures, selected


def run_pilot(*, root: Path, config_path: Path, audit_dir: Path,
              output_dir: Path, resume: bool = False) -> dict:
    import torch
    import pandas as pd
    from ..data import load_fixed8, reconstruct_gsm8k_records
    from ..model import build_model_input, input_device, load_model
    from ..grading import canonical_grade_text
    from .audit import audit_loaded_model
    from .controller import LayerwiseKController
    from .fidelity import verified_contract_from_loaded_model
    from .patch import install_layerwise_patch
    from .io import BudgetLedger
    from .settings import ResourceCaps
    from .stochastic_k1_policy import build_sparse_k1_plan

    raw = _config(config_path)
    audit_path = audit_dir / "manifest.json"
    if not audit_path.is_file():
        raise FileNotFoundError("run the K=1 audit stage before the policy pilot")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    commit = _git(root, "rev-parse", "HEAD")
    config_digest = _sha(config_path)
    if audit.get("status") != "passed" or audit.get("git_commit") != commit or audit.get("config_sha256") != config_digest:
        raise RuntimeError("K=1 audit does not match this source commit and config")
    ids = [int(x) for x in audit["prompt_ids"]]
    records = reconstruct_gsm8k_records(ids)
    model, tokenizer = load_model()
    model_audit = audit_loaded_model(model)
    contract, semantic_evidence = verified_contract_from_loaded_model(model, raw["model"]["revision"])
    if not contract.router_semantics_verified:
        raise RuntimeError("router semantics changed since the passed audit")
    from dataclasses import replace
    contract = replace(contract, native_fixed8_equivalence_verified=True)
    names = tuple(model_audit["ordered_moe_paths"])

    frame = load_fixed8(root / raw["dataset"]["historical_fixed8_csv"])
    prompt_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    identity = {"protocol_version": raw["protocol_version"], "git_commit": commit,
                "config_sha256": config_digest, "audit_manifest_sha256": _sha(audit_path),
                "fixed8_csv_sha256": _sha(root / raw["dataset"]["historical_fixed8_csv"]),
                "prompt_ids": ids, "prompt_ids_sha256": prompt_hash,
                "checkpoint_revision": raw["model"]["revision"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        prior_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if not resume or prior_identity != identity:
            raise RuntimeError("pilot output exists; use --resume with identical code/config/audit or choose a new output directory")
    else:
        if resume:
            raise RuntimeError("cannot resume pilot without its original run identity")
        identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    caps_raw = raw["resources"]
    caps = ResourceCaps(max_gpu_hours=float(caps_raw["max_gpu_hours"]),
                        max_branches=int(caps_raw["max_branches"]),
                        max_prompts=int(caps_raw["max_prompts"]),
                        max_wall_hours=float(caps_raw["max_wall_hours"]),
                        max_output_gb=float(caps_raw["max_output_gb"]))
    progress = output_dir / "progress.json"
    ledger = BudgetLedger.resume_from(caps, output_dir) if resume and progress.is_file() else BudgetLedger(caps, output_dir)
    existing = _read_jsonl(output_dir / "branch_outcomes.jsonl")
    completed = {(str(row["prompt_id"]), str(row["arm"])) for row in existing if row.get("status") == "completed"}
    controller = LayerwiseKController(names, min_k=1, max_k=8)
    arms = ["fixed8_control", "confidence_weighted_k1", "matched_random_k1"]
    seed_base = int(raw["statistics"]["bootstrap_seed"])
    error = None
    try:
        for prompt_index, prompt_id_int in enumerate(ids):
            prompt_key = str(prompt_id_int)
            _, prompt_ids = build_model_input(records[prompt_id_int]["harness_context"], tokenizer)
            prompt_ids = prompt_ids.to(input_device(model))
            arm_order = list(arms)
            random.Random(seed_base + prompt_id_int).shuffle(arm_order)
            for arm in arm_order:
                if (prompt_key, arm) in completed:
                    continue
                ledger.check_before_branch(
                    prompt_key,
                    reserve_gpu_seconds=float(caps_raw["branch_reserve_gpu_seconds"]),
                    reserve_output_bytes=65536,
                )
                branch_seed = seed_base + prompt_id_int
                try:
                    row = _run_one_arm(
                        model=model, tokenizer=tokenizer, prompt_ids=prompt_ids,
                        prompt_id=prompt_key, arm=arm, controller=controller,
                        names=names, contract=contract, config=raw,
                        config_digest=config_digest, seed=branch_seed,
                        output_dir=output_dir,
                    )
                    grade = canonical_grade_text(row["clean_text"], records[prompt_id_int]["gold_full_answer"])
                    row["canonical_correct"] = int(grade["canonical_correct"])
                    row["canonical_prediction"] = grade.get("prediction")
                    row["historical_fixed8_correct_for_stratum"] = int(audit["baseline_correct_strata"][prompt_key])
                    if row["cuda_event_seconds"] is not None:
                        ledger.add_gpu_time(row["cuda_event_seconds"])
                    (row["exposures_by_confidence_band"],
                     row["k1_assignments_by_confidence_band"]) = _aggregate_policy_exposure(
                         row["policy_rows"])
                    for forward in row.pop("policy_rows"):
                        _append_jsonl(output_dir / "forward_policy_summaries.jsonl", forward)
                    for event in row.pop("selected_events"):
                        _append_jsonl(output_dir / "k1_selected_token_layer_events.jsonl", event)
                    trace = pd.DataFrame(row.pop("trace_records"))
                    trace_dir = output_dir / "traces"
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    trace.to_csv(trace_dir / f"{prompt_key}_{arm}.csv", index=False)
                    _append_jsonl(output_dir / "branch_outcomes.jsonl", row)
                    completed.add((prompt_key, arm))
                except BaseException as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    _append_jsonl(output_dir / "failures.jsonl", {
                        "prompt_id": prompt_key, "arm": arm, "error": error,
                        "branches_started": ledger.branches_started, "wall_unix": time.time(),
                    })
                    raise
    finally:
        for _, module in model.named_modules():
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
                delattr(module, "_jrc_v4_original_forward")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        del model, tokenizer

    results = _read_jsonl(output_dir / "branch_outcomes.jsonl")
    summary = _summarize(results, raw)
    summary.update({"identity": identity, "resource_ledger": ledger.snapshot(),
                    "audit_manifest": str(audit_path), "audit_status": audit["status"],
                    "router_semantics": semantic_evidence,
                    "failure": error})
    from .io import atomic_write_json
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "manifest.json", {
        "protocol_version": raw["protocol_version"], "stage": "policy_pilot",
        "status": summary["status"], "git_commit": commit,
        "config_sha256": config_digest, "prompt_count": summary["prompt_count"],
        "branch_count": summary["branch_count"], "resource_ledger": ledger.snapshot(),
        "scientific_claims": "development_only_no_confirmatory_claims",
        "summary": str(output_dir / "summary.json"),
    })
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "pilot"), required=True)
    parser.add_argument("--config", default="configs/jrc_v4/stochastic_k1_sparse.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--audit-dir", default="outputs/jrc_v4_stochastic_k1/audit")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    raw = _config(config_path)
    if args.stage == "audit":
        output = Path(args.output_dir or "outputs/jrc_v4_stochastic_k1/audit")
        if not output.is_absolute():
            output = root / output
        result = run_audit(root=root, config_path=config_path, output_dir=output)
    else:
        output = Path(args.output_dir or "outputs/jrc_v4_stochastic_k1/pilot")
        if not output.is_absolute():
            output = root / output
        audit_dir = Path(args.audit_dir)
        if not audit_dir.is_absolute():
            audit_dir = root / audit_dir
        result = run_pilot(root=root, config_path=config_path, audit_dir=audit_dir,
                           output_dir=output, resume=args.resume)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") in {"passed", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
