"""Bounded paired rollout experiment for confidence-adaptive router mass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _config(path: Path) -> dict:
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol_version") != "jrc-v4-confidence-router-mass-1":
        raise ValueError("unexpected router-mass protocol configuration")
    policy = raw["policy"]
    if policy["base_mass"] != 0.6 or policy["low_mass"] != 0.1 or policy["high_mass"] != 0.9:
        raise ValueError("this pilot is frozen to P=.6 control and adaptive P=.1/.9")
    return raw


def _providers(*, config: dict, prompt_id: str, seed: int, num_layers: int):
    from .mass_policy import build_mass_plan
    policy = config["policy"]

    def provider(variant):
        def allocate(*, prompt_len, block_idx, unresolved_abs, confidence_history,
                     global_nfe, **kwargs):
            return build_mass_plan(
                sequence_length=prompt_len + int(config["decoder"]["generation_length"]),
                num_layers=num_layers, unresolved_positions=unresolved_abs,
                confidence_history=confidence_history, variant=variant, seed=seed,
                prompt_id=prompt_id, block_idx=block_idx, global_nfe=global_nfe,
                base_mass=float(policy["base_mass"]), low_mass=float(policy["low_mass"]),
                high_mass=float(policy["high_mass"]),
                low_confidence=float(policy["low_confidence"]),
                high_confidence=float(policy["high_confidence"]),
                low_mass_probability=float(policy["low_confidence_p_low"]),
                high_mass_probability=float(policy["high_confidence_p_low"]),
            )
        return allocate
    return provider("fixed"), provider("confidence_adaptive")


def _load_runtime(root: Path, config: dict, config_path: Path):
    # Reuse the previously validated checkpoint loader, strata sampling, native
    # Fixed8 parity check, and environment audit. The additional K=1 probe is a
    # harmless gate from the preserved prior study, not an arm in this experiment.
    from .stochastic_k1_experiment import _load_runtime as load_validated_runtime
    return load_validated_runtime(root, config, config_path)


def _mass_forward_gate(runtime: dict, config: dict) -> dict:
    import torch
    from torch.nn import functional as F
    from .controller import LayerwiseKController
    from .patch import install_layerwise_patch

    model = runtime["model"]
    names = runtime["names"]
    controller = LayerwiseKController(names, min_k=1, max_k=64,
                                      monitored_layers=tuple(range(len(names))),
                                      monitored_positions=(0,))
    install_layerwise_patch(model, controller, runtime["contract"])
    input_ids = runtime["first_prompt_ids"]
    mask_input = input_ids.clone()
    from ..config import MASK_ID, GEN_LENGTH
    mask_input = torch.cat([mask_input, torch.full((1, GEN_LENGTH), MASK_ID,
                             dtype=mask_input.dtype, device=mask_input.device)], dim=1)
    target = float(config["policy"]["base_mass"])
    plan = [[target] * int(mask_input.shape[1]) for _ in names]
    try:
        controller.begin_forward(plan, global_nfe=1)
        with torch.inference_mode():
            first_logits = model(mask_input).logits.detach().float().cpu()
        result = controller.end_forward()
        route_counts = [len(record["experts"]) for record in result["routing_records"].values()]
        masses = [sum(record["weights"]) for record in result["routing_records"].values()]
        # Independent route selection on actual captured router softmaxes.
        checked = [record for record in result["routing_records"].values()
                   if abs(float(record["router_statistics"]["top_k_gate_mass"]) -
                          sum(record["weights"])) <= 5e-4]
        if not torch.isfinite(first_logits).all() or not route_counts:
            raise RuntimeError("P=.6 mass forward produced nonfinite logits or no route records")
        if any(mass + 5e-4 < target for mass in masses):
            raise RuntimeError("actual selected router mass fell below its P target")
        mass_summary = result["mass_summary"]
        if mass_summary["mass_target_violations"]:
            raise RuntimeError("one or more GPU-routed tokens failed to reach the requested cumulative mass")
        return {"passed": True, "target_mass": target,
                "recorded_tokens": len(route_counts),
                "mean_experts_at_p06": sum(route_counts) / len(route_counts),
                "max_experts_at_p06": max(route_counts),
                "mean_achieved_mass": sum(masses) / len(masses),
                "min_achieved_mass": min(masses),
                "all_layer_token_count": int(mass_summary["token_count"]),
                "all_layer_mean_experts": sum(int(k) * int(v) for k, v in
                    mass_summary["expert_count_histogram"].items()) / int(mass_summary["token_count"]),
                "all_layer_mean_achieved_mass": mass_summary["achieved_mass_sum"] / mass_summary["token_count"],
                "all_layer_expert_count_histogram": mass_summary["expert_count_histogram"],
                "finite_logits": True,
                "router_mass_accounting_checks": len(checked)}
    finally:
        for _, module in model.named_modules():
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
                delattr(module, "_jrc_v4_original_forward")
        if controller.enabled:
            controller.abort_forward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def run_audit(*, root: Path, config_path: Path, output_dir: Path) -> dict:
    import torch
    from .audit import collect_environment
    from .io import atomic_write_json
    raw = _config(config_path)
    started = time.perf_counter()
    runtime = _load_runtime(root, raw, config_path)
    try:
        gate = _mass_forward_gate(runtime, raw)
        # The plan itself is a deterministic replay contract; regenerate an
        # identical draw matrix before allowing expensive full rollouts.
        fixed, adaptive = _providers(config=raw, prompt_id=str(runtime["ids"][0]),
                                     seed=int(raw["statistics"]["bootstrap_seed"]),
                                     num_layers=len(runtime["names"]))
        position = [int(runtime["first_prompt_ids"].shape[1])]
        confidence = {position[0]: [0.37]}
        left = adaptive(prompt_len=int(runtime["first_prompt_ids"].shape[1]), block_idx=0,
                        unresolved_abs=position, confidence_history=confidence, global_nfe=1)
        right = adaptive(prompt_len=int(runtime["first_prompt_ids"].shape[1]), block_idx=0,
                         unresolved_abs=position, confidence_history=confidence, global_nfe=1)
        if left != right or fixed(prompt_len=int(runtime["first_prompt_ids"].shape[1]), block_idx=0,
                                  unresolved_abs=position, confidence_history=confidence,
                                  global_nfe=1)[0] != [[0.6] * (int(runtime["first_prompt_ids"].shape[1]) + int(raw["decoder"]["generation_length"])) for _ in runtime["names"]]:
            raise RuntimeError("mass allocation replay/fixed-control gate failed")
        manifest = {
            "protocol_version": raw["protocol_version"], "stage": "mass_audit",
            "status": "passed", "config": str(config_path),
            "config_sha256": _sha(config_path),
            "git_commit": __import__("subprocess").run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                 check=True, capture_output=True, text=True).stdout.strip(),
            "git_branch": __import__("subprocess").run(["git", "-C", str(root), "branch", "--show-current"],
                                 check=True, capture_output=True, text=True).stdout.strip(),
            "checkpoint": f'{raw["model"]["id"]}@{raw["model"]["revision"]}',
            "prompt_ids": runtime["ids"], "router_semantics": runtime["semantic_evidence"],
            "native_fixed8_parity": runtime["fixed8_parity"],
            "mass_forward_gate": gate, "allocation_replay_equal": True,
            "environment": collect_environment(root),
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / "manifest.json", manifest)
        return manifest
    finally:
        del runtime["model"], runtime["tokenizer"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def _run_arm(*, model, tokenizer, prompt_ids, prompt_id: str, arm: str,
             controller, runtime: dict, config: dict, config_digest: str,
             seed: int) -> dict:
    import torch
    from ..generation import generate_with_routing_mode, postprocess_generated_region
    from ..grading import canonical_grade_text
    from ..model import input_device
    from .patch import install_layerwise_patch
    from .controller import LayerwiseKController

    install_layerwise_patch(model, controller, runtime["contract"])
    fixed, adaptive = _providers(config=config, prompt_id=prompt_id, seed=seed,
                                 num_layers=len(runtime["names"]))
    provider = fixed if arm == "fixed_p06" else adaptive
    policy_rows, selected = [], []
    def observe(payload):
        meta = payload.get("allocation_meta") or {}
        policy_rows.append({"prompt_id": prompt_id, "arm": arm,
                            "block_idx": payload["block_idx"], "global_nfe": payload["global_nfe"],
                            "unresolved_positions": len(payload["unresolved_positions"]),
                            "mass_summary": payload.get("mass_summary"), **meta})
        selected.extend({**event, "prompt_id": prompt_id, "arm": arm,
                         "block_idx": payload["block_idx"], "global_nfe": payload["global_nfe"]}
                        for event in meta.get("selected", []))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None
    start = time.perf_counter()
    try:
        output_ids, trace, meta = generate_with_routing_mode(
            model, prompt_ids, "fixed8", controller, allocation_provider=provider,
            forward_observer=observe, checkpoint_revision=config["model"]["revision"],
            config_digest=config_digest)
        if trace.empty or int(trace["block_idx"].nunique()) != 8:
            raise RuntimeError("rollout did not complete all eight decoding blocks")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_event.record()
            torch.cuda.synchronize()
            cuda_seconds = float(start_event.elapsed_time(end_event) / 1000)
        else:
            cuda_seconds = None
        prompt_len = int(prompt_ids.shape[1])
        generated = output_ids[0, prompt_len:].detach().cpu().tolist()
        raw_text, clean_text = postprocess_generated_region(generated, tokenizer)
        return {"prompt_id": prompt_id, "arm": arm, "status": "completed",
                "generated_ids": generated,
                "generated_ids_sha256": hashlib.sha256(json.dumps(generated, separators=(",", ":")).encode()).hexdigest(),
                "raw_text": raw_text, "clean_text": clean_text,
                "total_nfe": int(meta["total_nfe"]), "total_pairs": int(meta["total_pairs"]),
                "avg_k": float(meta["avg_k"]), "wall_seconds": time.perf_counter() - start,
                "cuda_event_seconds": cuda_seconds, "policy_rows": policy_rows,
                "selected_events": selected, "trace_records": trace.to_dict(orient="records"),
                "seed": int(seed)}
    finally:
        for _, module in model.named_modules():
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
                delattr(module, "_jrc_v4_original_forward")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def _summarize(rows: list[dict], config: dict) -> dict:
    import numpy as np
    from ..grading import canonical_grade_text
    paired = {}
    for row in rows:
        paired.setdefault(str(row["prompt_id"]), {})[row["arm"]] = row
    comparisons = {}
    for metric in ("total_pairs", "total_nfe", "avg_k", "wall_seconds", "cuda_event_seconds"):
        differences = [float(arms["confidence_adaptive"][metric] - arms["fixed_p06"][metric])
                       for arms in paired.values() if set(arms) == {"fixed_p06", "confidence_adaptive"}
                       and arms["confidence_adaptive"][metric] is not None and arms["fixed_p06"][metric] is not None]
        rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]) + len(comparisons))
        vals = np.asarray(differences)
        sims = vals[rng.integers(0, len(vals), size=(int(config["statistics"]["bootstrap_resamples"]), len(vals)))].mean(axis=1) if len(vals) else []
        comparisons[metric + "_adaptive_minus_fixed"] = {
            "mean": float(vals.mean()) if len(vals) else None,
            "ci95": [float(x) for x in np.quantile(sims, [.025, .975])] if len(vals) else [None, None],
            "n_prompts": len(vals)}
    band_exposure = {band: {"eligible": 0, "p01": 0, "p06": 0, "p09": 0}
                     for band in ("low", "middle", "high")}
    for row in rows:
        if row["arm"] != "confidence_adaptive":
            continue
        for forward in row.get("policy_summary", []):
            for band in band_exposure:
                band_exposure[band]["eligible"] += int(forward.get("exposures_by_confidence_band", {}).get(band, 0))
            assignments = forward.get("assignments_by_band", {})
            for band in band_exposure:
                band_exposure[band]["p01"] += int(assignments.get(band, {}).get("p01", 0))
                band_exposure[band]["p09"] += int(assignments.get(band, {}).get("p09", 0))
    for record in band_exposure.values():
        record["p06"] = record["eligible"] - record["p01"] - record["p09"]
        record["p01_rate"] = record["p01"] / record["eligible"] if record["eligible"] else None
        record["p09_rate"] = record["p09"] / record["eligible"] if record["eligible"] else None
    all_complete = all(set(v) == {"fixed_p06", "confidence_adaptive"} for v in paired.values())
    return {"status": "complete" if len(paired) == int(config["dataset"]["prompts"]) and all_complete else "partial",
            "prompt_count": len(paired), "branch_count": len(rows),
            "arms": {arm: {"n": sum(r["arm"] == arm for r in rows),
                           "mean_correct": sum(r.get("canonical_correct", 0) for r in rows if r["arm"] == arm) /
                           max(1, sum(r["arm"] == arm for r in rows)),
                           "mean_avg_k": sum(r["avg_k"] for r in rows if r["arm"] == arm) /
                           max(1, sum(r["arm"] == arm for r in rows)),
                           "mean_realized_experts_per_token": sum(r.get("mass_summary", {}).get("expert_count_sum", 0) /
                               max(1, r.get("mass_summary", {}).get("token_count", 0)) for r in rows if r["arm"] == arm) /
                               max(1, sum(r["arm"] == arm for r in rows)),
                           "mean_achieved_router_mass": sum(r.get("mass_summary", {}).get("achieved_mass_sum", 0) /
                               max(1, r.get("mass_summary", {}).get("token_count", 0)) for r in rows if r["arm"] == arm) /
                               max(1, sum(r["arm"] == arm for r in rows)),
                           "mass_target_violations": sum(r.get("mass_summary", {}).get("mass_target_violations", 0)
                               for r in rows if r["arm"] == arm),
                           "mean_total_pairs": sum(r["total_pairs"] for r in rows if r["arm"] == arm) /
                           max(1, sum(r["arm"] == arm for r in rows)),
                           "mean_wall_seconds": sum(r["wall_seconds"] for r in rows if r["arm"] == arm) /
                           max(1, sum(r["arm"] == arm for r in rows))}
                     for arm in ("fixed_p06", "confidence_adaptive")},
            "paired_vs_fixed_p06": comparisons,
            "adaptive_mass_by_prior_confidence": band_exposure,
            "changed_sequences": sum(v["fixed_p06"]["generated_ids_sha256"] != v["confidence_adaptive"]["generated_ids_sha256"]
                                     for v in paired.values() if set(v) == {"fixed_p06", "confidence_adaptive"}),
            "correctness_regressions": sum(v["fixed_p06"].get("canonical_correct", 0) == 1 and
                                            v["confidence_adaptive"].get("canonical_correct", 0) == 0
                                            for v in paired.values() if set(v) == {"fixed_p06", "confidence_adaptive"}),
            "correctness_repairs": sum(v["fixed_p06"].get("canonical_correct", 0) == 0 and
                                        v["confidence_adaptive"].get("canonical_correct", 0) == 1
                                        for v in paired.values() if set(v) == {"fixed_p06", "confidence_adaptive"}),
            "interpretation": "bounded development pilot; no predictive-benefit or acceleration claim without confirmatory evidence"}


def run_pilot(*, root: Path, config_path: Path, audit_dir: Path,
              output_dir: Path, resume: bool = False) -> dict:
    import torch
    import pandas as pd
    from ..data import load_fixed8, reconstruct_gsm8k_records
    from ..model import build_model_input, input_device
    from ..grading import canonical_grade_text
    from .stochastic_k1_experiment import _git
    from .controller import LayerwiseKController
    from .io import BudgetLedger, atomic_write_json
    from .settings import ResourceCaps

    raw = _config(config_path)
    audit_path = audit_dir / "manifest.json"
    if not audit_path.is_file():
        raise FileNotFoundError("run mass audit before the pilot")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    commit, digest = _git(root, "rev-parse", "HEAD"), _sha(config_path)
    if audit.get("status") != "passed" or audit.get("git_commit") != commit or audit.get("config_sha256") != digest:
        raise RuntimeError("mass audit does not match this source commit and configuration")
    ids = [int(x) for x in audit["prompt_ids"]]
    records = reconstruct_gsm8k_records(ids)
    from .stochastic_k1_experiment import _load_runtime
    runtime = _load_runtime(root, raw, config_path)
    model, tokenizer = runtime["model"], runtime["tokenizer"]
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_version": raw["protocol_version"], "git_commit": commit,
                "config_sha256": digest, "prompt_ids": ids,
                "checkpoint_revision": raw["model"]["revision"]}
    identity_file = output_dir / "run_identity.json"
    if identity_file.exists():
        if not resume or json.loads(identity_file.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("existing output identity differs; use matching --resume or new output directory")
    else:
        if resume:
            raise RuntimeError("cannot resume without the original run identity")
        identity_file.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    caps_raw = raw["resources"]
    caps = ResourceCaps(max_gpu_hours=float(caps_raw["max_gpu_hours"]),
                        max_branches=int(caps_raw["max_branches"]), max_prompts=int(caps_raw["max_prompts"]),
                        max_wall_hours=float(caps_raw["max_wall_hours"]), max_output_gb=float(caps_raw["max_output_gb"]))
    ledger = BudgetLedger.resume_from(caps, output_dir) if resume and (output_dir / "progress.json").exists() else BudgetLedger(caps, output_dir)
    rows = _read(output_dir / "branch_outcomes.jsonl")
    completed = {(str(r["prompt_id"]), str(r["arm"])) for r in rows if r.get("status") == "completed"}
    controller = LayerwiseKController(runtime["names"], min_k=1, max_k=64)
    arms = ("fixed_p06", "confidence_adaptive")
    try:
        for pid in ids:
            _, prompt_ids = build_model_input(records[pid]["harness_context"], tokenizer)
            prompt_ids = prompt_ids.to(input_device(model))
            order = list(arms)
            random.Random(int(raw["statistics"]["bootstrap_seed"]) + pid).shuffle(order)
            for arm in order:
                key = (str(pid), arm)
                if key in completed:
                    continue
                ledger.check_before_branch(str(pid), reserve_gpu_seconds=float(caps_raw["branch_reserve_gpu_seconds"]), reserve_output_bytes=65536)
                result = _run_arm(model=model, tokenizer=tokenizer, prompt_ids=prompt_ids,
                                  prompt_id=str(pid), arm=arm, controller=controller,
                                  runtime=runtime, config=raw, config_digest=digest,
                                  seed=int(raw["statistics"]["bootstrap_seed"]) + pid)
                grade = canonical_grade_text(result["clean_text"], records[pid]["gold_full_answer"])
                result["canonical_correct"] = int(grade["canonical_correct"])
                result["canonical_prediction"] = grade.get("prediction")
                result["historical_fixed8_correct"] = int(audit["baseline_correct_strata"][str(pid)])
                policy_rows = result["policy_rows"]
                result["policy_summary"] = []
                aggregate_mass = {"token_count": 0, "expert_count_sum": 0,
                                  "achieved_mass_sum": 0.0, "mass_target_violations": 0,
                                  "expert_count_histogram": {}, "target_mass_histogram": {}}
                for forward in policy_rows:
                    mass = forward.get("mass_summary") or {}
                    aggregate_mass["token_count"] += int(mass.get("token_count", 0))
                    aggregate_mass["expert_count_sum"] += sum(
                        int(k) * int(count) for k, count in mass.get("expert_count_histogram", {}).items())
                    aggregate_mass["achieved_mass_sum"] += float(mass.get("achieved_mass_sum", 0.0))
                    aggregate_mass["mass_target_violations"] += int(mass.get("mass_target_violations", 0))
                    for key, value in mass.get("expert_count_histogram", {}).items():
                        aggregate_mass["expert_count_histogram"][key] = aggregate_mass["expert_count_histogram"].get(key, 0) + int(value)
                    for key, value in mass.get("target_mass_histogram", {}).items():
                        aggregate_mass["target_mass_histogram"][key] = aggregate_mass["target_mass_histogram"].get(key, 0) + int(value)
                    counts = {band: {"p01": 0, "p09": 0} for band in ("low", "middle", "high")}
                    for event in forward.get("selected", []):
                        counts[event["confidence_band"]]["p01" if event["target_mass"] == .1 else "p09"] += 1
                    result["policy_summary"].append({
                        "exposures_by_confidence_band": forward.get("exposures_by_confidence_band", {}),
                        "assignments_by_band": counts,
                        "assignments_by_mass": forward.get("assignments_by_mass", {})})
                result["mass_summary"] = aggregate_mass
                if result["cuda_event_seconds"] is not None:
                    ledger.add_gpu_time(result["cuda_event_seconds"])
                for forward in result.pop("policy_rows"):
                    _append(output_dir / "forward_policy_summaries.jsonl", forward)
                for event in result.pop("selected_events"):
                    _append(output_dir / "selected_mass_events.jsonl", event)
                trace = pd.DataFrame(result.pop("trace_records"))
                (output_dir / "traces").mkdir(exist_ok=True)
                trace.to_csv(output_dir / "traces" / f"{pid}_{arm}.csv", index=False)
                _append(output_dir / "branch_outcomes.jsonl", result)
                rows.append(result)
                completed.add(key)
    finally:
        for _, module in model.named_modules():
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
                delattr(module, "_jrc_v4_original_forward")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        del runtime["model"], runtime["tokenizer"]
    summary = _summarize(rows, raw)
    summary.update({"identity": identity, "resource_ledger": ledger.snapshot(), "audit_status": audit["status"]})
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "manifest.json", {"protocol_version": raw["protocol_version"],
        "stage": "mass_pilot", "status": summary["status"], "git_commit": commit,
        "config_sha256": digest, "prompt_count": summary["prompt_count"],
        "branch_count": summary["branch_count"], "resource_ledger": ledger.snapshot(),
        "scientific_claims": "bounded_development_only", "summary": str(output_dir / "summary.json")})
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "pilot"), required=True)
    parser.add_argument("--config", default="configs/jrc_v4/router_mass_adaptive.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--audit-dir", default="outputs/jrc_v4_router_mass/audit")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    raw = _config(config_path)
    if args.stage == "audit":
        output = Path(args.output_dir or "outputs/jrc_v4_router_mass/audit")
        if not output.is_absolute():
            output = root / output
        result = run_audit(root=root, config_path=config_path, output_dir=output)
    else:
        output = Path(args.output_dir or "outputs/jrc_v4_router_mass/pilot")
        audit = Path(args.audit_dir)
        if not output.is_absolute():
            output = root / output
        if not audit.is_absolute():
            audit = root / audit
        result = run_pilot(root=root, config_path=config_path, audit_dir=audit,
                           output_dir=output, resume=args.resume)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") in {"passed", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
