"""Bounded CLI for natural REFLEX smoke, collection, analysis, and packaging."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .natural_storage import (CampaignLedger, NaturalRunStore, atomic_write_json,
                              canonical_json_bytes, directory_bytes, sha256_file)


def _forward_output_bytes(config: dict) -> int:
    return int(float(config.get("resources", {}).get("max_forward_record_bytes_mib", 128)) * 1024**2)


def _forward_output_check(ledger: CampaignLedger, transaction,
                          baseline_bytes: int, config: dict):
    bound = _forward_output_bytes(config)
    # Reserve one complete current forward plus a full next-forward margin.
    reserve = 2 * bound + 16 * 1024**2
    return lambda: ledger.check_output_headroom(
        reserve_bytes=reserve, baseline_bytes=baseline_bytes,
        incremental_bytes=transaction.bytes_on_disk)


def _repo_root() -> Path:
    for path in (Path.cwd(), *Path.cwd().parents):
        if (path / "pyproject.toml").is_file():
            return path
    return Path.cwd()


def _load_config(path: Path) -> dict:
    import yaml
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("natural study configuration must be a YAML mapping")
    return value


def _path(root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _git_commit(root: Path) -> str:
    proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _nvidia_smi() -> dict:
    proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"nvidia-smi GPU query failed: {proc.stderr.strip()}")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True)
    if apps.returncode:
        raise RuntimeError(f"nvidia-smi process query failed: {apps.stderr.strip()}")
    processes = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
    if processes:
        raise RuntimeError("GPU is occupied; refusing to start without disturbing existing jobs: " +
                           "; ".join(processes))
    return {"gpu_query": proc.stdout.strip(), "active_compute_processes": processes,
            "checked_unix": time.time()}


def _ledger(root: Path, config: dict) -> CampaignLedger:
    resources = config.get("resources", {})
    return CampaignLedger(root,
                          max_gpu_seconds=float(resources.get("max_gpu_hours", 4.0)) * 3600,
                          max_output_bytes=int(float(resources.get("max_output_gib", 10.0)) * 1024**3))


def _stage_begin(ledger: CampaignLedger, name: str, reserve_seconds: float,
                 *, gpu_stage: bool = True) -> dict:
    if gpu_stage:
        reserve_seconds = min(float(reserve_seconds),
                              float(ledger.check()["gpu_seconds_remaining"]))
        if reserve_seconds <= 0:
            raise RuntimeError("no GPU wall-time remains in the campaign ledger")
        return ledger.start(name, gpu_stage=True, reserve_seconds=reserve_seconds)
    return ledger.start(name, gpu_stage=False)


def _stage_fail(ledger: CampaignLedger, name: str, exc: BaseException) -> None:
    try:
        ledger.finish(exit_code=1, details={"error": f"{type(exc).__name__}: {exc}", "stage": name})
    except Exception:
        pass


def _torch_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ids_sha256(ids) -> str:
    import numpy as np
    array = np.asarray(ids, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _trace_signature(trace) -> list[dict]:
    signature = []
    columns = ("block_idx", "local_nfe", "global_nfe", "masked_before", "num_above_threshold",
               "fallback_used", "num_committed", "masked_after", "direct_pairs",
               "allocation_sha256", "eligible_positions_relative",
               "threshold_positions_relative", "committed_positions_relative",
               "fallback_position_relative")
    for record in trace.to_dict(orient="records"):
        row = {}
        for key in columns:
            value = record.get(key)
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, float) and value != value:
                value = None
            elif isinstance(value, (list, tuple)):
                value = [None if isinstance(item, float) and item != item else item for item in value]
            elif hasattr(value, "item"):
                value = value.item()
            row[key] = value
        signature.append(row)
    return signature


def _prepare_inputs(fixed8_csv: Path, limit: int, tokenizer):
    from .data import load_fixed8
    from .runner import _prepare
    fixed8 = load_fixed8(fixed8_csv).sort_values("diagnostic_ordinal").head(limit).copy()
    indices, records, prepared = _prepare(fixed8, tokenizer)
    return fixed8, indices, records, prepared


def _manifest_identity(root: Path, config_path: Path, fixed8_csv: Path) -> dict:
    cfg_digest = sha256_file(config_path)
    fixed8_digest = sha256_file(fixed8_csv)
    commit = _git_commit(root)
    from .config import MODEL_ID, MODEL_REVISION
    source = f"{commit}\0{cfg_digest}\0{fixed8_digest}\0{MODEL_ID}\0{MODEL_REVISION}"
    run_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    return {"run_id": run_id, "source_commit": commit, "config_sha256": cfg_digest,
            "fixed8_csv_sha256": fixed8_digest, "model_id": MODEL_ID,
            "checkpoint_revision": MODEL_REVISION,
            "checkpoint_identity": f"{MODEL_ID}@{MODEL_REVISION}"}


def _attention_audit(args, root: Path, config: dict, config_path: Path,
                     output_root: Path, ledger: CampaignLedger) -> int:
    from .jrc_v4.audit import collect_environment
    audit_dir = output_root / "attention_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    output = audit_dir / "qk_parity.json"
    _nvidia_smi()
    output_bound = _forward_output_bytes(config)
    active_stage = _stage_begin(ledger, "attention_audit", float(config.get("resources", {}).get("attention_audit_reserve_seconds", 900)))
    started = time.time()
    try:
        ledger.check_output_headroom(reserve_bytes=2 * output_bound + 16 * 1024**2)
        audit_script = root / "scripts" / "verify_jrc_v4_attention.py"
        deadline = active_stage.get("deadline_unix")
        timeout = (max(1, int(float(deadline) - time.time()))
                   if deadline is not None else None)
        proc = subprocess.run([sys.executable, str(audit_script), "--output", str(output)],
                              capture_output=True, text=True, timeout=timeout)
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.returncode not in (0, 2) or not output.is_file():
            raise RuntimeError(f"bounded attention-audit subprocess failed ({proc.returncode}): "
                               f"{proc.stderr[-4000:]}")
        report = json.loads(output.read_text(encoding="utf-8"))
        if (proc.returncode == 0) != bool(report.get("passed")):
            raise RuntimeError("attention-audit process exit code disagrees with its saved result")
        report["environment"] = collect_environment(root)
        report["config_sha256"] = sha256_file(config_path)
        report["occupied_wall_seconds"] = time.time() - started
        atomic_write_json(output, report)
        ledger.finish(exit_code=0 if report.get("passed") else 2, details={"status": report.get("status"),
                                            "attention_calls": report.get("sdpa_calls"),
                                            "output": str(output)})
        print(json.dumps({"status": report.get("status"), "output": str(output)}, indent=2))
        return 0 if report.get("passed") else 2
    except BaseException as exc:
        _stage_fail(ledger, "attention_audit", exc)
        raise


def _smoke(args, root: Path, config: dict, config_path: Path,
           fixed8_csv: Path, output_root: Path, ledger: CampaignLedger) -> int:
    if args.max_prompts < 3 or args.max_prompts > 5:
        raise ValueError("invariance smoke requires 3–5 prompts")
    from .config import GEN_LENGTH, BLOCK_LENGTH, THETA
    from .data import load_fixed8
    from .model import load_model
    from .routing import VariableKController, install_variable_k_patch
    from .generation import generate_with_routing_mode, validate_trace, postprocess_generated_region
    from .grading import canonical_grade_text
    from .natural_features import NaturalFeatureObserver
    from .observational import validate_trajectory
    from .config import NUM_MOE_LAYERS

    gpu = _nvidia_smi()
    output_bound = _forward_output_bytes(config)
    identity = _manifest_identity(root, config_path, fixed8_csv)
    old_smoke = output_root / "smoke" / "smoke_manifest.json"
    if old_smoke.is_file() and args.resume:
        cached = json.loads(old_smoke.read_text(encoding="utf-8"))
        if cached.get("run_id") == identity["run_id"] and cached.get("gate_status") == "passed":
            print(json.dumps({"status": "resumed", "smoke_manifest": str(old_smoke)}, indent=2))
            return 0
    _stage_begin(ledger, "invariance_smoke", float(config.get("resources", {}).get("smoke_reserve_seconds", 3 * 3600)))
    started = time.time()
    parity = []
    preflight = {}
    try:
        ledger.check_output_headroom(reserve_bytes=2 * output_bound + 16 * 1024**2)
        model, tokenizer = load_model()
        fixed8 = load_fixed8(fixed8_csv).sort_values("diagnostic_ordinal").head(args.max_prompts).copy()
        from .runner import _prepare
        indices, records, prepared = _prepare(fixed8, tokenizer)
        preflight = {"gpu": gpu}
        control_input = prepared[indices[0]]["input_ids"]
        controller = VariableKController(NUM_MOE_LAYERS)
        from .model import fixed8_patch_equivalence
        preflight["fixed8_patch_equivalence"] = fixed8_patch_equivalence(model, control_input, controller)
        controller = VariableKController(NUM_MOE_LAYERS)
        install_variable_k_patch(model, controller)
        from .jrc_v4.audit import audit_loaded_model
        preflight["model_audit"] = audit_loaded_model(model)
        audit_path = output_root / "attention_audit" / "qk_parity.json"
        audit = None
        if audit_path.is_file():
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                audit = None
        attention_enabled = bool(audit and audit.get("passed"))
        store = NaturalRunStore(output_root)
        completed = store.completed()
        modes = tuple(config.get("study", {}).get("modes", ("reflex", "fixed8")))
        if set(modes) != {"reflex", "fixed8"}:
            raise ValueError("natural study modes must be exactly reflex and fixed8")
        for idx in indices:
            row = fixed8.loc[fixed8.test_idx == idx].iloc[0]
            prompt_id = f"gsm8k:{int(idx)}"
            seed = int(config.get("study", {}).get("seed_base", 20261011)) + int(idx)
            for mode in modes:
                unit = (identity["run_id"], prompt_id, mode, seed)
                if unit in completed:
                    if not args.resume:
                        raise FileExistsError(f"completed smoke trajectory exists; pass --resume: {unit}")
                    saved = completed[unit].get("validation", {}).get("parity_gate")
                    if not saved:
                        raise RuntimeError(f"saved smoke trajectory lacks parity evidence: {unit}")
                    parity.append(saved)
                    if saved.get("gate_status") != "passed":
                        raise RuntimeError(f"saved instrumentation parity failure for {prompt_id}/{mode}")
                    continue
                prompt_ids = prepared[idx]["input_ids"]
                no_hook_seed = seed
                import torch
                _torch_seed(torch, no_hook_seed)
                base_ids, base_trace, base_meta = generate_with_routing_mode(
                    model, prompt_ids, mode, controller, position_trace=True,
                    budget_check=ledger.check_active, cuda_event_profile=True,
                )
                validate_trace(base_trace, mode)
                base_generated = base_ids[0, prompt_ids.shape[1]:].detach().cpu().tolist()
                base_signature = _trace_signature(base_trace)
                output_baseline_bytes = directory_bytes(output_root)
                tx = store.start(run_id=identity["run_id"], prompt_id=prompt_id, mode=mode,
                                 seed=seed, metadata={**identity, "config_path": str(config_path),
                                                       "fixed8_csv": str(fixed8_csv), "instrumentation": "attention_router_if_audit_passes",
                                                       "mode_policy": mode})
                event_metadata = {**identity, "prompt_id": prompt_id, "seed": seed,
                                  "config_hash": identity["config_sha256"],
                                  "checkpoint_hash": identity["checkpoint_revision"],
                                  "prompt_length": int(prompt_ids.shape[1]),
                                  "diagnostic_test_idx": int(idx)}
                observer = NaturalFeatureObserver(
                    tx, prompt_len=int(prompt_ids.shape[1]),
                    router_layers=tuple(config.get("features", {}).get("router_layers", (4, 12))),
                    router_stride=int(config.get("features", {}).get("router_stride", 1)),
                    attention_layers=tuple(config.get("features", {}).get("attention_layers", (4, 12))),
                    attention_heads=tuple(config.get("features", {}).get("attention_heads", (0, 4, 8, 12))),
                    landmark_count=int(config.get("features", {}).get("landmark_count", 8)),
                    attention_stride=int(config.get("features", {}).get("attention_stride", 1)),
                    attention_audit=audit, attention_enabled=attention_enabled,
                    max_forward_record_bytes=output_bound,
                )
                install_info = observer.install(model)
                _torch_seed(torch, seed)
                try:
                    obs_ids, obs_trace, obs_meta = generate_with_routing_mode(
                        model, prompt_ids, mode, controller, observational_observer=observer,
                        event_metadata=event_metadata, position_trace=True,
                        budget_check=ledger.check_active, cuda_event_profile=True,
                        output_budget_check=_forward_output_check(
                            ledger, tx, output_baseline_bytes, config),
                    )
                    validate_trace(obs_trace, mode)
                    trajectory_gate = validate_trajectory(observer.events,
                                                          generated_length=GEN_LENGTH,
                                                          block_length=BLOCK_LENGTH)
                    obs_generated = obs_ids[0, prompt_ids.shape[1]:].detach().cpu().tolist()
                    base_equal = _ids_sha256(base_generated) == _ids_sha256(obs_generated)
                    trace_equal = _trace_signature(obs_trace) == base_signature
                    base_raw, base_clean = postprocess_generated_region(base_generated, tokenizer)
                    obs_raw, obs_clean = postprocess_generated_region(obs_generated, tokenizer)
                    base_grade = canonical_grade_text(base_clean, records[idx]["gold_full_answer"])
                    obs_grade = canonical_grade_text(obs_clean, records[idx]["gold_full_answer"])
                    route_hash_equal = [r.get("allocation_sha256") for r in _trace_signature(obs_trace)] == [
                        r.get("allocation_sha256") for r in base_signature]
                    gate_passed = bool(base_equal and trace_equal and route_hash_equal and
                                       base_grade["canonical_correct"] == obs_grade["canonical_correct"])
                    parity_row = {
                        **identity, "prompt_id": prompt_id, "mode": mode, "seed": seed,
                        "prompt_length": int(prompt_ids.shape[1]), "generation_length": GEN_LENGTH,
                        "baseline_nfe": int(base_meta["total_nfe"]), "observed_nfe": int(obs_meta["total_nfe"]),
                        "generated_ids_sha256_baseline": _ids_sha256(base_generated),
                        "generated_ids_sha256_observed": _ids_sha256(obs_generated),
                        "token_ids_identical": base_equal, "commit_trace_identical": trace_equal,
                        "allocation_hashes_identical": route_hash_equal,
                        "fallback_flags_identical": [x["fallback_used"] for x in base_signature] ==
                                                     [x["fallback_used"] for x in _trace_signature(obs_trace)],
                        "baseline_correct": int(base_grade["canonical_correct"]),
                        "observed_correct": int(obs_grade["canonical_correct"]),
                        "attention_capture_enabled": attention_enabled,
                        "attention_install": install_info,
                        "trajectory_validation": trajectory_gate,
                        "hooked_forward_median_seconds": float(obs_trace["forward_seconds"].median()),
                        "hooked_forward_p95_seconds": float(obs_trace["forward_seconds"].quantile(0.95)),
                        "baseline_forward_median_seconds": float(base_trace["forward_seconds"].median()),
                        "baseline_forward_p95_seconds": float(base_trace["forward_seconds"].quantile(0.95)),
                        "hooked_cuda_event_median_seconds": float(obs_trace["forward_cuda_seconds"].median()),
                        "hooked_cuda_event_p95_seconds": float(obs_trace["forward_cuda_seconds"].quantile(0.95)),
                        "baseline_cuda_event_median_seconds": float(base_trace["forward_cuda_seconds"].median()),
                        "baseline_cuda_event_p95_seconds": float(base_trace["forward_cuda_seconds"].quantile(0.95)),
                        "baseline_generation_wall_seconds": float(base_meta["generation_seconds"]),
                        "hooked_generation_wall_seconds": float(obs_meta["generation_seconds"]),
                        "instrumentation_wall_overhead_seconds": float(obs_meta["generation_seconds"] - base_meta["generation_seconds"]),
                        "instrumentation_wall_overhead_fraction": (float(obs_meta["generation_seconds"] / base_meta["generation_seconds"] - 1.0)
                                                                   if base_meta["generation_seconds"] > 0 else None),
                        "baseline_meta": base_meta, "observed_meta": obs_meta,
                        "gate_status": "passed" if gate_passed else "failed",
                    }
                    parity.append(parity_row)
                    output_metadata = {"prompt_id": prompt_id, "seed": seed,
                                       "test_idx": int(idx), "diagnostic_ordinal": int(row.diagnostic_ordinal),
                                       "prompt_length": int(prompt_ids.shape[1]),
                                       "question_id": int(idx)}
                    generation = {"gold_full_answer": records[idx]["gold_full_answer"], **output_metadata}
                    observer.close()
                    generated = obs_ids[0, prompt_ids.shape[1]:].detach().cpu().tolist()
                    raw_text, clean_text = postprocess_generated_region(generated, tokenizer)
                    grade = canonical_grade_text(clean_text, records[idx]["gold_full_answer"])
                    prompt_output = {**output_metadata, "generated_token_ids": [int(x) for x in generated],
                                     "generated_token_strings": tokenizer.convert_ids_to_tokens(generated),
                                     "generated_ids_sha256": _ids_sha256(generated),
                                     "raw_text": raw_text, "clean_text": clean_text, **grade,
                                     "generation_meta": obs_meta,
                                     "peak_gpu_allocated_bytes": (int(torch.cuda.max_memory_allocated())
                                                                   if torch.cuda.is_available() else None)}
                    tx.commit(prompt_output=prompt_output,
                              validation={**trajectory_gate, "instrumentation_parity": "passed" if gate_passed else "failed",
                                          "attention_features_enabled": attention_enabled,
                                          "parity_gate": parity_row})
                    ledger.check_output_cap_active()
                    if not gate_passed:
                        raise RuntimeError(f"instrumentation parity failed for {prompt_id}/{mode}; stop gate")
                except BaseException:
                    observer.close()
                    if not tx._closed:
                        tx.abort(reason="smoke parity or trajectory error")
                    raise
        passed = len(parity) >= 6 and all(row["gate_status"] == "passed" for row in parity)
        smoke_dir = output_root / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(smoke_dir / "parity_details.json", {"parity": parity, "preflight": preflight})
        report = {**identity, "gate_status": "passed" if passed else "failed",
                  "prompts": sorted({row["prompt_id"] for row in parity}),
                  "modes": sorted({row["mode"] for row in parity}),
                  "trajectory_count": len(parity), "attention_audit_status": "passed" if attention_enabled else "not_passed_or_missing",
                  "gpu": gpu, "elapsed_wall_seconds": time.time() - started,
                  "resource_ledger": ledger.read(), "caps": config.get("resources", {}),
                  "parity_path": str(smoke_dir / "parity_details.json"),
                  "fixed8_csv_sha256": identity["fixed8_csv_sha256"],
                  "prompt_lengths": {row["prompt_id"]: row["prompt_length"] for row in parity}}
        atomic_write_json(old_smoke, report)
        ledger.finish(exit_code=0 if passed else 2,
                      details={"gate_status": report["gate_status"], "trajectories": len(parity),
                               "manifest": str(old_smoke)})
        del model
        print(json.dumps(report, indent=2))
        return 0 if passed else 2
    except BaseException as exc:
        smoke_dir = output_root / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(smoke_dir / "parity_details.json",
                          {"run_id": identity["run_id"], "parity": parity,
                           "preflight": preflight, "error": f"{type(exc).__name__}: {exc}"})
        atomic_write_json(old_smoke,
                          {**identity, "gate_status": "failed", "parity_path": str(smoke_dir / "parity_details.json"),
                           "error": f"{type(exc).__name__}: {exc}",
                           "elapsed_wall_seconds": time.time() - started})
        _stage_fail(ledger, "invariance_smoke", exc)
        raise


def _collect(args, root: Path, config: dict, config_path: Path,
             fixed8_csv: Path, output_root: Path, ledger: CampaignLedger) -> int:
    if not 1 <= args.max_prompts <= 50:
        raise ValueError("--max-prompts must be from 1 through 50")
    identity = _manifest_identity(root, config_path, fixed8_csv)
    smoke_path = output_root / "smoke" / "smoke_manifest.json"
    if not smoke_path.is_file():
        raise RuntimeError("run the 3–5 prompt instrumentation smoke before collection")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("run_id") != identity["run_id"] or smoke.get("gate_status") != "passed":
        raise RuntimeError("instrumentation parity gate failed or does not match current sources/config/data")
    gpu = _nvidia_smi()
    output_bound = _forward_output_bytes(config)
    reserve = float(config.get("resources", {}).get("collect_reserve_seconds", 3 * 3600))
    _stage_begin(ledger, "natural_collection", reserve)
    started = time.time()
    try:
        ledger.check_output_headroom(reserve_bytes=2 * output_bound + 16 * 1024**2)
        from .model import load_model, fixed8_patch_equivalence
        from .routing import VariableKController, install_variable_k_patch
        from .config import NUM_MOE_LAYERS
        from .data import load_fixed8
        from .runner import _prepare
        from .generation import generate_with_routing_mode, postprocess_generated_region
        from .grading import canonical_grade_text
        from .observational import validate_trajectory, stable_prompt_split
        from .natural_features import NaturalFeatureObserver
        import torch

        model, tokenizer = load_model()
        fixed8 = load_fixed8(fixed8_csv).sort_values("diagnostic_ordinal").head(args.max_prompts).copy()
        indices, records, prepared = _prepare(fixed8, tokenizer)
        controller = VariableKController(NUM_MOE_LAYERS)
        equivalence = fixed8_patch_equivalence(model, prepared[indices[0]]["input_ids"], controller)
        controller = VariableKController(NUM_MOE_LAYERS)
        install_variable_k_patch(model, controller)
        store = NaturalRunStore(output_root)
        completed = store.completed()
        modes = tuple(config.get("study", {}).get("modes", ("reflex", "fixed8")))
        audit_path = output_root / "attention_audit" / "qk_parity.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
        attention_enabled = bool(audit and audit.get("passed"))
        feature = config.get("features", {})
        trajectories = []
        for idx in indices:
            prompt_id = f"gsm8k:{int(idx)}"
            seed = int(config.get("study", {}).get("seed_base", 20261011)) + int(idx)
            for mode in modes:
                key = (identity["run_id"], prompt_id, mode, seed)
                if key in completed:
                    if args.resume:
                        continue
                    raise FileExistsError(f"trajectory already exists; use --resume: {key}")
                output_baseline_bytes = directory_bytes(output_root)
                tx = store.start(run_id=identity["run_id"], prompt_id=prompt_id, mode=mode,
                                 seed=seed, metadata={**identity, "config_path": str(config_path),
                                                       "fixed8_csv": str(fixed8_csv), "gpu": gpu,
                                                       "mode_policy": mode, "split": stable_prompt_split(prompt_id),
                                                       "attention_audit_passed": attention_enabled,
                                                       "feature_config": feature})
                event_metadata = {**identity, "prompt_id": prompt_id, "seed": seed,
                                  "config_hash": identity["config_sha256"],
                                  "checkpoint_hash": identity["checkpoint_revision"],
                                  "prompt_length": int(prepared[idx]["input_ids"].shape[1]),
                                  "diagnostic_test_idx": int(idx)}
                observer = NaturalFeatureObserver(
                    tx, prompt_len=int(prepared[idx]["input_ids"].shape[1]),
                    router_layers=tuple(feature.get("router_layers", (4, 12))),
                    router_stride=int(feature.get("router_stride", 1)),
                    attention_layers=tuple(feature.get("attention_layers", (4, 12))),
                    attention_heads=tuple(feature.get("attention_heads", (0, 4, 8, 12))),
                    landmark_count=int(feature.get("landmark_count", 8)),
                    attention_stride=int(feature.get("attention_stride", 1)),
                    attention_audit=audit, attention_enabled=attention_enabled,
                    max_forward_record_bytes=output_bound,
                )
                install_info = observer.install(model)
                _torch_seed(torch, seed)
                try:
                    out, trace, meta = generate_with_routing_mode(
                        model, prepared[idx]["input_ids"], mode, controller,
                        observational_observer=observer, event_metadata=event_metadata,
                        position_trace=True, budget_check=ledger.check_active,
                        cuda_event_profile=True,
                        output_budget_check=_forward_output_check(
                            ledger, tx, output_baseline_bytes, config),
                    )
                    observer.close()
                    validate_trace(trace, mode)
                    validation = validate_trajectory(observer.events)
                    generated = out[0, prepared[idx]["input_ids"].shape[1]:].detach().cpu().tolist()
                    raw_text, clean_text = postprocess_generated_region(generated, tokenizer)
                    grade = canonical_grade_text(clean_text, records[idx]["gold_full_answer"])
                    prompt_output = {"prompt_id": prompt_id, "seed": seed, "test_idx": int(idx),
                                     "diagnostic_ordinal": int(fixed8.loc[fixed8.test_idx == idx, "diagnostic_ordinal"].iloc[0]),
                                     "prompt_length": int(prepared[idx]["input_ids"].shape[1]),
                                     "generated_token_ids": [int(x) for x in generated],
                                     "generated_token_strings": tokenizer.convert_ids_to_tokens(generated),
                                     "generated_ids_sha256": _ids_sha256(generated),
                                     "raw_text": raw_text, "clean_text": clean_text, **grade,
                                     "generation_meta": meta,
                                     "peak_gpu_allocated_bytes": (int(torch.cuda.max_memory_allocated())
                                                                   if torch.cuda.is_available() else None)}
                    index = tx.commit(prompt_output=prompt_output,
                                      validation={**validation, "attention_features_enabled": attention_enabled,
                                                  "observer_install": install_info})
                    ledger.check_output_cap_active()
                    trajectories.append({"prompt_id": prompt_id, "mode": mode,
                                         "seed": seed, "total_nfe": meta["total_nfe"],
                                         "generation_seconds": meta["generation_seconds"],
                                         "correct": grade["canonical_correct"],
                                         "attention_enabled": attention_enabled,
                                         "record_counts": index["record_counts"]})
                    print(json.dumps(trajectories[-1]))
                except BaseException:
                    observer.close()
                    if not tx._closed:
                        tx.abort(reason="collection trajectory failed")
                    raise
        split = {f"gsm8k:{int(i)}": stable_prompt_split(f"gsm8k:{int(i)}") for i in indices}
        run_completed = {key: row for key, row in store.completed().items() if key[0] == identity["run_id"]}
        manifest = {**identity, "status": "complete" if len(run_completed) >= len(indices) * len(modes) else "partial",
                    "requested_prompts": args.max_prompts,
                    "new_trajectories_completed": len(trajectories),
                    "all_completed_trajectories": len(run_completed),
                    "modes": list(modes), "attention_audit_passed": attention_enabled,
                    "prompt_split": split, "fixed8_patch_equivalence": equivalence,
                    "gpu": gpu, "elapsed_wall_seconds": time.time() - started,
                    "disk_bytes": directory_bytes(output_root),
                    "resource_ledger": ledger.read(), "collection_output": str(output_root)}
        atomic_write_json(output_root / "run_manifest.json", manifest)
        ledger.finish(exit_code=0, details={"status": manifest["status"],
                                            "trajectories": len(trajectories),
                                            "manifest": str(output_root / "run_manifest.json")})
        del model
        print(json.dumps(manifest, indent=2))
        return 0
    except BaseException as exc:
        _stage_fail(ledger, "natural_collection", exc)
        raise


def _dispatch(args) -> int:
    root = _repo_root()
    config_path = _path(root, args.config)
    config = _load_config(config_path)
    output_root = _path(root, args.out or config.get("output_root", "outputs/natural_reflex"))
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = _ledger(output_root, config)
    fixed8_csv = _path(root, args.fixed8_csv or config.get("data", {}).get("fixed8_csv", "data/diagnostic_50_results.csv"))
    if args.command == "attention-audit":
        return _attention_audit(args, root, config, config_path, output_root, ledger)
    if args.command == "smoke":
        return _smoke(args, root, config, config_path, fixed8_csv, output_root, ledger)
    if args.command == "collect":
        return _collect(args, root, config, config_path, fixed8_csv, output_root, ledger)
    if args.command == "analyze":
        from .natural_analysis import analyze
        _stage_begin(ledger, "cpu_analysis", 0, gpu_stage=False)
        try:
            result = analyze(output_root, config=config)
            ledger.finish(exit_code=0, details={"status": result.get("status"),
                                                "report": str(output_root / "analysis" / "natural_report.md")})
            print(json.dumps({"status": result.get("status"),
                              "report": str(output_root / "analysis" / "natural_report.md")}, indent=2))
            return 0
        except BaseException as exc:
            _stage_fail(ledger, "cpu_analysis", exc)
            raise
    if args.command == "package":
        from .natural_package import package_review
        result = package_review(output_root, output_path=_path(root, args.output) if args.output else None)
        print(json.dumps(result, indent=2))
        return 0
    raise ValueError(args.command)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="jrc-moe natural")
    parser.add_argument("command", choices=("attention-audit", "smoke", "collect", "analyze", "package"))
    parser.add_argument("--config", default="configs/natural_reflex/pilot.yaml")
    parser.add_argument("--fixed8-csv")
    parser.add_argument("--out")
    parser.add_argument("--output", help="review ZIP path for package")
    parser.add_argument("--max-prompts", type=int, default=3,
                        help="smoke: 3–5 prompts; collect: up to 50 prompts")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    return _dispatch(args)

