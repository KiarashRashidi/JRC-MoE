"""Command line entrypoint with fail-closed scientific and resource gates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time


from .audit import audit_loaded_model, collect_environment
from .io import BudgetLedger, atomic_write_json, record_failure
from .settings import ResourceCaps, V4Settings


STAGES = ("audit", "smoke", "pilot", "analyze", "preregister", "signal",
          "communities", "allocate", "systems", "report")
DEFAULT_ROOTS = {"audit": "audit", "smoke": "smoke", "pilot": "pilot", "signal": "signal",
                "communities": "community", "allocate": "allocator", "systems": "systems"}


def _repo_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jrc-moe jrc-v4")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default="configs/jrc_v4/pilot.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--input-dir")
    parser.add_argument("--fixed8-csv")
    parser.add_argument("--max-gpu-hours", type=float)
    parser.add_argument("--max-branches", type=int)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-wall-hours", type=float)
    parser.add_argument("--max-output-gb", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true",
                        help="record cache/environment only; do not allocate the checkpoint")
    return parser


def _load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the JRC-v4 YAML configuration") from exc
    if not path.is_file():
        raise FileNotFoundError(f"configuration not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a YAML mapping")
    return value


def _settings(raw: dict) -> V4Settings:
    model = raw.get("model", {})
    intervention = raw.get("intervention", {})
    features = raw.get("features", {})
    settings = V4Settings(
        model_id=model.get("id"), model_revision=model.get("revision"),
        num_layers=int(model.get("num_layers", 16)), num_experts=int(model.get("num_experts", 64)),
        native_top_k=int(model.get("native_top_k", 8)), min_k=int(intervention.get("min_k", 2)),
        max_k=int(intervention.get("max_k", 8)), legal_k=tuple(intervention.get("legal_k", (2, 4, 6, 8))),
        monitored_layers=tuple(intervention.get("monitored_layers", (4, 12))),
        heads=tuple(features.get("heads", (0, 4, 8, 12))),
        landmark_count=int(features.get("positional_landmarks", 16)),
        neighbor_cap=int(features.get("candidate_neighbor_cap", 8)),
        history_window=int(features.get("temporal_history_window", 4)),
        graph_refresh_every=int(features.get("graph_refresh_every", 2)),
        seed=int(raw.get("statistics", {}).get("bootstrap_seed", 20261009)),
    )
    settings.validate()
    return settings


def _caps(args, raw: dict) -> ResourceCaps:
    configured = raw.get("resources", {})
    caps = ResourceCaps(
        max_gpu_hours=float(args.max_gpu_hours if args.max_gpu_hours is not None else configured.get("max_gpu_hours", 24)),
        max_branches=int(args.max_branches if args.max_branches is not None else configured.get("max_branches", 200)),
        max_prompts=int(args.max_prompts if args.max_prompts is not None else configured.get("max_prompts", 50)),
        max_wall_hours=float(args.max_wall_hours if args.max_wall_hours is not None else configured.get("max_wall_hours", 30)),
        max_output_gb=float(args.max_output_gb if args.max_output_gb is not None else configured.get("max_output_gb", 10)),
    )
    caps.validate()
    return caps


def _resolve_path(root: Path, value: str | None, fallback: str) -> Path:
    path = Path(value or fallback).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _cached_model_files(model_id: str, revision: str) -> dict[str, str | None]:
    from huggingface_hub import try_to_load_from_cache
    names = ("config.json", "modeling_lladamoe.py", "configuration_lladamoe.py",
             "model.safetensors.index.json", "model-00001-of-00003.safetensors",
             "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors",
             "tokenizer.json", "tokenizer_config.json")
    result = {}
    for name in names:
        try:
            path = try_to_load_from_cache(model_id, name, revision=revision)
            result[name] = str(path) if isinstance(path, str) and Path(path).is_file() else None
        except Exception:
            result[name] = None
    return result


def _validate_fixed8(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = set(reader.fieldnames or ())
    aliases = {"prompt_length": {"prompt_length", "prompt_token_count"},
               "total_pairs": {"total_pairs", "selected_expert_token_pairs"},
               "pairs_m": {"pairs_m", "selected_expert_token_pairs_m"}}
    required = {"test_idx", "diagnostic_ordinal", "total_nfe", "clean_generated_text"}
    missing = sorted(required - columns)
    missing.extend(name for name, options in aliases.items() if not columns.intersection(options))
    return {"status": "valid" if not missing and len(rows) == 50 else "invalid",
            "path": str(path), "rows": len(rows), "columns": sorted(columns),
            "missing": missing, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _historical_replication(root: Path) -> dict:
    target = root / "outputs" / "reflex_paper_replication50"
    summary = target / "summary.csv"
    manifest = target / "manifest.json"
    return {"path": str(target), "summary_present": summary.is_file(),
            "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest() if summary.is_file() else None,
            "manifest_present": manifest.is_file(),
            "manifest": json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None}


def _audit(args, root: Path, raw: dict, config_path: Path, output_dir: Path, caps: ResourceCaps) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = collect_environment(root)
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    fixed8_path = _resolve_path(root, args.fixed8_csv, raw.get("dataset", {}).get("historical_fixed8_csv", "data/diagnostic_50_results.csv"))
    fixed8 = _validate_fixed8(fixed8_path)
    settings = _settings(raw)
    cache = _cached_model_files(settings.model_id, settings.model_revision)
    expected_cache_files = set(cache)
    cache_complete = all(cache.get(name) for name in expected_cache_files)
    gates = {
        "checkpoint_cache": "passed" if cache_complete else "blocked",
        "fixed8_input": "passed" if fixed8["status"] == "valid" else "blocked",
        "loaded_architecture": "not_run",
        "router_semantics": "not_run",
        "native_patch_fidelity": "not_run",
        "legal_k_forwards": "not_run",
        "legacy_reproduction": "not_run",
        "fresh_fixed8_profile": "not_run",
        "replay": "not_run",
        "synchronized_profile": "not_run",
    }
    model_audit = None
    model_error = None
    if cache_complete and not args.skip_model_load:
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from ..model import load_model
            model, tokenizer = load_model()
            model_audit = audit_loaded_model(model)
            gates["loaded_architecture"] = "passed" if model_audit["config_matches_frozen_expectations"] and model_audit["moe_block_count_matches"] else "failed"
            from .fidelity import verified_contract_from_loaded_model
            contract, semantic_evidence = verified_contract_from_loaded_model(model, settings.model_revision)
            gates["router_semantics"] = "passed" if contract.router_semantics_verified else "failed"
            # Detailed multi-prompt phase parity, full rollout replay, and fresh baseline
            # profiling remain separate operations; no audit is promoted by config alone.
            if gates["router_semantics"] == "passed":
                gates["native_patch_fidelity"] = "not_run"
            del model, tokenizer
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"
            record_failure(output_dir, prompt_id=None, branch_id=None, error=exc, stage="audit_model_load")
    g0_passed = all(gates[key] == "passed" for key in
                    ("loaded_architecture", "router_semantics", "native_patch_fidelity", "legal_k_forwards",
                     "legacy_reproduction", "fresh_fixed8_profile", "replay", "synchronized_profile"))
    manifest = {
        "protocol_version": "jrc-moe-v4-intervention-1", "created_unix": time.time(),
        "stage": "audit", "status": "passed" if g0_passed else "blocked",
        "g0": "passed" if g0_passed else "blocked", "gates": gates,
        "config_path": str(config_path), "config_sha256": config_digest,
        "config_resolved": raw, "settings_resolved": settings.__dict__, "resource_caps": caps.__dict__,
        "environment": "environment.json", "fixed8_input": fixed8,
        "historical_replication": _historical_replication(root),
        "checkpoint_cache": cache, "checkpoint_cache_complete": cache_complete,
        "model_audit": model_audit, "model_load_error": model_error,
        "branches_started": 0,
        "blocker": None if g0_passed else "G0 is not complete: native parity, legal-K forwards, legacy reproduction, fresh Fixed8 profile, replay and synchronized GPU profiling have not all passed.",
    }
    atomic_write_json(output_dir / "environment.json", env)
    atomic_write_json(output_dir / "config_resolved.json", {"config": raw, "sha256": config_digest,
                                                              "settings": settings.__dict__, "caps": caps.__dict__})
    (output_dir / "git_commit.txt").write_text(f"{env.get('git_commit')}\n", encoding="utf-8")
    (output_dir / "model_revision.txt").write_text(f"{settings.model_id}@{settings.model_revision}\n", encoding="utf-8")
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"stage": "audit", "status": manifest["status"], "g0": manifest["g0"],
                      "checkpoint_cache_complete": cache_complete, "fixed8_input": fixed8["status"],
                      "gates": gates, "output_dir": str(output_dir)}, indent=2))
    return 0


def _gate_manifest(output_dir: Path) -> dict | None:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _blocked_stage(stage: str, output_dir: Path, reason: str, caps: ResourceCaps,
                   config_digest: str, resume: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol_version": "jrc-moe-v4-intervention-1", "stage": stage,
                "status": "blocked", "reason": reason, "config_sha256": config_digest,
                "resource_caps": caps.__dict__, "resume_requested": resume,
                "branches_started": 0, "prompts_started": 0,
                "artifacts_not_created": {
                    "state_index.jsonl": "no valid G0 state collection",
                    "interventions.jsonl": "no intervention branches were permitted",
                    "features.jsonl": "no verified feature hook observations",
                    "branch_outcomes.jsonl": "no full rollouts were run",
                    "paired_exchanges.csv": "no valid causal labels",
                    "interaction_samples.csv": "no four-arm rollouts",
                    "cost_ledger.csv": "no branches; no measured costs",
                    "figures": "no measurements to plot",
                }}
    atomic_write_json(output_dir / "manifest.json", manifest)
    if stage in {"smoke", "pilot"}:
        (output_dir / "cost_ledger.csv").write_text(
            "branch_id,prompt_id,arm,started_unix,elapsed_seconds,gpu_seconds,nfe,executed_pairs,status\n",
            encoding="utf-8")
    print(json.dumps({"stage": stage, "status": "blocked", "reason": reason,
                      "branches_started": 0, "output_dir": str(output_dir)}, indent=2))
    return 2


def _stage(args, root: Path, raw: dict, config_path: Path, output_dir: Path, caps: ResourceCaps) -> int:
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if args.stage == "audit":
        return _audit(args, root, raw, config_path, output_dir, caps)
    if args.stage == "preregister":
        source = root / "docs" / "jrc_v4" / "preregistration_draft.yaml"
        destination = _resolve_path(root, args.output_dir, "docs/jrc_v4/preregistration_draft.yaml")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        print(json.dumps({"stage": "preregister", "status": "draft_pending_pilot",
                          "confirmatory_results_accessed": False, "output": str(destination)}, indent=2))
        return 0
    if args.stage == "report":
        statuses = {}
        for name, directory in DEFAULT_ROOTS.items():
            status = _gate_manifest(root / "outputs" / "jrc_v4" / directory)
            statuses[name] = status.get("status", "not_run") if status else "not_run"
        report = {"status": "incomplete", "gates": statuses, "scientific_claims": "none_established"}
        atomic_write_json(output_dir / "report.json", report)
        print(json.dumps(report, indent=2))
        return 0
    if args.stage == "analyze":
        input_dir = _resolve_path(root, args.input_dir, "outputs/jrc_v4/pilot")
        outcomes = input_dir / "paired_exchanges.csv"
        if not outcomes.is_file():
            return _blocked_stage("analyze", output_dir, f"no measured paired-exchange file at {outcomes}", caps, digest, args.resume)
        # Analysis is intentionally unavailable until there are measured rows and
        # their provenance/splits have been validated by a successful pilot manifest.
        pilot = _gate_manifest(input_dir)
        if not pilot or pilot.get("status") != "passed":
            return _blocked_stage("analyze", output_dir, "pilot manifest has not passed provenance and split checks", caps, digest, args.resume)
        return _blocked_stage("analyze", output_dir, "analysis adapters are gated pending the first validated pilot", caps, digest, args.resume)

    required_gate = {"smoke": "g0", "pilot": "g0", "signal": "g1", "communities": "g1",
                     "allocate": "g1", "systems": "g3"}[args.stage]
    gate_source = "audit" if args.stage in {"smoke", "pilot"} else "pilot" if args.stage in {"signal", "communities", "allocate"} else "allocator"
    audit_dir = root / "outputs" / "jrc_v4" / gate_source
    evidence = _gate_manifest(audit_dir)
    if not evidence or evidence.get(required_gate) != "passed":
        status = evidence.get(required_gate, "not_run") if evidence else "not_run"
        return _blocked_stage(args.stage, output_dir,
                              f"required {required_gate.upper()} gate is {status}; no branches or later-stage experiments started",
                              caps, digest, args.resume)
    if args.stage in {"communities", "allocate", "systems"}:
        preceding = {"communities": ("pilot", "g1"), "allocate": ("pilot", "g1"), "systems": ("pilot", "g3")}[args.stage]
        prior = _gate_manifest(root / "outputs" / "jrc_v4" / preceding[0])
        if not prior or prior.get(preceding[1]) != "passed":
            return _blocked_stage(args.stage, output_dir, f"required {preceding[1].upper()} evidence is missing", caps, digest, args.resume)
    return _blocked_stage(args.stage, output_dir,
                          "gate is passed but stage execution adapter is not enabled in this build; no outcome is fabricated",
                          caps, digest, args.resume)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _repo_root()
    config_path = _resolve_path(root, args.config, "configs/jrc_v4/pilot.yaml")
    raw = _load_config(config_path)
    caps = _caps(args, raw)
    if args.output_dir:
        output_dir = _resolve_path(root, args.output_dir, "outputs/jrc_v4/audit")
    elif args.stage in DEFAULT_ROOTS:
        output_dir = root / "outputs" / "jrc_v4" / DEFAULT_ROOTS[args.stage]
    else:
        output_dir = root / "outputs" / "jrc_v4" / args.stage
    return _stage(args, root, raw, config_path, output_dir, caps)


if __name__ == "__main__":
    sys.exit(main())
