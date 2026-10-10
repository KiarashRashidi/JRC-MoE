"""Bounded development smoke/pilot for paired full-rollout K exchanges."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch

from ..config import BLOCK_LENGTH, GEN_LENGTH, MASK_ID, NUM_MOE_LAYERS
from ..data import load_fixed8, reconstruct_gsm8k_records
from ..generation import generate_with_routing_mode, postprocess_generated_region
from ..grading import canonical_grade_text
from ..model import build_model_input, input_device, load_model
from .audit import audit_loaded_model
from .controller import LayerwiseKController
from .features import FeatureRecord
from .interventions import paired_exchange_rollout
from .io import (BudgetExhausted, BudgetLedger, append_jsonl, atomic_write_json,
                 bind_run_identity, record_failure)
from .patch import install_layerwise_patch
from .replay import ReplayState
from .sampling import IndexedRandom, prompt_split_manifest, sample_equal_budget_exchanges
from .statistics import paired_prompt_bootstrap


def _append_csv(path: Path, fields: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def _rewrite_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _gpu_seconds_since(start_event, end_event, cuda: bool) -> float:
    if not cuda:
        return 0.0
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    return float(start_event.elapsed_time(end_event) / 1000.0)


def _jsd(logp: torch.Tensor, logq: torch.Tensor) -> tuple[float, float]:
    """Mean tokenwise Jensen-Shannon divergence and top-probability change."""
    p, q = logp.exp(), logq.exp()
    logm = torch.logaddexp(logp, logq) - 0.6931471805599453
    values = 0.5 * (p * (logp - logm)).sum(dim=-1) + 0.5 * (q * (logq - logm)).sum(dim=-1)
    top_delta = (p.max(dim=-1).values - q.max(dim=-1).values).abs()
    return float(values.mean()), float(top_delta.max())


def _read_jsonl_ids(path: Path, key: str) -> set[str]:
    if not path.is_file():
        return set()
    found = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line).get(key)
        if value is not None:
            found.add(str(value))
    return found


def run_pilot(*, root: Path, output_dir: Path, raw_config: dict, config_digest: str,
              caps, resume: bool, prompt_limit: int, audit_manifest: dict,
              audit_manifest_path: Path, stage_name: str) -> dict:
    """Run uniformly sampled K=6 exchanges with full continuations under fixed8.

    The 50 historical prompts are treated as development/pilot data only. The
    function cannot start unless the same frozen configuration has passed G0.
    """
    if audit_manifest.get("g0") != "passed":
        raise RuntimeError("G0 is not passed")
    if audit_manifest.get("config_sha256") != config_digest:
        raise RuntimeError("audit config digest differs from this run")
    current_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                    capture_output=True, text=True).stdout.strip()
    if audit_manifest.get("git_commit") != current_commit:
        raise RuntimeError("source commit differs from the commit that passed G0; rerun audit")
    settings_raw = raw_config.get("model", {})
    checkpoint_revision = str(settings_raw.get("revision"))
    audited_settings = audit_manifest.get("settings_resolved", {})
    if (audited_settings.get("model_id") != settings_raw.get("id")
            or audited_settings.get("model_revision") != checkpoint_revision):
        raise RuntimeError("audit checkpoint revision differs from pilot config")
    historical = audit_manifest.get("fixed8_input", {})
    fixed8_value = historical.get("path") or raw_config.get("dataset", {}).get(
        "historical_fixed8_csv", "data/diagnostic_50_results.csv")
    fixed8_path = Path(fixed8_value).expanduser()
    if not fixed8_path.is_absolute():
        fixed8_path = root / fixed8_path
    fixed8_path = fixed8_path.resolve()
    if not fixed8_path.is_file():
        raise FileNotFoundError(f"historical development input is missing: {fixed8_path}")
    fixed8_hash = hashlib.sha256(fixed8_path.read_bytes()).hexdigest()
    if historical.get("sha256") != fixed8_hash:
        raise RuntimeError("Fixed8 input checksum differs from the passed G0 audit")
    audit_hash = hashlib.sha256(audit_manifest_path.read_bytes()).hexdigest()
    identity = {"protocol_version": "jrc-moe-v4-intervention-1", "config_sha256": config_digest,
                "git_commit": current_commit, "checkpoint_revision": checkpoint_revision,
                "fixed8_input_sha256": fixed8_hash, "audit_manifest_sha256": audit_hash,
                "resource_caps": caps.__dict__, "prompt_limit": prompt_limit,
                "stage": stage_name}
    caps.validate()
    progress_path = bind_run_identity(output_dir, identity, resume=resume)

    if resume and progress_path.is_file():
        ledger = BudgetLedger.resume_from(caps, output_dir)
    else:
        ledger = BudgetLedger(caps, output_dir)
    atomic_write_json(progress_path, ledger.snapshot())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("JRC-v4 execution requires exactly one visible CUDA GPU")
    os.environ["HF_HUB_OFFLINE"] = "1"
    model, tokenizer = load_model()
    model_audit = audit_loaded_model(model)
    audit_arch = audit_manifest.get("model_audit") or {}
    if model_audit.get("ordered_moe_paths") != audit_arch.get("ordered_moe_paths"):
        raise RuntimeError("live MoE module order differs from the G0 audit")
    from dataclasses import replace
    from .fidelity import verified_contract_from_loaded_model
    contract, semantic = verified_contract_from_loaded_model(model, checkpoint_revision)
    runtime = audit_manifest.get("runtime_evidence") or {}
    if not semantic["runtime_semantics_gate"] or not runtime.get("native_patch_fidelity", {}).get("passed"):
        raise RuntimeError("runtime routing semantics/fidelity no longer match passed G0")
    contract = replace(contract, native_fixed8_equivalence_verified=True)

    frame = load_fixed8(fixed8_path).sort_values("diagnostic_ordinal").head(prompt_limit).copy()
    ids = frame["test_idx"].astype(int).tolist()
    records = reconstruct_gsm8k_records(ids)
    expected_lengths = frame.set_index("test_idx")["prompt_token_count"].astype(int).to_dict()
    prepared = {}
    for prompt_id in ids:
        rendered, prompt_tokens = build_model_input(records[prompt_id]["harness_context"], tokenizer)
        if int(prompt_tokens.shape[1]) != expected_lengths[prompt_id]:
            raise RuntimeError(f"prompt token count changed for test_idx={prompt_id}")
        prepared[prompt_id] = {"rendered": rendered, "input_ids": prompt_tokens.to(input_device(model))}

    split = prompt_split_manifest(ids, seed=int(raw_config["dataset"]["prompt_partition_seed"]),
                                  confirmatory_ids=raw_config["dataset"].get("confirmatory_prompt_ids", []))
    split["dataset_sha256"] = fixed8_hash
    split["partition"] = "development_pilot_only"
    atomic_write_json(output_dir / "dataset_split_manifest.json", split)

    names = tuple(model_audit["ordered_moe_paths"])
    monitored_layers = tuple(int(x) for x in raw_config["intervention"]["monitored_layers"])
    controller = LayerwiseKController(names, min_k=int(raw_config["intervention"]["min_k"]),
                                      max_k=int(raw_config["intervention"]["max_k"]),
                                      monitored_layers=())
    install_layerwise_patch(model, controller, contract)
    streams = IndexedRandom(int(raw_config["dataset"]["prompt_partition_seed"]))
    branch_reserve = float(raw_config.get("resources", {}).get("branch_reserve_gpu_seconds", 600))
    setup_reserve = float(raw_config.get("resources", {}).get("state_capture_reserve_gpu_seconds", 300))
    output_reserve = int(raw_config.get("resources", {}).get("output_reserve_bytes", 65536))
    exchanges_per_state = int(raw_config.get("sampling", {}).get("exchanges_per_state", 2))
    base_k = int(raw_config["intervention"].get("operational_base_k", 6))
    target_nfe_range = tuple(raw_config.get("sampling", {}).get("target_nfe_range", (1, 8)))
    if len(target_nfe_range) != 2 or target_nfe_range[0] < 1 or target_nfe_range[1] < target_nfe_range[0]:
        raise ValueError("sampling.target_nfe_range must contain an increasing positive range")
    complete_pairs = _read_jsonl_ids(output_dir / "paired_exchange_records.jsonl", "exchange_id")
    intervention_ids = _read_jsonl_ids(output_dir / "interventions.jsonl", "exchange_id")
    planned_by_prompt: dict[str, set[str]] = {}
    interventions_path = output_dir / "interventions.jsonl"
    if interventions_path.is_file():
        for line in interventions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prior = json.loads(line)
            if prior.get("exchange_id") is not None and prior.get("prompt_id") is not None:
                planned_by_prompt.setdefault(str(prior["prompt_id"]), set()).add(str(prior["exchange_id"]))
    branch_fields = ["branch_id", "exchange_id", "prompt_id", "arm", "correctness", "canonical_prediction",
                     "canonical_rule", "total_nfe", "total_pairs", "generated_text", "raw_generated_text"]
    cost_fields = ["branch_id", "exchange_id", "prompt_id", "arm", "started_unix", "elapsed_seconds",
                   "gpu_seconds", "nfe", "executed_pairs", "status"]
    pair_fields = ["exchange_id", "prompt_id", "global_nfe", "block_idx", "layer_idx", "recipient_position",
                   "donor_position", "selection_probability", "control_correct", "treatment_correct", "effect",
                   "control_nfe", "treatment_nfe", "control_total_pairs", "treatment_total_pairs",
                   "trajectory_pair_delta", "target_layer_control_pairs", "target_layer_treatment_pairs",
                   "target_layer_pair_delta", "tier1_mean_jsd", "tier1_max_top_probability_delta"]
    interaction_path = output_dir / "interaction_samples.csv"
    if not interaction_path.exists():
        with interaction_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                ["interaction_id", "prompt_id", "global_nfe", "layer_idx", "position_i", "position_j",
                 "u00", "u10", "u01", "u11", "interaction", "total_pairs_by_arm"])
    (output_dir / "failures.jsonl").touch(exist_ok=True)
    completed_prompts = set()
    stop_reason = None
    try:
        for prompt_id in ids:
            prompt_key = str(prompt_id)
            already_planned = planned_by_prompt.get(prompt_key, set())
            if len(already_planned) >= exchanges_per_state and already_planned.issubset(complete_pairs):
                completed_prompts.add(prompt_key)
                continue
            prompt_tokens = prepared[prompt_id]["input_ids"]
            prompt_length = int(prompt_tokens.shape[1])
            total_length = prompt_length + GEN_LENGTH
            target_rng = streams.stream(prompt_id=prompt_key, stream_id="pilot-state", step=0,
                                        block=0, position=-1, purpose="target-nfe")
            target_nfe = target_rng.randint(int(target_nfe_range[0]), int(target_nfe_range[1]))
            layer_rng = streams.stream(prompt_id=prompt_key, stream_id="pilot-state", step=0,
                                        block=0, position=-1, purpose="target-layer")
            target_layer = layer_rng.choice(monitored_layers)
            state_box: dict[str, ReplayState] = {}
            provider = lambda **kw: [[base_k] * total_length for _ in names]

            # Replay the shared prefix under K=6 and stop immediately before the
            # selected forward. Prefix work is timed and charged, but is not a
            # counterfactual branch or an outcome label.
            cuda = torch.cuda.is_available()
            setup_start = torch.cuda.Event(enable_timing=True) if cuda else None
            setup_end = torch.cuda.Event(enable_timing=True) if cuda else None
            try:
                ledger.check_before_branch(prompt_key, reserve_gpu_seconds=setup_reserve,
                                           reserve_output_bytes=output_reserve, count_branch=False)
            except BudgetExhausted as exc:
                stop_reason = str(exc)
                break
            setup_branch_id = f"p{prompt_key}:state-capture"
            setup_started_unix = time.time()
            setup_wall_start = time.perf_counter()
            if cuda:
                torch.cuda.synchronize()
                setup_start.record()
            _append_csv(output_dir / "cost_ledger.csv", cost_fields,
                        {"branch_id": setup_branch_id, "exchange_id": "", "prompt_id": prompt_key,
                         "arm": "state_capture", "started_unix": setup_started_unix,
                         "elapsed_seconds": "", "gpu_seconds": "", "nfe": "",
                         "executed_pairs": "", "status": "started"})
            def capture_at_target(state):
                if state.global_nfe == target_nfe:
                    state_box["state"] = state.clone()
                    return True
                return False
            try:
                _, _, paused_meta = generate_with_routing_mode(
                    model, prompt_tokens, "jrc_v4_base_k6", controller,
                    allocation_provider=provider, before_forward=capture_at_target,
                    checkpoint_revision=checkpoint_revision, config_digest=config_digest,
                    before_forward_nfe=target_nfe)
                if not paused_meta.get("paused_before_forward") or "state" not in state_box:
                    raise RuntimeError(f"failed to capture requested NFE {target_nfe} for prompt {prompt_key}")
            except BaseException as exc:
                if cuda:
                    torch.cuda.synchronize()
                    setup_end.record()
                    torch.cuda.synchronize()
                    failed_gpu_seconds = setup_start.elapsed_time(setup_end) / 1000.0
                    ledger.add_gpu_time(failed_gpu_seconds)
                else:
                    failed_gpu_seconds = 0.0
                _append_csv(output_dir / "cost_ledger.csv", cost_fields,
                            {"branch_id": setup_branch_id, "exchange_id": "", "prompt_id": prompt_key,
                             "arm": "state_capture", "started_unix": "",
                             "elapsed_seconds": time.perf_counter() - setup_wall_start,
                             "gpu_seconds": failed_gpu_seconds, "nfe": "", "executed_pairs": "",
                             "status": "failed"})
                record_failure(output_dir, prompt_id=prompt_key, branch_id=None, error=exc, stage="pilot_state_capture")
                raise
            setup_gpu_seconds = _gpu_seconds_since(setup_start, setup_end, cuda)
            ledger.add_gpu_time(setup_gpu_seconds)
            state = state_box["state"]
            _append_csv(output_dir / "cost_ledger.csv", cost_fields,
                        {"branch_id": setup_branch_id, "exchange_id": "", "prompt_id": prompt_key,
                         "arm": "state_capture", "started_unix": "",
                         "elapsed_seconds": paused_meta["continuation_seconds"],
                         "gpu_seconds": setup_gpu_seconds, "nfe": state.global_nfe,
                         "executed_pairs": state.decoder_counters.get("total_pairs", 0),
                         "status": "completed"})
            if state.block_idx >= GEN_LENGTH // BLOCK_LENGTH:
                raise RuntimeError("captured replay state has an invalid active block")
            block_start = prompt_length + state.block_idx * BLOCK_LENGTH
            block_end = block_start + BLOCK_LENGTH
            unresolved = torch.where(state.input_ids[0, block_start:block_end] == MASK_ID)[0]
            unresolved_positions = [block_start + int(i) for i in unresolved.detach().cpu().tolist()]
            k_by_position = [state.k_by_layer[target_layer][position] for position in unresolved_positions]
            sample_seed = int(streams.stream(prompt_id=prompt_key, stream_id="pilot-exchange", step=state.global_nfe,
                                             block=state.block_idx, position=target_layer,
                                             purpose="ordered-pair-sample").randrange(2**31))
            sampled = sample_equal_budget_exchanges(
                prompt_id=prompt_key, global_nfe=state.global_nfe, layer_idx=target_layer,
                unresolved_positions=unresolved_positions, k_by_position=k_by_position,
                count=exchanges_per_state, seed=sample_seed, base_k=base_k,
                min_k=int(raw_config["intervention"]["min_k"]),
                max_k=int(raw_config["intervention"]["max_k"]))
            if not sampled:
                raise RuntimeError(f"no legal unresolved K={base_k} pair at prompt={prompt_key} NFE={state.global_nfe}")

            state_hash = hashlib.sha256(state.input_ids.detach().cpu().numpy().tobytes()).hexdigest()
            state_record = {"prompt_id": prompt_key, "global_nfe": state.global_nfe,
                            "local_nfe": state.local_nfe, "block_idx": state.block_idx,
                            "layer_idx": target_layer, "unresolved_count": len(unresolved_positions),
                            "state_input_sha256": state_hash, "prefix_total_pairs": state.decoder_counters.get("total_pairs", 0),
                            "checkpoint_revision": checkpoint_revision, "config_sha256": config_digest,
                            "state_capture_gpu_seconds": setup_gpu_seconds}
            append_jsonl(output_dir / "state_index.jsonl", state_record)

            for sample_idx, item in enumerate(sampled):
                exchange = item["exchange"]
                exchange_id = f"p{prompt_key}-n{state.global_nfe}-l{target_layer}-r{exchange.recipient_position}-d{exchange.donor_position}"
                if exchange_id in complete_pairs:
                    continue
                record = {**{k: v for k, v in item.items() if k != "exchange"},
                          "exchange_id": exchange_id, "sample_index": sample_idx,
                          "block_idx": state.block_idx, "state_input_sha256": state_hash,
                          "continuation_policy": "fixed8_after_target_nfe",
                          "operational_base_schedule": f"K={base_k} through target decision",
                          "decision_local_target_layer_budget_neutral": True,
                          "full_trajectory_pair_equality_assumed": False}
                if exchange_id not in intervention_ids:
                    append_jsonl(output_dir / "interventions.jsonl", record)
                    intervention_ids.add(exchange_id)
                planned_by_prompt.setdefault(prompt_key, set()).add(exchange_id)
                controller.monitored_layers = frozenset((target_layer,))
                controller.monitored_positions = frozenset((exchange.recipient_position, exchange.donor_position))
                branch_ids: dict[str, str] = {}
                branch_text: dict[str, dict] = {}
                control_logp: dict[str, torch.Tensor] = {}
                tier1: dict[str, float] = {}
                active_branch = {"name": None}
                later_provider = lambda **kw: [[8] * total_length for _ in names]

                def before_branch(arm):
                    ledger.check_before_branch(prompt_key, reserve_gpu_seconds=branch_reserve,
                                               reserve_output_bytes=output_reserve)
                    branch_id = f"{exchange_id}:{arm}:a{ledger.branches_started}"
                    branch_ids[arm] = branch_id
                    active_branch["name"] = arm
                    _append_csv(output_dir / "cost_ledger.csv", cost_fields,
                                {"branch_id": branch_id, "exchange_id": exchange_id, "prompt_id": prompt_key,
                                 "arm": arm, "started_unix": time.time(), "elapsed_seconds": "",
                                 "gpu_seconds": "", "nfe": "", "executed_pairs": "", "status": "started"})

                def utility(output):
                    arm = active_branch["name"]
                    generated = output[0, prompt_length:].detach().cpu().tolist()
                    raw_text, clean_text = postprocess_generated_region(generated, tokenizer)
                    grade = canonical_grade_text(clean_text, records[prompt_id]["gold_full_answer"])
                    branch_text[arm] = {"raw_text": raw_text, "clean_text": clean_text, **grade}
                    return int(grade["canonical_correct"])

                def after_branch(arm, output, trace, meta, value, *, elapsed_seconds, gpu_seconds):
                    ledger.add_gpu_time(gpu_seconds)
                    details = branch_text[arm]
                    trace_path = output_dir / "traces" / f"{branch_ids[arm].replace(':', '_')}.csv"
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace.to_csv(trace_path, index=False)
                    append_jsonl(output_dir / "branch_outcomes.jsonl", {
                        "branch_id": branch_ids[arm], "exchange_id": exchange_id,
                        "prompt_id": prompt_key, "arm": arm, "correctness": int(value),
                        "total_nfe": meta["total_nfe"], "total_pairs": meta["total_pairs"],
                        "avg_k": meta["avg_k"], "generation_seconds": meta["generation_seconds"],
                        "continuation_seconds": meta.get("continuation_seconds"),
                        "canonical_prediction": details["canonical_prediction"],
                        "canonical_rule": details["canonical_rule"],
                        "generated_text": details["clean_text"],
                        "raw_generated_text": details["raw_text"], "trace_path": str(trace_path),
                    })
                    _append_csv(output_dir / "cost_ledger.csv", cost_fields,
                                {"branch_id": branch_ids[arm], "exchange_id": exchange_id,
                                 "prompt_id": prompt_key, "arm": arm, "started_unix": "",
                                 "elapsed_seconds": elapsed_seconds, "gpu_seconds": gpu_seconds,
                                 "nfe": meta["total_nfe"], "executed_pairs": meta["total_pairs"],
                                 "status": "completed"})

                def branch_error(arm, exc, elapsed_seconds, gpu_seconds):
                    ledger.add_gpu_time(gpu_seconds)
                    branch_id = branch_ids.get(arm)
                    record_failure(output_dir, prompt_id=prompt_key, branch_id=branch_id,
                                   error=exc, stage=f"pilot_{arm}")
                    _append_csv(output_dir / "cost_ledger.csv", cost_fields,
                                {"branch_id": branch_id, "exchange_id": exchange_id,
                                 "prompt_id": prompt_key, "arm": arm, "started_unix": "",
                                 "elapsed_seconds": elapsed_seconds, "gpu_seconds": gpu_seconds,
                                 "nfe": "", "executed_pairs": "", "status": "failed"})

                def forward_observer(observation):
                    if observation["branch"] != "control" or observation["global_nfe"] != state.global_nfe:
                        return
                    positions = (exchange.recipient_position, exchange.donor_position)
                    unresolved_count = observation["masked_before"]
                    mask_ratio = unresolved_count / float(BLOCK_LENGTH)
                    for role, position in (("recipient", exchange.recipient_position),
                                           ("donor", exchange.donor_position)):
                        routing = observation["routing_records"].get((target_layer, position))
                        if routing is None:
                            raise RuntimeError(f"missing pre-decision routing feature for layer={target_layer} pos={position}")
                        token_unresolved = bool(state.input_ids[0, position].item() == MASK_ID)
                        record_feature = FeatureRecord(
                            prompt_id=prompt_key, global_nfe=state.global_nfe, layer_idx=target_layer,
                            position=position, block_idx=state.block_idx, mask_ratio=mask_ratio,
                            is_prompt_token=position < prompt_length, is_target_token=position >= prompt_length,
                            active_block=block_start <= position < block_end,
                            relative_position=(position - prompt_length) / float(GEN_LENGTH),
                            unresolved=token_unresolved, provisional=position >= prompt_length and not token_unresolved,
                            confidence_history=tuple(state.confidence_history.get(position, [])),
                            router=routing["router_statistics"],
                            router_history=(tuple(routing["experts"]),),
                            hidden_state_norm=routing.get("hidden_state_norm"),
                            attention_scores=None, feature_age=0, source_nfe=state.global_nfe,
                            extra={"exchange_id": exchange_id, "role": role,
                                   "attention_enabled": False, "router_k": routing["k"]})
                        record_feature.validate()
                        append_jsonl(output_dir / "features.jsonl", record_feature.__dict__)

                def sensitivity_observer(observation):
                    if observation["global_nfe"] != state.global_nfe:
                        return
                    positions = list(observation["positions"])
                    selected = [exchange.recipient_position, exchange.donor_position]
                    row_indices = [positions.index(position) for position in selected]
                    logits = observation["logits"][row_indices].float()
                    logp = torch.log_softmax(logits, dim=-1).detach().cpu()
                    if observation["branch"] == "control":
                        control_logp["value"] = logp
                    else:
                        if "value" not in control_logp:
                            raise RuntimeError("treatment sensitivity arrived before paired control")
                        mean_jsd, top_delta = _jsd(control_logp["value"], logp)
                        tier1.update(mean_jsd=mean_jsd, max_top_probability_delta=top_delta)

                try:
                    exchange_state = state.clone()
                    exchange_state.selected_layer_idx = target_layer
                    exchange_state.recipient_position = exchange.recipient_position
                    exchange_state.donor_position = exchange.donor_position
                    pair_result = paired_exchange_rollout(
                        model=model, prompt_ids=prompt_tokens, mode="jrc_v4_exchange",
                        controller=controller, state=exchange_state, exchange=exchange,
                        allocation_provider=later_provider, utility=utility,
                        checkpoint_revision=checkpoint_revision, config_digest=config_digest,
                        before_branch=before_branch, after_branch=after_branch,
                        on_branch_error=branch_error, forward_observer=forward_observer,
                        sensitivity_observer=sensitivity_observer)
                except BudgetExhausted as exc:
                    stop_reason = str(exc)
                    break
                except BaseException as exc:
                    stop_reason = f"{type(exc).__name__}: {exc}"
                    break

                def target_pairs(trace):
                    row = trace[trace["global_nfe"] == state.global_nfe]
                    if len(row) != 1:
                        raise RuntimeError("paired trace does not contain exactly one target NFE")
                    budgets = row.iloc[0].get("layer_budgets")
                    if budgets is None:
                        raise RuntimeError("paired trace is missing actual per-layer expert-token counts")
                    return int(budgets[target_layer])

                control_target_pairs = target_pairs(pair_result["control_trace"])
                treatment_target_pairs = target_pairs(pair_result["treatment_trace"])
                if control_target_pairs != treatment_target_pairs:
                    raise RuntimeError("target-layer exchange did not preserve its executed pair count")
                control, treatment = pair_result["control_meta"], pair_result["treatment_meta"]
                outcome = {
                    "exchange_id": exchange_id, "prompt_id": prompt_key,
                    "global_nfe": state.global_nfe, "block_idx": state.block_idx,
                    "layer_idx": target_layer, "recipient_position": exchange.recipient_position,
                    "donor_position": exchange.donor_position,
                    "selection_probability": item["selection_probability"],
                    "control_correct": int(pair_result["control_utility"]),
                    "treatment_correct": int(pair_result["treatment_utility"]),
                    "effect": int(pair_result["effect"]),
                    "control_nfe": control["total_nfe"], "treatment_nfe": treatment["total_nfe"],
                    "control_total_pairs": control["total_pairs"],
                    "treatment_total_pairs": treatment["total_pairs"],
                    "trajectory_pair_delta": treatment["total_pairs"] - control["total_pairs"],
                    "target_layer_control_pairs": control_target_pairs,
                    "target_layer_treatment_pairs": treatment_target_pairs,
                    "target_layer_pair_delta": treatment_target_pairs - control_target_pairs,
                    "tier1_mean_jsd": tier1.get("mean_jsd"),
                    "tier1_max_top_probability_delta": tier1.get("max_top_probability_delta"),
                    "branch_ids": branch_ids,
                }
                append_jsonl(output_dir / "paired_exchange_records.jsonl", outcome)
                _append_csv(output_dir / "paired_exchanges.csv", pair_fields, outcome)
                complete_pairs.add(exchange_id)
            if stop_reason:
                break
            completed_prompts.add(prompt_key)
    finally:
        for _, module in model.named_modules():
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        del model

    expected_pairs = len(ids) * exchanges_per_state
    status = "passed" if len(complete_pairs) >= expected_pairs and stop_reason is None else "partial"
    paired_rows = []
    paired_path = output_dir / "paired_exchange_records.jsonl"
    if paired_path.is_file():
        paired_rows = [json.loads(line) for line in paired_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _rewrite_csv(output_dir / "paired_exchanges.csv", pair_fields, paired_rows)
    state_rows = []
    state_path = output_dir / "state_index.jsonl"
    if state_path.is_file():
        state_rows = [json.loads(line) for line in state_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique_states = {(str(row["prompt_id"]), int(row["global_nfe"]), int(row["layer_idx"]))
                     for row in state_rows}
    effect_counts = {"negative": 0, "zero": 0, "positive": 0}
    for row in paired_rows:
        effect_counts["negative" if row["effect"] < 0 else "positive" if row["effect"] > 0 else "zero"] += 1
    effect_interval = (paired_prompt_bootstrap(paired_rows, treatment_key="treatment_correct",
                                               comparator_key="control_correct",
                                               resamples=int(raw_config["statistics"].get("bootstrap_resamples", 2000)),
                                               seed=int(raw_config["statistics"].get("bootstrap_seed", 20261009)))
                       if paired_rows else None)
    runtime_rows = []
    cost_path = output_dir / "cost_ledger.csv"
    if cost_path.is_file():
        with cost_path.open("r", encoding="utf-8", newline="") as stream:
            runtime_rows = [row for row in csv.DictReader(stream)
                            if row.get("arm") in {"control", "treatment"} and row.get("status") == "completed"]
    branch_wall = sorted(float(row["elapsed_seconds"]) for row in runtime_rows if row.get("elapsed_seconds"))
    branch_gpu = sorted(float(row["gpu_seconds"]) for row in runtime_rows if row.get("gpu_seconds"))
    def median_p95(values):
        if not values:
            return {"median_seconds": None, "p95_seconds": None}
        return {"median_seconds": float(values[(len(values) - 1) // 2]) if len(values) % 2 else
                float((values[len(values) // 2 - 1] + values[len(values) // 2]) / 2),
                "p95_seconds": float(values[max(0, __import__("math").ceil(0.95 * len(values)) - 1)])}
    pilot_summary = {
        "paired_exchanges": len(paired_rows), "effect_counts": effect_counts,
        "prompt_clustered_effect": effect_interval,
        "control_correct_rate": (sum(int(r["control_correct"]) for r in paired_rows) / len(paired_rows)) if paired_rows else None,
        "treatment_correct_rate": (sum(int(r["treatment_correct"]) for r in paired_rows) / len(paired_rows)) if paired_rows else None,
        "mean_control_nfe": (sum(int(r["control_nfe"]) for r in paired_rows) / len(paired_rows)) if paired_rows else None,
        "mean_treatment_nfe": (sum(int(r["treatment_nfe"]) for r in paired_rows) / len(paired_rows)) if paired_rows else None,
        "mean_control_total_pairs": (sum(int(r["control_total_pairs"]) for r in paired_rows) / len(paired_rows)) if paired_rows else None,
        "mean_treatment_total_pairs": (sum(int(r["treatment_total_pairs"]) for r in paired_rows) / len(paired_rows)) if paired_rows else None,
        "trajectory_pair_deltas": [int(r["trajectory_pair_delta"]) for r in paired_rows],
        "branch_wall_runtime": median_p95(branch_wall), "branch_cuda_event_runtime": median_p95(branch_gpu),
        "sensitivity_measure": "mean JSD across sampled recipient/donor next-forward logits; diagnostic only",
    }
    atomic_write_json(output_dir / "pilot_summary.json", pilot_summary)
    (output_dir / "pilot_report.md").write_text(
        "# Bounded JRC-MoE v4 pilot\n\n"
        f"Status: **{status}**. Paired exchanges: {len(paired_rows)}. "
        f"Effects: {effect_counts['negative']} negative, {effect_counts['zero']} zero, "
        f"{effect_counts['positive']} positive.\n\n"
        f"Prompt-clustered treatment-minus-control correctness: `{json.dumps(effect_interval)}`.\n\n"
        "Decision-local target-layer pair budgets are checked equal for every exchange. "
        "Whole-rollout executed-pair differences are reported per exchange and are not assumed to be zero. "
        "The Tier-1 JSD is a sensitivity diagnostic, not a utility label or predictor result.\n",
        encoding="utf-8")
    manifest = {
        "protocol_version": "jrc-moe-v4-intervention-1", "stage": stage_name,
        "status": status, "g0": "passed", "g1": "not_run", "config_sha256": config_digest,
        "audit_manifest": str(audit_manifest_path), "fixed8_input_sha256": fixed8_hash,
        "checkpoint_revision": checkpoint_revision, "prompt_limit": prompt_limit,
        "git_commit": current_commit, "audit_manifest_sha256": audit_hash,
        "prompts_completed": len(completed_prompts), "prompts_requested": len(ids),
        "states_captured": len(unique_states),
        "paired_exchanges_completed": len(complete_pairs), "expected_exchanges": expected_pairs,
        "branches_started": ledger.branches_started, "gpu_seconds": ledger.gpu_seconds,
        "gpu_hours": ledger.gpu_seconds / 3600.0, "resource_caps": ledger.caps.__dict__,
        "stop_reason": stop_reason, "continuation_policy": "fixed8_after_target_nfe",
        "pilot_summary": pilot_summary,
        "target_exchange_budget_neutral": True,
        "whole_trajectory_pair_matching": "not_assumed; per-arm pair totals and delta are recorded",
        "attention_features": "disabled until post-QK-normalization/post-RoPE hook fidelity is verified",
        "gates": {"G1": "not_evaluated", "G2": "not_run", "G3": "not_run", "G4": "not_run"},
        "artifacts": ["dataset_split_manifest.json", "state_index.jsonl", "interventions.jsonl",
                      "features.jsonl", "branch_outcomes.jsonl", "paired_exchange_records.jsonl",
                      "paired_exchanges.csv", "interaction_samples.csv", "cost_ledger.csv",
                      "failures.jsonl", "progress.json", "pilot_summary.json", "pilot_report.md",
                      "run_identity.json", "traces/"],
        "interaction_samples": "header_only; four-arm interaction branches were outside the bounded paired-exchange pilot budget",
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    if status != "passed":
        raise RuntimeError(stop_reason or f"pilot incomplete: completed {len(complete_pairs)}/{expected_pairs} exchanges")
    return manifest
