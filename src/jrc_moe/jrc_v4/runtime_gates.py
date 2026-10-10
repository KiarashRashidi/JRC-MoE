"""Runtime evidence required by G0, using the repository's existing decoder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from ..config import GEN_LENGTH, MASK_ID, MODE_FGER, MODE_REFLEX, MODE_RRB, NUM_MOE_LAYERS
from ..data import load_fixed8, reconstruct_gsm8k_records
from ..generation import generate_with_routing_mode, postprocess_generated_region, validate_trace
from ..grading import canonical_grade_text
from ..model import build_model_input, fixed8_patch_equivalence, input_device
from ..routing import VariableKController, install_variable_k_patch
from .controller import LayerwiseKController
from .fidelity import legal_k_forward_gate, make_mask_phase_cases, native_patch_fidelity
from .interventions import verify_noop_replay
from .patch import install_layerwise_patch
from .profile import profile_callable


def _prepare_audit_prompts(model, tokenizer, fixed8_path: Path, count: int = 3):
    frame = load_fixed8(fixed8_path).sort_values("diagnostic_ordinal").head(count).copy()
    indices = frame["test_idx"].astype(int).tolist()
    records = reconstruct_gsm8k_records(indices)
    prepared = {}
    expected_lengths = frame.set_index("test_idx")["prompt_token_count"].astype(int).to_dict()
    for index in indices:
        rendered, ids = build_model_input(records[index]["harness_context"], tokenizer)
        length = int(ids.shape[1])
        if length != expected_lengths[index]:
            raise RuntimeError(f"prompt length mismatch for test_idx={index}: {length} != {expected_lengths[index]}")
        prepared[index] = {"rendered": rendered, "input_ids": ids.to(input_device(model))}
    return frame, records, prepared


def _restore_legacy_patch(model) -> None:
    for _, module in model.named_modules():
        if hasattr(module, "_reflex_original_forward"):
            module.forward = module._reflex_original_forward


def _reproduce_legacy_rows(*, root: Path, model, tokenizer, frame, records, prepared, count: int = 3) -> dict:
    result_file = root / "outputs" / "reflex_paper_replication50" / "results.jsonl"
    trace_root = root / "outputs" / "reflex_paper_replication50" / "traces"
    if not result_file.is_file():
        raise FileNotFoundError(f"committed REFLEX result rows are unavailable: {result_file}")
    prior_rows = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    prior = {(int(row["test_idx"]), row["mode"]): row for row in prior_rows if row.get("error") is None}
    picked = frame.sort_values("diagnostic_ordinal").iloc[[0, len(frame) // 2, len(frame) - 1]]
    selected = []
    modes = (MODE_REFLEX, MODE_RRB, MODE_FGER)
    for (_, row), mode in zip(picked.iterrows(), modes):
        key = (int(row["test_idx"]), mode)
        if key not in prior:
            raise FileNotFoundError(f"historical result row is missing for test_idx={key[0]} mode={mode}")
        selected.append((key, prior[key]))

    controller = VariableKController(NUM_MOE_LAYERS)
    install_variable_k_patch(model, controller)
    observed = []
    try:
        for (index, mode), expected in selected:
            output_ids, trace, meta = generate_with_routing_mode(
                model, prepared[index]["input_ids"], mode, controller)
            validate_trace(trace, mode)
            prompt_len = int(prepared[index]["input_ids"].shape[1])
            generated = output_ids[0, prompt_len:].detach().cpu().tolist()
            _, clean_text = postprocess_generated_region(generated, tokenizer)
            grade = canonical_grade_text(clean_text, records[index]["gold_full_answer"])
            for field in ("total_nfe", "total_pairs", "canonical_correct"):
                actual = grade.get(field) if field == "canonical_correct" else meta[field]
                if actual != expected[field]:
                    raise RuntimeError(f"legacy replay mismatch at ({index}, {mode}) {field}: {actual} != {expected[field]}")
            trace_path = trace_root / f"test_{index:04d}_{mode}.csv"
            if not trace_path.is_file():
                raise FileNotFoundError(f"historical trace is unavailable: {trace_path}")
            prior_trace = pd.read_csv(trace_path)
            stable = ["block_idx", "local_nfe", "global_nfe", "masked_before", "num_committed",
                      "fallback_used", "masked_after", "direct_pairs"]
            if not set(stable).issubset(trace.columns) or not set(stable).issubset(prior_trace.columns):
                raise RuntimeError("legacy trace is missing required commit/accounting fields")
            if not trace[stable].reset_index(drop=True).equals(prior_trace[stable].reset_index(drop=True)):
                raise RuntimeError(f"legacy commit/pair trace differs at ({index}, {mode})")
            observed.append({"test_idx": index, "mode": mode, "total_nfe": meta["total_nfe"],
                             "total_pairs": meta["total_pairs"], "canonical_correct": grade["canonical_correct"],
                             "stable_trace_rows": len(trace)})
    finally:
        _restore_legacy_patch(model)
    return {"passed": True, "rows": observed, "result_source": str(result_file)}


def run_g0_runtime(*, root: Path, model, tokenizer, fixed8_path: Path, checkpoint_revision: str,
                   config_digest: str, model_audit: dict, contract, settings) -> dict:
    """Run only G0 checks. Any failed prerequisite prevents downstream checks."""
    frame, records, prepared = _prepare_audit_prompts(model, tokenizer, fixed8_path, count=3)
    chosen_ids = frame["test_idx"].astype(int).tolist()
    fidelity_cases = []
    for index in chosen_ids:
        prompt = prepared[index]["input_ids"]
        short_prompt = prompt[:, :min(128, prompt.shape[1])]
        fidelity_cases.extend(make_mask_phase_cases(short_prompt, mask_id=MASK_ID))

    # The v4 fidelity helper deliberately allows its own equivalence probe before
    # setting the contract's normal intervention gate.
    fidelity = native_patch_fidelity(model, None, contract, fidelity_cases)
    if not fidelity["passed"]:
        return {"gates": {"native_patch_fidelity": "failed", "legal_k_forwards": "not_run",
                          "legacy_reproduction": "not_run", "fresh_fixed8_profile": "not_run",
                          "replay": "not_run", "synchronized_profile": "not_run"},
                "native_patch_fidelity": fidelity}

    from dataclasses import replace
    verified = replace(contract, native_fixed8_equivalence_verified=True)
    short_case = fidelity_cases[0][0]
    k_forward = legal_k_forward_gate(model, LayerwiseKController(
        model_audit["ordered_moe_paths"], min_k=settings.min_k, max_k=settings.max_k,
        monitored_layers=settings.monitored_layers), verified, short_case, legal_k=settings.legal_k)

    legacy_patch = VariableKController(NUM_MOE_LAYERS)
    fixed8_equivalence = fixed8_patch_equivalence(model, prepared[chosen_ids[0]]["input_ids"], legacy_patch)
    legacy = _reproduce_legacy_rows(root=root, model=model, tokenizer=tokenizer,
                                    frame=frame, records=records, prepared=prepared)

    # Measure complete Fixed8 denoising runs through the existing REFLEX dispatcher,
    # with synchronized host/CUDA timing and no feature hooks enabled.
    profile_prompt = prepared[chosen_ids[0]]["input_ids"]
    profile_controller = VariableKController(NUM_MOE_LAYERS)
    install_variable_k_patch(model, profile_controller)
    try:
        profile = profile_callable(
            lambda: generate_with_routing_mode(model, profile_prompt, "fixed8", profile_controller),
            warmup=1, repetitions=3, tokens_per_call=GEN_LENGTH)
    finally:
        _restore_legacy_patch(model)
    profile["workload"] = "full_256_token_fixed8_denoising"
    profile["dispatcher"] = "existing_variable_k_python_expert_loop"
    profile["prompt_token_count"] = int(profile_prompt.shape[1])

    # A v4 Fixed8 replay from the exact pre-forward state must reproduce the
    # entire reference continuation, including threshold/fallback decisions.
    names = tuple(model_audit["ordered_moe_paths"])
    replay_controller = LayerwiseKController(
        names, min_k=settings.min_k, max_k=settings.max_k,
        monitored_layers=())
    install_layerwise_patch(model, replay_controller, verified)
    state_box = {}
    provider = lambda **kw: [[8] * (int(profile_prompt.shape[1]) + GEN_LENGTH) for _ in names]
    def capture(state):
        if "state" not in state_box:
            state_box["state"] = state
    try:
        baseline_ids, baseline_trace, baseline_meta = generate_with_routing_mode(
            model, profile_prompt, "fixed8", replay_controller, allocation_provider=provider,
            before_forward=capture, checkpoint_revision=checkpoint_revision,
            config_digest=config_digest, before_forward_nfe=1)
        replay = verify_noop_replay(
            model=model, prompt_ids=profile_prompt, mode="fixed8", controller=replay_controller,
            state=state_box["state"], allocation_provider=provider,
            checkpoint_revision=checkpoint_revision, config_digest=config_digest,
            expected_output=baseline_ids, expected_trace=baseline_trace)
    finally:
        for _, module in model.named_modules():
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    gates = {"native_patch_fidelity": "passed", "legal_k_forwards": "passed",
             "legacy_reproduction": "passed", "fresh_fixed8_profile": "passed",
             "replay": "passed", "synchronized_profile": "passed" if profile["synchronized"] else "failed"}
    return {"gates": gates, "native_patch_fidelity": fidelity,
            "legal_k_forwards": k_forward, "legacy_fixed8_equivalence": fixed8_equivalence,
            "legacy_reproduction": legacy, "fresh_fixed8_profile": profile,
            "replay": {**replay, "reference_total_nfe": baseline_meta["total_nfe"],
                       "reference_total_pairs": baseline_meta["total_pairs"]},
            "prompt_ids": chosen_ids,
            "input_hashes": {str(i): hashlib.sha256(prepared[i]["input_ids"].detach().cpu().numpy().tobytes()).hexdigest()
                             for i in chosen_ids}}
