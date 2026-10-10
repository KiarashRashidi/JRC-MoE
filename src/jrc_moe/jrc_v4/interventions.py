"""Paired full-continuation exchanges using the shared decoder implementation."""
from __future__ import annotations

from typing import Callable

from .controller import BudgetExchange, LayerwiseKController
from .replay import ReplayState
from .statistics import exchange_effect


def verify_noop_replay(*, model, prompt_ids, mode: str, controller: LayerwiseKController,
                       state: ReplayState, allocation_provider: Callable,
                       checkpoint_revision: str, config_digest: str,
                       expected_output=None, expected_trace=None) -> dict:
    """Run an identical control replay twice and require exact outputs and traces."""
    from ..generation import generate_with_routing_mode
    arms = []
    for _ in range(2):
        arms.append(generate_with_routing_mode(
            model=model, prompt_ids=prompt_ids, mode=mode, controller=controller,
            allocation_provider=allocation_provider, resume_state=state.clone(),
            branch="control", exchange=None,
            checkpoint_revision=checkpoint_revision, config_digest=config_digest,
        ))
    left, right = arms
    import torch
    if not torch.equal(left[0], right[0]):
        raise RuntimeError("no-op replay produced different final token IDs")
    stable_left = left[1].drop(columns=["forward_seconds"], errors="ignore")
    stable_right = right[1].drop(columns=["forward_seconds"], errors="ignore")
    if not stable_left.equals(stable_right):
        raise RuntimeError("no-op replay produced a different confidence/commit trace")
    for key in ("total_nfe", "total_pairs", "avg_k"):
        if left[2][key] != right[2][key]:
            raise RuntimeError(f"no-op replay changed {key}")
    matches_reference = None
    if expected_output is not None:
        import torch
        matches_reference = bool(torch.equal(left[0], expected_output))
        if not matches_reference:
            raise RuntimeError("replayed tokens differ from the original full-rollout continuation")
    if expected_trace is not None:
        stable_expected = expected_trace.drop(columns=["forward_seconds"], errors="ignore")
        if not stable_left.equals(stable_expected):
            raise RuntimeError("replay confidence/commit decisions differ from the original rollout")
    return {"passed": True, "final_tokens_identical": True, "trace_identical": True,
            "matches_original_rollout": matches_reference,
            "total_nfe": left[2]["total_nfe"], "total_pairs": left[2]["total_pairs"]}


def paired_exchange_rollout(*, model, prompt_ids, mode: str, controller: LayerwiseKController,
                            state: ReplayState, exchange: BudgetExchange, allocation_provider: Callable,
                            utility: Callable, checkpoint_revision: str, config_digest: str,
                            sampler_settings: dict | None = None,
                            before_branch: Callable | None = None,
                            after_branch: Callable | None = None,
                            on_branch_error: Callable | None = None,
                            forward_observer: Callable | None = None,
                            sensitivity_observer: Callable | None = None) -> dict:
    """Run both arms to completion from the same exact pre-forward state.

    The generation routine owns threshold/fallback commits and later policy
    continuation. This helper intentionally does not synthesize per-forward labels.
    """
    if state.global_nfe != exchange.global_nfe:
        raise ValueError("exchange target NFE must match the saved pre-forward state")
    if state.selected_layer_idx is not None and (
            state.selected_layer_idx != exchange.layer_idx
            or state.recipient_position != exchange.recipient_position
            or state.donor_position != exchange.donor_position):
        raise ValueError("exchange identity differs from the selected site recorded in ReplayState")
    from ..generation import generate_with_routing_mode
    import time
    import torch

    arm_results = {}
    for branch in ("control", "treatment"):
        if before_branch is not None:
            before_branch(branch)
        cuda = torch.cuda.is_available()
        start_event = torch.cuda.Event(enable_timing=True) if cuda else None
        end_event = torch.cuda.Event(enable_timing=True) if cuda else None
        if cuda:
            torch.cuda.synchronize()
            start_event.record()
        started = time.perf_counter()
        try:
            output, trace, meta = generate_with_routing_mode(
                model=model, prompt_ids=prompt_ids, mode=mode, controller=controller,
                allocation_provider=allocation_provider, resume_state=state.clone(),
                branch=branch, exchange=exchange,
                checkpoint_revision=checkpoint_revision, config_digest=config_digest,
                sampler_settings=sampler_settings, forward_observer=forward_observer,
                sensitivity_observer=sensitivity_observer,
            )
        except BaseException as exc:
            if cuda:
                torch.cuda.synchronize()
                end_event.record()
                torch.cuda.synchronize()
            if on_branch_error is not None:
                on_branch_error(branch, exc, time.perf_counter() - started,
                                start_event.elapsed_time(end_event) / 1000.0 if cuda else 0.0)
            raise
        if cuda:
            torch.cuda.synchronize()
            end_event.record()
            torch.cuda.synchronize()
        gpu_seconds = start_event.elapsed_time(end_event) / 1000.0 if cuda else 0.0
        try:
            value = float(utility(output))
        except BaseException as exc:
            if on_branch_error is not None:
                on_branch_error(branch, exc, time.perf_counter() - started, gpu_seconds)
            raise
        elapsed_seconds = time.perf_counter() - started
        if after_branch is not None:
            after_branch(branch, output, trace, meta, value,
                         elapsed_seconds=elapsed_seconds, gpu_seconds=gpu_seconds)
        arm_results[branch] = {"output": output, "trace": trace, "meta": meta, "utility": value}
    control, treatment = arm_results["control"], arm_results["treatment"]
    return {
        "global_nfe": exchange.global_nfe,
        "layer_idx": exchange.layer_idx,
        "recipient_position": exchange.recipient_position,
        "donor_position": exchange.donor_position,
        "amount": exchange.amount,
        "control_utility": control["utility"],
        "treatment_utility": treatment["utility"],
        "effect": exchange_effect(control["utility"], treatment["utility"]),
        "control_meta": control["meta"],
        "treatment_meta": treatment["meta"],
        "control_trace": control["trace"],
        "treatment_trace": treatment["trace"],
        "control_output": control["output"],
        "treatment_output": treatment["output"],
        "paired_rng_restore": True,
        "complete_rollout": True,
    }
