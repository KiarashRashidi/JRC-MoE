"""Paired full-continuation exchanges using the shared decoder implementation."""
from __future__ import annotations

from typing import Callable

from .controller import BudgetExchange, LayerwiseKController
from .replay import ReplayState
from .statistics import exchange_effect


def verify_noop_replay(*, model, prompt_ids, mode: str, controller: LayerwiseKController,
                       state: ReplayState, allocation_provider: Callable,
                       checkpoint_revision: str, config_digest: str) -> dict:
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
    if not left[1].equals(right[1]):
        raise RuntimeError("no-op replay produced a different confidence/commit trace")
    for key in ("total_nfe", "total_pairs", "avg_k"):
        if left[2][key] != right[2][key]:
            raise RuntimeError(f"no-op replay changed {key}")
    return {"passed": True, "final_tokens_identical": True, "trace_identical": True,
            "total_nfe": left[2]["total_nfe"], "total_pairs": left[2]["total_pairs"]}


def paired_exchange_rollout(*, model, prompt_ids, mode: str, controller: LayerwiseKController,
                            state: ReplayState, exchange: BudgetExchange, allocation_provider: Callable,
                            utility: Callable, checkpoint_revision: str, config_digest: str,
                            sampler_settings: dict | None = None) -> dict:
    """Run both arms to completion from the same exact pre-forward state.

    The generation routine owns threshold/fallback commits and later policy
    continuation. This helper intentionally does not synthesize per-forward labels.
    """
    if state.global_nfe != exchange.global_nfe:
        raise ValueError("exchange target NFE must match the saved pre-forward state")
    from ..generation import generate_with_routing_mode

    arm_results = {}
    for branch in ("control", "treatment"):
        output, trace, meta = generate_with_routing_mode(
            model=model, prompt_ids=prompt_ids, mode=mode, controller=controller,
            allocation_provider=allocation_provider, resume_state=state.clone(),
            branch=branch, exchange=exchange,
            checkpoint_revision=checkpoint_revision, config_digest=config_digest,
            sampler_settings=sampler_settings,
        )
        value = float(utility(output))
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
