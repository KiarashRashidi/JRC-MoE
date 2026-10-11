"""Predeclared sparse scale slate for development-only follow-up experiments."""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

from .multi_interventions import MultiTokenReallocation


def initial_scale_slate() -> list[dict[str, int | str]]:
    """Return the sparse comparisons; m, c, and A are separate factors."""
    return [
        {"m": 2, "A": 1, "c": 2, "condition": "dose_m2_A1"},
        {"m": 2, "A": 2, "c": 2, "condition": "dose_m2_A2"},
        {"m": 2, "A": 3, "c": 2, "condition": "dose_m2_A3"},
        {"m": 5, "A": 2, "c": 2, "condition": "sparsity_m5_A2_c2"},
        {"m": 5, "A": 2, "c": 4, "condition": "context_m5_A2_c4"},
        {"m": 8, "A": 2, "c": 2, "condition": "sparsity_m8_A2_c2"},
        {"m": 8, "A": 2, "c": 4, "condition": "context_m8_A2_c4"},
        {"m": 16, "A": 2, "c": 2, "condition": "sparsity_m16_A2_c2"},
        {"m": 16, "A": 2, "c": 4, "condition": "context_m16_A2_c4"},
        {"m": 5, "A": 4, "c": 4, "condition": "scale_m5_A4_c4"},
        {"m": 8, "A": 4, "c": 4, "condition": "scale_m8_A4_c4"},
        {"m": 16, "A": 4, "c": 4, "condition": "scale_m16_A4_c4"},
    ]


def _choose_changed(
    candidates: Sequence[int],
    changed_count: int,
    *,
    rng: random.Random,
    selection_strategy: str,
    confidence: Mapping[int, float] | None,
    admission_margin: Mapping[int, float] | None,
) -> tuple[list[int], list[int], str, float | None]:
    if changed_count > len(candidates):
        raise ValueError("infeasible action: candidate group is smaller than changed-token count")
    if changed_count % 2:
        raise ValueError("balanced slate currently requires an even changed-token count")
    half = changed_count // 2
    if selection_strategy == "uniform":
        changed = rng.sample(list(candidates), changed_count)
        rng.shuffle(changed)
        return changed[:half], changed[half:], "uniform changed subset and signs", 1.0 / (
            math.comb(len(candidates), changed_count) * math.comb(changed_count, half)
        )
    if selection_strategy == "token_only":
        if confidence is None or any(position not in confidence for position in candidates):
            raise ValueError("infeasible token_only action: valid pre-decision confidence missing")
        ordered = sorted(candidates, key=lambda position: (float(confidence[position]), position))
        recipients, donors = ordered[:half], ordered[-half:]
        return recipients, donors, "recipient=lowest past REFLEX confidence; donor=highest", None
    if selection_strategy == "router_margin":
        if admission_margin is None or any(position not in admission_margin for position in candidates):
            raise ValueError("infeasible router_margin action: pre-decision router margins missing")
        ordered = sorted(candidates, key=lambda position: (float(admission_margin[position]), position))
        recipients, donors = ordered[:half], ordered[-half:]
        return recipients, donors, "recipient=smallest router K-boundary margin; donor=largest", None
    raise ValueError(f"unsupported follow-up selection strategy: {selection_strategy}")


def build_action(
    *,
    global_nfe: int,
    layer_idx: int,
    candidate_positions: Sequence[int],
    base_k: int,
    amount: int,
    changed_count: int,
    selection_strategy: str,
    rng: random.Random,
    sampling_key: Sequence[str],
    confidence: Mapping[int, float] | None = None,
    admission_margin: Mapping[int, float] | None = None,
) -> tuple[MultiTokenReallocation, dict]:
    """Construct one legal sparse delta without clipping or hidden resampling."""
    candidates = tuple(int(position) for position in candidate_positions)
    if len(set(candidates)) != len(candidates) or len(candidates) < 2:
        raise ValueError("candidate pool must contain at least two unique positions")
    if amount < 1 or amount % (changed_count // 2) != 0:
        raise ValueError("requested A cannot be divided evenly across balanced recipient/donor tokens")
    recipients, donors, rationale, probability = _choose_changed(
        candidates, changed_count, rng=rng, selection_strategy=selection_strategy,
        confidence=confidence, admission_margin=admission_margin,
    )
    per_token = amount // len(recipients)
    deltas = tuple((position, per_token) for position in recipients) + tuple(
        (position, -per_token) for position in donors
    )
    action = MultiTokenReallocation(
        global_nfe=int(global_nfe), layer_idx=int(layer_idx),
        candidate_positions=candidates, k_before=tuple([int(base_k)] * len(candidates)),
        deltas=deltas, selection_strategy=selection_strategy,
        sampling_key=tuple(str(part) for part in sampling_key),
    )
    if action.A != amount or action.c != changed_count:
        raise RuntimeError("constructed action does not match requested A/c")
    return action, {
        "action": action.to_record(),
        "selection_rationale": rationale,
        "conditional_action_probability": probability,
        "candidate_population_size": len(candidates),
    }


def nested_candidate_pools(eligible_positions: Sequence[int], *, max_m: int,
                           rng: random.Random) -> dict[int, tuple[int, ...]]:
    """Make matched, nested candidate contexts from one uniform permutation."""
    if max_m < 2:
        raise ValueError("max_m must be at least two")
    if len(set(eligible_positions)) < 2:
        return {}
    permutation = rng.sample(list(eligible_positions), len(eligible_positions))
    return {m: tuple(permutation[:m]) for m in (2, 5, 8, 16)
            if m <= max_m and m <= len(permutation)}
