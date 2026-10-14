"""Deterministic, confidence-conditioned cumulative router-mass allocations."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence


def adaptive_mass_probabilities(confidence: float, *, low_confidence: float = 0.50,
                                high_confidence: float = 0.90,
                                low_mass_probability: float = 0.50,
                                high_mass_probability: float = 0.02) -> tuple[float, float]:
    """Return mutually exclusive probabilities for P=.1 and P=.9.

    The .1 probability decreases linearly from 0.50 to 0.02 as confidence
    increases; the .9 probability is its mirror. Remaining draws use P=.6.
    """
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and within [0, 1]")
    if not 0.0 <= low_confidence < high_confidence <= 1.0:
        raise ValueError("confidence bounds must satisfy 0 <= low < high <= 1")
    if not 0.0 <= high_mass_probability <= low_mass_probability <= 1.0:
        raise ValueError("mass probabilities must satisfy 0 <= high <= low <= 1")
    fraction = min(1.0, max(0.0, (confidence - low_confidence) /
                            (high_confidence - low_confidence)))
    p_low = low_mass_probability + fraction * (high_mass_probability - low_mass_probability)
    return float(p_low), float(low_mass_probability + high_mass_probability - p_low)


def _uniform_draw(*, seed: int, prompt_id: str, block_idx: int,
                  global_nfe: int, layer_idx: int, position: int) -> float:
    payload = json.dumps(["jrc-mass-v1", int(seed), str(prompt_id), int(block_idx),
                          int(global_nfe), int(layer_idx), int(position)],
                         separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def build_mass_plan(*, sequence_length: int, num_layers: int,
                    unresolved_positions: Sequence[int],
                    confidence_history: Mapping[int, Sequence[float]],
                    variant: str, seed: int, prompt_id: str, block_idx: int,
                    global_nfe: int, base_mass: float = 0.60,
                    low_mass: float = 0.10, high_mass: float = 0.90,
                    low_confidence: float = 0.50, high_confidence: float = 0.90,
                    low_mass_probability: float = 0.50,
                    high_mass_probability: float = 0.02) -> tuple[list[list[float]], dict]:
    """Build [layer, sequence position] P targets; unknown confidence stays P=.6."""
    if variant not in {"fixed", "confidence_adaptive"}:
        raise ValueError("variant must be fixed or confidence_adaptive")
    for name, value in (("base_mass", base_mass), ("low_mass", low_mass), ("high_mass", high_mass)):
        if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1]")
    if sequence_length < 1 or num_layers < 1:
        raise ValueError("sequence length and number of layers must be positive")
    positions = tuple(int(p) for p in unresolved_positions)
    if len(set(positions)) != len(positions) or any(p < 0 or p >= sequence_length for p in positions):
        raise ValueError("unresolved positions must be unique and within the sequence")

    plan = [[float(base_mass)] * sequence_length for _ in range(num_layers)]
    exposure = {"low": 0, "middle": 0, "high": 0}
    chosen = {"low_mass": 0, "base_mass": 0, "high_mass": 0}
    events = []
    if variant == "fixed":
        chosen["base_mass"] = sequence_length * num_layers
        return plan, {
            "variant": variant, "base_mass": float(base_mass),
            "low_mass": float(low_mass), "high_mass": float(high_mass),
            "eligible_positions": len(positions),
            "positions_with_prior_confidence": sum(bool(confidence_history.get(p)) for p in positions),
            "exposures_by_confidence_band": exposure,
            "assignments_by_mass": chosen, "nonbase_assignment_count": 0,
            "selected": [],
        }
    for position in positions:
        history = confidence_history.get(position)
        if not history:
            continue
        confidence = float(history[-1])
        p_low, p_high = adaptive_mass_probabilities(
            confidence, low_confidence=low_confidence, high_confidence=high_confidence,
            low_mass_probability=low_mass_probability,
            high_mass_probability=high_mass_probability)
        band = "low" if confidence <= low_confidence else "high" if confidence >= high_confidence else "middle"
        exposure[band] += num_layers
        for layer in range(num_layers):
            draw = _uniform_draw(seed=seed, prompt_id=prompt_id, block_idx=block_idx,
                                 global_nfe=global_nfe, layer_idx=layer, position=position)
            target = low_mass if draw < p_low else high_mass if draw < p_low + p_high else base_mass
            plan[layer][position] = float(target)
            label = "low_mass" if target == low_mass else "high_mass" if target == high_mass else "base_mass"
            chosen[label] += 1
            if target != base_mass:
                events.append({"layer_idx": layer, "position": position,
                               "confidence": confidence, "confidence_band": band,
                               "uniform_draw": draw, "p_low_mass": p_low,
                               "p_high_mass": p_high, "target_mass": target})
    return plan, {
        "variant": variant, "base_mass": float(base_mass),
        "low_mass": float(low_mass), "high_mass": float(high_mass),
        "eligible_positions": len(positions),
        "positions_with_prior_confidence": sum(bool(confidence_history.get(p)) for p in positions),
        "exposures_by_confidence_band": exposure,
        "assignments_by_mass": chosen, "nonbase_assignment_count": len(events),
        "selected": events,
    }


def experts_for_cumulative_mass(probabilities: Sequence[float], target_mass: float) -> tuple[int, float]:
    """Smallest top-ranked expert prefix whose original gate mass reaches P."""
    values = [float(x) for x in probabilities]
    target_mass = float(target_mass)
    if not values or any(not math.isfinite(x) or x < 0.0 for x in values):
        raise ValueError("router probabilities must be a nonempty finite nonnegative vector")
    if not math.isfinite(target_mass) or not 0.0 < target_mass <= 1.0:
        raise ValueError("target mass must be in (0, 1]")
    total = sum(values)
    if abs(total - 1.0) > 1e-4:
        raise ValueError("router probabilities must sum to one")
    running = 0.0
    for count, weight in enumerate(sorted(values, reverse=True), 1):
        running += weight
        if running + 1e-8 >= target_mass:
            return count, running
    return len(values), running
