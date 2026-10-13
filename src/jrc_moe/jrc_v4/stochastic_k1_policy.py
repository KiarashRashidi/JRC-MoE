"""Reproducible confidence-guided sparse K=1 assignments."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence


def confidence_probability(confidence: float, *, low_confidence: float = 0.50,
                           high_confidence: float = 0.90,
                           low_probability: float = 0.12,
                           high_probability: float = 0.02) -> float:
    """Return a linearly decreasing K=1 probability as confidence increases.

    At or below ``low_confidence`` use ``low_probability``; at or above
    ``high_confidence`` use ``high_probability``. Values between are
    interpolated. This ensures that high confidence never increases the chance
    of dropping to one expert.
    """
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and within [0, 1]")
    if not 0.0 <= low_confidence < high_confidence <= 1.0:
        raise ValueError("confidence bounds must satisfy 0 <= low < high <= 1")
    if not 0.0 <= high_probability <= low_probability <= 1.0:
        raise ValueError("probabilities must satisfy 0 <= high <= low <= 1")
    if confidence <= low_confidence:
        return float(low_probability)
    if confidence >= high_confidence:
        return float(high_probability)
    fraction = (confidence - low_confidence) / (high_confidence - low_confidence)
    return float(low_probability + fraction * (high_probability - low_probability))


def _uniform_draw(*, seed: int, prompt_id: str, block_idx: int,
                  global_nfe: int, layer_idx: int, position: int) -> float:
    payload = json.dumps(
        ["jrc-k1-sparse-v1", int(seed), str(prompt_id), int(block_idx),
         int(global_nfe), int(layer_idx), int(position)],
        separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / 2**64


def build_sparse_k1_plan(*, sequence_length: int, num_layers: int,
                         unresolved_positions: Sequence[int],
                         confidence_history: Mapping[int, Sequence[float]],
                         variant: str, seed: int, prompt_id: str,
                         block_idx: int, global_nfe: int, base_k: int = 8,
                         low_confidence: float = 0.50,
                         high_confidence: float = 0.90,
                         low_probability: float = 0.12,
                         high_probability: float = 0.02) -> tuple[list[list[int]], dict]:
    """Build a layer-by-position plan and an auditable selection summary.

    Bernoulli decisions are made independently for each eligible
    (MoE-layer, unresolved-token) pair. In the matched-random arm, each forward
    and block uses the mean confidence-policy probability as the flat rate.
    Both arms use the same counter-based uniform draw for each pair.
    """
    if variant not in {"confidence_weighted", "matched_random"}:
        raise ValueError("variant must be confidence_weighted or matched_random")
    if sequence_length < 1 or num_layers < 1 or not 1 <= base_k:
        raise ValueError("sequence length, layers, and base K must be positive")
    positions = tuple(int(p) for p in unresolved_positions)
    if len(set(positions)) != len(positions) or any(p < 0 or p >= sequence_length for p in positions):
        raise ValueError("unresolved positions must be unique and inside the sequence")

    confidence_by_position: dict[int, float] = {}
    for position in positions:
        values = confidence_history.get(position)
        if values:
            confidence_by_position[position] = float(values[-1])
    probabilities = {
        position: confidence_probability(
            confidence, low_confidence=low_confidence,
            high_confidence=high_confidence, low_probability=low_probability,
            high_probability=high_probability,
        )
        for position, confidence in confidence_by_position.items()
    }
    flat_probability = (sum(probabilities.values()) / len(probabilities)) if probabilities else 0.0

    plan = [[int(base_k)] * sequence_length for _ in range(num_layers)]
    selected = []
    exposure_by_band = {"low": 0, "middle": 0, "high": 0}
    selected_by_band = {"low": 0, "middle": 0, "high": 0}
    for position, confidence in confidence_by_position.items():
        band = "low" if confidence <= low_confidence else "high" if confidence >= high_confidence else "middle"
        exposure_by_band[band] += num_layers
        probability = probabilities[position] if variant == "confidence_weighted" else flat_probability
        for layer_idx in range(num_layers):
            draw = _uniform_draw(seed=seed, prompt_id=prompt_id, block_idx=block_idx,
                                 global_nfe=global_nfe, layer_idx=layer_idx, position=position)
            if draw < probability:
                plan[layer_idx][position] = 1
                selected_by_band[band] += 1
                selected.append({"layer_idx": layer_idx, "position": position,
                                 "confidence": confidence, "selection_probability": probability,
                                 "uniform_draw": draw, "confidence_band": band})
    metadata = {
        "variant": variant,
        "prompt_id": str(prompt_id),
        "block_idx": int(block_idx),
        "global_nfe": int(global_nfe),
        "eligible_positions": len(positions),
        "positions_with_prior_confidence": len(confidence_by_position),
        "positions_without_prior_confidence": len(positions) - len(confidence_by_position),
        "confidence_probability_mean": flat_probability,
        "exposures_by_confidence_band": exposure_by_band,
        "k1_assignments_by_confidence_band": selected_by_band,
        "k1_assignment_count": len(selected),
        "selected": selected,
    }
    return plan, metadata
