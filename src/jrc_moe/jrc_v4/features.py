"""Leakage-aware router summaries and auditable attention-proxy primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


def router_statistics(logits: Sequence[float], k: int, *, selected_experts: Sequence[int] | None = None,
                      expert_bias_enabled: bool = False,
                      selection_scores: Sequence[float] | None = None) -> dict[str, float | int]:
    """Compute full-distribution router controls before admission.

    ``logits`` define the softmax gate mass. When the checkpoint adds an expert
    bias before top-K admission, ``selection_scores`` must contain those exact
    adjusted scores so the admission margin follows executed routing semantics.
    """
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("router logits must be finite and include at least two experts")
    if expert_bias_enabled and selection_scores is None:
        raise RuntimeError("expert-bias routing requires the exact post-bias admission scores")
    if not 1 <= int(k) < values.size:
        raise ValueError("K must leave a K+1 admission boundary")
    shifted = values - values.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    admission = (np.asarray(selection_scores, dtype=np.float64).reshape(-1)
                 if selection_scores is not None else probs)
    if admission.shape != values.shape or not np.isfinite(admission).all():
        raise ValueError("selection scores must be finite and match the router logits")
    order = np.argsort(-admission, kind="stable")
    if selected_experts is None:
        chosen = order[:k]
        next_idx = int(order[k])
    else:
        chosen = np.asarray(selected_experts, dtype=np.int64)
        if chosen.size != k or np.any(chosen < 0) or np.any(chosen >= values.size):
            raise ValueError("selected expert indices do not match K")
        if len(set(int(x) for x in chosen)) != k:
            raise ValueError("selected experts must be unique")
        remaining = [int(i) for i in order if i not in set(int(x) for x in chosen)]
        if not remaining:
            raise ValueError("no next expert remains")
        next_idx = remaining[0]
    sorted_probs = probs[order]
    sorted_admission = admission[order]
    boundary = float(sorted_admission[k - 1] - sorted_admission[k])
    if boundary < -1e-12:
        raise RuntimeError("sorted router probability margin became negative")
    entropy = float(-(probs * np.log(np.maximum(probs, np.finfo(np.float64).tiny))).sum())
    return {
        "router_entropy": entropy,
        "top_k_gate_mass": float(probs[chosen].sum()),
        "admission_margin": max(0.0, boundary),
        "next_expert_gate": float(probs[next_idx]),
        "selected_k": int(k),
    }


def positional_landmarks(sequence_length: int, count: int) -> tuple[int, ...]:
    if sequence_length < 1 or count < 1:
        raise ValueError("sequence_length and count must be positive")
    n = min(sequence_length, count)
    # Integer midpoint per deterministic segment; unique for n <= sequence_length.
    return tuple(((2 * i + 1) * sequence_length) // (2 * n) for i in range(n))


def post_rope_landmark_scores(query: Sequence[float], keys: Sequence[Sequence[float]], *,
                              source: str, landmark_positions: Sequence[int]) -> tuple[float, ...]:
    """Restricted QK scores; refuse raw/pre-RoPE projections instead of relabeling them."""
    if source != "post_qk_norm_post_rope":
        raise RuntimeError("attention proxy requires verified post-normalization, post-RoPE Q/K")
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    k = np.asarray(keys, dtype=np.float64)
    if k.ndim != 2 or k.shape[1] != q.size or not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError("Q/K dimensions or values are invalid")
    if any(p < 0 or p >= k.shape[0] for p in landmark_positions):
        raise ValueError("landmark index outside key sequence")
    dh = q.size
    return tuple(float(np.dot(q, k[p]) / math.sqrt(dh)) for p in landmark_positions)


@dataclass(frozen=True)
class FeatureRecord:
    prompt_id: str
    global_nfe: int
    layer_idx: int
    position: int
    block_idx: int
    mask_ratio: float
    is_prompt_token: bool
    is_target_token: bool
    active_block: bool
    relative_position: float
    unresolved: bool
    provisional: bool
    confidence_history: tuple[float, ...]
    router: Mapping[str, float | int]
    fps_components: Mapping[str, float] = field(default_factory=dict)
    router_history: tuple[tuple[int, ...], ...] = ()
    hidden_state_norm: float | None = None
    attention_scores: tuple[float, ...] | None = None
    feature_age: int = 0
    source_nfe: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt_id or self.global_nfe < 1 or self.layer_idx < 0 or self.position < 0:
            raise ValueError("feature identity/counters invalid")
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("mask ratio must be in [0,1]")
        if not math.isfinite(self.relative_position) or self.feature_age < 0:
            raise ValueError("feature position or age invalid")
        if self.source_nfe is not None and self.source_nfe > self.global_nfe:
            raise ValueError("feature source is from the future")
        if any(not math.isfinite(v) for v in self.confidence_history):
            raise ValueError("confidence history must be finite")
        if self.hidden_state_norm is not None and not math.isfinite(self.hidden_state_norm):
            raise ValueError("hidden state norm must be finite")
        if any(not math.isfinite(float(v)) for v in self.fps_components.values()):
            raise ValueError("FPS components must be finite")
        if self.attention_scores is not None and any(not math.isfinite(v) for v in self.attention_scores):
            raise ValueError("attention scores must be finite")
        forbidden = {"label", "outcome", "future_confidence", "post_intervention_logits", "later_hidden_state"}
        if forbidden.intersection(self.extra):
            raise ValueError("post-decision labels or future values cannot be feature inputs")


def affinity_channels(attention_similarity: float, routing_similarity: float) -> tuple[float, float, float, float]:
    a, r = float(attention_similarity), float(routing_similarity)
    if not (math.isfinite(a) and math.isfinite(r) and 0 <= a <= 1 and 0 <= r <= 1):
        raise ValueError("calibrated similarities must be finite and in [0,1]")
    return (a * r, a * (1 - r), (1 - a) * r, (1 - a) * (1 - r))
