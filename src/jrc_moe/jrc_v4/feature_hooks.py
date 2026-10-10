"""Hooks that refuse to label pre-RoPE or post-decision tensors as causal attention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .features import post_rope_landmark_scores


@dataclass(frozen=True)
class AttentionProbeContract:
    source: str
    layer_idx: int
    head_idx: int
    checkpoint_revision: str
    reference_attention_fidelity: float | None = None

    def validate(self) -> None:
        if self.source != "post_qk_norm_post_rope":
            raise RuntimeError("raw q_proj/k_proj hooks are not accepted as attention features")
        if self.layer_idx < 0 or self.head_idx < 0 or not self.checkpoint_revision:
            raise ValueError("attention probe identity is incomplete")
        if self.reference_attention_fidelity is None:
            raise RuntimeError("short-sequence reference-attention fidelity has not been measured")


def restricted_attention_scores(query: Sequence[float], keys: Sequence[Sequence[float]],
                                positions: Sequence[int], contract: AttentionProbeContract) -> tuple[float, ...]:
    contract.validate()
    return post_rope_landmark_scores(query, keys, source=contract.source, landmark_positions=positions)
