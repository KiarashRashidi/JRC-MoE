"""Typed, budget-neutral sparse reallocations across token candidate groups."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


@dataclass(frozen=True)
class MultiTokenReallocation:
    """An immutable integer K allocation change on an ordered candidate pool.

    ``deltas`` is sparse: omitted candidate positions have zero change. Candidate
    order is retained for interpretation, while the stable digest canonicalizes
    each (position, K_before, delta) tuple so a pure permutation does not alter
    the action identity.
    """

    global_nfe: int
    layer_idx: int
    candidate_positions: tuple[int, ...]
    k_before: tuple[int, ...]
    deltas: tuple[tuple[int, int], ...]
    selection_strategy: str = "uniform"
    sampling_key: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_positions", tuple(int(p) for p in self.candidate_positions))
        object.__setattr__(self, "k_before", tuple(int(k) for k in self.k_before))
        object.__setattr__(self, "deltas", tuple((int(p), int(d)) for p, d in self.deltas))
        object.__setattr__(self, "sampling_key", tuple(str(part) for part in self.sampling_key))
        if not self.selection_strategy:
            raise ValueError("selection_strategy must be a nonempty provenance label")
        self.validate()

    @property
    def m(self) -> int:
        return len(self.candidate_positions)

    @property
    def delta_by_position(self) -> dict[int, int]:
        return dict(self.deltas)

    @property
    def position_deltas(self) -> tuple[tuple[int, int], ...]:
        return self.deltas

    @property
    def c(self) -> int:
        return sum(delta != 0 for _, delta in self.deltas)

    @property
    def A(self) -> int:
        return sum(delta for _, delta in self.deltas if delta > 0)

    @property
    def k_after(self) -> tuple[int, ...]:
        changes = self.delta_by_position
        return tuple(k + changes.get(position, 0)
                     for position, k in zip(self.candidate_positions, self.k_before))

    @property
    def action_digest(self) -> str:
        before = dict(zip(self.candidate_positions, self.k_before))
        payload = {
            "schema": "jrc-v4-multitoken-action-v1",
            "global_nfe": self.global_nfe,
            "layer_idx": self.layer_idx,
            "candidate_k_delta": sorted(
                (position, before[position], self.delta_by_position.get(position, 0))
                for position in self.candidate_positions
            ),
            "selection_strategy": self.selection_strategy,
            "sampling_key": self.sampling_key,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def action_id(self) -> str:
        return self.action_digest[:16]

    @property
    def recipient_position(self) -> int:
        return next(position for position, delta in self.deltas if delta > 0)

    @property
    def donor_position(self) -> int:
        return next(position for position, delta in self.deltas if delta < 0)

    @property
    def amount(self) -> int:
        return self.A

    def validate(self, num_layers: int | None = None, sequence_length: int | None = None,
                 min_k: int = 2, max_k: int = 8,
                 current_k: Iterable[int] | None = None) -> None:
        if self.global_nfe < 1 or self.layer_idx < 0 or (num_layers is not None and self.layer_idx >= num_layers):
            raise ValueError("reallocation NFE/layer is out of range")
        if self.m < 2 or len(set(self.candidate_positions)) != self.m:
            raise ValueError("candidate positions must contain at least two unique positions")
        if len(self.k_before) != self.m:
            raise ValueError("K_before must contain one value per candidate position")
        if any(position < 0 or (sequence_length is not None and position >= sequence_length)
               for position in self.candidate_positions):
            raise ValueError("candidate position lies outside the router input sequence")
        if min_k < 1 or max_k < min_k or any(k < min_k or k > max_k for k in self.k_before):
            raise ValueError("K_before contains a value outside configured legal bounds")
        delta_positions = [position for position, _ in self.deltas]
        if not self.deltas or len(set(delta_positions)) != len(delta_positions):
            raise ValueError("sparse deltas must be nonempty and contain unique positions")
        if any(position not in set(self.candidate_positions) for position in delta_positions):
            raise ValueError("every changed position must be part of the candidate pool")
        if any(delta == 0 for _, delta in self.deltas):
            raise ValueError("zero adjustments must be omitted from sparse deltas")
        if sum(delta for _, delta in self.deltas) != 0:
            raise ValueError("sparse K adjustment must sum to zero")
        if self.c < 2 or self.A < 1:
            raise ValueError("a nontrivial zero-sum action must change at least two positions")
        after = self.k_after
        if any(k < min_k or k > max_k for k in after):
            raise ValueError("K_after is infeasible; the action is rejected without clipping")
        if sum(self.k_before) != sum(after):
            raise ValueError("candidate group expert-token budget is not neutral")
        if current_k is not None:
            observed = tuple(int(current_k[position]) for position in self.candidate_positions)
            if observed != self.k_before:
                raise ValueError("action K_before differs from the live saved-state allocation")

    def to_record(self) -> dict:
        return {
            "schema": "jrc-v4-multitoken-action-v1",
            "global_nfe": self.global_nfe,
            "layer_idx": self.layer_idx,
            "candidate_positions": list(self.candidate_positions),
            "K_before": list(self.k_before),
            "delta": [[position, delta] for position, delta in self.deltas],
            "K_after": list(self.k_after),
            "m": self.m,
            "c": self.c,
            "A": self.A,
            "selection_strategy": self.selection_strategy,
            "sampling_key": list(self.sampling_key),
            "action_id": self.action_id,
            "action_sha256": self.action_digest,
        }


def legacy_exchange_as_multi(exchange, *, k_before: tuple[int, int],
                             selection_strategy: str = "legacy_two_token") -> MultiTokenReallocation:
    """Losslessly expose the historical recipient/donor API as the new action."""
    positions = (int(exchange.recipient_position), int(exchange.donor_position))
    return MultiTokenReallocation(
        global_nfe=int(exchange.global_nfe),
        layer_idx=int(exchange.layer_idx),
        candidate_positions=positions,
        k_before=tuple(int(value) for value in k_before),
        deltas=((positions[0], int(exchange.amount)), (positions[1], -int(exchange.amount))),
        selection_strategy=selection_strategy,
        sampling_key=("legacy",),
    )
