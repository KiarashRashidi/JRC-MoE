"""Bounded deterministic online groups using only edges observed by the current NFE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class TemporalEdge:
    source: int
    target: int
    weight: float
    observed_nfe: int

    def validate(self, n_tokens: int) -> None:
        if self.source == self.target or not (0 <= self.source < n_tokens and 0 <= self.target < n_tokens):
            raise ValueError("edge endpoint invalid")
        if not 0 <= self.weight <= 1 or self.observed_nfe < 1:
            raise ValueError("edge weight/NFE invalid")


def bounded_candidate_pairs(n_tokens: int, *, preferred_experts: list[set[tuple[int, int]]],
                            positional_window: int = 2, retained_edges: Iterable[TemporalEdge] = (),
                            neighbor_cap: int = 8, current_nfe: int) -> tuple[tuple[int, int], ...]:
    """Union position, shared layer-expert, and past-edge proposals with a hard degree cap."""
    if len(preferred_experts) != n_tokens or min(n_tokens, neighbor_cap) < 1 or positional_window < 0:
        raise ValueError("candidate graph dimensions/caps are invalid")
    candidates: dict[int, set[int]] = {i: set() for i in range(n_tokens)}
    for i in range(n_tokens):
        for j in range(max(0, i - positional_window), min(n_tokens, i + positional_window + 1)):
            if i != j:
                candidates[i].add(j)
    inverted: dict[tuple[int, int], list[int]] = {}
    for position, experts in enumerate(preferred_experts):
        for identity in experts:
            # The layer index is part of identity; expert 17 in two layers differs.
            inverted.setdefault((int(identity[0]), int(identity[1])), []).append(position)
    for positions in inverted.values():
        for i in positions:
            candidates[i].update(j for j in positions if j != i)
    for edge in retained_edges:
        if edge.observed_nfe > current_nfe:
            raise ValueError("future edge cannot enter the current candidate graph")
        edge.validate(n_tokens)
        candidates[edge.source].add(edge.target)
        candidates[edge.target].add(edge.source)
    # Deterministic locality-first cap; scoring/calibration is performed later.
    proposals = set()
    for i in range(n_tokens):
        ordered = sorted(candidates[i], key=lambda j: (abs(i - j), j))[:neighbor_cap]
        proposals.update((min(i, j), max(i, j)) for j in ordered)
    degree = [0] * n_tokens
    bounded = []
    for i, j in sorted(proposals, key=lambda edge: (abs(edge[0] - edge[1]), edge[0], edge[1])):
        if degree[i] < neighbor_cap and degree[j] < neighbor_cap:
            bounded.append((i, j))
            degree[i] += 1
            degree[j] += 1
    return tuple(bounded)


def popularity_corrected_routing_similarity(experts_left: set[tuple[int, int]],
                                             experts_right: set[tuple[int, int]],
                                             expert_frequency: dict[tuple[int, int], float]) -> float:
    if not experts_left or not experts_right:
        return 0.0
    if any(not 0 <= p <= 1 for p in expert_frequency.values()):
        raise ValueError("expert frequencies must be probabilities")
    # Keep layer identity in every key; expected overlap is computed from dev-only marginals.
    union = experts_left | experts_right
    jaccard = len(experts_left & experts_right) / len(union)
    expected = sum(expert_frequency.get(e, 0.0) ** 2 for e in experts_left & experts_right)
    expected = min(1.0, expected)
    if expected >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (jaccard - expected) / (1.0 - expected)))


class EmpiricalCalibrator:
    """Monotone empirical-CDF scale learned only from development similarities."""

    def __init__(self):
        self.values: tuple[float, ...] | None = None

    def fit(self, values: Iterable[float], *, source_split: str) -> "EmpiricalCalibrator":
        if source_split != "development":
            raise RuntimeError("similarity calibration may use development prompts only")
        data = sorted(float(v) for v in values)
        if not data or any(not 0 <= v <= 1 for v in data):
            raise ValueError("calibration values must be nonempty and in [0,1]")
        self.values = tuple(data)
        return self

    def transform(self, value: float) -> float:
        if self.values is None:
            raise RuntimeError("calibrator is not fit")
        if not 0 <= value <= 1:
            raise ValueError("similarity must be in [0,1]")
        import bisect
        left = bisect.bisect_left(self.values, value)
        right = bisect.bisect_right(self.values, value)
        return ((left + right) * 0.5) / len(self.values)


@dataclass
class PastOnlyCommunities:
    n_tokens: int
    neighbor_cap: int = 8
    max_group_size: int = 16
    passes: int = 2
    history_window: int = 4
    observed_nfe: int = 0
    edges: list[TemporalEdge] = field(default_factory=list)
    assignments: tuple[int, ...] | None = None
    pairwise_churn: float = 0.0

    def update(self, global_nfe: int, new_edges: Iterable[TemporalEdge]) -> tuple[int, ...]:
        if global_nfe <= self.observed_nfe:
            raise ValueError("community updates must be strictly chronological")
        new = list(new_edges)
        for edge in new:
            edge.validate(self.n_tokens)
            if edge.observed_nfe != global_nfe:
                raise ValueError("edge timestamp must equal the current update NFE")
        self.edges.extend(new)
        floor = global_nfe - self.history_window + 1
        self.edges = [e for e in self.edges if floor <= e.observed_nfe <= global_nfe]
        self.observed_nfe = global_nfe
        adjacency: dict[int, dict[int, float]] = {i: {} for i in range(self.n_tokens)}
        for edge in self.edges:
            adjacency[edge.source][edge.target] = max(adjacency[edge.source].get(edge.target, 0.0), edge.weight)
            adjacency[edge.target][edge.source] = max(adjacency[edge.target].get(edge.source, 0.0), edge.weight)
        # Keep only each token's strongest q neighbors; deterministic tie break by token index.
        bounded = {i: {j for j, _ in sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))[:self.neighbor_cap]}
                   for i, row in adjacency.items()}
        labels = list(range(self.n_tokens)) if self.assignments is None else list(self.assignments)
        for _ in range(max(1, self.passes)):
            for token in range(self.n_tokens):
                candidates = sorted({labels[j] for j in bounded[token]})
                if not candidates:
                    continue
                scores = []
                for group in candidates:
                    members = [i for i, label in enumerate(labels) if label == group]
                    if len(members) > self.max_group_size:
                        continue
                    score = sum(adjacency[token].get(j, 0.0) for j in members if j in bounded[token])
                    scores.append((score, -group, group))
                if scores:
                    best = max(scores)[2]
                    if sum(label == best for label in labels) < self.max_group_size:
                        labels[token] = best
        new_assignments = tuple(labels)
        if self.assignments is not None:
            old_pairs = {(i, j) for i in range(self.n_tokens) for j in range(i + 1, self.n_tokens)
                         if self.assignments[i] == self.assignments[j]}
            new_pairs = {(i, j) for i in range(self.n_tokens) for j in range(i + 1, self.n_tokens)
                         if new_assignments[i] == new_assignments[j]}
            denom = max(1, self.n_tokens * (self.n_tokens - 1) // 2)
            self.pairwise_churn = len(old_pairs ^ new_pairs) / denom
        self.assignments = new_assignments
        return new_assignments

    def context(self, token: int, decision_nfe: int) -> tuple[int, ...]:
        if decision_nfe != self.observed_nfe or self.assignments is None:
            raise RuntimeError("communities are only available at the latest observed NFE")
        if not 0 <= token < self.n_tokens:
            raise IndexError(token)
        label = self.assignments[token]
        return tuple(i for i, group in enumerate(self.assignments) if group == label)
