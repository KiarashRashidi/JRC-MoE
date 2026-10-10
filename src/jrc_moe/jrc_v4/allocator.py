"""Transparent prefix-feasible allocation for separable surrogate benefits."""
import heapq
from typing import Mapping, Sequence


def prefix_greedy(marginal_benefits: Mapping[int, Sequence[float]], *, k_min: int, k_max: int,
                  extra_budget: int, stop_at_nonpositive: bool = True) -> dict:
    if k_min < 1 or k_max < k_min or extra_budget < 0:
        raise ValueError("invalid allocation constraints")
    positions = sorted(marginal_benefits)
    if not positions:
        return {"allocation": {}, "used_budget": 0, "monotonicity_violations": 0}
    curves = {}
    violations = 0
    for position in positions:
        values = [float(v) for v in marginal_benefits[position]]
        if len(values) < k_max - k_min:
            raise ValueError("each position needs all marginal benefits through Kmax")
        curves[position] = values[:k_max - k_min]
        violations += sum(curves[position][i + 1] > curves[position][i] for i in range(len(curves[position]) - 1))
    allocation = {position: k_min for position in positions}
    heap = []
    for position in positions:
        if curves[position]:
            heapq.heappush(heap, (-curves[position][0], position, 0))
    used = 0
    while heap and used < extra_budget:
        neg_value, position, increment = heapq.heappop(heap)
        value = -neg_value
        if stop_at_nonpositive and value <= 0:
            break
        allocation[position] += 1
        used += 1
        nxt = increment + 1
        if nxt < len(curves[position]):
            heapq.heappush(heap, (-curves[position][nxt], position, nxt))
    return {"allocation": allocation, "used_budget": used, "monotonicity_violations": violations,
            "surrogate_assumptions": {"separable": True, "equal_pair_cost": True,
                                      "nonincreasing_marginals": violations == 0}}
