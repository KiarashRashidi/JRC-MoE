"""Prompt-clustered summaries and sign/tie-aware ranking utilities."""
import math
import random
from collections import defaultdict
from typing import Iterable, Mapping


def prompt_cluster_bootstrap(rows: Iterable[Mapping], *, value_key: str, prompt_key: str = "prompt_id",
                             resamples: int = 2000, seed: int = 20261009) -> dict:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[prompt_key]].append(float(row[value_key]))
    if not grouped:
        raise ValueError("no rows to summarize")
    prompt_values = [sum(values) / len(values) for _, values in sorted(grouped.items(), key=lambda x: str(x[0]))]
    observed = sum(prompt_values) / len(prompt_values)
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [prompt_values[rng.randrange(len(prompt_values))] for _ in prompt_values]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    lo = estimates[max(0, math.floor(0.025 * resamples))]
    hi = estimates[min(resamples - 1, math.ceil(0.975 * resamples) - 1)]
    return {"mean_prompt_value": observed, "ci95": [lo, hi], "prompts": len(prompt_values),
            "rows": sum(len(v) for v in grouped.values()), "resamples": resamples, "seed": seed}


def paired_prompt_bootstrap(rows: Iterable[Mapping], *, treatment_key: str, comparator_key: str,
                            prompt_key: str = "prompt_id", resamples: int = 2000,
                            seed: int = 20261009) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[prompt_key]].append(float(row[treatment_key]) - float(row[comparator_key]))
    if not grouped or resamples < 1:
        raise ValueError("paired bootstrap needs rows and positive resamples")
    per_prompt = [sum(values) / len(values) for _, values in sorted(grouped.items(), key=lambda x: str(x[0]))]
    observed = sum(per_prompt) / len(per_prompt)
    rng = random.Random(seed)
    estimates = [sum(per_prompt[rng.randrange(len(per_prompt))] for _ in per_prompt) / len(per_prompt)
                 for _ in range(resamples)]
    estimates.sort()
    lo = estimates[max(0, math.floor(0.025 * resamples))]
    hi = estimates[min(resamples - 1, math.ceil(0.975 * resamples) - 1)]
    return {"mean_prompt_paired_difference": observed, "ci95": [lo, hi],
            "prompts": len(per_prompt), "rows": sum(len(v) for v in grouped.values()),
            "resamples": resamples, "seed": seed, "independence_unit": "prompt"}


def exchange_effect(control_utility: float, treatment_utility: float) -> int | float:
    """Keep exact zero outcomes; correctness utilities yield -1, 0, or +1."""
    delta = float(treatment_utility) - float(control_utility)
    if delta in (-1.0, 0.0, 1.0):
        return int(delta)
    return delta


def four_arm_interaction(y00: float, y10: float, y01: float, y11: float) -> float:
    return float(y11 - y10 - y01 + y00)


def kendall_tau_b(left: Iterable[float], right: Iterable[float]) -> float | None:
    x, y = list(left), list(right)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("rank inputs must have equal length >= 2")
    concordant = discordant = tie_x = tie_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = (x[i] > x[j]) - (x[i] < x[j]), (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    return None if denom == 0 else (concordant - discordant) / denom
