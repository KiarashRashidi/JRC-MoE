"""Prompt-disjoint splits and probability-recorded stratified sampling."""
import hashlib
import json
import random
from collections import defaultdict
from typing import Hashable, Iterable, Mapping

from .controller import BudgetExchange


class IndexedRandom:
    """Order-independent random streams addressed by the full scientific key."""

    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)

    def stream(self, *, prompt_id: str, stream_id: str, step: int, block: int,
               position: int, purpose: str) -> random.Random:
        key = [self.master_seed, prompt_id, stream_id, step, block, position, purpose]
        digest = hashlib.sha256(json.dumps(key, separators=(",", ":"), default=str).encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))


def prompt_split_manifest(prompt_ids: Iterable[Hashable], *, seed: int,
                          development_fraction: float = 0.7,
                          validation_fraction: float = 0.3,
                          confirmatory_ids: Iterable[Hashable] = ()) -> dict:
    ids = sorted(set(prompt_ids), key=lambda x: str(x))
    if not ids:
        raise ValueError("prompt_ids cannot be empty")
    if development_fraction <= 0 or validation_fraction <= 0 or abs(development_fraction + validation_fraction - 1.0) > 1e-9:
        raise ValueError("development and validation fractions must be positive and sum to one")
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_dev = max(1, min(len(ids) - 1, round(len(ids) * development_fraction))) if len(ids) > 1 else 1
    dev, val = shuffled[:n_dev], shuffled[n_dev:]
    confirmatory = sorted(set(confirmatory_ids), key=lambda x: str(x))
    if set(dev) & set(val) or (set(dev) | set(val)) & set(confirmatory):
        raise RuntimeError("prompt split leakage")
    payload = {"seed": seed, "development": dev, "validation": val, "confirmatory": confirmatory}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    return {**payload, "sha256": digest, "confirmatory_locked": bool(confirmatory)}


def stratified_sample(items: Iterable[Mapping], *, strata_key: str, sample_size: int, seed: int) -> list[dict]:
    rows = [dict(row) for row in items]
    if sample_size < 1 or sample_size > len(rows):
        raise ValueError("sample_size must be between one and the population size")
    groups = defaultdict(list)
    for row in rows:
        if strata_key not in row:
            raise ValueError(f"missing stratum key {strata_key}")
        groups[row[strata_key]].append(row)
    rng = random.Random(seed)
    selected = []
    keys = sorted(groups, key=str)
    # Allocate proportionally by largest remainder; inclusion probabilities are logged.
    raw = {key: sample_size * len(groups[key]) / len(rows) for key in keys}
    quotas = {key: int(raw[key]) for key in keys}
    remaining = sample_size - sum(quotas.values())
    for key in sorted(keys, key=lambda k: (-(raw[k] - quotas[k]), str(k)))[:remaining]:
        quotas[key] += 1
    for key in keys:
        bucket = groups[key]
        rng.shuffle(bucket)
        take = quotas[key]
        probability = take / len(bucket)
        for row in bucket[:take]:
            selected.append({**row, "sampling_stratum": key, "selection_probability": probability,
                             "inverse_probability_weight": 1.0 / probability})
    rng.shuffle(selected)
    return selected


def sample_equal_budget_exchanges(*, prompt_id: str, global_nfe: int, layer_idx: int,
                                  unresolved_positions: Iterable[int], k_by_position: Iterable[int],
                                  count: int, seed: int, base_k: int = 6,
                                  min_k: int = 2, max_k: int = 8) -> list[dict]:
    """Sample unique legal recipient/donor pairs and log their draw probabilities."""
    positions, ks = list(unresolved_positions), list(k_by_position)
    if len(positions) != len(ks) or count < 1:
        raise ValueError("positions/K vectors must align and count must be positive")
    if any(k < min_k or k > max_k for k in ks):
        raise ValueError("base K outside legal interval")
    eligible_recipient = [p for p, k in zip(positions, ks) if k == base_k and k < max_k]
    eligible_donor = [p for p, k in zip(positions, ks) if k == base_k and k > min_k]
    pairs = [(r, d) for r in eligible_recipient for d in eligible_donor if r != d]
    if not pairs:
        return []
    rng = random.Random(seed)
    rng.shuffle(pairs)
    take = min(count, len(pairs))
    probability = take / len(pairs)
    return [{"prompt_id": str(prompt_id), "global_nfe": global_nfe, "layer_idx": layer_idx,
             "recipient_position": r, "donor_position": d, "base_k": base_k,
             "recipient_k_after": base_k + 1, "donor_k_after": base_k - 1,
             "selection_probability": probability, "sampling_stratum": "uniform_legal_pair",
             "exchange": BudgetExchange(global_nfe, layer_idx, r, d, 1)}
            for r, d in pairs[:take]]
