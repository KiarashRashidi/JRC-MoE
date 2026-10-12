"""Leakage-aware event schema and CPU-side validation for natural REFLEX runs."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

SCHEMA_VERSION = "natural-reflex-events-v1"
DEFAULT_HORIZONS = (1, 2, 4)


def stable_prompt_split(prompt_id: str, *, seed: int = 20261011, validation_fraction: float = 0.30) -> str:
    """Stable prompt-level split; every nested event for a prompt stays together."""
    if not prompt_id or not 0.0 < validation_fraction < 1.0:
        raise ValueError("prompt_id and a validation fraction in (0,1) are required")
    digest = hashlib.sha256(f"{seed}:{prompt_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "validation" if value < validation_fraction else "development"


def confidence_summary(values: Sequence[float]) -> dict:
    """Summarize only values already observed before the prediction event."""
    xs = [float(x) for x in values]
    if any(not math.isfinite(x) for x in xs):
        raise ValueError("confidence history contains a non-finite value")
    n = len(xs)
    slope = None
    if n >= 2:
        xbar = (n - 1) / 2.0
        ybar = sum(xs) / n
        den = sum((i - xbar) ** 2 for i in range(n))
        slope = sum((i - xbar) * (y - ybar) for i, y in enumerate(xs)) / den if den else 0.0
    return {
        "history_count": n,
        "confidence_last": xs[-1] if n else None,
        "confidence_prev": xs[-2] if n >= 2 else None,
        "confidence_slope": slope,
        "confidence_variance": (sum((x - sum(xs) / n) ** 2 for x in xs) / n) if n else None,
        "confidence_delta": (xs[-1] - xs[-2]) if n >= 2 else None,
        "confidence_history": xs,
        "confidence_available": bool(n),
    }


def _event_key(event: Mapping) -> tuple:
    return (str(event["run_id"]), str(event["prompt_id"]), str(event["mode"]), int(event["block_idx"]))


def validate_trajectory(events: Sequence[Mapping], *, generated_length: int = 256,
                        block_length: int = 32) -> dict:
    """Enforce exact once-only commits and blockwise threshold/fallback semantics."""
    if generated_length < 1 or block_length < 1 or generated_length % block_length:
        raise ValueError("generated_length must be a positive multiple of block_length")
    if not events:
        raise ValueError("trajectory has no events")
    events = sorted(events, key=lambda e: (int(e["block_idx"]), int(e["local_nfe"])))
    identity = {(str(e.get("run_id")), str(e.get("prompt_id")), str(e.get("mode"))) for e in events}
    if len(identity) != 1:
        raise ValueError("events mix trajectories")
    committed: list[int] = []
    block_counts = defaultdict(int)
    per_block_local = defaultdict(list)
    for event in events:
        block = int(event["block_idx"])
        local = int(event["local_nfe"])
        eligible = [int(p) for p in event["eligible_positions_relative"]]
        threshold = [int(p) for p in event["threshold_positions_relative"]]
        newly = [int(p) for p in event["committed_positions_relative"]]
        if not 0 <= block < generated_length // block_length or local < 1:
            raise ValueError("block or local NFE is out of range")
        start, end = block * block_length, (block + 1) * block_length
        if any(not start <= p < end for p in eligible + threshold + newly):
            raise ValueError("inactive/future-block position included in current event")
        if not set(threshold).issubset(eligible) or not set(newly).issubset(eligible):
            raise ValueError("threshold/commit positions must be eligible before the forward")
        fallback = bool(event["fallback_used"])
        if not threshold:
            if not fallback or len(newly) != 1:
                raise ValueError("no-threshold forward must commit exactly one fallback position")
            if event.get("fallback_position_relative") != newly[0]:
                raise ValueError("fallback identity differs from the committed singleton")
        elif fallback or newly != threshold:
            raise ValueError("threshold forward must commit every and only threshold-eligible position")
        if int(event["masked_before"]) != len(eligible):
            raise ValueError("masked_before disagrees with eligible active positions")
        if int(event["masked_after"]) != len(eligible) - len(newly):
            raise ValueError("masked_after disagrees with the committed wave")
        if int(event.get("features_available_through_local_nfe", -1)) > local - 1:
            raise ValueError("event features include contemporaneous or future information")
        if local in per_block_local[block]:
            raise ValueError("duplicate local NFE")
        per_block_local[block].append(local)
        committed.extend(newly)
        block_counts[block] += len(newly)
    expected_blocks = set(range(generated_length // block_length))
    if set(block_counts) != expected_blocks:
        raise ValueError("one or more generated blocks are missing")
    for block in expected_blocks:
        local = sorted(per_block_local[block])
        if local != list(range(1, len(local) + 1)):
            raise ValueError(f"block {block} local NFE is not contiguous from one")
        if block_counts[block] != block_length:
            raise ValueError(f"block {block} committed {block_counts[block]} positions, expected {block_length}")
    counts = defaultdict(int)
    for p in committed:
        counts[p] += 1
    if set(counts) != set(range(generated_length)) or any(n != 1 for n in counts.values()):
        raise ValueError("every generated position must be committed exactly once")
    return {"status": "passed", "events": len(events), "blocks": len(expected_blocks),
            "positions_committed": len(committed), "fallback_events": sum(bool(e["fallback_used"]) for e in events),
            "multi_position_waves": sum(len(e["committed_positions_relative"]) > 1 for e in events)}


def build_at_risk_rows(events: Sequence[Mapping], positions: Sequence[Mapping], *,
                       horizons: Sequence[int] = DEFAULT_HORIZONS) -> list[dict]:
    """Create future-commit labels from past-only features; excludes inactive blocks."""
    hs = tuple(sorted(set(int(h) for h in horizons)))
    if not hs or hs[0] < 1:
        raise ValueError("horizons must be positive integers")
    first_commit = {}
    terminal = {}
    event_by_key = {}
    for event in events:
        key = _event_key(event)
        local = int(event["local_nfe"])
        event_by_key[(key, local)] = event
        terminal[key] = max(terminal.get(key, 0), local)
        for pos in event["committed_positions_relative"]:
            pos_key = (*key, int(pos))
            if pos_key in first_commit:
                raise ValueError("position committed more than once")
            first_commit[pos_key] = local
    output = []
    for pos in positions:
        key = _event_key(pos)
        local = int(pos["local_nfe"])
        position = int(pos["position_relative"])
        if not bool(pos.get("candidate_at_risk", True)):
            continue
        if pos.get("active_block") is not True:
            raise ValueError("inactive future-block positions cannot be prediction risk-set rows")
        if int(pos.get("available_through_local_nfe", -1)) > local - 1:
            raise ValueError("position feature was not available before the predicted event")
        commit_at = first_commit.get((*key, position))
        if commit_at is None or commit_at < local:
            raise ValueError("eligible position has no valid future first-commit event")
        event = event_by_key.get((key, local))
        if event is None or position not in set(int(x) for x in event["eligible_positions_relative"]):
            raise ValueError("risk-set row is not unresolved in its active block at this event")
        for horizon in hs:
            effective = min(horizon, terminal[key] - local + 1)
            if effective < 1:
                continue
            output.append({**dict(pos), "horizon": horizon, "effective_exposure": effective,
                           "label": int(commit_at <= local + effective - 1),
                           "commit_local_nfe": commit_at,
                           "label_same_forward": int(commit_at == local)})
    return output


def build_pair_labels(events: Sequence[Mapping], positions: Sequence[Mapping], *,
                      horizons: Sequence[int] = DEFAULT_HORIZONS,
                      max_pairs_per_event: int = 4096) -> list[dict]:
    """Label same/near future waves only for pairs jointly at risk in an active block."""
    if max_pairs_per_event < 1:
        raise ValueError("max_pairs_per_event must be positive")
    commit_at = {}
    terminals = defaultdict(int)
    lookup = {}
    eligible_by_event = {}
    for event in events:
        key = _event_key(event)
        local = int(event["local_nfe"])
        terminals[key] = max(terminals[key], local)
        eligible_by_event[(key, local)] = set(map(int, event["eligible_positions_relative"]))
        for p in event["committed_positions_relative"]:
            kp = (*key, int(p))
            if kp in commit_at:
                raise ValueError("duplicate first commit in pair labels")
            commit_at[kp] = local
    for row in positions:
        key = _event_key(row)
        lookup.setdefault((key, int(row["local_nfe"])), {})[int(row["position_relative"])] = row
    output = []
    for event in events:
        key = _event_key(event)
        local = int(event["local_nfe"])
        eligible = sorted(eligible_by_event[(key, local)])
        pairs = [(a, b) for i, a in enumerate(eligible) for b in eligible[i + 1:]]
        if len(pairs) > max_pairs_per_event:
            # Deterministic, label-independent stride sample with inverse inclusion weight.
            stride = math.ceil(len(pairs) / max_pairs_per_event)
            sampled = pairs[::stride][:max_pairs_per_event]
            weight = len(pairs) / len(sampled)
        else:
            sampled, weight = pairs, 1.0
        for a, b in sampled:
            ca, cb = commit_at.get((*key, a)), commit_at.get((*key, b))
            if ca is None or cb is None:
                continue
            ra = lookup.get((key, local), {}).get(a)
            rb = lookup.get((key, local), {}).get(b)
            if ra is None or rb is None:
                continue
            if max(int(ra.get("available_through_local_nfe", -1)),
                   int(rb.get("available_through_local_nfe", -1))) > local - 1:
                raise ValueError("pair feature leakage")
            for horizon in horizons:
                h = int(horizon)
                exposure = min(h, terminals[key] - local + 1)
                ya = ca <= local + exposure - 1
                yb = cb <= local + exposure - 1
                output.append({"run_id": key[0], "prompt_id": key[1], "mode": key[2],
                               "block_idx": key[3], "local_nfe": local,
                               "position_i": a, "position_j": b, "horizon": h,
                               "effective_exposure": exposure,
                               "same_wave": int(ya and yb and ca == cb),
                               "near_wave": int(ya and yb and abs(ca - cb) <= 1),
                               "label_available": int(ya and yb),
                               "sampling_weight": weight,
                               "position_distance": abs(a - b)})
    return output


def assert_past_only(feature_timestamps: Mapping[str, int], *, prediction_local_nfe: int) -> None:
    """Fail closed if any feature is recorded after the prediction decision time."""
    limit = int(prediction_local_nfe) - 1
    future = {name: int(stamp) for name, stamp in feature_timestamps.items() if int(stamp) > limit}
    if future:
        raise ValueError(f"features observed after t-1: {future}")


