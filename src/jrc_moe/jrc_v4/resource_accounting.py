"""Helpers for conservative, attempt-inclusive campaign resource ledgers."""
from __future__ import annotations

from collections.abc import Iterable


def aggregate_stage_attempts(rows: Iterable[tuple[str, int, int]]) -> dict[str, dict]:
    """Sum every attempt per stage while preserving the latest exit code."""
    grouped: dict[str, list[dict[str, int]]] = {}
    for name, wall_seconds, exit_code in rows:
        if wall_seconds < 0:
            raise ValueError("stage wall time cannot be negative")
        grouped.setdefault(str(name), []).append({
            "wall_seconds": int(wall_seconds),
            "exit_code": int(exit_code),
        })
    return {
        name: {
            "wall_seconds": sum(attempt["wall_seconds"] for attempt in attempts),
            "exit_code": attempts[-1]["exit_code"],
            "attempts": attempts,
        }
        for name, attempts in grouped.items()
    }
