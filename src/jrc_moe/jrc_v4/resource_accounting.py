"""Helpers for conservative, attempt-inclusive campaign resource ledgers."""
from __future__ import annotations

from collections.abc import Iterable
import csv
import json
from pathlib import Path


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


def write_campaign_budget(
    run_root: str | Path,
    *,
    sensitivity_exit_code: int,
    attention_exit_code: int,
    signal_exit_code: int,
) -> dict:
    """Build a retry-inclusive, output-inclusive ledger and enforce campaign caps."""
    root = Path(run_root)
    attempts_path = root / "stage_wall_seconds.tsv"
    with attempts_path.open("r", encoding="utf-8", newline="") as stream:
        stage_rows = [
            (name, int(seconds), int(status))
            for name, seconds, status in csv.reader(stream, delimiter="\t")
        ]
    stages = aggregate_stage_attempts(stage_rows)
    branches = 0
    sensitivity_branches = 0
    for name in ("smoke", "sensitivity"):
        progress_path = root / name / "progress.json"
        if progress_path.is_file():
            stage_branches = int(json.loads(progress_path.read_text(encoding="utf-8")).get("branches_started", 0))
            branches += stage_branches
            if name == "sensitivity":
                sensitivity_branches = stage_branches
    total_wall = sum(stage["wall_seconds"] for stage in stages.values())
    ledger = {
        "protocol": "jrc-v4-followup-d1-1",
        "stages": stages,
        "total_wall_seconds": total_wall,
        "allocated_gpu_wall_hours_upper_bound": total_wall / 3600,
        "intervention_branches_started": branches,
        "output_bytes": 0,
        "caps": {
            "total_wall_hours": 12,
            "total_branches": 400,
            "output_gb": 30,
            "audit_and_smoke_seconds": 3600,
            "sensitivity_seconds": 10800,
            "sensitivity_branches": 250,
            "attention_seconds": 7200,
        },
        "sensitivity_exit_code": int(sensitivity_exit_code),
        "attention_exit_code": int(attention_exit_code),
        "signal_exit_code": int(signal_exit_code),
    }
    budget_path = root / "campaign_budget.json"
    for _ in range(10):
        budget_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        measured = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if measured == ledger["output_bytes"]:
            break
        ledger["output_bytes"] = measured
    else:
        raise RuntimeError("campaign output-byte ledger did not converge")

    if total_wall > 12 * 3600 or branches > 400 or measured > 30 * 1024**3:
        raise RuntimeError("cumulative campaign cap exceeded")
    if stages.get("sensitivity", {}).get("wall_seconds", 0) > 10800:
        raise RuntimeError("sensitivity three-hour cap exceeded")
    if sensitivity_branches > 250:
        raise RuntimeError("sensitivity branch cap exceeded")
    if stages.get("attention", {}).get("wall_seconds", 0) > 7200:
        raise RuntimeError("attention two-hour cap exceeded")
    if sum(stages.get(name, {}).get("wall_seconds", 0) for name in ("audit", "smoke")) > 3600:
        raise RuntimeError("audit/smoke one-hour cap exceeded")
    return ledger
