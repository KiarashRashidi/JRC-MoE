"""Atomic artifacts and pre-branch resource-limit enforcement."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from .settings import ResourceCaps


class BudgetExhausted(RuntimeError):
    pass


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


@dataclass
class BudgetLedger:
    caps: ResourceCaps
    output_dir: Path
    started_at: float = field(default_factory=time.monotonic)
    gpu_seconds: float = 0.0
    branches_started: int = 0
    prompts_started: set[str] = field(default_factory=set)
    stopped_reason: str | None = None

    @classmethod
    def resume_from(cls, caps: ResourceCaps, output_dir: str | Path) -> "BudgetLedger":
        path = Path(output_dir) / "progress.json"
        if not path.exists():
            raise FileNotFoundError(f"resume requested but no progress file exists: {path}")
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("caps") != asdict(caps):
            raise ValueError("resume caps differ from the caps recorded with prior progress")
        wall_seconds = float(snapshot.get("wall_seconds", 0.0))
        ledger = cls(caps=caps, output_dir=Path(output_dir), started_at=time.monotonic() - wall_seconds,
                     gpu_seconds=float(snapshot.get("gpu_seconds", 0.0)),
                     branches_started=int(snapshot.get("branches_started", 0)),
                     prompts_started=set(snapshot.get("prompts_started", [])),
                     stopped_reason=None)
        return ledger

    def check_before_branch(self, prompt_id: str, *, reserve_gpu_seconds: float = 0.0,
                            reserve_wall_seconds: float = 0.0, reserve_output_bytes: int = 0) -> None:
        """Call before any model work for a new branch; failed attempts count."""
        self.caps.validate()
        wall_h = (time.monotonic() - self.started_at) / 3600.0
        next_prompts = len(self.prompts_started | {str(prompt_id)})
        output_gb = directory_size_bytes(self.output_dir) / (1024 ** 3)
        checks = (
            (self.branches_started >= self.caps.max_branches, "max_branches"),
            (next_prompts > self.caps.max_prompts, "max_prompts"),
            (self.gpu_seconds + max(0.0, reserve_gpu_seconds) >= self.caps.max_gpu_hours * 3600, "max_gpu_hours"),
            (wall_h + max(0.0, reserve_wall_seconds) / 3600.0 >= self.caps.max_wall_hours, "max_wall_hours"),
            ((output_gb * (1024 ** 3) + max(0, reserve_output_bytes)) >= self.caps.max_output_gb * (1024 ** 3), "max_output_gb"),
        )
        for reached, name in checks:
            if reached:
                self.stopped_reason = name
                atomic_write_json(self.output_dir / "progress.json", self.snapshot())
                raise BudgetExhausted(f"resource cap reached before next branch: {name}")
        self.branches_started += 1
        self.prompts_started.add(str(prompt_id))
        atomic_write_json(self.output_dir / "progress.json", self.snapshot())

    def add_gpu_time(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("GPU duration cannot be negative")
        self.gpu_seconds += float(seconds)
        atomic_write_json(self.output_dir / "progress.json", self.snapshot())

    def snapshot(self) -> dict:
        return {"caps": asdict(self.caps), "gpu_seconds": self.gpu_seconds,
                "branches_started": self.branches_started,
                "prompts_started": sorted(self.prompts_started),
                "wall_seconds": time.monotonic() - self.started_at,
                "output_bytes": directory_size_bytes(self.output_dir),
                "stopped_reason": self.stopped_reason}


def record_failure(output_dir: str | Path, *, prompt_id: str | None, branch_id: str | None,
                   error: BaseException, stage: str) -> None:
    append_jsonl(Path(output_dir) / "failures.jsonl", {
        "prompt_id": prompt_id, "branch_id": branch_id, "stage": stage,
        "error_type": type(error).__name__, "error": str(error), "time_unix": time.time(),
    })
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
