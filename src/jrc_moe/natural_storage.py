"""Atomic, resumable, compressed storage for natural REFLEX observations."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Mapping


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_safe(v) for v in value]
    if hasattr(value, "detach"):
        return json_safe(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(json_safe(value), indent=2, sort_keys=True,
                                        ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path: str | Path) -> int:
    root = Path(path)
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def trajectory_id(run_id: str, prompt_id: str, mode: str, seed: int) -> str:
    identity = f"{run_id}\0{prompt_id}\0{mode}\0{int(seed)}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:24]


class TrajectoryTransaction:
    def __init__(self, store: "NaturalRunStore", *, run_id: str, prompt_id: str,
                 mode: str, seed: int, metadata: Mapping):
        self.store = store
        self.run_id, self.prompt_id, self.mode, self.seed = run_id, prompt_id, mode, int(seed)
        self.identity = trajectory_id(run_id, prompt_id, mode, seed)
        self.final_dir = store.raw_dir / self.identity
        self.partial_dir = store.raw_dir / f".{self.identity}.{os.getpid()}.partial"
        if self.final_dir.exists():
            raise FileExistsError(f"trajectory directory exists without resumable index: {self.final_dir}")
        if self.partial_dir.exists():
            shutil.rmtree(self.partial_dir)
        self.partial_dir.mkdir(parents=True, exist_ok=False)
        self._streams = {}
        self._counts = {"events": 0, "positions": 0, "relations": 0}
        for name in self._counts:
            path = self.partial_dir / f"{name}.jsonl.gz"
            self._streams[name] = gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=6)
        self.metadata = dict(metadata)
        self._closed = False

    def _write(self, name: str, record: Mapping) -> None:
        if self._closed:
            raise RuntimeError("trajectory transaction is closed")
        self._streams[name].write(json.dumps(json_safe(record), sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False, allow_nan=False) + "\n")
        self._counts[name] += 1

    def preflight_forward(self, records, *, max_bytes: int) -> int:
        """Bound one serialized forward before any of its records reach disk."""
        total = sum(len(canonical_json_bytes(record)) for record in records)
        if total > int(max_bytes):
            raise RuntimeError(f"one observation forward needs {total} uncompressed bytes; "
                               f"limit is {int(max_bytes)}")
        return total

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError("trajectory transaction is closed")
        for stream in self._streams.values():
            stream.flush()

    @property
    def bytes_on_disk(self) -> int:
        return sum(path.stat().st_size for path in self.partial_dir.iterdir() if path.is_file())

    def write_event(self, record: Mapping) -> None:
        self._write("events", record)

    def write_position(self, record: Mapping) -> None:
        self._write("positions", record)

    def write_relation(self, record: Mapping) -> None:
        self._write("relations", record)

    def abort(self, *, reason: str | None = None) -> None:
        if not self._closed:
            for stream in self._streams.values():
                stream.close()
            self._closed = True
        if reason:
            atomic_write_json(self.partial_dir / "failure.json", {"reason": reason, "time": time.time()})

    def commit(self, *, prompt_output: Mapping, validation: Mapping) -> dict:
        if self._closed:
            raise RuntimeError("trajectory transaction already closed")
        for stream in self._streams.values():
            stream.flush()
            stream.close()
        self._closed = True
        output_path = self.partial_dir / "prompt_output.json"
        atomic_write_json(output_path, prompt_output)
        hashes = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in sorted(self.partial_dir.iterdir()) if p.is_file()}
        meta = {"schema_version": "natural-reflex-trajectory-v1", "trajectory_id": self.identity,
                "run_id": self.run_id, "prompt_id": self.prompt_id, "mode": self.mode,
                "seed": self.seed, "created_unix": time.time(), "record_counts": self._counts,
                "validation": json_safe(validation), "metadata": json_safe(self.metadata),
                "files": hashes}
        atomic_write_json(self.partial_dir / "trajectory.json", meta)
        os.replace(self.partial_dir, self.final_dir)
        index = {**meta, "raw_path": str(self.final_dir),
                 "output_path": str(self.final_dir / "prompt_output.json")}
        atomic_write_json(self.store.index_dir / f"{self.identity}.json", index)
        self.store.refresh_indexes()
        return index

    def __enter__(self) -> "TrajectoryTransaction":
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None and not self._closed:
            self.abort(reason=f"{type(exc).__name__}: {exc}")
        return False


class NaturalRunStore:
    """Completed prompt/mode units are immutable and resumable after SHA checks."""
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.raw_dir = self.root / "trajectories" / "raw"
        self.index_dir = self.root / "trajectories" / "index"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def completed(self) -> dict[tuple[str, str, str, int], dict]:
        result = {}
        for path in sorted(self.index_dir.glob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                raw = Path(row["raw_path"])
                valid = raw.is_dir()
                for filename, detail in row.get("files", {}).items():
                    item = raw / filename
                    valid = valid and item.is_file() and item.stat().st_size == detail["bytes"] and sha256_file(item) == detail["sha256"]
                if valid:
                    result[(str(row["run_id"]), str(row["prompt_id"]),
                            str(row["mode"]), int(row["seed"]))] = row
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue
        return result

    def start(self, *, run_id: str, prompt_id: str, mode: str, seed: int,
              metadata: Mapping) -> TrajectoryTransaction:
        return TrajectoryTransaction(self, run_id=run_id, prompt_id=prompt_id,
                                     mode=mode, seed=seed, metadata=metadata)

    def refresh_indexes(self) -> None:
        rows = list(self.completed().values())
        rows.sort(key=lambda r: (r["prompt_id"], r["mode"], int(r["seed"])))
        csv_path = self.root / "trajectories" / "trajectory_index.csv"
        fields = ("trajectory_id", "run_id", "prompt_id", "mode", "seed", "raw_path",
                  "output_path", "event_count", "position_count", "relation_count", "validation_status")
        import io
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            counts = row.get("record_counts", {})
            writer.writerow({"trajectory_id": row["trajectory_id"], "run_id": row["run_id"],
                             "prompt_id": row["prompt_id"], "mode": row["mode"], "seed": row["seed"],
                             "raw_path": row["raw_path"], "output_path": row["output_path"],
                             "event_count": counts.get("events", 0), "position_count": counts.get("positions", 0),
                             "relation_count": counts.get("relations", 0),
                             "validation_status": row.get("validation", {}).get("status", "unknown")})
        atomic_write_bytes(csv_path, buffer.getvalue().encode("utf-8"))
        out_path = self.root / "trajectories" / "prompt_outputs.jsonl"
        lines = []
        for row in rows:
            p = Path(row["output_path"])
            if p.is_file():
                lines.append(p.read_bytes())
        atomic_write_bytes(out_path, b"".join(lines))


class CampaignLedger:
    """Persistent conservative wall/disk ledger shared across natural-study stages."""
    def __init__(self, root: str | Path, *, max_gpu_seconds: float = 4 * 3600,
                 max_output_bytes: int = 10 * 1024**3):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "campaign_ledger.json"
        self.max_gpu_seconds = float(max_gpu_seconds)
        self.max_output_bytes = int(max_output_bytes)
        self.lock_path = self.root / ".campaign_ledger.lock"

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        stream = self.lock_path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if os.fstat(stream.fileno()).st_size == 0:
                    stream.seek(0)
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()

    def read(self) -> dict:
        if not self.path.exists():
            return {"schema_version": "natural-reflex-campaign-v1", "stages": [],
                    "gpu_wall_seconds": 0.0, "output_bytes": directory_bytes(self.root)}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def check(self, *, reserve_seconds: float = 0.0) -> dict:
        with self._lock():
            return self._check_unlocked(reserve_seconds=reserve_seconds)

    def _check_unlocked(self, *, reserve_seconds: float = 0.0) -> dict:
        ledger = self.read()
        spent = float(ledger.get("gpu_wall_seconds", 0.0))
        used = directory_bytes(self.root)
        if spent + reserve_seconds > self.max_gpu_seconds:
            raise RuntimeError("campaign GPU wall-time cap would be exceeded")
        if used >= self.max_output_bytes:
            raise RuntimeError("campaign output-disk cap is exhausted")
        if ledger.get("active_stage"):
            raise RuntimeError(f"campaign stage already active: {ledger['active_stage'].get('name')}")
        return {"gpu_wall_seconds": spent, "output_bytes": used,
                "gpu_seconds_remaining": self.max_gpu_seconds - spent,
                "output_bytes_remaining": self.max_output_bytes - used}

    def start(self, name: str, *, gpu_stage: bool = True, reserve_seconds: float = 0.0) -> dict:
        with self._lock():
            self._check_unlocked(reserve_seconds=reserve_seconds)
            ledger = self.read()
            now = time.time()
            remaining = max(0.0, self.max_gpu_seconds - float(ledger.get("gpu_wall_seconds", 0.0)))
            stage_budget = float(reserve_seconds) if reserve_seconds > 0 else remaining
            active = {"name": name, "gpu_stage": bool(gpu_stage), "started_unix": now,
                      "deadline_unix": now + min(stage_budget, remaining) if gpu_stage else None,
                      "output_bytes_start": directory_bytes(self.root)}
            ledger["active_stage"] = active
            atomic_write_json(self.path, ledger)
            return active

    def check_active(self) -> dict:
        """Cheap per-forward wall deadline check; disk is checked at unit boundaries."""
        with self._lock():
            ledger = self.read()
            active = ledger.get("active_stage")
            if not active:
                raise RuntimeError("no active campaign stage")
            now = time.time()
            deadline = active.get("deadline_unix")
            if active.get("gpu_stage") and deadline is not None and now > float(deadline):
                raise RuntimeError(f"campaign deadline reached during stage {active.get('name')}")
            return {"stage": active["name"], "seconds_to_deadline": None if deadline is None else max(0.0, float(deadline) - now)}

    def check_output_cap_active(self) -> int:
        with self._lock():
            ledger = self.read()
            if not ledger.get("active_stage"):
                raise RuntimeError("no active campaign stage")
            used = directory_bytes(self.root)
            if used > self.max_output_bytes:
                raise RuntimeError("campaign output-disk cap was exceeded")
            return used

    def check_output_headroom(self, *, reserve_bytes: int,
                              baseline_bytes: int | None = None,
                              incremental_bytes: int = 0) -> int:
        """Require disk headroom before a bounded forward; optional baseline avoids full scans."""
        with self._lock():
            ledger = self.read()
            if not ledger.get("active_stage"):
                raise RuntimeError("no active campaign stage")
            used = directory_bytes(self.root) if baseline_bytes is None else int(baseline_bytes)
            projected = used + int(incremental_bytes) + int(reserve_bytes)
            if projected > self.max_output_bytes:
                raise RuntimeError(f"campaign output cap headroom exhausted: "
                                   f"{projected} projected bytes > {self.max_output_bytes}")
            return projected

    def finish(self, *, exit_code: int, details: Mapping | None = None) -> dict:
        with self._lock():
            ledger = self.read()
            active = ledger.get("active_stage")
            if not active:
                raise RuntimeError("no active campaign stage to finish")
            elapsed = max(0.0, time.time() - float(active["started_unix"]))
            output = directory_bytes(self.root)
            stage = {**active, "ended_unix": time.time(), "elapsed_wall_seconds": elapsed,
                     "gpu_wall_seconds": elapsed if active["gpu_stage"] else 0.0,
                     "output_bytes_end": output, "output_bytes_delta": output - int(active["output_bytes_start"]),
                     "exit_code": int(exit_code), "details": json_safe(details or {})}
            ledger.setdefault("stages", []).append(stage)
            ledger["gpu_wall_seconds"] = float(ledger.get("gpu_wall_seconds", 0.0)) + stage["gpu_wall_seconds"]
            ledger["output_bytes"] = output
            ledger["active_stage"] = None
            if ledger["gpu_wall_seconds"] > self.max_gpu_seconds:
                ledger["cap_violation"] = "gpu_wall_time"
            if output > self.max_output_bytes:
                ledger["cap_violation"] = "output_disk"
            atomic_write_json(self.path, ledger)
            if ledger.get("cap_violation"):
                raise RuntimeError(f"campaign cap exceeded: {ledger['cap_violation']}")
            return ledger
