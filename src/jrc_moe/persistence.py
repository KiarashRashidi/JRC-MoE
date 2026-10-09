"""Crash-resumable local result and trace storage."""
import json
import os
from pathlib import Path
import pandas as pd


def load_completed(path: Path) -> dict[tuple[int, str], dict]:
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed[(int(row["test_idx"]), row["mode"])] = row
            except Exception as exc:
                raise RuntimeError(f"Invalid checkpoint JSONL line {line_no}: {exc}") from exc
    return completed


def append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_trace(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
