"""Synchronized GPU and host timing helpers; never report unsynchronized CUDA time."""
from __future__ import annotations

import math
import statistics
import time
from typing import Callable


def profile_callable(fn: Callable, *, warmup: int = 2, repetitions: int = 10,
                     tokens_per_call: int | None = None) -> dict:
    if warmup < 0 or repetitions < 1:
        raise ValueError("invalid timing repetition counts")
    import torch
    cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    wall_ms, cuda_ms = [], []
    for _ in range(repetitions):
        if cuda:
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - start) * 1000.0)
            cuda_ms.append(start_event.elapsed_time(end_event))
        else:
            start = time.perf_counter()
            fn()
            wall_ms.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(wall_ms)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    wall_median = statistics.median(wall_ms)
    result = {"repetitions": repetitions, "warmup": warmup, "synchronized": cuda,
            "wall_median_ms": wall_median, "wall_p95_ms": p95,
            "cuda_event_median_ms": statistics.median(cuda_ms) if cuda_ms else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None}
    if tokens_per_call is not None:
        if tokens_per_call < 1:
            raise ValueError("tokens_per_call must be positive")
        result["tokens_per_call"] = int(tokens_per_call)
        result["tokens_per_second_at_wall_median"] = float(tokens_per_call / (wall_median / 1000.0))
    return result
