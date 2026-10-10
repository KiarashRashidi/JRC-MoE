"""Deep-copyable denoising replay state and coupled RNG snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import random
from typing import Any, Mapping


@dataclass
class RNGSnapshot:
    python_state: object
    numpy_state: object | None = None
    torch_cpu_state: object | None = None
    torch_cuda_states: object | None = None

    @classmethod
    def capture(cls) -> "RNGSnapshot":
        np_state = None
        torch_cpu = None
        torch_cuda = None
        try:
            import numpy as np
            np_state = copy.deepcopy(np.random.get_state())
        except ImportError:
            pass
        try:
            import torch
            torch_cpu = torch.random.get_rng_state().clone()
            if torch.cuda.is_available():
                torch_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
        except ImportError:
            pass
        return cls(random.getstate(), np_state, torch_cpu, torch_cuda)

    def restore(self) -> None:
        random.setstate(copy.deepcopy(self.python_state))
        if self.numpy_state is not None:
            import numpy as np
            np.random.set_state(copy.deepcopy(self.numpy_state))
        if self.torch_cpu_state is not None:
            import torch
            torch.random.set_rng_state(self.torch_cpu_state.clone())
            if self.torch_cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([state.clone() for state in self.torch_cuda_states])


@dataclass
class ReplayState:
    """State captured immediately before one denoising model forward."""

    input_ids: Any
    attention_mask: Any | None
    block_idx: int
    global_nfe: int
    local_nfe: int
    confidence_history: Mapping[int, list[float]]
    decoder_counters: Mapping[str, int]
    sampler: Mapping[str, Any]
    threshold: float
    commit_policy: str
    k_by_layer: tuple[tuple[int, ...], ...]
    checkpoint_revision: str
    config_digest: str
    rng: RNGSnapshot = field(default_factory=RNGSnapshot.capture)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.global_nfe < 1 or self.local_nfe < 1 or self.block_idx < 0:
            raise ValueError("invalid replay counters")
        if not self.checkpoint_revision or not self.config_digest:
            raise ValueError("replay requires checkpoint revision and resolved-config digest")
        if self.commit_policy not in {"threshold_with_argmax_fallback"}:
            raise ValueError("unsupported commit policy")

    def clone(self) -> "ReplayState":
        """Copy tensors and mutable histories so branch order cannot contaminate state."""
        result = copy.deepcopy(self)
        if hasattr(self.input_ids, "clone"):
            result.input_ids = self.input_ids.clone()
        if hasattr(self.attention_mask, "clone"):
            result.attention_mask = self.attention_mask.clone()
        return result

    def restore_rng(self) -> None:
        self.rng.restore()


def capture_replay_state(**kwargs) -> ReplayState:
    kwargs.setdefault("rng", RNGSnapshot.capture())
    kwargs["input_ids"] = kwargs["input_ids"].clone() if hasattr(kwargs["input_ids"], "clone") else copy.deepcopy(kwargs["input_ids"])
    mask = kwargs.get("attention_mask")
    if hasattr(mask, "clone"):
        kwargs["attention_mask"] = mask.clone()
    kwargs["confidence_history"] = copy.deepcopy(kwargs["confidence_history"])
    kwargs["decoder_counters"] = copy.deepcopy(kwargs["decoder_counters"])
    kwargs["sampler"] = copy.deepcopy(kwargs["sampler"])
    kwargs["metadata"] = copy.deepcopy(kwargs.get("metadata", {}))
    return ReplayState(**kwargs)
