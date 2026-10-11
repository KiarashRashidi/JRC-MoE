"""Deep-copyable denoising replay state and coupled RNG snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import json
import pickle
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
    selected_layer_idx: int | None = None
    recipient_position: int | None = None
    donor_position: int | None = None
    selected_positions: tuple[int, ...] = ()
    selected_action_digest: str | None = None
    selected_action_record: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.global_nfe < 1 or self.local_nfe < 1 or self.block_idx < 0:
            raise ValueError("invalid replay counters")
        if not self.checkpoint_revision or not self.config_digest:
            raise ValueError("replay requires checkpoint revision and resolved-config digest")
        if self.commit_policy not in {"threshold_with_argmax_fallback"}:
            raise ValueError("unsupported commit policy")
        selected = (self.selected_layer_idx, self.recipient_position, self.donor_position,
                    self.selected_positions, self.selected_action_digest)
        if any(value is not None and value != () for value in selected):
            if self.selected_layer_idx is None:
                raise ValueError("selected intervention identity requires a layer")
            if self.selected_positions:
                if len(self.selected_positions) < 2 or len(set(self.selected_positions)) != len(self.selected_positions):
                    raise ValueError("replay candidate positions must be unique and include at least two tokens")
                if self.recipient_position is not None or self.donor_position is not None:
                    raise ValueError("generalized replay identity uses selected_positions, not legacy recipient/donor fields")
                if not self.selected_action_digest:
                    raise ValueError("generalized replay identity requires an action digest")
                if (not self.selected_action_record
                        or self.selected_action_record.get("action_sha256") != self.selected_action_digest):
                    raise ValueError("generalized replay identity requires the complete matching sparse action record")
                positions = self.selected_positions
            else:
                if self.recipient_position is None or self.donor_position is None:
                    raise ValueError("legacy selected layer, recipient, and donor must be recorded together")
                if self.recipient_position == self.donor_position:
                    raise ValueError("replay recipient and donor must differ")
                positions = (self.recipient_position, self.donor_position)
            if not 0 <= self.selected_layer_idx < len(self.k_by_layer) or any(
                    not 0 <= position < len(self.k_by_layer[0]) for position in positions):
                raise ValueError("replay intervention identity is out of range")

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


def replay_state_digest(state: ReplayState, *, identity_fields: Mapping[str, Any] | None = None) -> str:
    """Hash the complete replay state plus explicit policy/cache identity fields."""
    digest = hashlib.sha256()
    ids = state.input_ids
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().contiguous().numpy().tobytes()
    elif hasattr(ids, "tobytes"):
        ids = ids.tobytes()
    else:
        ids = json.dumps(ids, separators=(",", ":"), default=str).encode()
    digest.update(ids)
    state_payload = {
        "block_idx": state.block_idx,
        "global_nfe": state.global_nfe,
        "local_nfe": state.local_nfe,
        "confidence_history": state.confidence_history,
        "decoder_counters": state.decoder_counters,
        "sampler": state.sampler,
        "threshold": state.threshold,
        "commit_policy": state.commit_policy,
        "k_by_layer": state.k_by_layer,
        "checkpoint_revision": state.checkpoint_revision,
        "config_digest": state.config_digest,
        "metadata": state.metadata,
        "policy_identity": identity_fields or {},
    }
    digest.update(json.dumps(state_payload, sort_keys=True, separators=(",", ":"), default=str).encode())
    if state.attention_mask is not None:
        mask = state.attention_mask
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().contiguous().numpy().tobytes()
        elif hasattr(mask, "tobytes"):
            mask = mask.tobytes()
        else:
            mask = json.dumps(mask, separators=(",", ":"), default=str).encode()
        digest.update(mask)
    digest.update(pickle.dumps(state.rng.python_state, protocol=4))
    if state.rng.numpy_state is not None:
        import numpy as np
        numpy_state = state.rng.numpy_state
        digest.update(str(numpy_state[0]).encode())
        digest.update(np.asarray(numpy_state[1]).tobytes())
        digest.update(json.dumps(list(numpy_state[2:]), separators=(",", ":"), default=str).encode())
    if state.rng.torch_cpu_state is not None:
        digest.update(state.rng.torch_cpu_state.detach().cpu().contiguous().numpy().tobytes())
    if state.rng.torch_cuda_states is not None:
        for tensor in state.rng.torch_cuda_states:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def control_cache_identity(state: ReplayState, *, baseline_k: int | tuple[tuple[int, ...], ...],
                           continuation_policy: str, branch_stream_id: str) -> str:
    """Build a control-cache key that cannot cross states, K schedules, or policies."""
    if not continuation_policy or not branch_stream_id:
        raise ValueError("control cache requires named continuation and stream identities")
    plan = tuple(tuple(int(value) for value in row) for row in state.k_by_layer)
    if isinstance(baseline_k, int):
        expected = tuple(tuple([int(baseline_k)] * len(row)) for row in plan)
    else:
        expected = tuple(tuple(int(value) for value in row) for row in baseline_k)
    if plan != expected:
        raise ValueError("control cache baseline K does not match the saved replay state's K plan")
    return replay_state_digest(
        state,
        identity_fields={
            "baseline_k": expected,
            "continuation_policy": continuation_policy,
            "branch_stream_id": branch_stream_id,
            "checkpoint_revision": state.checkpoint_revision,
            "config_digest": state.config_digest,
        },
    )


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
