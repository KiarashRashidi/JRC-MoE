"""Frozen defaults and resource gates for exploratory JRC-MoE v4 runs."""
from dataclasses import dataclass
from pathlib import Path

from ..config import MODEL_ID, MODEL_REVISION, NUM_EXPERTS, NUM_MOE_LAYERS


@dataclass(frozen=True)
class V4Settings:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    num_layers: int = NUM_MOE_LAYERS
    num_experts: int = NUM_EXPERTS
    native_top_k: int = 8
    min_k: int = 2
    max_k: int = 8
    legal_k: tuple[int, ...] = (2, 4, 6, 8)
    monitored_layers: tuple[int, ...] = (4, 12)
    heads: tuple[int, ...] = (0, 4, 8, 12)
    landmark_count: int = 16
    neighbor_cap: int = 8
    history_window: int = 4
    graph_refresh_every: int = 2
    seed: int = 20261009
    pilot_prompt_ids: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.min_k < 1 or self.max_k > self.num_experts or self.min_k > self.max_k:
            raise ValueError("invalid K bounds")
        if any(k < self.min_k or k > self.max_k for k in self.legal_k):
            raise ValueError("legal_k outside K bounds")
        if any(layer < 0 or layer >= self.num_layers for layer in self.monitored_layers):
            raise ValueError("monitored layer outside model")
        if self.landmark_count < 1 or self.neighbor_cap < 1 or self.history_window < 1:
            raise ValueError("feature and graph caps must be positive")


@dataclass(frozen=True)
class ResourceCaps:
    max_gpu_hours: float = 24.0
    max_branches: int = 200
    max_prompts: int = 50
    max_wall_hours: float = 30.0
    max_output_gb: float = 10.0

    def validate(self) -> None:
        if min(self.max_gpu_hours, self.max_wall_hours, self.max_output_gb) <= 0:
            raise ValueError("time and output caps must be positive")
        if min(self.max_branches, self.max_prompts) < 1:
            raise ValueError("branch and prompt caps must be positive")


DEFAULT_CONFIG_PATH = Path("configs/jrc_v4/pilot.yaml")
