"""Read-only environment and loaded-model architecture inventory."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from ..config import MODEL_ID, MODEL_REVISION, NUM_EXPERTS, NUM_MOE_LAYERS, NATIVE_TOPK


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False, timeout=15).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_environment(repo_root: str | Path) -> dict:
    root = Path(repo_root).resolve()
    git_head = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    git_branch = _run(["git", "-C", str(root), "branch", "--show-current"])
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])
    torch_meta = {}
    try:
        import torch
        torch_meta = {"version": torch.__version__, "cuda_runtime": torch.version.cuda,
                      "cuda_available": torch.cuda.is_available(),
                      "devices": [{"name": torch.cuda.get_device_name(i),
                                   "compute_capability": list(torch.cuda.get_device_capability(i))}
                                  for i in range(torch.cuda.device_count())]}
    except ImportError:
        torch_meta = {"unavailable": True}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "repository": str(root),
        "git_commit": git_head,
        "git_branch": git_branch,
        "git_status": _run(["git", "-C", str(root), "status", "--short", "--branch"]),
        "torch": torch_meta,
        "transformers": _package_version("transformers"),
        "huggingface_hub": _package_version("huggingface_hub"),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        # Record only non-secret configuration knobs; never serialize HF_TOKEN or credentials.
        "cuda_environment": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "PYTORCH_CUDA_ALLOC_CONF",
            "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HUB_OFFLINE", "VIRTUAL_ENV")
            if os.environ.get(k) is not None},
        "nvidia_smi": _run(["nvidia-smi"]),
        "nvidia_query": _run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.free,memory.used,utilization.gpu", "--format=csv"]),
        "nvcc": _run(["nvcc", "--version"]),
        "pip_freeze": pip_freeze.splitlines() if pip_freeze else None,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_architecture_expectations": {"num_moe_layers": NUM_MOE_LAYERS,
                                             "num_experts": NUM_EXPERTS, "native_top_k": NATIVE_TOPK},
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def audit_loaded_model(model: Any) -> dict:
    cfg = model.config
    moe = []
    attention = []
    source_hashes = {}
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if "SparseMoeBlock" in cls_name or cls_name == "LLaDAMoESparseMoeBlock":
            gate = getattr(module, "gate", None)
            gate_weight = getattr(gate, "weight", None)
            bias = getattr(module, "expert_bias", None)
            try:
                source_path = inspect.getsourcefile(module.__class__)
                source = inspect.getsource(module.__class__.forward)
                forward_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            except (OSError, TypeError, AttributeError):
                source_path, source, forward_hash = None, "", None
            moe.append({"path": name, "class": cls_name,
                        "gate_class": gate.__class__.__name__ if gate else None,
                        "gate_weight_shape": list(gate_weight.shape) if gate_weight is not None else None,
                        "num_experts": getattr(module, "num_experts", None),
                        "num_experts_per_tok": getattr(module, "top_k", getattr(module, "num_experts_per_tok", None)),
                        "norm_topk_prob": getattr(module, "norm_topk_prob", None),
                        "expert_bias_present": bias is not None,
                        "expert_bias_nonzero": bool(bias.detach().count_nonzero().item()) if bias is not None else False,
                        "routed_scaling_factor": getattr(module, "routed_scaling_factor", None),
                        "shared_pathways": [attr for attr in ("shared_expert", "shared_experts", "shared_expert_gate")
                                            if getattr(module, attr, None) is not None],
                        "class_source_path": source_path, "forward_source_sha256": forward_hash,
                        "forward_semantic_markers": {marker: marker in source for marker in
                                                      ("softmax", "topk", "expert_bias", "norm_topk_prob",
                                                       "routed_scaling_factor", "index_add")}})
            if source_path and Path(source_path).is_file():
                source_hashes[source_path] = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        if any(hasattr(module, key) for key in ("q_proj", "k_proj")):
            attention.append({"path": name, "class": cls_name,
                              "q_proj": _shape(getattr(module, "q_proj", None)),
                              "k_proj": _shape(getattr(module, "k_proj", None)),
                              "q_norm": type(getattr(module, "q_norm", None)).__name__ if getattr(module, "q_norm", None) is not None else None,
                              "k_norm": type(getattr(module, "k_norm", None)).__name__ if getattr(module, "k_norm", None) is not None else None})
            try:
                source_path = inspect.getsourcefile(module.__class__)
                if source_path and Path(source_path).is_file():
                    source_hashes[source_path] = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
            except (OSError, TypeError):
                pass
    expected = {"num_hidden_layers": NUM_MOE_LAYERS, "num_experts": NUM_EXPERTS,
                "num_experts_per_tok": NATIVE_TOPK, "moe_router_score_function": "softmax"}
    config_values = {key: getattr(cfg, key, None) for key in expected}
    config_ok = all(config_values[k] == v for k, v in expected.items())
    for cls in (type(cfg), type(model)):
        try:
            path = inspect.getsourcefile(cls)
            if path and Path(path).is_file():
                source_hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except (OSError, TypeError):
            pass
    return {"model_class": type(model).__name__, "config_class": type(cfg).__name__,
            "config": config_values, "config_matches_frozen_expectations": config_ok,
            "moe_blocks": moe, "ordered_moe_paths": [m["path"] for m in moe],
            "attention_projection_modules": attention,
            "attention_head_count": getattr(cfg, "num_attention_heads", None),
            "key_value_head_count": getattr(cfg, "num_key_value_heads", None),
            "head_dim": getattr(cfg, "head_dim", None),
            "rope": {key: getattr(cfg, key, None) for key in ("rope_theta", "rope_scaling", "partial_rotary_factor", "qk_layernorm")},
            "moe_layer_frequency": list(getattr(cfg, "moe_layer_freq", []) or []),
            "attention_mask_behavior": "must be established from loaded attention source and runtime; not inferred here",
            "attention_backend": getattr(cfg, "_attn_implementation", None),
            "use_cache": getattr(cfg, "use_cache", None), "source_hashes": source_hashes,
            "moe_block_count_matches": len(moe) == NUM_MOE_LAYERS}


def _shape(module: Any) -> list[int] | None:
    weight = getattr(module, "weight", None)
    return list(weight.shape) if weight is not None else None
