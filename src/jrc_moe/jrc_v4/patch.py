"""Fail-closed layerwise MoE interception for the audited LLaDA checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
import types
from typing import Any

import torch
import torch.nn.functional as F

from .controller import LayerwiseKController


@dataclass(frozen=True)
class VerifiedMoEContract:
    checkpoint_revision: str
    num_layers: int
    num_experts: int
    native_top_k: int
    score_function: str
    norm_topk_prob: bool
    expert_bias_enabled: bool
    routed_scaling_factor: float
    router_semantics_verified: bool
    native_fixed8_equivalence_verified: bool

    def require_supported(self, *, allow_equivalence_probe: bool = False) -> None:
        if not self.router_semantics_verified:
            raise RuntimeError("G0 router semantics gate must pass before installing the patch")
        if not self.native_fixed8_equivalence_verified and not allow_equivalence_probe:
            raise RuntimeError("native Fixed8 fidelity must pass before any intervention run")
        if self.score_function != "softmax" or self.norm_topk_prob or self.expert_bias_enabled:
            raise RuntimeError("unsupported checkpoint routing semantics; refusing to patch")
        if self.routed_scaling_factor != 1.0:
            raise RuntimeError("non-unit routed scaling is not yet verified by this dispatcher")


def install_layerwise_patch(model: Any, controller: LayerwiseKController,
                            contract: VerifiedMoEContract, *, allow_equivalence_probe: bool = False) -> tuple[str, ...]:
    """Patch only after the caller supplies a runtime-verified checkpoint contract."""
    contract.require_supported(allow_equivalence_probe=allow_equivalence_probe)
    if len(controller.module_names) != contract.num_layers:
        raise ValueError("controller layer count differs from verified model contract")
    for _, module in model.named_modules():
        if hasattr(module, "_jrc_v4_original_forward"):
            module.forward = module._jrc_v4_original_forward
    blocks = [(name, module) for name, module in model.named_modules()
              if module.__class__.__name__ == "LLaDAMoESparseMoeBlock"]
    names = tuple(name for name, _ in blocks)
    if len(blocks) != contract.num_layers or names != controller.module_names:
        raise RuntimeError("runtime MoE module identity/order differs from audited controller mapping")
    for name, module in blocks:
        if getattr(module, "num_experts", contract.num_experts) != contract.num_experts:
            raise RuntimeError(f"unexpected expert count in {name}")
        if bool(getattr(module, "norm_topk_prob", False)) != contract.norm_topk_prob:
            raise RuntimeError(f"normalization flag changed in {name}")
        bias = getattr(module, "expert_bias", None)
        if contract.expert_bias_enabled or (bias is not None and bool(bias.detach().count_nonzero().item())):
            raise RuntimeError(f"nonzero router expert bias in {name}")
    for layer_idx, (name, module) in enumerate(blocks):
        module._jrc_v4_original_forward = module.forward

        def variable_forward(self, hidden_states, _name=name, _layer_idx=layer_idx):
            if not controller.enabled:
                return self._jrc_v4_original_forward(hidden_states)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            if batch_size != 1:
                raise RuntimeError("JRC-v4 intervention patch requires batch size 1")
            flat = hidden_states.reshape(-1, hidden_dim)
            k_values = controller.for_layer(_layer_idx, _name, flat.shape[0])
            k_counts = torch.as_tensor(k_values, dtype=torch.long, device=flat.device)
            max_k = int(k_counts.max().item())
            router_logits = self.gate(flat)
            full_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
            selected_weights, selected_experts = torch.topk(full_probs, max_k, dim=-1)
            ranks = torch.arange(max_k, device=flat.device).unsqueeze(0)
            active_rank = ranks < k_counts.unsqueeze(1)
            selected_weights = selected_weights * active_rank
            selected_weights = selected_weights.to(flat.dtype)

            if _layer_idx in controller.monitored_layers:
                ids_cpu = selected_experts.detach().cpu().tolist()
                weights_cpu = selected_weights.detach().float().cpu().tolist()
                counts_cpu = k_counts.detach().cpu().tolist()
                logits_cpu = router_logits.detach().float().cpu().tolist()
                hidden_norm_cpu = flat.detach().float().norm(dim=-1).cpu().tolist()
                for position, k in enumerate(counts_cpu):
                    controller.record_routing(
                        _layer_idx, position, ids_cpu[position][:k], weights_cpu[position][:k],
                        router_logits=logits_cpu[position], hidden_state_norm=hidden_norm_cpu[position])

            expert_mask = F.one_hot(selected_experts, num_classes=contract.num_experts).permute(2, 1, 0).bool()
            expert_mask = expert_mask & active_rank.T.unsqueeze(0)
            output = torch.zeros((flat.shape[0], hidden_dim), dtype=flat.dtype, device=flat.device)
            observed_pairs = 0
            for expert_idx in range(contract.num_experts):
                rank_idx, token_idx = torch.where(expert_mask[expert_idx])
                if token_idx.numel() == 0:
                    continue
                observed_pairs += int(token_idx.numel())
                expert_out = self.experts[expert_idx](flat[token_idx])
                weighted = expert_out * selected_weights[token_idx, rank_idx, None]
                output.index_add_(0, token_idx, weighted.to(flat.dtype))
            controller.record_layer_pairs(_layer_idx, _name, observed_pairs)
            return output.reshape(batch_size, seq_len, hidden_dim) * contract.routed_scaling_factor

        module.forward = types.MethodType(variable_forward, module)
    return names
