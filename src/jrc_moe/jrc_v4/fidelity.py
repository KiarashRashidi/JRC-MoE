"""Native-versus-patched gates on actual model forwards."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from .audit import audit_loaded_model
from .controller import LayerwiseKController
from .patch import VerifiedMoEContract, install_layerwise_patch
from .controller import LayerwiseKController


def verified_contract_from_loaded_model(model, checkpoint_revision: str) -> tuple[VerifiedMoEContract, dict]:
    """Verify runtime config and Python block attributes; native equivalence is a separate gate."""
    audit = audit_loaded_model(model)
    cfg = model.config
    norm = bool(getattr(cfg, "norm_topk_prob", False))
    bias_enabled = bool(getattr(cfg, "moe_router_enable_expert_bias", False))
    scale = float(getattr(cfg, "routed_scaling_factor", 1.0) or 1.0)
    blocks = [m for _, m in model.named_modules() if m.__class__.__name__ == "LLaDAMoESparseMoeBlock"]
    per_block_norm = {bool(getattr(m, "norm_topk_prob", False)) for m in blocks}
    bias_nonzero = any(m.expert_bias is not None and bool(m.expert_bias.detach().count_nonzero().item())
                       for m in blocks if hasattr(m, "expert_bias"))
    scale_values = {float(getattr(m, "routed_scaling_factor", scale) or 1.0) for m in blocks}
    source_available = bool(audit["source_hashes"]) and all(m["forward_source_sha256"] for m in audit["moe_blocks"])
    semantics_ok = (audit["config_matches_frozen_expectations"] and audit["moe_block_count_matches"]
                    and len(per_block_norm) == 1 and next(iter(per_block_norm), None) == norm
                    and not bias_enabled and not bias_nonzero and scale == 1.0 and scale_values == {1.0}
                    and source_available)
    contract = VerifiedMoEContract(
        checkpoint_revision=checkpoint_revision,
        num_layers=len(blocks),
        num_experts=int(getattr(cfg, "num_experts", 0)),
        native_top_k=int(getattr(cfg, "num_experts_per_tok", 0)),
        score_function=str(getattr(cfg, "moe_router_score_function", "")),
        norm_topk_prob=norm,
        expert_bias_enabled=bias_enabled or bias_nonzero,
        routed_scaling_factor=scale,
        router_semantics_verified=bool(semantics_ok and not norm and str(getattr(cfg, "moe_router_score_function", "")) == "softmax"),
        native_fixed8_equivalence_verified=False,
    )
    return contract, {"runtime_semantics_gate": contract.router_semantics_verified,
                     "expert_bias_enabled": bias_enabled, "expert_bias_nonzero": bias_nonzero,
                     "norm_topk_prob": norm, "routed_scaling_factor": scale,
                     "block_scaling_values": sorted(scale_values), "audit": audit}


def _output_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def make_mask_phase_cases(prompt_ids: torch.Tensor, *, mask_id: int, generation_length: int = 256,
                          block_length: int = 32) -> list[tuple[torch.Tensor, tuple[int, ...]]]:
    """Build three explicit masked-denoising phases from one real encoded prompt."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("fidelity prompt must be one [1, sequence] tensor")
    prompt_len = int(prompt_ids.shape[1])
    total_len = prompt_len + generation_length
    token = int(prompt_ids[0, -1])
    cases = []
    for phase in ("initial", "partial_block", "later_block"):
        x = torch.full((1, total_len), int(mask_id), dtype=torch.long, device=prompt_ids.device)
        x[:, :prompt_len] = prompt_ids
        if phase == "initial":
            positions = tuple(range(prompt_len, prompt_len + block_length))
        elif phase == "partial_block":
            x[:, prompt_len:prompt_len + block_length // 2] = token
            positions = tuple(range(prompt_len + block_length // 2, prompt_len + block_length))
        else:
            x[:, prompt_len:prompt_len + block_length] = token
            x[:, prompt_len + block_length:prompt_len + block_length + block_length // 2] = token
            positions = tuple(range(prompt_len + block_length + block_length // 2,
                                    prompt_len + 2 * block_length))
        cases.append((x, positions))
    return cases


@torch.inference_mode()
def legal_k_forward_gate(model, controller: LayerwiseKController, contract: VerifiedMoEContract,
                         input_ids: torch.Tensor, legal_k=(2, 4, 6, 8)) -> dict:
    """Exercise every in-protocol K and verify measured per-layer pair counts."""
    contract.require_supported()
    blocks = [(name, module) for name, module in model.named_modules()
              if module.__class__.__name__ == "LLaDAMoESparseMoeBlock"]
    install_layerwise_patch(model, controller, contract)
    observed = []
    try:
        for nfe, k in enumerate(legal_k, start=1):
            if not 2 <= k <= 8:
                raise ValueError("v4 K test must stay inside [2,8]")
            controller.begin_forward([k] * input_ids.shape[1], global_nfe=nfe)
            model(input_ids)
            result = controller.end_forward()
            expected = k * input_ids.shape[1]
            if result["layer_pair_counts"] != (expected,) * len(blocks):
                raise RuntimeError(f"K={k}: observed pair accounting mismatch")
            observed.append({"k": k, "per_layer_pairs": expected, "direct_pairs": result["direct_pairs"]})
    finally:
        for _, module in blocks:
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return {"passed": True, "forwards": observed}


@torch.inference_mode()
def native_patch_fidelity(model, controller_factory, contract: VerifiedMoEContract,
                          cases: Iterable[tuple[torch.Tensor, tuple[int, ...]]], *,
                          max_logit_diff: float = 1e-5, min_argmax_match: float = 1.0,
                          max_confidence_diff: float = 1e-6, max_weight_diff: float = 1e-6,
                          max_block_output_diff: float = 1e-5) -> dict:
    """Compare block outputs, route identities/weights, and whole-model logits.

    `cases` is a sequence of (input_ids, positions_to_compare). Use representative
    prompt and diffusion-phase inputs selected from the frozen development set.
    """
    contract.require_supported(allow_equivalence_probe=True)
    blocks = [(name, module) for name, module in model.named_modules()
              if module.__class__.__name__ == "LLaDAMoESparseMoeBlock"]
    names = tuple(name for name, _ in blocks)
    if controller_factory is None:
        controller_factory = lambda module_names, positions: LayerwiseKController(
            module_names, min_k=2, max_k=8, monitored_layers=range(len(module_names)),
            monitored_positions=positions)
    cases = list(cases)
    if not cases:
        raise ValueError("at least one fidelity case is required")
    controller = controller_factory(names, ())
    install_layerwise_patch(model, controller, contract, allow_equivalence_probe=True)
    results = []
    nfe = 0
    try:
        for case_idx, (input_ids, positions) in enumerate(cases):
            if not positions:
                raise ValueError("each fidelity case needs at least one comparison position")
            native_inputs, native_blocks = {}, {}
            patch_blocks = {}
            handles = []
            for layer_idx, (name, module) in enumerate(blocks):
                def native_pre(_module, args, _i=layer_idx):
                    native_inputs[_i] = args[0].detach()
                def native_post(_module, _args, output, _i=layer_idx):
                    native_blocks[_i] = _output_tensor(output).detach().float().cpu()
                handles.extend((module.register_forward_pre_hook(native_pre), module.register_forward_hook(native_post)))
            native = model(input_ids).logits[:, list(positions), :].detach().float().cpu()
            for handle in handles:
                handle.remove()

            controller.monitored_positions = frozenset(range(input_ids.shape[1]))
            for layer_idx, (name, module) in enumerate(blocks):
                def patched_post(_module, _args, output, _i=layer_idx):
                    patch_blocks[_i] = _output_tensor(output).detach().float().cpu()
                handles.append(module.register_forward_hook(patched_post))
            nfe += 1
            controller.begin_forward([contract.native_top_k] * input_ids.shape[1], global_nfe=nfe, branch="control")
            if input_ids.is_cuda:
                torch.cuda.synchronize()
            patched = model(input_ids).logits[:, list(positions), :].detach().float().cpu()
            if input_ids.is_cuda:
                torch.cuda.synchronize()
            pair_meta = controller.end_forward()
            for handle in handles:
                handle.remove()
            handles = []

            logit_diff = float((native - patched).abs().max())
            argmax_match = float((native.argmax(-1) == patched.argmax(-1)).float().mean())
            native_conf = F.softmax(native.double(), dim=-1).max(-1).values
            patched_conf = F.softmax(patched.double(), dim=-1).max(-1).values
            confidence_diff = float((native_conf - patched_conf).abs().max())
            per_layer_diff = {names[i]: float((native_blocks[i] - patch_blocks[i]).abs().max()) for i in range(len(names))}

            route_diff_max = 0.0
            rank_matches = 0
            rank_total = 0
            expert_counts = [[0] * contract.num_experts for _ in blocks]
            for layer_idx, (_, module) in enumerate(blocks):
                hidden = native_inputs[layer_idx].reshape(-1, native_inputs[layer_idx].shape[-1])
                logits = module.gate(hidden)
                probs = F.softmax(logits, dim=1, dtype=torch.float32)
                weights, experts = torch.topk(probs, contract.native_top_k, dim=-1)
                for position in range(input_ids.shape[1]):
                    got = pair_meta["routing_records"].get((layer_idx, position))
                    if got is None:
                        continue
                    expected_ids = experts[position].detach().cpu().tolist()
                    expected_weights = weights[position].to(hidden.dtype).float().detach().cpu().tolist()
                    for expert in got["experts"]:
                        expert_counts[layer_idx][int(expert)] += 1
                    rank_matches += int(tuple(expected_ids) == got["experts"])
                    rank_total += 1
                    if expected_weights:
                        route_diff_max = max(route_diff_max, max(abs(a - b) for a, b in zip(expected_weights, got["weights"])))
            case_result = {"case_idx": case_idx, "sequence_length": int(input_ids.shape[1]),
                           "positions": list(positions),
                           "max_abs_logit_diff": logit_diff, "argmax_match_fraction": argmax_match,
                           "max_confidence_diff": confidence_diff, "per_layer_max_abs_output_diff": per_layer_diff,
                           "selected_rank_match_fraction": rank_matches / rank_total if rank_total else None,
                           "selected_weight_max_abs_diff": route_diff_max,
                           "per_layer_executed_pairs": list(pair_meta["layer_pair_counts"]),
                           "per_layer_expert_token_pairs": expert_counts,
                           "active_experts_by_layer": [sum(count > 0 for count in row) for row in expert_counts],
                           "load_balance_cv_by_layer": [
                               float(torch.tensor(row, dtype=torch.float64).std(unbiased=False)
                                     / torch.tensor(row, dtype=torch.float64).mean())
                               if sum(row) else None for row in expert_counts],
                           "nonempty_expert_calls_by_layer": [sum(count > 0 for count in row) for row in expert_counts],
                           "direct_pairs": pair_meta["direct_pairs"]}
            results.append(case_result)
            del native, patched, native_inputs, native_blocks, patch_blocks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for _, module in blocks:
            for handle in handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            if hasattr(module, "_jrc_v4_original_forward"):
                module.forward = module._jrc_v4_original_forward
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    maxima = {key: max(row[key] for row in results) for key in
              ("max_abs_logit_diff", "max_confidence_diff", "selected_weight_max_abs_diff")}
    block_max = max(max(value for value in row["per_layer_max_abs_output_diff"].values()) for row in results)
    argmax_min = min(row["argmax_match_fraction"] for row in results)
    ranks = [row["selected_rank_match_fraction"] for row in results if row["selected_rank_match_fraction"] is not None]
    rank_min = min(ranks) if ranks else None
    pair_counts_valid = all(all(count == contract.native_top_k * row["sequence_length"]
                                for count in row["per_layer_executed_pairs"])
                            for row in results)
    passed = (maxima["max_abs_logit_diff"] <= max_logit_diff and argmax_min >= min_argmax_match
              and maxima["max_confidence_diff"] <= max_confidence_diff
              and maxima["selected_weight_max_abs_diff"] <= max_weight_diff
              and block_max <= max_block_output_diff and rank_min == 1.0 and pair_counts_valid)
    return {"passed": passed, "cases": results, "maxima": maxima,
            "max_block_output_diff": block_max, "per_layer_pair_counts_valid": pair_counts_valid,
            "minimum_argmax_match_fraction": argmax_min, "minimum_selected_rank_match_fraction": rank_min,
            "tolerances": {"max_logit_diff": max_logit_diff, "min_argmax_match": min_argmax_match,
                           "max_confidence_diff": max_confidence_diff, "max_weight_diff": max_weight_diff,
                           "max_block_output_diff": max_block_output_diff}}
