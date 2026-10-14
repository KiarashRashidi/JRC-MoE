"""Variable-K MoE patch and paper RRB/FPS-FGER allocation logic."""
import math
import types
import torch
import torch.nn.functional as F
from .config import *

class VariableKController:
    def __init__(self, expected_layers):
        self.expected_layers = expected_layers
        self.enabled = False
        self.k_counts_cpu = None
        self.layer_pair_counts = []
        self.layer_names = []

    def begin_forward(self, k_counts):
        k = torch.as_tensor(k_counts, dtype=torch.long).detach().cpu()
        if k.ndim != 1:
            raise ValueError("k_counts must be a 1-D vector")
        if int(k.min()) < 1 or int(k.max()) > NUM_EXPERTS:
            raise ValueError("invalid expert count")
        self.k_counts_cpu = k
        self.layer_pair_counts = []
        self.layer_names = []
        self.enabled = True

    def record_layer(self, name, pair_count):
        self.layer_names.append(name)
        self.layer_pair_counts.append(int(pair_count))

    def end_forward(self):
        self.enabled = False
        if len(self.layer_pair_counts) != self.expected_layers:
            raise RuntimeError(
                f"Expected {self.expected_layers} MoE calls, "
                f"got {len(self.layer_pair_counts)}"
            )
        direct = int(sum(self.layer_pair_counts))
        expected = int(
            self.expected_layers * int(self.k_counts_cpu.sum())
        )
        if direct != expected:
            raise RuntimeError(
                f"Direct pairs {direct} != allocated pairs {expected}"
            )
        return {
            "direct_pairs": direct,
            "expected_pairs": expected,
            "layer_pair_counts": list(self.layer_pair_counts),
        }




def install_variable_k_patch(model, controller):
    patched = []

    # Restore native forwards if this cell is rerun.
    for _, module in model.named_modules():
        if hasattr(module, "_reflex_original_forward"):
            module.forward = module._reflex_original_forward

    for name, module in model.named_modules():
        if module.__class__.__name__ != "LLaDAMoESparseMoeBlock":
            continue

        module._reflex_original_forward = module.forward

        def variable_forward(self, hidden_states, _name=name):
            ctrl = controller

            if not ctrl.enabled or ctrl.k_counts_cpu is None:
                return self._reflex_original_forward(hidden_states)

            batch_size, seq_len, hidden_dim = hidden_states.shape
            if batch_size != 1:
                raise RuntimeError("Mechanism reproduction currently uses batch=1")

            flat = hidden_states.view(-1, hidden_dim)

            if len(ctrl.k_counts_cpu) != flat.shape[0]:
                raise RuntimeError("allocation length != sequence length")

            k_counts = ctrl.k_counts_cpu.to(flat.device)

            router_logits = self.gate(flat)
            full_weights = F.softmax(
                router_logits,
                dim=1,
                dtype=torch.float,
            )

            if self.expert_bias is not None:
                full_weights = full_weights + self.expert_bias

            max_k = int(k_counts.max().item())

            selected_weights, selected_experts = torch.topk(
                full_weights,
                max_k,
                dim=-1,
            )

            ranks = torch.arange(
                max_k,
                device=flat.device,
            ).unsqueeze(0)

            active_rank = ranks < k_counts.unsqueeze(1)

            # Native checkpoint has norm_topk_prob=False.
            if self.norm_topk_prob:
                selected_weights = selected_weights * active_rank
                selected_weights = selected_weights / (
                    selected_weights.sum(dim=-1, keepdim=True)
                    .clamp_min(1e-12)
                )

            selected_weights = selected_weights.to(flat.dtype)

            expert_mask = F.one_hot(
                selected_experts,
                num_classes=self.num_experts,
            ).permute(2, 1, 0).bool()

            # [E, rank, token] AND [1, rank, token]
            expert_mask = (
                expert_mask
                & active_rank.T.unsqueeze(0)
            )

            layer_pairs = int(active_rank.sum().item())
            ctrl.record_layer(_name, layer_pairs)

            out = torch.zeros(
                (batch_size * seq_len, hidden_dim),
                dtype=flat.dtype,
                device=flat.device,
            )

            for expert_idx in range(self.num_experts):
                rank_idx, token_idx = torch.where(
                    expert_mask[expert_idx]
                )

                if token_idx.numel() == 0:
                    continue

                current = flat[token_idx]
                expert_out = (
                    self.experts[expert_idx](current)
                    * selected_weights[
                        token_idx,
                        rank_idx,
                        None,
                    ]
                )

                out.index_add_(
                    0,
                    token_idx,
                    expert_out.to(flat.dtype),
                )

            return out.reshape(batch_size, seq_len, hidden_dim)

        module.forward = types.MethodType(
            variable_forward,
            module,
        )
        patched.append(name)

    if len(patched) != NUM_MOE_LAYERS:
        raise RuntimeError(
            f"Patched {len(patched)} MoE blocks; expected {NUM_MOE_LAYERS}"
        )

    return patched


def position_block_ids(prompt_len, generation_length=None):
    generated_length = GEN_LENGTH if generation_length is None else int(generation_length)
    if generated_length <= 0 or generated_length % BLOCK_LENGTH:
        raise ValueError(f"generation_length must be a positive multiple of {BLOCK_LENGTH}")
    ids = torch.empty(
        prompt_len + generated_length,
        dtype=torch.long,
    )
    ids[:prompt_len] = 0

    target_rel = torch.arange(
        generated_length,
        dtype=torch.long,
    )
    ids[prompt_len:] = (
        1 + target_rel // BLOCK_LENGTH
    )
    return ids


def rrb_counts_for_forward(prompt_len, active_block_idx, generation_length=None):
    block_ids = position_block_ids(prompt_len, generation_length=generation_length)
    active_paper = active_block_idx + 1
    distance = (block_ids - active_paper).abs()

    counts = torch.full_like(
        block_ids,
        K_BASE - DELTA_ROLE,  # 4
    )
    counts[distance == 2] = K_BASE  # 8
    counts[distance <= 1] = K_BASE + DELTA_ROLE  # 12

    return counts, block_ids, distance


def sigmoid_scalar(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def fps_from_history(history):
    # No c^{t-1}: no dynamic evidence yet.
    if len(history) == 0:
        return None

    c_prev = float(history[-1])

    # c^{t-2} unavailable: paper uses v=0, c_hat=c_prev, P=1.
    if len(history) == 1:
        v_raw = 0.0
        v_bar = 0.0
        c_hat = c_prev

        frontier = (
            sigmoid_scalar(
                (c_hat - THETA + W_CF) / TAU
            )
            * sigmoid_scalar(
                (THETA - c_hat) / TAU
            )
        )

        return {
            "c_prev": c_prev,
            "c_prev2": None,
            "v_raw": v_raw,
            "v_bar": v_bar,
            "c_hat": c_hat,
            "frontier": frontier,
            "progress": 1.0,
            "score": frontier,
            "history_case": "frontier_only",
        }

    c_prev2 = float(history[-2])
    v_raw = c_prev - c_prev2

    v_bar = max(
        -W_CF / 2,
        min(W_CF / 2, v_raw),
    )

    c_hat = max(
        0.0,
        min(1.0, c_prev + v_bar),
    )

    frontier = (
        sigmoid_scalar(
            (c_hat - THETA + W_CF) / TAU
        )
        * sigmoid_scalar(
            (THETA - c_hat) / TAU
        )
    )

    progress = sigmoid_scalar(
        (V_REF - v_bar) / TAU
    )

    return {
        "c_prev": c_prev,
        "c_prev2": c_prev2,
        "v_raw": v_raw,
        "v_bar": v_bar,
        "c_hat": c_hat,
        "frontier": frontier,
        "progress": progress,
        "score": frontier * progress,
        "history_case": "fps",
    }


def apply_fger(
    counts,
    unresolved_abs_positions,
    confidence_history,
    baseline_active_k,
):
    counts = counts.clone()
    unresolved = [int(p) for p in unresolved_abs_positions]

    score_rows = []

    for pos in unresolved:
        info = fps_from_history(
            confidence_history.get(pos, [])
        )
        if info is not None:
            score_rows.append({
                "position": pos,
                **info,
            })

    # No history => active tokens retain baseline.
    if not score_rows:
        return counts, {
            "rankable_count": 0,
            "group_size": 0,
            "high_positions": [],
            "low_positions": [],
            "score_rows": [],
            "budget_before": baseline_active_k * len(unresolved),
            "budget_after": int(counts[unresolved].sum()) if unresolved else 0,
        }

    ranked = sorted(
        score_rows,
        key=lambda r: (-r["score"], r["position"]),
    )

    group_size = int(
        math.floor(Q_OUTER * len(ranked))
    )

    high = ranked[:group_size] if group_size else []
    low = ranked[-group_size:] if group_size else []

    high_positions = [r["position"] for r in high]
    low_positions = [r["position"] for r in low]

    if high_positions:
        counts[high_positions] = baseline_active_k + DELTA_CUR

    if low_positions:
        counts[low_positions] = baseline_active_k - DELTA_CUR

    before = baseline_active_k * len(unresolved)
    after = int(counts[unresolved].sum()) if unresolved else 0

    if before != after:
        raise RuntimeError(
            f"FGER active budget changed: {before} != {after}"
        )

    return counts, {
        "rankable_count": len(ranked),
        "group_size": group_size,
        "high_positions": high_positions,
        "low_positions": low_positions,
        "score_rows": ranked,
        "budget_before": before,
        "budget_after": after,
    }


def allocation_for_forward(
    mode,
    prompt_len,
    active_block_idx,
    unresolved_abs_positions,
    confidence_history,
    generation_length=None,
):
    generated_length = GEN_LENGTH if generation_length is None else int(generation_length)
    if generated_length <= 0 or generated_length % BLOCK_LENGTH:
        raise ValueError(f"generation_length must be a positive multiple of {BLOCK_LENGTH}")
    total_len = prompt_len + generated_length

    if mode == "fixed8":
        counts = torch.full((total_len,), 8, dtype=torch.long)
        fger_meta = None

    elif mode == "rrb_only":
        counts, _, _ = rrb_counts_for_forward(
            prompt_len,
            active_block_idx,
            generation_length=generated_length,
        )
        fger_meta = None

    elif mode == "fger_only":
        counts = torch.full((total_len,), 8, dtype=torch.long)
        counts, fger_meta = apply_fger(
            counts,
            unresolved_abs_positions,
            confidence_history,
            baseline_active_k=8,
        )
        if int(counts.sum()) != 8 * total_len:
            raise RuntimeError("FGER-only changed total forward budget")

    elif mode == "reflex":
        counts, _, _ = rrb_counts_for_forward(
            prompt_len,
            active_block_idx,
            generation_length=generated_length,
        )
        rrb_budget = int(counts.sum())

        counts, fger_meta = apply_fger(
            counts,
            unresolved_abs_positions,
            confidence_history,
            baseline_active_k=K_BASE + DELTA_ROLE,  # 12
        )

        if int(counts.sum()) != rrb_budget:
            raise RuntimeError("FGER changed RRB total forward budget")

    else:
        raise ValueError(mode)

    allowed = {
        "fixed8": {8},
        "rrb_only": {4, 8, 12},
        "fger_only": {6, 8, 10},
        "reflex": {4, 8, 10, 12, 14},
    }[mode]

    actual = set(counts.unique().tolist())

    if not actual.issubset(allowed):
        raise RuntimeError(
            f"{mode}: unexpected expert counts {actual}"
        )

    return counts, fger_meta


