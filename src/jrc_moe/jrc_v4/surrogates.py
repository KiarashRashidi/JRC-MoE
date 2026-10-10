"""Tier-1 sensitivity measurements; these are not utility labels."""
from __future__ import annotations

import math


def distribution_sensitivity(control_logits, treatment_logits, *, positions=None, epsilon: float = 1e-12) -> dict:
    """Stable tokenwise KL/JSD summaries over the requested positions."""
    try:
        import torch
        c = control_logits.detach().double()
        t = treatment_logits.detach().double()
        if positions is not None:
            c, t = c[..., positions, :], t[..., positions, :]
        pc, pt = torch.softmax(c, dim=-1), torch.softmax(t, dim=-1)
        pm = (pc + pt) * 0.5
        kl_ct = (pc * ((pc + epsilon).log() - (pt + epsilon).log())).sum(-1)
        kl_tc = (pt * ((pt + epsilon).log() - (pc + epsilon).log())).sum(-1)
        jsd = 0.5 * (pc * ((pc + epsilon).log() - (pm + epsilon).log())).sum(-1)
        jsd += 0.5 * (pt * ((pt + epsilon).log() - (pm + epsilon).log())).sum(-1)
        return {"mean_kl_control_to_treatment": float(kl_ct.mean()),
                "mean_kl_treatment_to_control": float(kl_tc.mean()), "mean_jsd": float(jsd.mean()),
                "mean_argmax_change": float((c.argmax(-1) != t.argmax(-1)).float().mean()),
                "interpretation": "sensitivity_only_not_positive_utility"}
    except ImportError as exc:
        raise RuntimeError("Tier-1 distribution comparison requires PyTorch") from exc


def confidence_shift(control_confidence, treatment_confidence) -> float:
    left, right = list(control_confidence), list(treatment_confidence)
    if len(left) != len(right) or not left:
        raise ValueError("confidence arrays must align and be nonempty")
    return sum(float(b) - float(a) for a, b in zip(left, right)) / len(left)


def provisional_answer_stability(control_tokens, treatment_tokens) -> float:
    left, right = list(control_tokens), list(treatment_tokens)
    if len(left) != len(right) or not left:
        raise ValueError("token arrays must align and be nonempty")
    return sum(a == b for a, b in zip(left, right)) / len(left)
