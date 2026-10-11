"""Bounded paired logit diagnostics without persisting dense vocabulary tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class PairedTrajectoryLogitComparator:
    """Compare target and first downstream candidate-token logits on common states.

    Only candidate-token vectors from the intervention forward and a short,
    fixed number of subsequent aligned NFEs are retained, in CPU float16.
    Dense logits are reduced to compact metrics and discarded after a pair.
    """

    target_nfe: int
    candidate_positions: tuple[int, ...]
    lookahead_steps: int = 2
    _control: dict[tuple[int, int], tuple[int, Any]] = field(default_factory=dict)
    _rows: list[dict[str, Any]] = field(default_factory=list)
    _missing: dict[int, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.candidate_positions = tuple(int(position) for position in self.candidate_positions)
        if self.target_nfe < 1 or self.lookahead_steps < 0:
            raise ValueError("target NFE must be positive and lookahead nonnegative")
        if len(set(self.candidate_positions)) != len(self.candidate_positions):
            raise ValueError("candidate positions must be unique")

    def observe(self, *, branch: str, global_nfe: int, positions: Iterable[int],
                input_token_ids: Iterable[int], logits) -> None:
        if branch not in {"control", "treatment"}:
            raise ValueError("branch must be control or treatment")
        offset = int(global_nfe) - self.target_nfe
        if offset < 0 or offset > self.lookahead_steps:
            return
        import torch
        position_list = [int(position) for position in positions]
        token_list = [int(token) for token in input_token_ids]
        if len(position_list) != len(token_list) or len(position_list) != int(logits.shape[0]):
            raise ValueError("positions, input token IDs, and logits rows must align")
        candidate_index = {position: index for index, position in enumerate(position_list)
                           if position in self.candidate_positions}
        if branch == "control":
            for position, index in candidate_index.items():
                self._control[(int(global_nfe), position)] = (
                    token_list[index], logits[index].detach().to(device="cpu", dtype=torch.float16).clone()
                )
            return

        metrics: list[dict[str, float | int]] = []
        missing = {"not_unresolved": 0, "token_state_mismatch": 0, "missing_control": 0}
        for position in self.candidate_positions:
            index = candidate_index.get(position)
            baseline = self._control.get((int(global_nfe), position))
            if index is None:
                missing["not_unresolved"] += 1
                continue
            if baseline is None:
                missing["missing_control"] += 1
                continue
            control_token, control_logits_half = baseline
            if control_token != token_list[index]:
                missing["token_state_mismatch"] += 1
                continue
            left = control_logits_half.to(dtype=torch.float32)
            right = logits[index].detach().to(device="cpu", dtype=torch.float32)
            logp, logq = torch.log_softmax(left, dim=-1), torch.log_softmax(right, dim=-1)
            p, q = logp.exp(), logq.exp()
            logm = torch.logaddexp(logp, logq) - 0.6931471805599453
            jsd = 0.5 * (p * (logp - logm)).sum() + 0.5 * (q * (logq - logm)).sum()
            top_left = torch.topk(left, k=min(5, left.numel())).indices
            top_right = torch.topk(right, k=min(5, right.numel())).indices
            metrics.append({
                "jsd": float(jsd),
                "mean_abs_logit_delta": float((left - right).abs().mean()),
                "max_abs_logit_delta": float((left - right).abs().max()),
                "top1_flip": int(int(top_left[0]) != int(top_right[0])),
                "top1_confidence_delta": float((p.max() - q.max()).abs()),
                "top5_overlap": len(set(top_left.tolist()) & set(top_right.tolist())) / len(top_left),
            })
        row: dict[str, Any] = {
            "global_nfe": int(global_nfe),
            "offset_from_intervention": offset,
            "compared_positions": len(metrics),
            **{f"missing_{key}": value for key, value in missing.items()},
        }
        if metrics:
            for key in metrics[0]:
                row[key] = sum(float(item[key]) for item in metrics) / len(metrics)
            row["max_abs_logit_delta_across_positions"] = max(float(item["max_abs_logit_delta"]) for item in metrics)
            row["top1_flip_count"] = sum(int(item["top1_flip"]) for item in metrics)
        else:
            row.update({
                "jsd": None, "mean_abs_logit_delta": None, "max_abs_logit_delta": None,
                "max_abs_logit_delta_across_positions": None, "top1_flip": None,
                "top1_flip_count": None, "top1_confidence_delta": None, "top5_overlap": None,
            })
        self._rows.append(row)
        self._missing[int(global_nfe)] = missing

    def result(self) -> dict[str, Any]:
        return {
            "target_nfe": self.target_nfe,
            "candidate_positions": list(self.candidate_positions),
            "lookahead_steps": self.lookahead_steps,
            "steps": list(self._rows),
            "dense_logits_persisted": False,
            "control_vectors_retained_cpu_float16": len(self._control),
        }
