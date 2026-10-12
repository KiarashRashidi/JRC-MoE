"""Threshold decoder, routing-mode inference, and trace gates."""
from collections import defaultdict
import hashlib
import math, re, time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from .config import *
from .model import input_device as resolve_input_device
from .routing import allocation_for_forward

def extract_target_predictions_and_confidence(
    logits,
    prompt_len,
):
    target_pred = torch.empty(
        GEN_LENGTH,
        dtype=torch.long,
    )
    target_conf = torch.empty(
        GEN_LENGTH,
        dtype=torch.float64,
    )

    for b in range(NUM_BLOCKS):
        abs_start = prompt_len + b * BLOCK_LENGTH
        abs_end = abs_start + BLOCK_LENGTH

        block_logits = logits[0, abs_start:abs_end, :]

        pred = torch.argmax(block_logits, dim=-1)

        probs = F.softmax(
            block_logits.to(torch.float64),
            dim=-1,
        )

        conf = torch.gather(
            probs,
            dim=-1,
            index=pred.unsqueeze(-1),
        ).squeeze(-1)

        rel_start = b * BLOCK_LENGTH
        rel_end = rel_start + BLOCK_LENGTH

        target_pred[rel_start:rel_end] = pred.detach().cpu()
        target_conf[rel_start:rel_end] = conf.detach().cpu()

        del probs, block_logits

    return target_pred, target_conf


@torch.no_grad()
def generate_with_routing_mode(
    model,
    prompt_ids,
    mode,
    controller,
    *,
    allocation_provider=None,
    before_forward=None,
    resume_state=None,
    branch="control",
    exchange=None,
    checkpoint_revision=None,
    config_digest=None,
    sampler_settings=None,
    before_forward_nfe=None,
    forward_observer=None,
    sensitivity_observer=None,
    observational_observer=None,
    event_metadata=None,
    position_trace=False,
    budget_check=None,
    output_budget_check=None,
    cuda_event_profile=False,
):
    prompt_ids = prompt_ids.to(resolve_input_device(model))

    prompt_len = int(prompt_ids.shape[1])
    total_len = prompt_len + GEN_LENGTH

    if resume_state is None:
        x = torch.full(
            (1, total_len),
            MASK_ID,
            dtype=torch.long,
            device=prompt_ids.device,
        )
        x[:, :prompt_len] = prompt_ids
        confidence_history = defaultdict(list)
        trace = []
        total_pairs = 0
        global_nfe = 0
        start_block = 0
        restored_local_nfe = 0
        prefix_seconds = 0.0
    else:
        if observational_observer is not None:
            raise ValueError("natural observations require a fresh complete trajectory, not a mid-block replay state")
        if tuple(resume_state.input_ids.shape) != (1, total_len):
            raise ValueError("ReplayState token shape does not match this prompt and generation length")
        x = resume_state.input_ids.to(prompt_ids.device).clone()
        confidence_history = defaultdict(list, copy.deepcopy(resume_state.confidence_history))
        trace = copy.deepcopy(resume_state.metadata.get("trace_prefix", []))
        total_pairs = int(resume_state.decoder_counters.get("total_pairs", 0))
        global_nfe = int(resume_state.global_nfe)
        start_block = int(resume_state.block_idx)
        restored_local_nfe = int(resume_state.local_nfe)
        prefix_seconds = float(resume_state.metadata.get("prefix_seconds", 0.0))
        resume_state.restore_rng()

    run_start = time.perf_counter()
    total_cuda_seconds = 0.0

    replay_first_forward = resume_state is not None
    observer_seconds = 0.0

    for block_idx in range(start_block, NUM_BLOCKS):
        abs_start = prompt_len + block_idx * BLOCK_LENGTH
        abs_end = abs_start + BLOCK_LENGTH
        local_nfe = restored_local_nfe if replay_first_forward else 0
        block_confidence_history = defaultdict(list)

        while True:
            active_tokens = x[0, abs_start:abs_end]
            active_mask = active_tokens == MASK_ID

            masked_before = int(active_mask.sum())

            if masked_before == 0:
                break
            if budget_check is not None:
                budget_check()
            if output_budget_check is not None:
                output_budget_check()

            restored_this_forward = replay_first_forward
            if not restored_this_forward:
                local_nfe += 1
                global_nfe += 1

            unresolved_local = (
                torch.where(active_mask)[0]
                .detach()
                .cpu()
                .tolist()
            )
            unresolved_abs = [
                abs_start + int(p)
                for p in unresolved_local
            ]

            pre_observation = None
            if observational_observer is not None:
                masked_target_rel_before = torch.where(
                    x[0, prompt_len:] == MASK_ID
                )[0].detach().cpu().tolist()
                active_set = set(unresolved_abs)
                future_masked_abs = [prompt_len + int(p) for p in masked_target_rel_before
                                     if not abs_start <= prompt_len + int(p) < abs_end]
                pre_observation = {
                    **dict(event_metadata or {}),
                    "mode": str(mode), "block_idx": int(block_idx),
                    "global_nfe": int(global_nfe), "local_nfe": int(local_nfe),
                    "prompt_length": int(prompt_len), "mask_token_id": int(MASK_ID),
                    "pre_forward_masked_positions_abs": tuple(prompt_len + int(p) for p in masked_target_rel_before),
                    "eligible_positions_abs": tuple(unresolved_abs),
                    "eligible_positions_relative": tuple(p - prompt_len for p in unresolved_abs),
                    "active_block_committed_before": int(BLOCK_LENGTH - masked_before),
                    "active_block_mask_ratio": float(masked_before / BLOCK_LENGTH),
                    "masked_before": int(masked_before),
                    "first_local_forward": bool(local_nfe == 1),
                    "features_available_through_local_nfe": int(local_nfe - 1),
                    "confidence_history_full_before": {
                        str(pos - prompt_len): tuple(float(v) for v in confidence_history.get(pos, ()))
                        for pos in (unresolved_abs + future_masked_abs)
                    },
                    "confidence_history_in_block_before": {
                        str(pos - prompt_len): tuple(float(v) for v in block_confidence_history.get(pos, ()))
                        for pos in unresolved_abs
                    },
                    "confidence_history_future_blocks_before": {
                        str(pos - prompt_len): tuple(float(v) for v in confidence_history.get(pos, ()))
                        for pos in future_masked_abs
                    },
                }

            # Existing REFLEX callers use the untouched allocator/controller path.
            # JRC-v4 may provide a layer-by-position allocation and restore a saved state.
            if restored_this_forward:
                k_counts = resume_state.k_by_layer
                fger_meta = copy.deepcopy(resume_state.metadata.get("fger_meta"))
            elif allocation_provider is not None:
                allocation = allocation_provider(
                    mode=mode, prompt_len=prompt_len, block_idx=block_idx,
                    unresolved_abs=unresolved_abs, confidence_history=confidence_history,
                    global_nfe=global_nfe,
                )
                if isinstance(allocation, tuple) and len(allocation) == 2:
                    k_counts, fger_meta = allocation
                else:
                    k_counts, fger_meta = allocation, None
            else:
                k_counts, fger_meta = allocation_for_forward(
                    mode,
                    prompt_len,
                    block_idx,
                    unresolved_abs,
                    confidence_history,
                )

            if observational_observer is not None:
                if hasattr(controller, "set_observation_context"):
                    router_stride = max(1, int(getattr(observational_observer, "router_stride", 1)))
                    capture_router = ((local_nfe - 1) % router_stride) == 0
                    controller.set_observation_context(
                        global_nfe=global_nfe,
                        monitored_positions=unresolved_abs,
                        monitored_layers=getattr(observational_observer, "router_layers", ()),
                        capture=capture_router,
                    )
                elif getattr(observational_observer, "router_layers", ()):
                    raise TypeError("router observations require a controller with sparse capture support")
                pre_observation["active_k_counts"] = tuple(int(k_counts[pos]) for pos in unresolved_abs)
                if fger_meta is not None:
                    pre_observation["fger_high_positions_relative"] = tuple(
                        int(pos) - prompt_len for pos in fger_meta.get("high_positions", ())
                    )
                    pre_observation["fger_low_positions_relative"] = tuple(
                        int(pos) - prompt_len for pos in fger_meta.get("low_positions", ())
                    )
                if hasattr(observational_observer, "features_for_positions"):
                    pre_observation["past_features_by_position"] = observational_observer.features_for_positions(
                        block_idx=int(block_idx), local_nfe=int(local_nfe),
                        positions=tuple(unresolved_abs), prompt_len=int(prompt_len),
                    )
                observational_observer.begin_forward(copy.deepcopy(pre_observation))

            from .jrc_v4.controller import LayerwiseKController
            if isinstance(controller, LayerwiseKController):
                controller.begin_forward(k_counts, global_nfe=global_nfe, branch=branch, exchange=exchange)
            else:
                if allocation_provider is not None or resume_state is not None:
                    raise TypeError("layerwise allocations and replay require LayerwiseKController")
                controller.begin_forward(k_counts)

            try:
                if (before_forward is not None and not restored_this_forward
                        and (before_forward_nfe is None or global_nfe == before_forward_nfe)):
                    from .jrc_v4.replay import capture_replay_state
                    k_matrix = k_counts
                    if len(k_counts) > 0 and isinstance(k_counts[0], (int, float)):
                        k_matrix = [list(k_counts) for _ in controller.module_names]
                    state = capture_replay_state(
                        input_ids=x,
                        attention_mask=None,
                        block_idx=block_idx,
                        global_nfe=global_nfe,
                        local_nfe=local_nfe,
                        confidence_history=dict(confidence_history),
                        decoder_counters={"total_pairs": total_pairs},
                        sampler=dict(sampler_settings or {"generation_length": GEN_LENGTH, "block_length": BLOCK_LENGTH}),
                        threshold=THETA,
                        commit_policy="threshold_with_argmax_fallback",
                        k_by_layer=tuple(tuple(int(v) for v in row) for row in k_matrix),
                        checkpoint_revision=checkpoint_revision or "unverified",
                        config_digest=config_digest or "unverified",
                        metadata={"trace_prefix": trace, "fger_meta": fger_meta,
                                  "prefix_seconds": time.perf_counter() - run_start,
                                  "mode": mode, "prompt_length": prompt_len},
                    )
                    pause = before_forward(state)
                    if pause:
                        if isinstance(controller, LayerwiseKController):
                            controller.abort_forward()
                        elif hasattr(controller, "abort_forward"):
                            controller.abort_forward()
                        elapsed = float(time.perf_counter() - run_start)
                        return x, pd.DataFrame(trace), {
                            "mode": mode, "prompt_length": prompt_len, "sequence_length": total_len,
                            "total_nfe": global_nfe, "total_pairs": int(total_pairs),
                            "paused_before_forward": True, "global_nfe": global_nfe,
                            "block_idx": block_idx, "continuation_seconds": elapsed,
                            "shared_prefix_seconds": prefix_seconds,
                        }

                synchronized_timing = allocation_provider is not None or restored_this_forward
                if synchronized_timing and torch.cuda.is_available():
                    torch.cuda.synchronize()
                cuda_start_event = cuda_end_event = None
                if cuda_event_profile and torch.cuda.is_available():
                    cuda_start_event = torch.cuda.Event(enable_timing=True)
                    cuda_end_event = torch.cuda.Event(enable_timing=True)
                    cuda_start_event.record()
                t0 = time.perf_counter()
                outputs = model(x)
                if cuda_end_event is not None:
                    cuda_end_event.record()
                if synchronized_timing and torch.cuda.is_available():
                    torch.cuda.synchronize()
            except BaseException:
                if isinstance(controller, LayerwiseKController):
                    controller.abort_forward()
                elif hasattr(controller, "abort_forward"):
                    controller.abort_forward()
                raise
            forward_seconds = time.perf_counter() - t0

            pair_meta = controller.end_forward()
            direct_pairs = int(pair_meta["direct_pairs"])
            total_pairs += direct_pairs

            # Router features are pre-admission records from the patched block;
            # post-forward logits are exposed through a separate sensitivity hook
            # and must never be fed into a pre-decision FeatureRecord.
            if forward_observer is not None:
                forward_observer({
                    "branch": branch, "global_nfe": global_nfe, "block_idx": block_idx,
                    "unresolved_positions": tuple(unresolved_abs),
                    "masked_before": masked_before,
                    "confidence_history": copy.deepcopy(confidence_history),
                    "k_by_layer": pair_meta["k_by_layer"],
                    "routing_records": pair_meta["routing_records"],
                })
            if sensitivity_observer is not None:
                active_logits = outputs.logits[0, unresolved_abs, :]
                sensitivity_observer({
                    "branch": branch, "global_nfe": global_nfe, "block_idx": block_idx,
                    "positions": tuple(unresolved_abs), "logits": active_logits,
                    "input_token_ids": tuple(int(x[0, position].item()) for position in unresolved_abs),
                })

            target_pred, target_conf = (
                extract_target_predictions_and_confidence(
                    outputs.logits,
                    prompt_len,
                )
            )
            forward_cuda_seconds = None
            if cuda_end_event is not None:
                # Target prediction extraction copies to CPU above, so the event
                # is complete without forcing a different attention backend.
                forward_cuda_seconds = float(cuda_start_event.elapsed_time(cuda_end_event) / 1000.0)
                total_cuda_seconds += forward_cuda_seconds

            # Observe confidence for every target that was masked BEFORE commit.
            target_before = x[0, prompt_len:].detach().cpu()
            masked_target_rel = (
                torch.where(target_before == MASK_ID)[0]
                .tolist()
            )

            for rel_pos in masked_target_rel:
                abs_pos = prompt_len + int(rel_pos)
                confidence_history[abs_pos].append(
                    float(target_conf[rel_pos])
                )

            rel_active_start = block_idx * BLOCK_LENGTH

            active_pred = target_pred[
                rel_active_start:
                rel_active_start + BLOCK_LENGTH
            ]
            active_conf = target_conf[
                rel_active_start:
                rel_active_start + BLOCK_LENGTH
            ]

            # Confidence observed on this forward is available only to the next
            # local decision. Keeping this history block-local prevents
            # provisional predictions from later blocks entering a candidate's
            # predictor features.
            for p in unresolved_local:
                abs_pos = abs_start + int(p)
                block_confidence_history[abs_pos].append(float(active_conf[p]))

            threshold_local = [
                p
                for p in unresolved_local
                if float(active_conf[p]) >= THETA
            ]

            fallback_used = False
            fallback_local = None

            if threshold_local:
                commit_local = threshold_local
            else:
                fallback_used = True
                best_local = max(
                    unresolved_local,
                    key=lambda p: float(active_conf[p]),
                )
                fallback_local = int(best_local)
                commit_local = [best_local]

            for p in commit_local:
                x[0, abs_start + p] = active_pred[p].to(x.device)

            masked_after = int(
                (x[0, abs_start:abs_end] == MASK_ID).sum()
            )

            k_tensor = torch.as_tensor(k_counts, dtype=torch.long)
            allocation_sha256 = hashlib.sha256(
                k_tensor.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            vals, freq = torch.unique(
                k_tensor,
                return_counts=True,
            )
            k_hist = dict(zip(vals.tolist(), freq.tolist()))
            if k_tensor.ndim == 1:
                forward_mean_k = float(k_tensor.float().mean())
                layer_budgets = [int(k_tensor.sum())] * NUM_MOE_LAYERS
                per_layer_budget = int(k_tensor.sum())
            elif k_tensor.ndim == 2:
                forward_mean_k = float(k_tensor.float().mean())
                layer_budgets = [int(row.sum()) for row in k_tensor]
                per_layer_budget = layer_budgets[0] if len(set(layer_budgets)) == 1 else None
            else:
                raise RuntimeError("K allocation must be [position] or [layer, position]")
            layer_budget_extra = {"layer_budgets": layer_budgets} if k_tensor.ndim == 2 else {}

            trace.append({
                "mode": mode,
                "block_idx": block_idx,
                "local_nfe": local_nfe,
                "global_nfe": global_nfe,
                "masked_before": masked_before,
                "num_above_threshold": len(threshold_local),
                "fallback_used": fallback_used,
                "num_committed": len(commit_local),
                "masked_after": masked_after,
                "active_conf_mean": float(
                    active_conf[unresolved_local].mean()
                ),
                "active_conf_max": float(
                    active_conf[unresolved_local].max()
                ),
                "active_conf_min": float(
                    active_conf[unresolved_local].min()
                ),
                "forward_mean_k": forward_mean_k,
                "per_layer_budget": per_layer_budget,
                **layer_budget_extra,
                "direct_pairs": direct_pairs,
                "k4_count": k_hist.get(4, 0),
                "k6_count": k_hist.get(6, 0),
                "k8_count": k_hist.get(8, 0),
                "k10_count": k_hist.get(10, 0),
                "k12_count": k_hist.get(12, 0),
                "k14_count": k_hist.get(14, 0),
                "fger_rankable_count": (
                    None if fger_meta is None
                    else fger_meta["rankable_count"]
                ),
                "fger_group_size": (
                    None if fger_meta is None
                    else fger_meta["group_size"]
                ),
                "fger_high_count": (
                    0 if fger_meta is None
                    else len(fger_meta["high_positions"])
                ),
                "fger_low_count": (
                    0 if fger_meta is None
                    else len(fger_meta["low_positions"])
                ),
                "forward_seconds": forward_seconds,
                "forward_cuda_seconds": forward_cuda_seconds,
                "allocation_sha256": allocation_sha256,
            })

            if position_trace:
                trace[-1].update({
                    "eligible_positions_relative": tuple(p - prompt_len for p in unresolved_abs),
                    "threshold_positions_relative": tuple(abs_start + p - prompt_len for p in threshold_local),
                    "committed_positions_relative": tuple(abs_start + p - prompt_len for p in commit_local),
                    "fallback_position_relative": (None if fallback_local is None
                                                    else abs_start + fallback_local - prompt_len),
                })

            if observational_observer is not None:
                remaining_relative = torch.where(x[0, prompt_len:] == MASK_ID)[0].detach().cpu().tolist()
                active_conf_values = {int(p): float(active_conf[p]) for p in unresolved_local}
                candidate_histories = {
                    int(p): tuple(float(v) for v in pre_observation["confidence_history_in_block_before"].get(
                        str(abs_start + int(p) - prompt_len), ()))
                    for p in unresolved_local
                }
                rank_histories = {int(p): [] for p in unresolved_local}
                history_steps = max((len(v) for v in candidate_histories.values()), default=0)
                for history_idx in range(history_steps):
                    past_values = [(p, values[history_idx]) for p, values in candidate_histories.items()
                                   if history_idx < len(values)]
                    ordered = sorted(past_values, key=lambda item: (item[1], -item[0]))
                    start = 0
                    while start < len(ordered):
                        end = start + 1
                        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
                            end += 1
                        percentile = ((start + end - 1) / 2) / max(1, len(ordered) - 1)
                        for p, _ in ordered[start:end]:
                            rank_histories[p].append(float(percentile))
                        start = end
                commit_local_set = set(int(p) for p in commit_local)
                threshold_local_set = set(int(p) for p in threshold_local)
                router_records = pair_meta.get("routing_records", {})
                event = {
                    **pre_observation,
                    "num_above_threshold": int(len(threshold_local)),
                    "threshold_positions_abs": tuple(abs_start + int(p) for p in threshold_local),
                    "threshold_positions_relative": tuple(abs_start + int(p) - prompt_len for p in threshold_local),
                    "committed_positions_abs": tuple(abs_start + int(p) for p in commit_local),
                    "committed_positions_relative": tuple(abs_start + int(p) - prompt_len for p in commit_local),
                    "committed_token_ids": tuple(int(active_pred[p]) for p in commit_local),
                    "fallback_used": bool(fallback_used),
                    "fallback_reason": "no_threshold_eligible_position" if fallback_used else None,
                    "fallback_position_abs": None if fallback_local is None else abs_start + fallback_local,
                    "fallback_position_relative": None if fallback_local is None else abs_start + fallback_local - prompt_len,
                    "masked_after": int(masked_after),
                    "positions_remain_masked_relative": tuple(int(p) for p in remaining_relative),
                    "active_conf_mean": float(active_conf[unresolved_local].mean()),
                    "active_conf_max": float(active_conf[unresolved_local].max()),
                    "active_conf_min": float(active_conf[unresolved_local].min()),
                    "position_decisions": tuple({
                        "position_relative": int(abs_start + int(p) - prompt_len),
                        "predicted_top1_token_id": int(active_pred[p]),
                        "predicted_confidence": active_conf_values[int(p)],
                        "threshold_eligible": bool(int(p) in threshold_local_set),
                        "committed_this_forward": bool(int(p) in commit_local_set),
                        "commit_type": ("threshold" if int(p) in commit_local_set and int(p) in threshold_local_set
                                        else "fallback" if int(p) in commit_local_set else None),
                    } for p in unresolved_local),
                    "direct_pairs": int(direct_pairs),
                    "layer_pair_counts": tuple(int(x) for x in pair_meta.get("layer_pair_counts", ())),
                    "allocation_sha256": allocation_sha256,
                    "active_k_counts": (tuple(int(k_tensor[p]) for p in unresolved_abs)
                                        if k_tensor.ndim == 1 else
                                        tuple(tuple(int(k_tensor[layer, p]) for p in unresolved_abs)
                                              for layer in range(k_tensor.shape[0]))),
                    "forward_seconds": float(forward_seconds),
                    "forward_cuda_seconds": forward_cuda_seconds,
                    "gpu_peak_allocated_bytes": (int(torch.cuda.max_memory_allocated())
                                                  if torch.cuda.is_available() else None),
                }
                position_rows = []
                for p in unresolved_local:
                    abs_pos = abs_start + int(p)
                    rel_pos = abs_pos - prompt_len
                    full_history = tuple(float(v) for v in pre_observation["confidence_history_full_before"].get(str(rel_pos), ()))
                    local_history = tuple(float(v) for v in pre_observation["confidence_history_in_block_before"].get(str(rel_pos), ()))
                    position_rows.append({
                        **dict(event_metadata or {}), "run_id": str((event_metadata or {}).get("run_id", "")),
                        "prompt_id": str((event_metadata or {}).get("prompt_id", "")), "mode": str(mode),
                        "block_idx": int(block_idx), "global_nfe": int(global_nfe), "local_nfe": int(local_nfe),
                        "prompt_length": int(prompt_len),
                        "mode_fixed8": int(mode == "fixed8"), "mode_reflex": int(mode == MODE_REFLEX),
                        "features_available_through_local_nfe": int(local_nfe - 1),
                        "available_through_local_nfe": int(local_nfe - 1),
                        "position_absolute": int(abs_pos), "position_relative": int(rel_pos),
                        "active_block": True, "candidate_at_risk": True,
                        "block_progress": float(1.0 - masked_before / BLOCK_LENGTH),
                        "position_fraction": float((rel_pos % BLOCK_LENGTH) / max(BLOCK_LENGTH - 1, 1)),
                        "active_mask_ratio": float(masked_before / BLOCK_LENGTH),
                        "masked_before": int(masked_before), "first_local_forward": bool(local_nfe == 1),
                        "previous_fallback": bool(trace[-2]["fallback_used"] if len(trace) >= 2 and
                                                   int(trace[-2]["block_idx"]) == block_idx else False),
                        "fger_high": int(rel_pos in set(pre_observation.get("fger_high_positions_relative", ()))),
                        "fger_low": int(rel_pos in set(pre_observation.get("fger_low_positions_relative", ()))),
                        "cold_start": int(not local_history),
                        "confidence_history_full_before": full_history,
                        "confidence_history_in_block_before": local_history,
                        "confidence_before_t": local_history[-1] if local_history else None,
                        "confidence_prev": local_history[-2] if len(local_history) >= 2 else None,
                        "confidence_slope": (None if len(local_history) < 2 else
                                              (local_history[-1] - local_history[0]) / max(1, len(local_history) - 1)),
                        "confidence_variance": (None if not local_history else
                                                float(np.var(np.asarray(local_history, dtype=np.float64)))),
                        "confidence_delta": (local_history[-1] - local_history[-2] if len(local_history) >= 2 else None),
                        "confidence_history_count": len(local_history),
                        "confidence_last": local_history[-1] if local_history else None,
                        "confidence_rank_previous": (rank_histories[int(p)][-1]
                                                       if rank_histories[int(p)] else None),
                        "confidence_rank_history_slope": (None if len(rank_histories[int(p)]) < 2 else
                                                           (rank_histories[int(p)][-1] - rank_histories[int(p)][0]) /
                                                           max(1, len(rank_histories[int(p)]) - 1)),
                        "confidence_rank_history_variance": (None if not rank_histories[int(p)] else
                                                             float(np.var(np.asarray(rank_histories[int(p)], dtype=np.float64)))),
                        "confidence_volatility": (None if not local_history else
                                                   float(np.std(np.asarray(local_history, dtype=np.float64)))),
                        "feature_available": {
                            "attention": bool(pre_observation.get("past_features_by_position", {}).get(str(abs_pos), {}).get("attention")),
                            "router": bool(pre_observation.get("past_features_by_position", {}).get(str(abs_pos), {}).get("router")),
                            "community": bool(pre_observation.get("past_features_by_position", {}).get(str(abs_pos), {}).get("community")),
                        },
                    })
                    past = pre_observation.get("past_features_by_position", {}).get(str(abs_pos), {})
                    for feature_name, value in past.get("features", {}).items():
                        position_rows[-1][feature_name] = value
                observe_start = time.perf_counter()
                observational_observer.end_forward(event, position_rows, router_records)
                observer_seconds += time.perf_counter() - observe_start
                if output_budget_check is not None:
                    output_budget_check()

            del outputs, target_pred, target_conf
            if restored_this_forward:
                resume_state = None
                replay_first_forward = False
                restored_local_nfe = 0
            elif replay_first_forward:
                # The state is consumed exactly once, at its saved global/local NFE.
                resume_state = None
                replay_first_forward = False
                restored_local_nfe = 0

    trace_df = pd.DataFrame(trace)

    avg_k = total_pairs / (
        NUM_MOE_LAYERS
        * total_len
        * global_nfe
    )

    elapsed = float(time.perf_counter() - run_start)
    meta = {
        "mode": mode,
        "prompt_length": prompt_len,
        "sequence_length": total_len,
        "total_nfe": global_nfe,
        "total_pairs": int(total_pairs),
        "pairs_m": float(total_pairs / 1e6),
        "avg_k": float(avg_k),
        "generation_seconds": float(prefix_seconds + elapsed),
        "cuda_event_forward_seconds": float(total_cuda_seconds) if cuda_event_profile else None,
    }
    if observational_observer is not None:
        meta["observation_callback_seconds"] = float(observer_seconds)
    if allocation_provider is not None or resume_state is not None or before_forward is not None:
        meta["continuation_seconds"] = elapsed
        meta["shared_prefix_seconds"] = prefix_seconds

    return x, trace_df, meta


def validate_trace(
    trace_df,
    mode,
):
    errors = []

    for b in range(
        NUM_BLOCKS
    ):
        g = (
            trace_df[
                trace_df[
                    "block_idx"
                ]
                == b
            ]
            .sort_values(
                "local_nfe"
            )
        )

        if len(g) == 0:
            errors.append(
                f"block {b}: empty trace"
            )
            continue

        if int(
            g[
                "num_committed"
            ].sum()
        ) != BLOCK_LENGTH:
            errors.append(
                f"block {b}: !=32 committed"
            )

        if int(
            g.iloc[-1][
                "masked_after"
            ]
        ) != 0:
            errors.append(
                f"block {b}: masks remain"
            )

        if not (
            g[
                "masked_after"
            ]
            <
            g[
                "masked_before"
            ]
        ).all():
            errors.append(
                f"block {b}: a forward made no progress"
            )

        if mode in {
            MODE_FGER,
            MODE_REFLEX,
        }:
            if not (
                g[
                    "fger_high_count"
                ]
                ==
                g[
                    "fger_low_count"
                ]
            ).all():
                errors.append(
                    f"block {b}: unequal FGER high/low groups"
                )

    if mode == MODE_FGER:
        if not np.allclose(
            trace_df[
                "forward_mean_k"
            ],
            8.0,
            rtol=0,
            atol=1e-12,
        ):
            errors.append(
                "FGER-only per-forward AvgK changed from 8"
            )

    allowed = {
        "fixed8": {
            8,
        },
        MODE_RRB: {
            4,
            8,
            12,
        },
        MODE_FGER: {
            6,
            8,
            10,
        },
        MODE_REFLEX: {
            4,
            8,
            10,
            12,
            14,
        },
    }[
        mode
    ]

    k_cols = [
        c
        for c in trace_df.columns
        if re.fullmatch(
            r"k\d+_count",
            c,
        )
    ]

    used_k = set()

    for col in k_cols:
        if (
            trace_df[
                col
            ].sum()
            > 0
        ):
            used_k.add(
                int(
                    re.findall(
                        r"\d+",
                        col,
                    )[0]
                )
            )

    if not used_k.issubset(
        allowed
    ):
        errors.append(
            f"{mode}: unexpected observed K set {sorted(used_k)}"
        )

    if errors:
        raise RuntimeError(
            "\n".join(
                errors
            )
        )


def postprocess_generated_region(generated_ids, tokenizer):
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    stopped = raw_text
    for stop_seq in STOP_SEQUENCES:
        if stop_seq in stopped: stopped = stopped.split(stop_seq)[0]
    clean = tokenizer.decode(tokenizer(stopped)["input_ids"], skip_special_tokens=True)
    return raw_text, clean
