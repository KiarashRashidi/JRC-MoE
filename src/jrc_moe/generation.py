"""Threshold decoder, routing-mode inference, and trace gates."""
from collections import defaultdict
import math, re, time
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
):
    prompt_ids = prompt_ids.to(resolve_input_device(model))

    prompt_len = int(prompt_ids.shape[1])
    total_len = prompt_len + GEN_LENGTH

    x = torch.full(
        (1, total_len),
        MASK_ID,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    x[:, :prompt_len] = prompt_ids

    # absolute target position -> prior confidence observations
    confidence_history = defaultdict(list)

    trace = []
    total_pairs = 0
    global_nfe = 0

    run_start = time.perf_counter()

    for block_idx in range(NUM_BLOCKS):
        abs_start = prompt_len + block_idx * BLOCK_LENGTH
        abs_end = abs_start + BLOCK_LENGTH
        local_nfe = 0

        while True:
            active_tokens = x[0, abs_start:abs_end]
            active_mask = active_tokens == MASK_ID

            masked_before = int(active_mask.sum())

            if masked_before == 0:
                break

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

            # Allocation is determined before this model forward.
            k_counts, fger_meta = allocation_for_forward(
                mode,
                prompt_len,
                block_idx,
                unresolved_abs,
                confidence_history,
            )

            controller.begin_forward(k_counts)

            t0 = time.perf_counter()
            outputs = model(x)
            forward_seconds = time.perf_counter() - t0

            pair_meta = controller.end_forward()
            direct_pairs = int(pair_meta["direct_pairs"])
            total_pairs += direct_pairs

            target_pred, target_conf = (
                extract_target_predictions_and_confidence(
                    outputs.logits,
                    prompt_len,
                )
            )

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

            threshold_local = [
                p
                for p in unresolved_local
                if float(active_conf[p]) >= THETA
            ]

            fallback_used = False

            if threshold_local:
                commit_local = threshold_local
            else:
                fallback_used = True
                best_local = max(
                    unresolved_local,
                    key=lambda p: float(active_conf[p]),
                )
                commit_local = [best_local]

            for p in commit_local:
                x[0, abs_start + p] = active_pred[p].to(x.device)

            masked_after = int(
                (x[0, abs_start:abs_end] == MASK_ID).sum()
            )

            vals, freq = torch.unique(
                k_counts,
                return_counts=True,
            )
            k_hist = dict(zip(vals.tolist(), freq.tolist()))

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
                "forward_mean_k": float(k_counts.float().mean()),
                "per_layer_budget": int(k_counts.sum()),
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
            })

            del outputs, target_pred, target_conf

    trace_df = pd.DataFrame(trace)

    avg_k = total_pairs / (
        NUM_MOE_LAYERS
        * total_len
        * global_nfe
    )

    meta = {
        "mode": mode,
        "prompt_length": prompt_len,
        "sequence_length": total_len,
        "total_nfe": global_nfe,
        "total_pairs": int(total_pairs),
        "pairs_m": float(total_pairs / 1e6),
        "avg_k": float(avg_k),
        "generation_seconds": float(
            time.perf_counter() - run_start
        ),
    }

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
