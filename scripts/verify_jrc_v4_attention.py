"""Runtime audit of the pinned LLaDA-MoE attention path on a short sequence.

The probe records only shapes, numerical error summaries, and a small table of
post-QK-normalization/post-RoPE landmark scores. It never saves model activations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path


def _source(module) -> str:
    try:
        return inspect.getsource(type(module).forward)
    except (OSError, TypeError, AttributeError):
        return ""


def verify(output_path: Path, *, atol: float = 0.03) -> dict:
    import torch
    import torch.nn.functional as F
    from jrc_moe.model import input_device, load_model
    from jrc_moe.jrc_v4.audit import audit_loaded_model
    from jrc_moe.jrc_v4.features import positional_landmarks, post_rope_landmark_scores
    from jrc_moe.jrc_v4.feature_hooks import AttentionProbeContract, restricted_attention_scores

    model, tokenizer = load_model()
    model.eval()
    audit = audit_loaded_model(model)
    modules = [(name, module) for name, module in model.named_modules()
               if hasattr(module, "q_proj") and hasattr(module, "k_proj")]
    if not modules:
        raise RuntimeError("loaded checkpoint has no identifiable attention modules")
    source_module = importlib.import_module(type(modules[0][1]).__module__)
    rotate = getattr(source_module, "apply_rotary_pos_emb", None)
    if rotate is None:
        raise RuntimeError("loaded attention module does not expose apply_rotary_pos_emb")
    forward_source = _source(modules[0][1])
    qnorm_index = forward_source.find("q_norm")
    knorm_index = forward_source.find("k_norm")
    rope_index = forward_source.find("apply_rotary_pos_emb")
    source_order_ok = (qnorm_index >= 0 and knorm_index >= 0
                       and rope_index > max(qnorm_index, knorm_index))

    rotated: list[tuple[torch.Tensor, torch.Tensor]] = []
    sdpa_rows: list[dict] = []
    feature_rows: list[dict] = []
    original_rotate = source_module.apply_rotary_pos_emb
    original_sdpa = F.scaled_dot_product_attention

    def rotated_wrapper(query, key, *args, **kwargs):
        q_out, k_out = original_rotate(query, key, *args, **kwargs)
        rotated.append((q_out.detach(), k_out.detach()))
        return q_out, k_out

    def sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, **kwargs):
        index = len(sdpa_rows)
        actual = original_sdpa(query, key, value, attn_mask=attn_mask,
                               dropout_p=dropout_p, is_causal=is_causal,
                               scale=scale, **kwargs)
        qf, kf, vf = query.float(), key.float(), value.float()
        used_scale = float(scale) if scale is not None else query.shape[-1] ** -0.5
        scores = torch.matmul(qf, kf.transpose(-2, -1)) * used_scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask.float()
        if is_causal:
            qlen, klen = scores.shape[-2:]
            causal = torch.ones((qlen, klen), dtype=torch.bool, device=scores.device).tril(
                diagonal=klen - qlen)
            scores = scores.masked_fill(~causal, float("-inf"))
        reference = torch.matmul(torch.softmax(scores, dim=-1), vf)
        absolute_error = float((actual.float() - reference).abs().max().item())
        rms_error = float(torch.sqrt(torch.mean((actual.float() - reference) ** 2)).item())
        qk_match = None
        if index < len(rotated):
            q_rot, k_rot = rotated[index]
            if q_rot.shape == query.shape and k_rot.shape == key.shape:
                q_diff = float((q_rot.float() - query.float()).abs().max().item())
                k_diff = float((k_rot.float() - key.float()).abs().max().item())
                qk_match = {"query_max_abs": q_diff, "key_max_abs": k_diff,
                            "passed": q_diff <= 1e-6 and k_diff <= 1e-6}
            else:
                qk_match = {"passed": False, "rope_query_shape": list(q_rot.shape),
                            "sdpa_query_shape": list(query.shape), "rope_key_shape": list(k_rot.shape),
                            "sdpa_key_shape": list(key.shape)}
        sdpa_rows.append({
            "layer_index": index,
            "query_shape": list(query.shape), "key_shape": list(key.shape),
            "value_shape": list(value.shape), "attention_mask_is_none": attn_mask is None,
            "is_causal": bool(is_causal), "dropout_p": float(dropout_p),
            "backend_reference_max_abs_error": absolute_error,
            "backend_reference_rms_error": rms_error,
            "post_rope_query_key_match": qk_match,
        })
        if index in {4, 12} and query.shape[1] > 12:
            seq_len = int(query.shape[-2])
            landmarks = positional_landmarks(seq_len, 16)
            query_positions = sorted(set((0, seq_len // 2, seq_len - 1)))
            for head in (0, 4, 8, 12):
                for query_position in query_positions:
                    q_vector = query[0, head, query_position].detach().float().cpu().tolist()
                    all_keys = key[0, head].detach().float().cpu().tolist()
                    contract = AttentionProbeContract(
                        source="post_qk_norm_post_rope", layer_idx=index, head_idx=head,
                        checkpoint_revision="783d3467f108d28ac0a78d3e41af16ab05cabd8d",
                        reference_attention_fidelity=max(0.0, 1.0 - absolute_error / max(atol, 1e-12)),
                    )
                    values = restricted_attention_scores(q_vector, all_keys, landmarks, contract)
                    reference_values = post_rope_landmark_scores(
                        q_vector, all_keys, source="post_qk_norm_post_rope", landmark_positions=landmarks)
                    feature_rows.append({
                        "layer_idx": index, "head_idx": head, "query_position": query_position,
                        "key_landmarks": list(landmarks), "scaled_qk_scores": list(values),
                        "proxy_vs_reference_max_abs": max(abs(a - b) for a, b in zip(values, reference_values)),
                    })
        return actual

    try:
        source_module.apply_rotary_pos_emb = rotated_wrapper
        F.scaled_dot_product_attention = sdpa_wrapper
        encoded = tokenizer("A short bidirectional attention fidelity probe.", return_tensors="pt")
        input_ids = encoded["input_ids"].to(input_device(model))
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=None, use_cache=False)
    finally:
        source_module.apply_rotary_pos_emb = original_rotate
        F.scaled_dot_product_attention = original_sdpa
        del model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    source_path = inspect.getsourcefile(type(modules[0][1]))
    source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest() if source_path else None
    mask_ok = len(sdpa_rows) == len(modules) and all(
        row["attention_mask_is_none"] and not row["is_causal"] and row["dropout_p"] == 0.0
        for row in sdpa_rows)
    qk_ok = len(rotated) == len(modules) and all(
        row["post_rope_query_key_match"] and row["post_rope_query_key_match"].get("passed")
        for row in sdpa_rows)
    score_error = max((row["proxy_vs_reference_max_abs"] for row in feature_rows), default=float("inf"))
    manual_error = max((row["backend_reference_max_abs_error"] for row in sdpa_rows), default=float("inf"))
    passed = (audit.get("attention_backend") == "sdpa" and source_order_ok and mask_ok and qk_ok
              and len(feature_rows) == 24 and score_error <= 1e-6 and manual_error <= atol)
    result = {
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "checkpoint_revision": "783d3467f108d28ac0a78d3e41af16ab05cabd8d",
        "loaded_backend": audit.get("attention_backend"),
        "runtime_attention_classes": sorted({type(module).__name__ for _, module in modules}),
        "attention_module_count": len(modules), "source_path": source_path,
        "source_sha256": source_hash, "source_qk_norm_precedes_rope": source_order_ok,
        "source_indicates_full_bidirectional_sdpa": "is_causal=False" in forward_source,
        "sequence_length": int(input_ids.shape[1]), "post_rope_capture_count": len(rotated),
        "sdpa_calls": len(sdpa_rows), "runtime_mask_is_none_and_is_causal_false": mask_ok,
        "post_rope_query_key_input_match": qk_ok,
        "reference_attention_atol": atol,
        "maximum_backend_reference_error": manual_error,
        "maximum_landmark_score_proxy_error": score_error,
        "landmark_feature_rows": feature_rows,
        "attention_calls": sdpa_rows,
        "activations_persisted": False,
        "feature_note": "Scores are exact post-QK-normalization/post-RoPE scaled QK values at selected query and key landmarks; no dense attention matrix was saved.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.03)
    args = parser.parse_args()
    result = verify(args.output, atol=args.atol)
    print(json.dumps({key: result[key] for key in
                      ("status", "loaded_backend", "attention_module_count", "sequence_length",
                       "maximum_backend_reference_error", "runtime_mask_is_none_and_is_causal_false")}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
