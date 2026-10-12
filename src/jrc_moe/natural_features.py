"""Sparse, time-stamped natural REFLEX router and post-RoPE QK observations.

Attention rows are scaled QK scores at selected edges, not probabilities. They are
recorded at forward t and exposed to predictors no earlier than decision t+1.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from statistics import mean, pstdev
from typing import Mapping, Sequence

from .jrc_v4.features import positional_landmarks


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _rank01(values: Mapping[int, float]) -> dict[int, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) <= 1:
        return {key: 0.5 for key, _ in ordered}
    output = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        mid = (start + end - 1) / 2
        rank = mid / (len(ordered) - 1)
        for key, _ in ordered[start:end]:
            output[key] = float(rank)
        start = end
    return output


def _router_summary(records: Mapping[tuple[int, int], Mapping], position: int,
                    previous: Mapping[tuple[int, int], Sequence[tuple[int, ...]]]) -> dict:
    selected = []
    stats = []
    overlaps = []
    histories = {}
    for (layer, pos), record in sorted(records.items()):
        if int(pos) != int(position):
            continue
        experts = tuple(int(e) for e in record["experts"])
        selected.extend((int(layer), expert) for expert in experts)
        stats.append(record["router_statistics"])
        prior = previous.get((int(layer), int(position)), ())
        if prior:
            overlaps.append(_jaccard(set(experts), set(prior[-1])))
        histories[(int(layer), int(position))] = experts
    if not stats:
        return {}
    def avg(name: str) -> float:
        return float(mean(float(row[name]) for row in stats))
    return {
        "router_entropy": avg("router_entropy"),
        "router_admission_margin": avg("admission_margin"),
        "router_top_k_gate_mass": avg("top_k_gate_mass"),
        "router_next_expert_gate": avg("next_expert_gate"),
        "router_preference_overlap": float(mean(overlaps)) if overlaps else None,
        "router_feature_age": 1,
        "router_selected_experts": tuple(sorted(set(selected))),
        "_router_histories": histories,
    }


def _attention_summary(relations: Sequence[Mapping], position: int,
                       eligible_positions: Sequence[int]) -> dict:
    by_key: dict[int, list[float]] = {}
    scores = []
    categories = {"unresolved_active": [], "already_committed_target": [],
                  "prompt": [], "key_to_committed": [], "key_to_prompt": []}
    eligible_set = set(map(int, eligible_positions))
    for row in relations:
        if int(row["focus_position_abs"]) != int(position):
            continue
        value = float(row["scaled_qk_score"])
        scores.append(value)
        category = row["edge_category"]
        if row["direction"] == "unresolved_to_key" and category in categories:
            categories[category].append(value)
        elif row["direction"] == "key_to_unresolved" and category in {"already_committed_target", "prompt"}:
            categories["key_to_committed" if category == "already_committed_target" else "key_to_prompt"].append(value)
        if (row["direction"] == "unresolved_to_key" and category == "unresolved_active"
                and int(row["key_position_abs"]) in eligible_set):
            by_key.setdefault(int(row["key_position_abs"]), []).append(value)
    if not scores:
        return {}
    key_means = {key: mean(values) for key, values in by_key.items()}
    ranked = _rank01(key_means)
    return {
        "attention_raw_mean": float(mean(scores)),
        "attention_raw_absmean": float(mean(abs(value) for value in scores)),
        "attention_raw_std": float(pstdev(scores)) if len(scores) > 1 else 0.0,
        "attention_relation_count": int(len(scores)),
        "attention_to_unresolved_mean": float(mean(categories["unresolved_active"])) if categories["unresolved_active"] else None,
        "attention_to_committed_mean": float(mean(categories["already_committed_target"])) if categories["already_committed_target"] else None,
        "attention_to_prompt_mean": float(mean(categories["prompt"])) if categories["prompt"] else None,
        "attention_from_committed_mean": float(mean(categories["key_to_committed"])) if categories["key_to_committed"] else None,
        "attention_from_prompt_mean": float(mean(categories["key_to_prompt"])) if categories["key_to_prompt"] else None,
        "attention_feature_age": 1,
        "_attention_peer_rank": ranked,
    }


class NaturalFeatureObserver:
    """Persist sparse native trajectory observations with strict one-forward lag."""

    def __init__(self, transaction, *, prompt_len: int, router_layers=(4, 12),
                 router_stride: int = 1, attention_layers=(4, 12),
                 attention_heads=(0, 4, 8, 12), landmark_count: int = 16,
                 attention_stride: int = 1,
                 attention_audit: str | Path | Mapping | None = None,
                 attention_enabled: bool = True, community_enabled: bool = False,
                 max_forward_record_bytes: int = 128 * 1024 * 1024):
        if router_stride < 1 or attention_stride < 1 or landmark_count < 1:
            raise ValueError("capture strides and landmark_count must be positive")
        self.transaction = transaction
        self.prompt_len = int(prompt_len)
        self.router_layers = tuple(sorted(set(map(int, router_layers))))
        self.router_stride = int(router_stride)
        self.attention_stride = int(attention_stride)
        self.attention_layers = tuple(sorted(set(map(int, attention_layers)))) if attention_enabled else ()
        if self.attention_layers and self.router_layers and router_stride != attention_stride:
            raise ValueError("joint feature capture currently requires aligned router/attention sampling strides")
        self.attention_heads = tuple(sorted(set(map(int, attention_heads))))
        self.landmark_count = int(landmark_count)
        self.max_forward_record_bytes = int(max_forward_record_bytes)
        if self.max_forward_record_bytes < 1:
            raise ValueError("max_forward_record_bytes must be positive")
        self.community_enabled = bool(community_enabled)
        self.attention_audit = self._load_audit(attention_audit) if attention_enabled else None
        if attention_enabled:
            self._validate_attention_audit(self.attention_audit)
        self._torch = None
        self._functional = None
        self._original_sdpa = None
        self._attention_module_count = 0
        self._active = False
        self._capture_attention_now = False
        self._call_idx = 0
        self._pre = None
        self._relations = []
        self._attention_contract_rows = []
        self._current_by_position = {}
        self._last_features = {}
        self._router_history = {}
        self.events = []

    @staticmethod
    def _load_audit(value):
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        import json
        return json.loads(Path(value).read_text(encoding="utf-8"))

    @staticmethod
    def _validate_attention_audit(audit):
        if not isinstance(audit, Mapping) or not audit.get("passed"):
            raise RuntimeError("attention features require a successful saved runtime audit")
        from .config import MODEL_REVISION
        if audit.get("checkpoint_revision") != MODEL_REVISION:
            raise RuntimeError("attention audit checkpoint revision does not match this model")
        if audit.get("loaded_backend") != "sdpa" or not audit.get("runtime_mask_is_none_and_is_causal_false"):
            raise RuntimeError("attention audit did not verify full bidirectional SDPA semantics")
        if not audit.get("post_rope_query_key_input_match"):
            raise RuntimeError("attention audit did not match post-RoPE Q/K to SDPA inputs")

    def install(self, model):
        """Patch SDPA only after the separate source/runtime attention audit passed."""
        if self.attention_layers and self._original_sdpa is not None:
            raise RuntimeError("attention observer is already installed")
        if not self.attention_layers:
            return
        import torch.nn.functional as functional
        modules = [(name, module) for name, module in model.named_modules()
                   if hasattr(module, "q_proj") and hasattr(module, "k_proj")]
        if not modules:
            raise RuntimeError("cannot identify attention modules for sparse QK capture")
        if max(self.attention_layers) >= len(modules):
            raise ValueError("selected attention layer is outside the audited model")
        source_path = inspect.getsourcefile(type(modules[0][1]))
        if source_path:
            current_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
            if current_hash != self.attention_audit.get("source_sha256"):
                raise RuntimeError("attention implementation differs from the source file in the saved audit")
        self._torch = __import__("torch")
        self._functional = functional
        self._attention_module_count = len(modules)
        self._original_sdpa = functional.scaled_dot_product_attention
        observer = self
        original = self._original_sdpa

        def sdpa_capture(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, **kwargs):
            index = observer._call_idx
            observer._call_idx += 1
            actual = original(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, **kwargs)
            if observer._active and observer._capture_attention_now and index in observer.attention_layers:
                observer._capture_attention(index, query, key, attn_mask,
                                            dropout_p, is_causal, scale)
            return actual

        functional.scaled_dot_product_attention = sdpa_capture
        return {"attention_module_count": len(modules),
                "attention_module_names": [name for name, _ in modules],
                "selected_layers": self.attention_layers,
                "source_sha256": self.attention_audit.get("source_sha256")}

    def close(self):
        if self._original_sdpa is not None:
            self._functional.scaled_dot_product_attention = self._original_sdpa
            self._original_sdpa = None
        self._active = False

    def _capture_attention(self, layer_idx, query, key, attn_mask, dropout_p, is_causal, scale):
        if query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1:
            raise RuntimeError("sparse attention capture expects [1, heads, sequence, head_dim]")
        q_len, k_len = int(query.shape[-2]), int(key.shape[-2])
        active_abs = tuple(int(p) for p in self._pre["eligible_positions_abs"])
        masked_abs = set(map(int, self._pre["pre_forward_masked_positions_abs"]))
        query_positions = tuple(p for p in active_abs if 0 <= p < q_len)
        if not query_positions:
            return
        landmarks = positional_landmarks(k_len, self.landmark_count)
        key_positions = tuple(sorted(set(landmarks) | {p for p in active_abs if 0 <= p < k_len}))
        heads = tuple(h for h in self.attention_heads if h < int(query.shape[1]))
        if not heads:
            raise RuntimeError("none of the requested attention heads exist")
        q_index = self._torch.as_tensor(query_positions, dtype=self._torch.long, device=query.device)
        k_index = self._torch.as_tensor(key_positions, dtype=self._torch.long, device=key.device)
        h_index_q = self._torch.as_tensor(heads, dtype=self._torch.long, device=query.device)
        h_index_k = self._torch.as_tensor(heads, dtype=self._torch.long, device=key.device)
        q = query[0].index_select(0, h_index_q).index_select(1, q_index).float()
        k = key[0].index_select(0, h_index_k).index_select(1, k_index).float()
        used_scale = float(scale) if scale is not None else float(query.shape[-1] ** -0.5)
        scores = self._torch.einsum("hqd,hkd->hqk", q, k) * used_scale
        score_rows = scores.detach().cpu().tolist()
        q_full = query[0].index_select(0, h_index_q).float()
        k_full = key[0].index_select(0, h_index_k).float()
        is_none = attn_mask is None
        contract = {"layer_idx": int(layer_idx), "attn_mask_is_none": bool(is_none),
                    "is_causal": bool(is_causal), "dropout_p": float(dropout_p),
                    "scale": used_scale, "query_shape": tuple(int(x) for x in query.shape),
                    "key_shape": tuple(int(x) for x in key.shape)}
        self._attention_contract_rows.append(contract)
        if not is_none or is_causal or float(dropout_p) != 0.0:
            raise RuntimeError(f"runtime SDPA contract differs from the audited bidirectional path: {contract}")
        local = int(self._pre["local_nfe"])
        global_nfe = int(self._pre["global_nfe"])
        for hi, head in enumerate(heads):
            for qi, qpos in enumerate(query_positions):
                current = self._current_by_position.setdefault(int(qpos), {"attention": [], "router": {}})
                for ki, kpos in enumerate(key_positions):
                    edge_category = ("prompt" if kpos < self.prompt_len else
                                     "unresolved_active" if kpos in active_abs else
                                     "unresolved_other_masked" if kpos in masked_abs else
                                     "already_committed_target")
                    score = float(score_rows[hi][qi][ki])
                    row = {
                        "run_id": self._pre.get("run_id"), "prompt_id": self._pre.get("prompt_id"),
                        "mode": self._pre.get("mode"), "block_idx": int(self._pre["block_idx"]),
                        "layer_idx": int(layer_idx), "observed_layer": int(layer_idx),
                        "head_idx": int(head), "head": int(head),
                        "global_nfe": global_nfe, "local_nfe": local,
                        "observed_nfe": global_nfe, "predicted_nfe": global_nfe + 1,
                        "observation_inclusion_probability": 1.0 / self.attention_stride,
                        "inverse_observation_weight": float(self.attention_stride),
                        "source_local_nfe": local, "feature_available_from_local_nfe": local + 1,
                        "query_position_abs": int(qpos), "query_position_relative": int(qpos) - self.prompt_len,
                        "key_position_abs": int(kpos), "key_position_relative": int(kpos) - self.prompt_len,
                        "focus_position_abs": int(qpos), "direction": "unresolved_to_key",
                        "edge_category": edge_category,
                        "query_was_masked": True, "key_was_masked": bool(kpos in masked_abs),
                        "available_position_mask": edge_category,
                        "scaled_qk_score": score, "raw_qk_score": score,
                        "score_semantics": "post_qk_norm_post_rope_scaled_dot_product_score",
                        "is_probability": False, "attention_mask_is_none": bool(is_none),
                        "is_causal": bool(is_causal), "dropout_p": float(dropout_p),
                        "scale": used_scale, "all_key_positions_visible": True,
                    }
                    self._relations.append(row)
                    current["attention"].append(row)
                    # Record the reverse edge for sampled prompt/committed keys;
                    # active unresolved pairs already have both directed rows.
                    if kpos not in active_abs and kpos < q_len and edge_category in {"prompt", "already_committed_target"}:
                        reverse = float((q_full[hi, kpos] * k_full[hi, qpos]).sum().item() * used_scale)
                        reverse_row = {
                            **{key: value for key, value in row.items()
                               if key not in {"query_position_abs", "query_position_relative",
                                              "key_position_abs", "key_position_relative",
                                              "focus_position_abs", "direction", "query_was_masked",
                                              "key_was_masked", "available_position_mask",
                                              "scaled_qk_score", "raw_qk_score"}},
                            "query_position_abs": int(kpos), "query_position_relative": int(kpos) - self.prompt_len,
                            "key_position_abs": int(qpos), "key_position_relative": int(qpos) - self.prompt_len,
                            "focus_position_abs": int(qpos), "direction": "key_to_unresolved",
                            "edge_category": edge_category,
                            "query_was_masked": bool(kpos in masked_abs), "key_was_masked": True,
                            "available_position_mask": f"{edge_category}_to_unresolved",
                            "scaled_qk_score": reverse, "raw_qk_score": reverse,
                        }
                        self._relations.append(reverse_row)
                        current["attention"].append(reverse_row)

    def features_for_positions(self, *, block_idx: int, local_nfe: int,
                                positions: Sequence[int], prompt_len: int) -> dict[str, dict]:
        result = {}
        for position in positions:
            cached = self._last_features.get(int(position))
            if cached is None or int(cached["block_idx"]) != int(block_idx):
                result[str(int(position))] = {}
                continue
            age = int(local_nfe) - int(cached["source_local_nfe"])
            if age < 1:
                raise RuntimeError("feature timestamp is contemporaneous or from the future")
            features = dict(cached["features"])
            for channel in ("attention", "router", "community"):
                age_name = f"{channel}_feature_age"
                if age_name in features:
                    features[age_name] = age
            features["feature_source_local_nfe"] = int(cached["source_local_nfe"])
            result[str(int(position))] = {"features": features,
                                           "attention": "attention_raw_mean" in features,
                                           "router": "router_entropy" in features,
                                           "community": "community_size" in features}
        return result

    def begin_forward(self, pre_observation: Mapping):
        if self._active:
            raise RuntimeError("observer received overlapping forward events")
        self._pre = dict(pre_observation)
        self._current_by_position = {}
        self._relations = []
        self._attention_contract_rows = []
        self._call_idx = 0
        local_nfe = int(self._pre["local_nfe"])
        self._capture_attention_now = bool(self.attention_layers) and ((local_nfe - 1) % self.attention_stride) == 0
        self._active = True

    def end_forward(self, event: Mapping, position_rows: Sequence[Mapping],
                    router_records: Mapping[tuple[int, int], Mapping]):
        if not self._active:
            raise RuntimeError("observer end_forward without begin_forward")
        try:
            if self.attention_layers and self._call_idx != self._attention_module_count:
                raise RuntimeError(f"captured {self._call_idx} SDPA calls; expected {self._attention_module_count}")
            identity = {"run_id": event.get("run_id"), "prompt_id": event.get("prompt_id"),
                        "mode": event.get("mode"), "block_idx": event.get("block_idx"),
                        "global_nfe": event.get("global_nfe"), "local_nfe": event.get("local_nfe")}
            event_record = {**dict(event),
                            "attention_contract": self._attention_contract_rows,
                            "attention_capture_call_count": self._call_idx,
                            "attention_sampled": bool(self._capture_attention_now),
                            "attention_sampling_weight": (float(self.attention_stride)
                                                          if self.attention_layers else None)}
            self.events.append(dict(event))
            current_rel_positions = tuple(map(int, event["eligible_positions_abs"]))
            router_rows = []
            for (layer, position), record in sorted(router_records.items()):
                row = {**identity, "layer_idx": int(layer), "position_absolute": int(position),
                       "position_relative": int(position) - self.prompt_len,
                       "observed_layer": int(layer), "observed_nfe": int(event["global_nfe"]),
                       "predicted_nfe": int(event["global_nfe"]) + 1,
                       "source_local_nfe": int(event["local_nfe"]),
                       "predicted_local_nfe": int(event["local_nfe"]) + 1,
                       "observation_inclusion_probability": 1.0 / self.router_stride,
                       "inverse_observation_weight": float(self.router_stride),
                       "feature_available_from_local_nfe": int(event["local_nfe"]) + 1,
                       "feature_type": "pre_admission_router_logits_and_topk",
                       "experts": tuple(record["experts"]), "weights": tuple(record["weights"]),
                       "router_logits": tuple(record["router_logits"]),
                       "selection_scores": tuple(record["selection_scores"]),
                       "expert_bias_enabled": bool(record["expert_bias_enabled"]),
                       "expert_ids_layer_local": True,
                       "router_statistics": dict(record["router_statistics"])}
                router_rows.append(row)
                slot = self._current_by_position.setdefault(int(position), {"attention": [], "router": {}})
                slot["router"][int(layer)] = dict(record)

            router_current = {}
            for key, value in router_records.items():
                router_current[(int(key[0]), int(key[1]))] = value
            attention_by_pos = {}
            for position in self._current_by_position:
                slot = self._current_by_position[position]
                attention_by_pos[position] = slot["attention"]
            current_cache = {}
            for position in current_rel_positions:
                features = {}
                attention = _attention_summary(attention_by_pos.get(position, ()), position,
                                               current_rel_positions)
                router = _router_summary(router_current, position, self._router_history)
                router_histories = router.pop("_router_histories", {})
                features.update(attention)
                features.update(router)
                features.pop("_attention_peer_rank", None)
                peer_rank = attention.get("_attention_peer_rank", {})
                selected_by_layer = {}
                for (layer, pos), record in router_current.items():
                    if int(pos) == int(position):
                        selected_by_layer[int(layer)] = set(map(int, record["experts"]))
                router_similarity = {}
                for peer in current_rel_positions:
                    if peer == position or peer not in peer_rank:
                        continue
                    peer_layers = {int(layer): set(map(int, record["experts"]))
                                   for (layer, pos), record in router_current.items() if int(pos) == int(peer)}
                    shared_layers = sorted(set(selected_by_layer) & set(peer_layers))
                    if shared_layers:
                        router_similarity[peer] = mean(_jaccard(selected_by_layer[layer], peer_layers[layer])
                                                       for layer in shared_layers)
                router_rank = _rank01(router_similarity)
                channel_rows = [(attention_sim, router_rank[peer])
                                for peer, attention_sim in peer_rank.items()
                                if peer in router_rank]
                if channel_rows:
                    features["attention_router_agreement"] = float(mean(1.0 - abs(a - r) for a, r in channel_rows))
                    features["attention_only_channel"] = float(mean(a * (1.0 - r) for a, r in channel_rows))
                    features["router_only_channel"] = float(mean((1.0 - a) * r for a, r in channel_rows))
                    features["neither_channel"] = float(mean((1.0 - a) * (1.0 - r) for a, r in channel_rows))
                for feature_name in ("_attention_peer_rank", "_router_histories", "router_selected_experts"):
                    features.pop(feature_name, None)
                if not features:
                    previous = self._last_features.get(position)
                    if previous is not None and previous["block_idx"] == int(event["block_idx"]):
                        current_cache[position] = previous
                    else:
                        current_cache[position] = {"block_idx": int(event["block_idx"]),
                                                   "source_local_nfe": int(event["local_nfe"]),
                                                   "features": {}}
                else:
                    current_cache[position] = {"block_idx": int(event["block_idx"]),
                                               "source_local_nfe": int(event["local_nfe"]),
                                               "features": features}
                for key, experts in router_histories.items():
                    self._router_history.setdefault(key, []).append(tuple(experts))
            for position, cache in current_cache.items():
                self._last_features[position] = cache

            # The generation layer adds current post-forward outcomes to these
            # past-only rows for descriptive analysis; predictor columns are
            # explicitly limited to pre-decision fields in natural_prediction.
            all_records = [event_record, *router_rows, *self._relations, *position_rows]
            self.transaction.preflight_forward(all_records, max_bytes=self.max_forward_record_bytes)
            self.transaction.write_event(event_record)
            for row in router_rows:
                self.transaction.write_relation(row)
            for row in self._relations:
                self.transaction.write_relation(row)
            for row in position_rows:
                self.transaction.write_position(row)
            self.transaction.flush()
        finally:
            self._active = False
            self._capture_attention_now = False
            self._pre = None

