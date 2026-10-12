"""Small prompt-disjoint logistic baselines for future commitment events."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

M0 = ("block_idx", "block_progress", "position_fraction", "active_mask_ratio", "masked_before",
      "prompt_length", "mode_fixed8", "mode_reflex", "first_local_forward",
      "previous_fallback", "fger_high", "fger_low", "cold_start")
M1 = M0 + ("confidence_last", "confidence_rank_previous")
M2 = M1 + ("confidence_prev", "confidence_slope", "confidence_variance", "confidence_delta",
           "confidence_history_count", "confidence_volatility",
           "confidence_rank_history_slope", "confidence_rank_history_variance")
ATTENTION = ("attention_raw_mean", "attention_raw_absmean", "attention_raw_std",
             "attention_relation_count", "attention_to_unresolved_mean",
             "attention_to_committed_mean", "attention_to_prompt_mean",
             "attention_from_committed_mean", "attention_from_prompt_mean",
             "attention_feature_age")
ROUTER = ("router_entropy", "router_admission_margin", "router_top_k_gate_mass",
          "router_next_expert_gate", "router_preference_overlap", "router_feature_age")
JOINT = ("attention_router_agreement", "attention_only_channel", "router_only_channel",
         "neither_channel")
COMMUNITY = ("community_size", "community_mean_confidence", "community_router_entropy",
             "community_attention_affinity", "community_router_affinity", "community_age")
FEATURES = {"M0": M0, "M1": M1, "M2": M2,
            "M3": M2 + ATTENTION, "M4": M2 + ROUTER,
            "M5": M2 + ATTENTION + ROUTER + JOINT,
            "M6": M2 + ATTENTION + ROUTER + JOINT + COMMUNITY}
REQUIRES = {"M0": (), "M1": (), "M2": (), "M3": ("attention",),
            "M4": ("router",), "M5": ("attention", "router"),
            "M6": ("attention", "router", "community")}


def _number(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _matrix(rows: Sequence[Mapping], columns: Sequence[str]):
    raw = np.asarray([[_number(r.get(name, np.nan)) for name in columns] for r in rows], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] == 0:
        raise ValueError("feature matrix is empty")
    observed = np.isfinite(raw)
    means = np.asarray([raw[observed[:, j], j].mean() if observed[:, j].any() else 0.0
                        for j in range(raw.shape[1])])
    filled = np.where(observed, raw, means)
    scale = np.asarray([filled[:, j].std() for j in range(filled.shape[1])])
    scale[scale < 1e-12] = 1.0
    return np.concatenate(((filled - means) / scale, (~observed).astype(np.float64)), axis=1), means, scale


def fit_logistic_ridge(x: np.ndarray, y: Sequence[int], *, l2: float = 1.0,
                       learning_rate: float = 0.2, max_iter: int = 400) -> dict:
    """Deterministic full-batch logistic regression; standardize from training rows only."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(y) != len(x) or not len(y):
        raise ValueError("training arrays have inconsistent or empty rows")
    if not np.isfinite(x).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("training data must be finite and binary")
    classes = np.unique(y)
    if len(classes) < 2:
        return {"status": "undefined_constant_labels", "classes": classes.tolist()}
    if l2 < 0 or learning_rate <= 0 or max_iter < 1:
        raise ValueError("invalid optimizer settings")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    z = np.column_stack((np.ones(len(z)), z))
    coef = np.zeros(z.shape[1], dtype=np.float64)
    penalty = np.ones(z.shape[1], dtype=np.float64)
    penalty[0] = 0.0
    # Frobenius norm bounds the spectral norm and avoids a costly SVD on large
    # position-level tables while keeping the full-batch step stable.
    lipschitz = float(np.sum(z * z) / len(z) + l2)
    lr = min(float(learning_rate), 1.0 / max(1.0, lipschitz))
    for _ in range(max_iter):
        p = _sigmoid(z @ coef)
        grad = z.T @ (p - y) / len(y) + l2 * penalty * coef
        step = lr * grad
        coef -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return {"status": "fitted", "coef": coef, "mean": mean, "scale": scale,
            "l2": l2, "n_train": int(len(y)), "n_iter": _ + 1}


def predict_logistic(model: Mapping, x: np.ndarray) -> np.ndarray:
    if model.get("status") != "fitted":
        raise ValueError("cannot predict from an unfitted model")
    z = (x - model["mean"]) / model["scale"]
    return _sigmoid(np.column_stack((np.ones(len(z)), z)) @ model["coef"])


def _auc_roc(y: np.ndarray, score: np.ndarray) -> float | None:
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ys, ss = y[order], score[order]
    tp = fp = 0
    ap = 0.0
    last_recall = 0.0
    i = 0
    while i < len(ys):
        j = i + 1
        while j < len(ys) and ss[j] == ss[i]:
            j += 1
        group_pos = int(ys[i:j].sum())
        tp += group_pos
        fp += (j - i) - group_pos
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - last_recall) * precision
        last_recall = recall
        i = j
    return float(ap)


def binary_metrics(y: Sequence[int], probability: Sequence[float], *, top_fraction: float = 0.10) -> dict:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid binary metric input")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0,1]")
    positives = int(y.sum())
    k = max(1, int(math.ceil(top_fraction * len(y))))
    chosen = np.argsort(-p, kind="mergesort")[:k]
    return {"n": int(len(y)), "positive": positives, "base_rate": float(y.mean()),
            "auroc": _auc_roc(y, p), "auprc": _average_precision(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(np.clip(p, 1e-12, 1)) +
                                       (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))),
            "top_q": k, "top_q_precision": float(y[chosen].mean()),
            "top_q_recall": float(y[chosen].sum() / positives) if positives else None}


def prompt_averaged_metrics(y: Sequence[int], probability: Sequence[float],
                            prompt_ids: Sequence[str], *, top_fraction: float = 0.10,
                            bootstrap_draws: int = 2000, seed: int = 20261012) -> dict:
    """Report prompt-cluster means and prompt-bootstrap intervals, never token p-values."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    ids = np.asarray([str(x) for x in prompt_ids])
    if len(y) != len(p) or len(y) != len(ids) or not len(y):
        raise ValueError("prompt-averaged metric inputs do not align")
    clusters = sorted(set(ids.tolist()))
    metric_names = ("auroc", "auprc", "brier", "log_loss", "top_q_precision", "top_q_recall")
    per_prompt = {}
    for prompt in clusters:
        mask = ids == prompt
        per_prompt[prompt] = binary_metrics(y[mask], p[mask], top_fraction=top_fraction)
    result = {"n_prompts": len(clusters), "n_rows": len(y), "prompts": clusters,
              "per_prompt": per_prompt, "metrics": {}}
    rng = np.random.default_rng(seed)
    for name in metric_names:
        values = [per_prompt[prompt][name] for prompt in clusters
                  if per_prompt[prompt][name] is not None and math.isfinite(per_prompt[prompt][name])]
        if not values:
            result["metrics"][name] = {"mean": None, "ci95": None, "defined_prompts": 0}
            continue
        samples = np.asarray(values, dtype=np.float64)
        means = np.empty(bootstrap_draws, dtype=np.float64)
        for i in range(bootstrap_draws):
            means[i] = rng.choice(samples, size=len(samples), replace=True).mean()
        result["metrics"][name] = {"mean": float(samples.mean()),
                                    "ci95": [float(np.quantile(means, 0.025)),
                                             float(np.quantile(means, 0.975))],
                                    "defined_prompts": len(samples)}
    return result


def _available(row: Mapping, group: str) -> bool:
    flags = row.get("feature_available", {})
    return bool(flags.get(group, False)) if isinstance(flags, Mapping) else False


def model_rows(rows: Sequence[Mapping], model_id: str) -> list[Mapping]:
    if model_id not in FEATURES:
        raise ValueError(f"unknown model {model_id}")
    return [row for row in rows if all(_available(row, group) for group in REQUIRES[model_id])]


def compare_models(rows: Sequence[Mapping], *, prompt_split: Mapping[str, str],
                   target_key: str = "label", prompt_key: str = "prompt_id",
                   l2: float = 1.0, minimum_prompts: int = 3) -> list[dict]:
    """Compare M0-M6 with prompt-disjoint splits and explicit coverage/status rows."""
    output = []
    evaluation_cache = {}
    for model_id in FEATURES:
        subset = model_rows(rows, model_id)
        train = [r for r in subset if prompt_split.get(str(r[prompt_key])) == "development"]
        valid = [r for r in subset if prompt_split.get(str(r[prompt_key])) == "validation"]
        train_prompts = sorted({str(r[prompt_key]) for r in train})
        valid_prompts = sorted({str(r[prompt_key]) for r in valid})
        base = {"model": model_id, "feature_columns": list(FEATURES[model_id]),
                "n_total": len(rows), "n_covered": len(subset), "coverage": len(subset) / len(rows) if rows else 0.0,
                "n_train": len(train), "n_validation": len(valid),
                "train_prompts": train_prompts, "validation_prompts": valid_prompts}
        if len(train_prompts) < minimum_prompts or len(valid_prompts) < 1:
            status = ("blocked_feature_view_missing" if REQUIRES[model_id] and not subset
                      else "insufficient_prompt_clusters")
            output.append({**base, "status": status})
            continue
        columns = FEATURES[model_id]
        x_train, train_mean, train_scale = _matrix(train, columns)
        # Impute and scale validation with training-only statistics, then append missingness bits.
        def train_transform(subset_rows):
            raw = np.asarray([[_number(r.get(name, np.nan)) for name in columns] for r in subset_rows], dtype=np.float64)
            observed = np.isfinite(raw)
            filled = np.where(observed, raw, train_mean)
            return np.concatenate(((filled - train_mean) / train_scale, (~observed).astype(np.float64)), axis=1)
        x_train = train_transform(train)
        x_valid = train_transform(valid)
        y_train = np.asarray([int(r[target_key]) for r in train], dtype=np.int64)
        y_valid = np.asarray([int(r[target_key]) for r in valid], dtype=np.int64)
        fitted = fit_logistic_ridge(x_train, y_train, l2=l2)
        if fitted["status"] != "fitted":
            output.append({**base, "status": fitted["status"], "train_positive": int(y_train.sum()),
                           "validation_positive": int(y_valid.sum())})
            continue
        probs = predict_logistic(fitted, x_valid)
        metrics = binary_metrics(y_valid, probs)
        prompt_metrics = prompt_averaged_metrics(
            y_valid, probs, [str(r[prompt_key]) for r in valid],
            seed=20261012 + sum(ord(ch) for ch in model_id),
        )
        record = {**base, "status": "evaluated", "train_positive": int(y_train.sum()),
                  "validation_positive": int(y_valid.sum()), "validation_metrics": metrics,
                  "validation_prompt_averaged": prompt_metrics,
                  "fit_iterations": fitted["n_iter"], "l2": fitted["l2"]}
        evaluation_cache[model_id] = {"rows": subset, "train": train, "valid": valid,
                                      "probability": probs, "record": record}
        output.append(record)

    # Refit M2 on each augmented model's exact eligible subset and score both
    # predictors on the same held-out position rows. This avoids coverage bias.
    for record in output:
        model_id = record["model"]
        if model_id not in {"M3", "M4", "M5", "M6"} or model_id not in evaluation_cache:
            continue
        cache = evaluation_cache[model_id]
        train, valid = cache["train"], cache["valid"]
        if not valid:
            continue
        _, train_mean, train_scale = _matrix(train, FEATURES["M2"])
        def transform_m2(subset_rows):
            raw = np.asarray([[_number(r.get(name, np.nan)) for name in FEATURES["M2"]]
                              for r in subset_rows], dtype=np.float64)
            observed = np.isfinite(raw)
            filled = np.where(observed, raw, train_mean)
            return np.concatenate(((filled - train_mean) / train_scale,
                                   (~observed).astype(np.float64)), axis=1)
        x2_train = transform_m2(train)
        x2_valid = transform_m2(valid)
        y_train = np.asarray([int(r[target_key]) for r in train], dtype=np.int64)
        y_valid = np.asarray([int(r[target_key]) for r in valid], dtype=np.int64)
        baseline = fit_logistic_ridge(x2_train, y_train, l2=l2)
        if baseline["status"] != "fitted":
            record["overlap_m2_status"] = baseline["status"]
            continue
        p2 = predict_logistic(baseline, x2_valid)
        p_aug = cache["probability"]
        m2_metrics = binary_metrics(y_valid, p2)
        aug_metrics = binary_metrics(y_valid, p_aug)
        record["overlap_m2_validation_metrics"] = m2_metrics
        record["overlap_augmented_validation_metrics"] = aug_metrics
        record["overlap_delta_m2_minus_aug_log_loss"] = float(m2_metrics["log_loss"] - aug_metrics["log_loss"])
        record["overlap_delta_aug_minus_m2_auprc"] = (
            None if m2_metrics["auprc"] is None or aug_metrics["auprc"] is None
            else float(aug_metrics["auprc"] - m2_metrics["auprc"])
        )
    return output


def m6_prerequisite(*, m5_dev_logloss_gain: float | None, covered_development_prompts: int,
                    minimum_logloss_gain: float = 0.01, minimum_prompts: int = 10,
                    enough_budget: bool = True) -> tuple[bool, str]:
    """Gate communities on development-only cross-validation evidence and budget."""
    if not enough_budget:
        return False, "insufficient_remaining_budget"
    if covered_development_prompts < minimum_prompts:
        return False, "insufficient_development_prompt_clusters"
    if m5_dev_logloss_gain is None or not math.isfinite(float(m5_dev_logloss_gain)):
        return False, "M5_development_cross_validation_unavailable"
    if float(m5_dev_logloss_gain) < minimum_logloss_gain:
        return False, "M5_did_not_improve_over_M2_on_development"
    return True, "M5_development_gate_passed"
