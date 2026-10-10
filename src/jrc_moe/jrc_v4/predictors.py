"""Low-capacity prompt-disjoint linear predictors for measured exchange effects."""
from __future__ import annotations

import numpy as np


class RidgeBenefitPredictor:
    """Ridge regression with train-only scaling and an explicit prompt split gate."""

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.training_prompts_: frozenset[str] = frozenset()

    def fit(self, features, labels, prompt_ids, *, allowed_training_prompts):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        ids = tuple(str(i) for i in prompt_ids)
        allowed = frozenset(str(i) for i in allowed_training_prompts)
        if x.ndim != 2 or len(x) != len(y) or len(y) != len(ids) or not len(y):
            raise ValueError("feature, label, and prompt arrays must align and be nonempty")
        if set(ids) - allowed:
            raise RuntimeError("training data includes a validation or confirmatory prompt")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training values must be finite")
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        z = (x - self.mean_) / self.scale_
        self.coef_ = np.linalg.solve(z.T @ z + self.alpha * np.eye(z.shape[1]), z.T @ y)
        self.training_prompts_ = frozenset(ids)
        return self

    def predict(self, features, *, prompt_ids=None):
        if self.coef_ is None:
            raise RuntimeError("predictor has not been fit")
        if prompt_ids is not None and self.training_prompts_ & {str(i) for i in prompt_ids}:
            raise RuntimeError("reporting held-out evaluation on a prompt used for fitting")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.coef_) or not np.isfinite(x).all():
            raise ValueError("prediction feature shape or values invalid")
        return ((x - self.mean_) / self.scale_) @ self.coef_


def exchange_design(recipient: np.ndarray, donor: np.ndarray, *, action_k: int, layer_idx: int) -> np.ndarray:
    """Explicit recipient, donor, difference, and local cross-product terms."""
    r = np.asarray(recipient, dtype=np.float64).reshape(-1)
    d = np.asarray(donor, dtype=np.float64).reshape(-1)
    if r.shape != d.shape or not np.isfinite(r).all() or not np.isfinite(d).all():
        raise ValueError("recipient/donor features must be aligned and finite")
    return np.concatenate((r, d, r - d, r * d, [float(action_k), float(layer_idx)]))
