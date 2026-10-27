"""Numerical gradient checking for parameters stored in a dict (NumPy only).

cost_fn(params) -> float is evaluated at perturbed copies of params. Gradients use the same keys as the
parameters with a "d" prefix (W1 -> dW1), as in the rest of this repository.
"""
from __future__ import annotations

import numpy as np


def numerical_gradients(cost_fn, params, eps: float = 1e-7):
    """Central differences: (J(w + eps) - J(w - eps)) / (2 * eps) for every single parameter entry."""
    grads = {}
    for key, value in params.items():
        g = np.zeros_like(value, dtype=float)
        for idx in np.ndindex(*value.shape):
            plus = {k: v.copy() for k, v in params.items()}
            minus = {k: v.copy() for k, v in params.items()}
            plus[key][idx] += eps
            minus[key][idx] -= eps
            g[idx] = (cost_fn(plus) - cost_fn(minus)) / (2 * eps)
        grads["d" + key] = g
    return grads


def relative_difference(g_num, g_ana) -> float:
    """||g_num - g_ana|| / (||g_num|| + ||g_ana||); 0 when both vanish."""
    a, b = np.asarray(g_num, dtype=float).ravel(), np.asarray(g_ana, dtype=float).ravel()
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    return 0.0 if denom == 0.0 else float(np.linalg.norm(a - b) / denom)


def gradient_check(cost_fn, params, grads, eps: float = 1e-7, threshold: float = 1e-6):
    """Compare analytic grads with numerical ones.

    Returns (difference over all parameters, {name: difference per parameter}, ok).
    Per-parameter differences point to where the backward pass is wrong.
    """
    num = numerical_gradients(cost_fn, params, eps)
    per_param = {k: relative_difference(num["d" + k], grads["d" + k]) for k in params}
    total = relative_difference(
        np.concatenate([num["d" + k].ravel() for k in params]),
        np.concatenate([np.asarray(grads["d" + k]).ravel() for k in params]),
    )
    return total, per_param, total < threshold
