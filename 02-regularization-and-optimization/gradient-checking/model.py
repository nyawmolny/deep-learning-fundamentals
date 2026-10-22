"""Small L-layer ReLU network: forward pass, cost and backward pass (NumPy only).

X : (n_x, m), one example per column;  Y : (1, m) with labels in {0, 1}.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def initialize_parameters(layer_dims, seed: int | None = None, random_biases: bool = False):
    """He initialization. Biases are zero unless random_biases=True (used for gradient checks)."""
    rng = np.random.default_rng(seed)
    params = {}
    for l in range(1, len(layer_dims)):
        params[f"W{l}"] = rng.standard_normal((layer_dims[l], layer_dims[l - 1])) * np.sqrt(2.0 / layer_dims[l - 1])
        params[f"b{l}"] = rng.standard_normal((layer_dims[l], 1)) if random_biases else np.zeros((layer_dims[l], 1))
    return params


def forward_propagation(X, params):
    L = len(params) // 2
    A, caches = X, []
    for l in range(1, L + 1):
        Z = params[f"W{l}"] @ A + params[f"b{l}"]
        caches.append((A, params[f"W{l}"], Z))
        A = np.maximum(0.0, Z) if l < L else 1.0 / (1.0 + np.exp(-Z))
    return A, caches


def compute_cost(AL, Y):
    A = np.clip(AL, EPS, 1.0 - EPS)
    return float(-np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A)))


def backward_propagation(AL, Y, caches):
    m = Y.shape[1]
    L = len(caches)
    grads = {}
    dZ = AL - Y
    for l in range(L, 0, -1):
        A_prev, W, _ = caches[l - 1]
        grads[f"dW{l}"] = dZ @ A_prev.T / m
        grads[f"db{l}"] = dZ.sum(axis=1, keepdims=True) / m
        if l > 1:
            dZ = (W.T @ dZ) * (caches[l - 2][2] > 0)
    return grads


def cost_from_params(params, X, Y):
    return compute_cost(forward_propagation(X, params)[0], Y)
