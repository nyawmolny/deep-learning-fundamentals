"""Small L-layer ReLU network with a mini-batch training loop (NumPy only).

X : (n_x, m), one example per column;  Y : (1, m) with labels in {0, 1}.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def initialize_parameters(layer_dims, seed: int | None = None):
    rng = np.random.default_rng(seed)
    params = {}
    for l in range(1, len(layer_dims)):
        params[f"W{l}"] = rng.standard_normal((layer_dims[l], layer_dims[l - 1])) * np.sqrt(2.0 / layer_dims[l - 1])
        params[f"b{l}"] = np.zeros((layer_dims[l], 1))
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


def predict(params, X):
    AL, _ = forward_propagation(X, params)
    return (AL > 0.5).astype(float)


def random_mini_batches(X, Y, batch_size: int, rng):
    """Shuffle the examples and split them into mini-batches."""
    m = X.shape[1]
    perm = rng.permutation(m)
    Xs, Ys = X[:, perm], Y[:, perm]
    return [(Xs[:, i:i + batch_size], Ys[:, i:i + batch_size]) for i in range(0, m, batch_size)]


def train(X, Y, layer_dims, optimizer, num_epochs: int, batch_size: int | None = None, seed: int = 0):
    """Returns the trained parameters and the mean cost per epoch.

    batch_size=None means full-batch gradient descent (one update per epoch).
    """
    params = initialize_parameters(layer_dims, seed)
    rng = np.random.default_rng(seed + 1000)
    costs = []
    for _ in range(num_epochs):
        batches = [(X, Y)] if batch_size is None else random_mini_batches(X, Y, batch_size, rng)
        total = 0.0
        for Xb, Yb in batches:
            AL, caches = forward_propagation(Xb, params)
            total += compute_cost(AL, Yb) * Xb.shape[1]
            params = optimizer.step(params, backward_propagation(AL, Yb, caches))
        costs.append(total / X.shape[1])
    return params, costs
