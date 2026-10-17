"""L-layer fully connected network: [LINEAR -> RELU] * (L-1) -> LINEAR -> SIGMOID.

Conventions
-----------
X : (n_x, m)   one example per column
Y : (1, m)     binary labels in {0, 1}
layer_dims = [n_x, n_1, ..., n_L]; parameters are W1, b1, ..., WL, bL
cache of layer l = (A_prev, W, Z)
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# ---------- activations ----------

def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def relu(Z):
    return np.maximum(0.0, Z)


def relu_backward(dA, Z):
    return dA * (Z > 0)


# ---------- initialization ----------

def initialize_parameters(layer_dims, init: str = "he", seed: int | None = None):
    """init='small': N(0, 0.01^2). init='he': N(0, 2/n_prev), suited to ReLU."""
    rng = np.random.default_rng(seed)
    params = {}
    for l in range(1, len(layer_dims)):
        n_prev, n = layer_dims[l - 1], layer_dims[l]
        scale = 0.01 if init == "small" else np.sqrt(2.0 / n_prev)
        if init not in ("small", "he"):
            raise ValueError(init)
        params[f"W{l}"] = rng.standard_normal((n, n_prev)) * scale
        params[f"b{l}"] = np.zeros((n, 1))
    return params


# ---------- forward ----------

def layer_forward(A_prev, W, b, activation):
    Z = W @ A_prev + b
    A = relu(Z) if activation == "relu" else sigmoid(Z)
    return A, (A_prev, W, Z)


def forward_propagation(X, params):
    """Returns AL (1, m) and the list of per-layer caches."""
    L = len(params) // 2
    A, caches = X, []
    for l in range(1, L):
        A, cache = layer_forward(A, params[f"W{l}"], params[f"b{l}"], "relu")
        caches.append(cache)
    AL, cache = layer_forward(A, params[f"W{L}"], params[f"b{L}"], "sigmoid")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    """Mean binary cross-entropy."""
    A = np.clip(AL, EPS, 1.0 - EPS)
    return float(-np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A)))


# ---------- backward ----------

def linear_backward(dZ, A_prev, W):
    m = A_prev.shape[1]
    dW = dZ @ A_prev.T / m
    db = dZ.sum(axis=1, keepdims=True) / m
    dA_prev = W.T @ dZ
    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    """Gradients dW1..dWL, db1..dbL.

    For sigmoid + cross-entropy, dZ_L simplifies to AL - Y, which avoids
    dividing by AL and 1 - AL.
    """
    L = len(caches)
    grads = {}

    A_prev, W, _ = caches[L - 1]
    dZ = AL - Y
    dA_prev, grads[f"dW{L}"], grads[f"db{L}"] = linear_backward(dZ, A_prev, W)

    for l in range(L - 1, 0, -1):
        A_prev, W, Z = caches[l - 1]
        dZ = relu_backward(dA_prev, Z)
        dA_prev, grads[f"dW{l}"], grads[f"db{l}"] = linear_backward(dZ, A_prev, W)
    return grads


def update_parameters(params, grads, learning_rate):
    return {k: v - learning_rate * grads["d" + k] for k, v in params.items()}


def predict(params, X, threshold=0.5):
    AL, _ = forward_propagation(X, params)
    return (AL > threshold).astype(float)


class DeepNetwork:
    """Fit/predict wrapper around the functions above."""

    def __init__(self, layer_dims, learning_rate=0.0075, num_iterations=2500,
                 init="he", seed=1, log_every=100):
        self.layer_dims = list(layer_dims)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.init = init
        self.seed = seed
        self.log_every = log_every
        self.params = None
        self.costs_ = []

    def fit(self, X, Y):
        params = initialize_parameters(self.layer_dims, self.init, self.seed)
        self.costs_ = []
        for i in range(self.num_iterations):
            AL, caches = forward_propagation(X, params)
            grads = backward_propagation(AL, Y, caches)
            params = update_parameters(params, grads, self.learning_rate)
            if i % self.log_every == 0:
                self.costs_.append(compute_cost(AL, Y))
        self.params = params
        return self

    def predict(self, X):
        return predict(self.params, X)

    def score(self, X, Y):
        return float(np.mean(self.predict(X) == Y))
