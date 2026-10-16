"""Neural network with one hidden layer (tanh) and a sigmoid output, NumPy only.

Conventions
-----------
X : (n_x, m)   one example per column
Y : (1, m)     binary labels in {0, 1}
W1: (n_h, n_x), b1: (n_h, 1), W2: (1, n_h), b2: (1, 1)
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def initialize_parameters(n_x: int, n_h: int, n_y: int = 1, seed: int | None = None):
    """Small random weights break the symmetry between hidden units; biases start at zero."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.standard_normal((n_h, n_x)) * 0.01,
        "b1": np.zeros((n_h, 1)),
        "W2": rng.standard_normal((n_y, n_h)) * 0.01,
        "b2": np.zeros((n_y, 1)),
    }


def forward_propagation(X, params):
    """Returns output probabilities A2 and a cache for the backward pass."""
    Z1 = params["W1"] @ X + params["b1"]
    A1 = np.tanh(Z1)
    Z2 = params["W2"] @ A1 + params["b2"]
    A2 = sigmoid(Z2)
    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


def compute_cost(A2, Y):
    """Mean binary cross-entropy."""
    A = np.clip(A2, EPS, 1.0 - EPS)
    return float(-np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A)))


def backward_propagation(params, cache, X, Y):
    """Gradients of the cost w.r.t. all parameters."""
    m = X.shape[1]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T / m
    db2 = dZ2.sum(axis=1, keepdims=True) / m

    dZ1 = (params["W2"].T @ dZ2) * (1 - A1 ** 2)   # tanh'(z) = 1 - tanh(z)^2
    dW1 = dZ1 @ X.T / m
    db1 = dZ1.sum(axis=1, keepdims=True) / m
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(params, grads, learning_rate):
    return {k: params[k] - learning_rate * grads["d" + k] for k in params}


def predict(params, X, threshold=0.5):
    A2, _ = forward_propagation(X, params)
    return (A2 > threshold).astype(float)


class ShallowNetwork:
    """Fit/predict wrapper around the functions above."""

    def __init__(self, n_hidden=4, learning_rate=1.2, num_iterations=10000,
                 log_every=100, seed=3):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.log_every = log_every
        self.seed = seed
        self.params = None
        self.costs_ = []

    def fit(self, X, Y):
        params = initialize_parameters(X.shape[0], self.n_hidden, Y.shape[0], self.seed)
        self.costs_ = []
        for i in range(self.num_iterations):
            A2, cache = forward_propagation(X, params)
            grads = backward_propagation(params, cache, X, Y)
            params = update_parameters(params, grads, self.learning_rate)
            if i % self.log_every == 0:
                self.costs_.append(compute_cost(A2, Y))
        self.params = params
        return self

    def predict(self, X):
        return predict(self.params, X)

    def score(self, X, Y):
        return float(np.mean(self.predict(X) == Y))
