"""Logistic regression implemented as a single-neuron network (NumPy only).

Conventions
-----------
X : (n_features, m)   one example per column
Y : (1, m)            binary labels in {0, 1}
w : (n_features, 1)   weights
b : float             bias
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12  # keeps log() finite when predictions saturate at 0 or 1


def sigmoid(z):
    """Sigmoid for scalars or arrays of any shape."""
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim: int):
    """Zero weights of shape (dim, 1) and zero bias.

    Zeros are fine here because logistic regression has no hidden layer,
    so there is no symmetry to break (this changes for deeper networks).
    """
    return np.zeros((dim, 1)), 0.0


def propagate(w, b, X, Y):
    """Forward pass (cost) and backward pass (gradients).

    Returns
    -------
    grads : dict with "dw" (same shape as w) and "db" (float)
    cost  : float, mean binary cross-entropy
    """
    m = X.shape[1]

    # forward
    A = sigmoid(w.T @ X + b)                       # (1, m)
    A_safe = np.clip(A, EPS, 1.0 - EPS)
    cost = -np.mean(Y * np.log(A_safe) + (1 - Y) * np.log(1 - A_safe))

    # backward: dJ/dz = A - Y
    dZ = A - Y
    dw = X @ dZ.T / m
    db = float(np.sum(dZ) / m)

    return {"dw": dw, "db": db}, float(cost)


def optimize(w, b, X, Y, num_iterations=2000, learning_rate=0.005, log_every=100):
    """Batch gradient descent. Returns params, last grads, recorded costs."""
    costs = []
    grads = {}
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        if i % log_every == 0:
            costs.append(cost)
    return {"w": w, "b": b}, grads, costs


def predict_proba(w, b, X):
    return sigmoid(w.T @ X + b)


def predict(w, b, X, threshold=0.5):
    """0/1 predictions of shape (1, m)."""
    return (predict_proba(w, b, X) > threshold).astype(float)


def accuracy(Y_pred, Y_true) -> float:
    return float(np.mean(Y_pred == Y_true))


class LogisticRegression:
    """Thin wrapper around the functions above."""

    def __init__(self, learning_rate=0.005, num_iterations=2000, log_every=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.log_every = log_every
        self.w = None
        self.b = None
        self.costs_ = []

    def fit(self, X, Y):
        w, b = initialize_with_zeros(X.shape[0])
        params, _, self.costs_ = optimize(
            w, b, X, Y, self.num_iterations, self.learning_rate, self.log_every
        )
        self.w, self.b = params["w"], params["b"]
        return self

    def predict_proba(self, X):
        return predict_proba(self.w, self.b, X)

    def predict(self, X, threshold=0.5):
        return predict(self.w, self.b, X, threshold)

    def score(self, X, Y):
        return accuracy(self.predict(X), Y)
