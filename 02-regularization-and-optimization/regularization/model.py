"""L-layer network with L2 regularization and inverted dropout (NumPy only).

Architecture: [LINEAR -> RELU (-> DROPOUT)] * (L-1) -> LINEAR -> SIGMOID.

Conventions
-----------
X : (n_x, m)   one example per column
Y : (1, m)     binary labels in {0, 1}
layer_dims = [n_x, n_1, ..., n_L]; parameters are W1, b1, ..., WL, bL
cache of layer l = (A_prev, W, Z, D), D is the dropout mask applied to the output of
layer l (None for the output layer or when dropout is off).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def relu(Z):
    return np.maximum(0.0, Z)


def initialize_parameters(layer_dims, seed: int | None = None):
    """He initialization: W ~ N(0, 2 / n_prev), zero biases."""
    rng = np.random.default_rng(seed)
    params = {}
    for l in range(1, len(layer_dims)):
        params[f"W{l}"] = rng.standard_normal((layer_dims[l], layer_dims[l - 1])) * np.sqrt(2.0 / layer_dims[l - 1])
        params[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return params


def forward_propagation(X, params, keep_prob: float = 1.0, rng=None):
    """Forward pass. Dropout is applied to hidden activations only when keep_prob < 1.

    Inverted dropout: kept activations are divided by keep_prob, so their expected value
    is unchanged and nothing has to be rescaled at test time.
    """
    L = len(params) // 2
    use_dropout = keep_prob < 1.0
    if use_dropout and rng is None:
        rng = np.random.default_rng()

    A, caches = X, []
    for l in range(1, L):
        Z = params[f"W{l}"] @ A + params[f"b{l}"]
        A_out = relu(Z)
        D = None
        if use_dropout:
            assert rng is not None
            D = rng.random(A_out.shape) < keep_prob
            A_out = A_out * D / keep_prob
        caches.append((A, params[f"W{l}"], Z, D))
        A = A_out

    ZL = params[f"W{L}"] @ A + params[f"b{L}"]
    AL = sigmoid(ZL)
    caches.append((A, params[f"W{L}"], ZL, None))
    return AL, caches


def compute_cost(AL, Y, params=None, lambd: float = 0.0):
    """Mean binary cross-entropy + (lambd / 2m) * sum of squared weights (biases excluded)."""
    m = Y.shape[1]
    A = np.clip(AL, EPS, 1.0 - EPS)
    cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    if lambd > 0.0:
        assert params is not None
        L = len(params) // 2
        l2 = sum(np.sum(np.square(params[f"W{l}"])) for l in range(1, L + 1))
        cost += lambd / (2 * m) * l2
    return float(cost)


def backward_propagation(AL, Y, caches, lambd: float = 0.0, keep_prob: float = 1.0):
    """Gradients of the (regularized) cost. Uses the same dropout masks as the forward pass."""
    m = Y.shape[1]
    L = len(caches)
    grads = {}

    A_prev, W, _, _ = caches[L - 1]
    dZ = AL - Y
    grads[f"dW{L}"] = dZ @ A_prev.T / m + lambd / m * W
    grads[f"db{L}"] = dZ.sum(axis=1, keepdims=True) / m
    dA = W.T @ dZ                                   # gradient w.r.t. the (dropped-out) output of layer L-1

    for l in range(L - 1, 0, -1):
        A_prev, W, Z, D = caches[l - 1]
        if D is not None:
            dA = dA * D / keep_prob                 # same mask and scaling as in the forward pass
        dZ = dA * (Z > 0)
        grads[f"dW{l}"] = dZ @ A_prev.T / m + lambd / m * W
        grads[f"db{l}"] = dZ.sum(axis=1, keepdims=True) / m
        dA = W.T @ dZ
    return grads


def update_parameters(params, grads, learning_rate):
    return {k: v - learning_rate * grads["d" + k] for k, v in params.items()}


def predict(params, X, threshold: float = 0.5):
    """Hard 0/1 predictions of shape (1, m). Dropout is never used at prediction time."""
    AL, _ = forward_propagation(X, params, keep_prob=1.0)
    return (AL > threshold).astype(float)


def weight_norm(params) -> float:
    """sqrt of the sum of squared weights (biases excluded)."""
    L = len(params) // 2
    return float(np.sqrt(sum(np.sum(np.square(params[f"W{l}"])) for l in range(1, L + 1))))


class RegularizedNetwork:
    """Fit/predict wrapper. lambd = L2 strength, keep_prob = dropout keep probability."""

    def __init__(self, layer_dims, learning_rate=0.3, num_iterations=30000,
                 lambd=0.0, keep_prob=1.0, seed=0, log_every=1000):
        self.layer_dims = list(layer_dims)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.seed = seed
        self.log_every = log_every
        self.params = None
        self.costs_ = []

    def fit(self, X, Y):
        params = initialize_parameters(self.layer_dims, self.seed)
        rng = np.random.default_rng(self.seed + 12345)   # dropout masks
        self.costs_ = []
        for i in range(self.num_iterations):
            AL, caches = forward_propagation(X, params, self.keep_prob, rng)
            grads = backward_propagation(AL, Y, caches, self.lambd, self.keep_prob)
            params = update_parameters(params, grads, self.learning_rate)
            if i % self.log_every == 0:
                self.costs_.append(compute_cost(AL, Y, params, self.lambd))
        self.params = params
        return self

    def predict(self, X):
        return predict(self.params, X)

    def score(self, X, Y):
        return float(np.mean(self.predict(X) == Y))
