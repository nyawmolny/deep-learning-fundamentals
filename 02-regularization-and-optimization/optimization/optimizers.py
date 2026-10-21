"""Optimizers from scratch (NumPy): gradient descent, momentum, RMSprop, Adam.

Each optimizer keeps its own state and exposes step(params, grads) -> new params.
params uses keys W1, b1, ...; grads uses the matching keys dW1, db1, ...
"""
from __future__ import annotations

import numpy as np


class GradientDescent:
    """w <- w - lr * g. Batch or mini-batch depends on what is fed to it, not on this class."""

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, params, grads):
        return {k: p - self.lr * grads["d" + k] for k, p in params.items()}


class Momentum:
    """v <- beta * v + (1 - beta) * g;  w <- w - lr * v."""

    def __init__(self, lr: float, beta: float = 0.9):
        self.lr, self.beta = lr, beta
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = {k: np.zeros_like(p) for k, p in params.items()}
        new = {}
        for k, p in params.items():
            g = grads["d" + k]
            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * g
            new[k] = p - self.lr * self.v[k]
        return new


class RMSprop:
    """s <- beta * s + (1 - beta) * g^2;  w <- w - lr * g / (sqrt(s) + eps)."""

    def __init__(self, lr: float, beta: float = 0.9, eps: float = 1e-8):
        self.lr, self.beta, self.eps = lr, beta, eps
        self.s = None

    def step(self, params, grads):
        if self.s is None:
            self.s = {k: np.zeros_like(p) for k, p in params.items()}
        new = {}
        for k, p in params.items():
            g = grads["d" + k]
            self.s[k] = self.beta * self.s[k] + (1 - self.beta) * g ** 2
            new[k] = p - self.lr * g / (np.sqrt(self.s[k]) + self.eps)
        return new


class Adam:
    """Momentum (first moment) + RMSprop (second moment) with bias correction."""

    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.v = self.s = None
        self.t = 0

    def step(self, params, grads):
        if self.v is None:
            self.v = {k: np.zeros_like(p) for k, p in params.items()}
            self.s = {k: np.zeros_like(p) for k, p in params.items()}
        self.t += 1
        new = {}
        for k, p in params.items():
            g = grads["d" + k]
            self.v[k] = self.beta1 * self.v[k] + (1 - self.beta1) * g
            self.s[k] = self.beta2 * self.s[k] + (1 - self.beta2) * g ** 2
            v_hat = self.v[k] / (1 - self.beta1 ** self.t)
            s_hat = self.s[k] / (1 - self.beta2 ** self.t)
            new[k] = p - self.lr * v_hat / (np.sqrt(s_hat) + self.eps)
        return new
