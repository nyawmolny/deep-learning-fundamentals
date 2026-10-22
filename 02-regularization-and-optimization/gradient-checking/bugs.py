"""Backward passes with deliberate bugs, used to show what gradient checking catches."""
from __future__ import annotations

import numpy as np

BUGS = {
    "missing 1/m in dW": "forgets to average over the batch in the weight gradients",
    "missing ReLU mask": "ignores the derivative of ReLU in the hidden layers",
    "wrong bias gradient": "db is the mean of dZ with the wrong sign",
    "dropped last-layer term": "output layer weight gradient is zero",
}


def broken_backward(AL, Y, caches, bug: str):
    m = Y.shape[1]
    L = len(caches)
    grads = {}
    dZ = AL - Y
    for l in range(L, 0, -1):
        A_prev, W, _ = caches[l - 1]
        scale = 1.0 if bug == "missing 1/m in dW" else 1.0 / m
        grads[f"dW{l}"] = dZ @ A_prev.T * scale
        grads[f"db{l}"] = dZ.sum(axis=1, keepdims=True) / m
        if bug == "wrong bias gradient":
            grads[f"db{l}"] = -grads[f"db{l}"]
        if bug == "dropped last-layer term" and l == L:
            grads[f"dW{l}"] = np.zeros_like(grads[f"dW{l}"])
        if l > 1:
            mask = 1.0 if bug == "missing ReLU mask" else (caches[l - 2][2] > 0)
            dZ = (W.T @ dZ) * mask
    return grads
