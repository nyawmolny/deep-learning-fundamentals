"""Moons dataset and decision-boundary plot."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons


def make_moons_dataset(m: int = 2000, noise: float = 0.25, seed: int = 0):
    """Returns X (2, m) and Y (1, m)."""
    X, y = make_moons(n_samples=m, noise=noise, random_state=seed)
    return X.T, y.reshape(1, -1).astype(float)


def plot_decision_boundary(predict_fn, X, Y, ax, title: str | None = None, step: float = 0.02):
    """predict_fn takes points (2, N) and returns 0/1 labels with N entries."""
    from matplotlib.colors import ListedColormap

    gx, gy = np.meshgrid(np.arange(X[0].min() - 0.5, X[0].max() + 0.5, step),
                         np.arange(X[1].min() - 0.5, X[1].max() + 0.5, step))
    Z = np.asarray(predict_fn(np.vstack([gx.ravel(), gy.ravel()]))).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5], cmap=ListedColormap(["#f4b6c2", "#b9c8f5"]))
    ax.scatter(X[0], X[1], c=Y.ravel(), s=8, cmap="coolwarm", edgecolors="none")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    if title:
        ax.set_title(title)
