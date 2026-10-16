"""Synthetic 2-D datasets and a decision-boundary plot helper."""
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_circles, make_moons


def make_flower(m: int = 400, petals: int = 4, radius: float = 4.0,
                noise: float = 0.2, seed: int = 1) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two-class 'flower': each class covers half of a rose curve r = a*sin(k*theta).

    Returns X of shape (2, m) and Y of shape (1, m).
    """
    rng = np.random.default_rng(seed)
    n = m // 2
    X = np.zeros((m, 2))
    Y = np.zeros((m, 1))
    for j in range(2):
        idx = slice(n * j, n * (j + 1))
        theta = np.linspace(j * np.pi, (j + 1) * np.pi, n) + rng.normal(0, noise, n)
        r = radius * np.sin(petals * theta) + rng.normal(0, noise, n)
        X[idx] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        Y[idx] = j
    return X.T, Y.T


def make_sklearn_dataset(name: str, m: int = 400, seed: int = 0):
    """'moons' or 'circles' in the same (2, m) / (1, m) layout."""
    if name == "moons":
        X, y = make_moons(n_samples=m, noise=0.2, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=m, factor=0.5, noise=0.15, random_state=seed)
    else:
        raise ValueError(name)
    return X.T, y.reshape(1, -1).astype(float)


def plot_decision_boundary(predict_fn, X, Y, ax=None, h=0.05):
    """predict_fn takes points of shape (2, N) and returns labels of shape (1, N)."""
    ax = ax or plt.gca()
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict_fn(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    ax.scatter(X[0], X[1], c=Y.ravel(), s=15, cmap=plt.cm.Spectral, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
