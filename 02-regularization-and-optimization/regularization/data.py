"""Utilities for loading and plotting the football dataset from train.csv.

X has shape (2, m), Y has shape (1, m).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_football(path: str | Path | None = None):
    """Load train.csv and return X (2, m) and Y (1, m)."""
    if path is None:
        path = Path(__file__).with_name("data") / "train.csv"

    raw = pd.read_csv(path)
    data = raw.apply(pd.to_numeric).to_numpy(dtype=float)

    labels = data[:, 2]

    X = data[:, :2].T 
    Y = labels.reshape(1, -1)
    return X, Y


def plot_football(X, Y, ax, cmap, title: str | None = None):
    """Plot the dataset in its existing normalized coordinate system."""
    ax.scatter(
        X[0],
        X[1],
        c=Y.ravel(),
        cmap=cmap
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if title:
        ax.set_title(title)


def plot_decision_boundary(
    predict_fn, X, Y, ax, title: str | None = None, step: float = 0.01
):
    """Plot a model's decision regions over the feature coordinates.

    predict_fn must accept points with shape (2, N) and return N or (1, N)
    binary predictions. Coordinates are passed to the model unchanged.
    """
    from matplotlib.colors import ListedColormap

    margin = 0.05
    x_min, x_max = X[0].min() - margin, X[0].max() + margin
    y_min, y_max = X[1].min() - margin, X[1].max() + margin
    gx, gy = np.meshgrid(
        np.arange(x_min, x_max + step, step),
        np.arange(y_min, y_max + step, step),
    )

    grid = np.vstack([gx.ravel(), gy.ravel()])
    predictions = np.asarray(predict_fn(grid)).reshape(-1)
    if predictions.size != grid.shape[1]:
        raise ValueError("predict_fn must return one prediction per input point.")
    Z = predictions.reshape(gx.shape)

    ax.contourf(
        gx,
        gy,
        Z,
        levels=[-0.5, 0.5, 1.5],
        cmap=ListedColormap(["#b0004b", "#5e4fa2"]),
        alpha=0.22,
    )
    plot_football(X, Y, cmap=plt.cm.Spectral, ax=ax)
    if title:
        ax.set_title(title)