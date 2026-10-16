"""Data loading and preprocessing helpers."""
from __future__ import annotations

import numpy as np


def load_h5_dataset(train_path: str, test_path: str):
    """Binary image dataset in two .h5 files -> images (m, H, W, 3), labels (1, m).

    Expected keys: "train_set_x"/"train_set_y" and "test_set_x"/"test_set_y".
    """
    from typing import cast

    import h5py

    with h5py.File(train_path, "r") as f:
        x_train = np.array(cast(h5py.Dataset, f["train_set_x"])[:])
        y_train = np.array(cast(h5py.Dataset, f["train_set_y"])[:]).reshape(1, -1)
    with h5py.File(test_path, "r") as f:
        x_test = np.array(cast(h5py.Dataset, f["test_set_x"])[:])
        y_test = np.array(cast(h5py.Dataset, f["test_set_y"])[:]).reshape(1, -1)
    return x_train, y_train, x_test, y_test


def load_cifar10_cat_vs_rest(root: str = "data", cat_class: int = 3):
    """CIFAR-10, label 1 = cat, 0 = the rest (needs torchvision, ~10% positives)."""
    from torchvision.datasets import CIFAR10

    train = CIFAR10(root=root, train=True, download=True)
    test = CIFAR10(root=root, train=False, download=True)
    y_train = (np.asarray(train.targets) == cat_class).astype(float).reshape(1, -1)
    y_test = (np.asarray(test.targets) == cat_class).astype(float).reshape(1, -1)
    return np.asarray(train.data), y_train, np.asarray(test.data), y_test


def flatten_and_normalize(images: np.ndarray) -> np.ndarray:
    """(m, H, W, C) uint8 -> (H*W*C, m) float in [0, 1]."""
    return images.reshape(images.shape[0], -1).T / 255.0
