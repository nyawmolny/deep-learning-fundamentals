"""Data loading and preprocessing helpers."""
from __future__ import annotations

import numpy as np


def load_h5_dataset(train_path: str, test_path: str):
    """Load a binary image dataset stored in two .h5 files.

    Expected keys: "train_set_x"/"train_set_y" (train file) and
    "test_set_x"/"test_set_y" (test file). Adjust here if your files differ.
    Returns images as (m, H, W, 3) uint8 and labels as (1, m).
    """
    import h5py

    with h5py.File(train_path, "r") as f:
        x_train = np.array(f["train_set_x"][:])
        y_train = np.array(f["train_set_y"][:]).reshape(1, -1)
    with h5py.File(test_path, "r") as f:
        x_test = np.array(f["test_set_x"][:])
        y_test = np.array(f["test_set_y"][:]).reshape(1, -1)
    return x_train, y_train, x_test, y_test


def load_cifar10_cat_vs_rest(root: str = "data", cat_class: int = 3):
    """Public alternative: CIFAR-10, label 1 = cat, 0 = everything else.

    Downloads automatically (needs torchvision). Note the classes are
    imbalanced (10% positives), so report more than plain accuracy.
    """
    from torchvision.datasets import CIFAR10

    train = CIFAR10(root=root, train=True, download=True)
    test = CIFAR10(root=root, train=False, download=True)
    x_train, x_test = np.asarray(train.data), np.asarray(test.data)
    y_train = (np.asarray(train.targets) == cat_class).astype(float).reshape(1, -1)
    y_test = (np.asarray(test.targets) == cat_class).astype(float).reshape(1, -1)
    return x_train, y_train, x_test, y_test


def flatten_and_normalize(images: np.ndarray) -> np.ndarray:
    """(m, H, W, C) uint8 -> (H*W*C, m) float in [0, 1]."""
    return images.reshape(images.shape[0], -1).T / 255.0
