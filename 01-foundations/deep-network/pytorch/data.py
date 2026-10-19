"""Data loading and preprocessing. Examples are rows: X (m, n), y (m,)."""
from __future__ import annotations

import numpy as np


def load_h5_dataset(train_path: str, test_path: str):
    """Binary image dataset in two .h5 files -> images (m, H, W, 3), labels (m,).

    Expected keys: "train_set_x"/"train_set_y" and "test_set_x"/"test_set_y".
    """
    import h5py

    with h5py.File(train_path, "r") as f:
        x_train, y_train = np.array(f["train_set_x"][:]), np.array(f["train_set_y"][:])
    with h5py.File(test_path, "r") as f:
        x_test, y_test = np.array(f["test_set_x"][:]), np.array(f["test_set_y"][:])
    return x_train, y_train.astype(float), x_test, y_test.astype(float)


def load_cifar10_cat_vs_rest(root: str = "data", cat_class: int = 3):
    """CIFAR-10, label 1 = cat, 0 = the rest (needs torchvision, ~10% positives)."""
    from torchvision.datasets import CIFAR10

    train = CIFAR10(root=root, train=True, download=True)
    test = CIFAR10(root=root, train=False, download=True)
    y_train = (np.asarray(train.targets) == cat_class).astype(float)
    y_test = (np.asarray(test.targets) == cat_class).astype(float)
    return np.asarray(train.data), y_train, np.asarray(test.data), y_test


def flatten_and_normalize(images: np.ndarray) -> np.ndarray:
    """(m, H, W, C) uint8 -> (m, H*W*C) float in [0, 1]."""
    return images.reshape(images.shape[0], -1) / 255.0


def to_tensors(X: np.ndarray, y: np.ndarray, dtype=None):
    """NumPy (m, n) and (m,) -> torch tensors of shapes (m, n) and (m, 1)."""
    import torch

    dtype = dtype or torch.float32
    return torch.as_tensor(X, dtype=dtype), torch.as_tensor(y, dtype=dtype).reshape(-1, 1)
