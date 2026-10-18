"""Breast cancer dataset: split, standardization and tensor conversion.

Labels follow sklearn: 0 = malignant, 1 = benign.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor


def load_breast_cancer_split(test_size: float = 0.2, seed: int = 0):
    """Returns X_train (m, n), X_test, y_train (m,), y_test as float64 NumPy arrays."""
    bc = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_test),
        y_train.astype(float),
        y_test.astype(float),
    )


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Test accuracy of always predicting the most frequent training class."""
    majority = float(y_train.mean() >= 0.5)
    return float(np.mean(y_test == majority))


def to_tensors(X: np.ndarray, y: np.ndarray, dtype=None) -> tuple[Tensor, Tensor]:
    """NumPy (m, n) and (m,) into torch tensors of shapes (m, n) and (m, 1)."""
    dtype = dtype or torch.float32
    return torch.as_tensor(X, dtype=dtype), torch.as_tensor(y, dtype=dtype).reshape(-1, 1)
