"""Logistic regression in PyTorch: model, training loop and evaluation helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import Tensor, nn


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


class LogisticRegressionTorch(nn.Module):
    """One linear layer followed by a sigmoid; outputs probabilities of shape (m, 1)."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.linear(x))

    def zero_init(self) -> None:
        """Zero weights and bias, as in the NumPy implementation."""
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


def train(model: nn.Module, X: Tensor, y: Tensor, optimizer, num_epochs: int = 100) -> list[float]:
    """Full-batch training. Returns the loss before each update (one value per epoch)."""
    criterion = nn.BCELoss()
    losses: list[float] = []

    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def predict(model: nn.Module, X: Tensor, threshold: float = 0.5) -> Tensor:
    return (model(X) > threshold).to(X.dtype)


@torch.no_grad()
def accuracy(model: nn.Module, X: Tensor, y: Tensor) -> float:
    return (predict(model, X) == y).to(torch.float32).mean().item()


def load_numpy_reference():
    """Load ../numpy/model.py (the from-scratch version) under a separate module name."""
    path = Path(__file__).resolve().parents[1] / "numpy" / "model.py"
    spec = importlib.util.spec_from_file_location("logreg_numpy_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module) 
    return module
