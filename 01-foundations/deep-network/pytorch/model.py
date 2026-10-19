"""L-layer fully connected network in PyTorch: [LINEAR -> RELU] * (L-1) -> LINEAR.

The model returns logits; the sigmoid is folded into the loss (BCEWithLogitsLoss),
which is numerically more stable than sigmoid + BCELoss.
layer_dims = [n_x, n_1, ..., n_L], e.g. [n_x, 4, 1] (shallow) or [n_x, 20, 7, 5, 1] (deep).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import Tensor, nn


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


class DeepNetworkTorch(nn.Module):
    def __init__(self, layer_dims: list[int], init: str = "he") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(1, len(layer_dims)):
            # (layer_dims[i], layer_dims[i - 1]) * (layer_dims[i - 1], m)
            # (layer_dims[i], m)
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            if i < len(layer_dims) - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.reset_parameters(init)

    @property
    def linears(self) -> list[nn.Linear]:
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def reset_parameters(self, init: str = "he") -> None:
        """'he': N(0, 2/n_in) (Kaiming, for ReLU). 'small': N(0, 0.01^2). Biases are zero."""
        if init not in ("he", "small"):
            raise ValueError(init)
        for layer in self.linears:
            if init == "he":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # logits, shape (m, 1)

    def predict_proba(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self(x))

    @torch.no_grad()
    def load_numpy_params(self, params: dict) -> None:
        """Copy W1, b1, ... from the NumPy implementation (W: (n, n_prev), b: (n, 1))."""
        for l, layer in enumerate(self.linears, start=1):
            layer.weight.copy_(torch.as_tensor(params[f"W{l}"]))
            layer.bias.copy_(torch.as_tensor(params[f"b{l}"]).reshape(-1))


def train(model: nn.Module, X: Tensor, y: Tensor, optimizer, num_epochs: int = 100) -> list[float]:
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []

    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def predict(model: DeepNetworkTorch, X: Tensor, threshold: float = 0.5) -> Tensor:
    return (model.predict_proba(X) > threshold).to(X.dtype)


@torch.no_grad()
def accuracy(model: DeepNetworkTorch, X: Tensor, y: Tensor) -> float:
    return (predict(model, X) == y).to(torch.float32).mean().item()


def load_numpy_reference():
    """Load ../numpy/model.py (the from-scratch version) under a separate module name."""
    path = Path(__file__).resolve().parents[1] / "numpy" / "model.py"
    spec = importlib.util.spec_from_file_location("deepnet_numpy_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
