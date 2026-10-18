import numpy as np
import pytest
import torch
from data import load_breast_cancer_split, to_tensors
from model import (
    LogisticRegressionTorch,
    accuracy,
    load_numpy_reference,
    set_seed,
    train,
)


def test_output_shape_and_range():
    set_seed(0)
    model = LogisticRegressionTorch(30)
    out = model(torch.randn(8, 30))
    assert out.shape == (8, 1)
    assert torch.all((out > 0) & (out < 1))


def test_zero_init():
    model = LogisticRegressionTorch(5)
    model.zero_init()
    assert torch.count_nonzero(model.linear.weight) == 0
    assert model.linear.bias.item() == 0.0


def test_training_reduces_loss_and_beats_chance():
    X_np, Xt_np, y_np, yt_np = load_breast_cancer_split()
    X, y = to_tensors(X_np, y_np)
    Xt, yt = to_tensors(Xt_np, yt_np)
    set_seed(0)
    model = LogisticRegressionTorch(X.shape[1])
    losses = train(model, X, y, torch.optim.SGD(model.parameters(), lr=0.1), num_epochs=300)
    assert losses[-1] < losses[0]
    assert accuracy(model, Xt, yt) > 0.9


def test_matches_numpy_implementation():
    """Same data, zero init, same lr, full batch: loss curves and weights must coincide."""
    X_np, _, y_np, _ = load_breast_cancer_split()
    lr, epochs = 0.1, 100

    ref = load_numpy_reference()
    np_model = ref.LogisticRegression(
        learning_rate=lr, num_iterations=epochs, log_every=1
    ).fit(X_np.T, y_np.reshape(1, -1))

    model = LogisticRegressionTorch(X_np.shape[1]).double()
    model.zero_init()
    X, y = to_tensors(X_np, y_np, dtype=torch.float64)
    losses = train(model, X, y, torch.optim.SGD(model.parameters(), lr=lr), epochs)

    np.testing.assert_allclose(losses, np_model.costs_, atol=1e-8)
    np.testing.assert_allclose(model.linear.weight.detach().numpy().T, np_model.w, atol=1e-8)
    assert model.linear.bias.item() == pytest.approx(np_model.b, abs=1e-8)
