import numpy as np
import pytest
from model import (
    DeepNetwork,
    forward_propagation,
    initialize_parameters,
    relu,
    relu_backward,
    sigmoid,
)


def test_initialize_parameters_shapes():
    p = initialize_parameters([5, 4, 3, 1], seed=0)
    assert len(p) == 6
    assert p["W1"].shape == (4, 5) and p["b1"].shape == (4, 1)
    assert p["W3"].shape == (1, 3) and p["b3"].shape == (1, 1)


def test_he_init_has_expected_scale():
    p = initialize_parameters([1000, 500, 1], init="he", seed=0)
    assert p["W1"].std() == pytest.approx(np.sqrt(2 / 1000), rel=0.05)


def test_activations():
    assert np.allclose(relu(np.array([-2.0, 0.0, 3.0])), [0, 0, 3])
    assert np.allclose(relu_backward(np.ones(3), np.array([-1.0, 0.0, 2.0])), [0, 0, 1])
    assert sigmoid(0.0) == 0.5


def test_forward_output_shape_and_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 9))
    AL, caches = forward_propagation(X, initialize_parameters([6, 5, 4, 1], seed=1))
    assert AL.shape == (1, 9) and np.all((AL > 0) & (AL < 1))
    assert len(caches) == 3


def test_network_learns_a_nonlinear_problem():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2, 400))
    Y = ((X[0] * X[1]) > 0).astype(float).reshape(1, -1)   # XOR-like, not linearly separable
    net = DeepNetwork([2, 8, 8, 1], learning_rate=0.5, num_iterations=3000).fit(X, Y)
    assert net.costs_[-1] < net.costs_[0]
    assert net.score(X, Y) > 0.9
