import numpy as np
from data import make_sklearn_dataset
from model import (
    ShallowNetwork,
    forward_propagation,
    initialize_parameters,
)


def test_parameter_and_output_shapes():
    p = initialize_parameters(2, 4, 1, seed=0)
    assert p["W1"].shape == (4, 2) and p["b1"].shape == (4, 1)
    assert p["W2"].shape == (1, 4) and p["b2"].shape == (1, 1)
    A2, cache = forward_propagation(np.zeros((2, 7)), p)
    assert A2.shape == (1, 7) and cache["A1"].shape == (4, 7)


def test_network_learns_nonlinear_data():
    X, Y = make_sklearn_dataset("moons", 300, seed=1)
    net = ShallowNetwork(n_hidden=4, num_iterations=3000).fit(X, Y)
    assert net.costs_[-1] < net.costs_[0]
    assert net.score(X, Y) > 0.85
