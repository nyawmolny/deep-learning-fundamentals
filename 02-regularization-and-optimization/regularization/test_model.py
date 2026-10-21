import numpy as np
import pytest
from model import (
    RegularizedNetwork,
    compute_cost,
    forward_propagation,
    initialize_parameters,
    predict,
    weight_norm,
)

DIMS = [3, 5, 4, 1]


def toy_problem(m=150, seed=1):
    """Small 2-D problem that is not linearly separable (XOR-like)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(2, m))
    Y = ((X[0] * X[1]) > 0).astype(float).reshape(1, -1)
    return X, Y


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10))
    Y = (rng.random((1, 10)) > 0.5).astype(float)
    params = initialize_parameters(DIMS, seed=1)
    return X, Y, params


def test_parameter_shapes_and_he_scale():
    p = initialize_parameters([1000, 500, 1], seed=0)
    assert p["W1"].shape == (500, 1000) and p["b1"].shape == (500, 1)
    assert p["W1"].std() == pytest.approx(np.sqrt(2 / 1000), rel=0.05)


def test_l2_term_matches_manual_computation(toy):
    X, Y, p = toy
    AL, _ = forward_propagation(X, p)
    m, lambd = Y.shape[1], 0.7
    l2 = sum(np.sum(p[f"W{l}"] ** 2) for l in (1, 2, 3))
    assert compute_cost(AL, Y, p, lambd) == pytest.approx(compute_cost(AL, Y) + lambd / (2 * m) * l2)


def test_zero_lambda_means_no_penalty(toy):
    X, Y, p = toy
    AL, _ = forward_propagation(X, p)
    assert compute_cost(AL, Y, p, 0.0) == compute_cost(AL, Y)


def test_inverted_dropout_keeps_expected_activation_and_drop_rate():
    rng = np.random.default_rng(0)
    p = initialize_parameters([20, 2000, 1], seed=0)
    X = rng.normal(size=(20, 3))
    _, caches_off = forward_propagation(X, p)
    _, caches_on = forward_propagation(X, p, keep_prob=0.8, rng=np.random.default_rng(1))
    mask = caches_on[0][3]
    assert mask.mean() == pytest.approx(0.8, abs=0.02)
    # activations fed to the last layer: same mean with and without dropout
    assert caches_on[1][0].mean() == pytest.approx(caches_off[1][0].mean(), rel=0.1)


def test_prediction_is_deterministic_and_ignores_dropout(toy):
    X, _, p = toy
    np.testing.assert_array_equal(predict(p, X), predict(p, X))


def test_l2_shrinks_weights_during_training():
    Xn, Y = toy_problem()
    kw = dict(layer_dims=[2, 20, 10, 1], learning_rate=0.1, num_iterations=1500, seed=0)
    plain = RegularizedNetwork(**kw).fit(Xn, Y)
    ridge = RegularizedNetwork(lambd=1.0, **kw).fit(Xn, Y)
    assert weight_norm(ridge.params) < weight_norm(plain.params)


def test_training_reduces_cost():
    X, Y = toy_problem()
    net = RegularizedNetwork([2, 20, 10, 1], learning_rate=0.1, num_iterations=2000,
                             keep_prob=0.9, seed=0, log_every=100).fit(X, Y)
    assert net.costs_[-1] < net.costs_[0]