import numpy as np
import pytest

from model import (LogisticRegression, initialize_with_zeros, predict,
                   propagate, sigmoid)


def test_sigmoid_known_values():
    assert np.allclose(sigmoid(np.array([0.0, 2.0])), [0.5, 0.88079708])


def test_zero_init_shapes():
    w, b = initialize_with_zeros(5)
    assert w.shape == (5, 1) and b == 0.0


def test_initial_cost_is_log_2_with_zero_weights():
    X = np.random.default_rng(1).normal(size=(3, 10))
    Y = np.ones((1, 10))
    w, b = initialize_with_zeros(3)
    _, cost = propagate(w, b, X, Y)
    assert cost == pytest.approx(np.log(2))


def test_training_reduces_cost_and_separates_easy_data():
    rng = np.random.default_rng(2)
    X = np.hstack([rng.normal(-2, 1, (2, 100)), rng.normal(2, 1, (2, 100))])
    Y = np.hstack([np.zeros((1, 100)), np.ones((1, 100))])
    clf = LogisticRegression(learning_rate=0.1, num_iterations=500).fit(X, Y)
    assert clf.costs_[-1] < clf.costs_[0]
    assert clf.score(X, Y) > 0.95
    assert predict(clf.w, clf.b, X).shape == (1, 200)
