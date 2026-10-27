import numpy as np
import pytest
from bugs import BUGS, broken_backward
from gradient_check import gradient_check, numerical_gradients, relative_difference
from model import (
    backward_propagation,
    cost_from_params,
    forward_propagation,
    initialize_parameters,
)
from sklearn.datasets import make_moons


def moons(m=20, seed=0):
    X, y = make_moons(m, noise=0.2, random_state=seed)
    return X.T, y.reshape(1, -1).astype(float)


@pytest.fixture
def setup():
    X, Y = moons()
    params = initialize_parameters([2, 5, 3, 1], seed=1, random_biases=True)
    AL, caches = forward_propagation(X, params)
    return X, Y, params, AL, caches


def flagged(per_param, threshold=1e-6):
    return {k for k, v in per_param.items() if v > threshold}


# ---------- the checker itself ----------

def test_numerical_gradients_of_a_quadratic():
    params = {"w": np.array([[1.0, -2.0], [0.5, 3.0]])}
    g = numerical_gradients(lambda p: float(np.sum(p["w"] ** 2)), params)
    np.testing.assert_allclose(g["dw"], 2 * params["w"], atol=1e-6)


def test_numerical_gradients_do_not_modify_params():
    params = {"w": np.array([[1.0, 2.0]])}
    before = params["w"].copy()
    numerical_gradients(lambda p: float(np.sum(p["w"] ** 3)), params)
    np.testing.assert_array_equal(params["w"], before)


def test_relative_difference_edge_cases():
    a = np.array([1.0, 2.0, 3.0])
    assert relative_difference(a, a) == 0.0
    assert relative_difference(np.zeros(3), np.zeros(3)) == 0.0
    assert relative_difference(a, -a) == pytest.approx(1.0)


# ---------- correct and broken backpropagation ----------

def test_correct_backpropagation_passes(setup):
    X, Y, params, AL, caches = setup
    grads = backward_propagation(AL, Y, caches)
    diff, per_param, ok = gradient_check(lambda p: cost_from_params(p, X, Y), params, grads)
    assert ok and diff < 1e-7
    assert flagged(per_param) == set()


@pytest.mark.parametrize("bug", list(BUGS))
def test_every_bug_is_caught(setup, bug):
    X, Y, params, AL, caches = setup
    diff, _, ok = gradient_check(lambda p: cost_from_params(p, X, Y), params, broken_backward(AL, Y, caches, bug))
    assert not ok and diff > 1e-3


def test_per_parameter_differences_localize_the_bug(setup):
    X, Y, params, AL, caches = setup
    check = lambda bug: gradient_check(lambda p: cost_from_params(p, X, Y), params,
                                       broken_backward(AL, Y, caches, bug))[1]
    assert flagged(check("dropped last-layer term")) == {"W3"}
    assert flagged(check("wrong bias gradient")) == {"b1", "b2", "b3"}
    assert flagged(check("missing 1/m in dW")) == {"W1", "W2", "W3"}
    assert flagged(check("missing ReLU mask")) == {"W1", "b1", "W2", "b2"}   # layers before the output


def test_moderate_step_beats_a_huge_step(setup):
    X, Y, params, AL, caches = setup
    grads = backward_propagation(AL, Y, caches)
    f = lambda p: cost_from_params(p, X, Y)
    assert gradient_check(f, params, grads, eps=1e-5)[0] < gradient_check(f, params, grads, eps=1e-1)[0]


# ---------- pitfall: the ReLU kink ----------

def test_zero_biases_can_cause_a_false_alarm_at_the_relu_kink():
    """Correct backprop, but with zero biases a pre-activation can be exactly 0 (the ReLU kink)."""
    r = np.random.default_rng(1)
    X = r.normal(size=(2, 6)); Y = (r.random((1, 6)) > 0.5).astype(float)
    f = lambda p: cost_from_params(p, X, Y)

    zero = initialize_parameters([2, 3, 2, 1], seed=1)
    AL, caches = forward_propagation(X, zero)
    diff, per_param, ok = gradient_check(f, zero, backward_propagation(AL, Y, caches))
    assert not ok and flagged(per_param) == {"b2"}

    rand = initialize_parameters([2, 3, 2, 1], seed=1, random_biases=True)
    AL, caches = forward_propagation(X, rand)
    assert gradient_check(f, rand, backward_propagation(AL, Y, caches))[2]
