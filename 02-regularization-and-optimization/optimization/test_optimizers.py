import numpy as np
import pytest
from data import make_moons_dataset
from model import predict, random_mini_batches, train
from optimizers import Adam, GradientDescent, Momentum, RMSprop


@pytest.fixture
def pg():
    rng = np.random.default_rng(0)
    params = {"W1": rng.normal(size=(3, 2)), "b1": rng.normal(size=(3, 1))}
    grads = [{"dW1": rng.normal(size=(3, 2)), "db1": rng.normal(size=(3, 1))} for _ in range(4)]
    return params, grads


# ---------- single steps against hand-written formulas ----------

def test_gradient_descent_step(pg):
    params, grads = pg
    new = GradientDescent(0.1).step(params, grads[0])
    np.testing.assert_allclose(new["W1"], params["W1"] - 0.1 * grads[0]["dW1"])


def test_momentum_two_steps(pg):
    params, grads = pg
    lr, beta = 0.1, 0.9
    opt = Momentum(lr, beta)
    p1 = opt.step(params, grads[0])
    p2 = opt.step(p1, grads[1])
    v1 = (1 - beta) * grads[0]["dW1"]
    v2 = beta * v1 + (1 - beta) * grads[1]["dW1"]
    np.testing.assert_allclose(p1["W1"], params["W1"] - lr * v1)
    np.testing.assert_allclose(p2["W1"], p1["W1"] - lr * v2)


def test_rmsprop_one_step(pg):
    params, grads = pg
    lr, beta, eps = 0.01, 0.9, 1e-8
    g = grads[0]["dW1"]
    s = (1 - beta) * g ** 2
    new = RMSprop(lr, beta, eps).step(params, grads[0])
    np.testing.assert_allclose(new["W1"], params["W1"] - lr * g / (np.sqrt(s) + eps))


def test_adam_first_step_is_about_lr_times_sign(pg):
    """With bias correction the first Adam step is lr * g / (|g| + eps), about lr * sign(g)."""
    params, grads = pg
    lr = 0.01
    new = Adam(lr).step(params, grads[0])
    np.testing.assert_allclose(new["W1"], params["W1"] - lr * np.sign(grads[0]["dW1"]), atol=1e-6)


def test_adam_two_steps_match_manual_formula(pg):
    params, grads = pg
    lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
    opt = Adam(lr, b1, b2, eps)
    p1 = opt.step(params, grads[0])
    p2 = opt.step(p1, grads[1])
    g1, g2 = grads[0]["dW1"], grads[1]["dW1"]
    v = (1 - b1) * g1; s = (1 - b2) * g1 ** 2
    v = b1 * v + (1 - b1) * g2; s = b2 * s + (1 - b2) * g2 ** 2
    expected = p1["W1"] - lr * (v / (1 - b1 ** 2)) / (np.sqrt(s / (1 - b2 ** 2)) + eps)
    np.testing.assert_allclose(p2["W1"], expected)


def test_momentum_equals_torch_style_sgd_with_rescaled_lr(pg):
    """Our v = beta*v + (1-beta)*g with lr equals the rule buf = beta*buf + g with lr*(1-beta)."""
    params, grads = pg
    lr, beta = 0.1, 0.9
    ours = Momentum(lr, beta)
    p_ours = params
    p_ref, buf = {k: v.copy() for k, v in params.items()}, {k: np.zeros_like(v) for k, v in params.items()}
    for g in grads:
        p_ours = ours.step(p_ours, g)
        for k in p_ref:
            buf[k] = beta * buf[k] + g["d" + k]
            p_ref[k] = p_ref[k] - lr * (1 - beta) * buf[k]
    for k in p_ref:
        np.testing.assert_allclose(p_ours[k], p_ref[k], atol=1e-12)


# ---------- mini-batches and training ----------

def test_mini_batches_cover_every_example_once():
    X = np.arange(2 * 10).reshape(2, 10).astype(float)
    Y = np.arange(10).reshape(1, 10).astype(float)
    batches = random_mini_batches(X, Y, 4, np.random.default_rng(0))
    assert [b[0].shape[1] for b in batches] == [4, 4, 2]
    seen = np.concatenate([b[1][0] for b in batches])
    np.testing.assert_array_equal(np.sort(seen), np.arange(10))
    for Xb, Yb in batches:                    # X and Y stay aligned
        np.testing.assert_array_equal(Xb[0], Yb[0])


def test_mini_batches_are_reproducible_and_reshuffled():
    X = np.random.default_rng(0).normal(size=(2, 20)); Y = np.zeros((1, 20))
    a = random_mini_batches(X, Y, 5, np.random.default_rng(1))
    b = random_mini_batches(X, Y, 5, np.random.default_rng(1))
    c = random_mini_batches(X, Y, 5, np.random.default_rng(2))
    np.testing.assert_array_equal(a[0][0], b[0][0])
    assert not np.array_equal(a[0][0], c[0][0])


@pytest.mark.parametrize("make_opt", [lambda: GradientDescent(0.05), lambda: Momentum(0.05),
                                      lambda: RMSprop(0.01), lambda: Adam(0.01)])
def test_every_optimizer_reduces_the_cost(make_opt):
    X, Y = make_moons_dataset(300, 0.2, seed=0)
    params, costs = train(X, Y, [2, 8, 4, 1], make_opt(), num_epochs=30, batch_size=32, seed=0)
    assert costs[-1] < costs[0]
    assert np.mean(predict(params, X) == Y) > 0.8
