import numpy as np
import pytest
import torch
from data import to_tensors
from model import DeepNetworkTorch, load_numpy_reference, set_seed, train
from torch import nn

DIMS = [6, 5, 4, 1]


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = (rng.random(40) > 0.5).astype(float)
    return X, y


def test_architecture_and_output_shape():
    model = DeepNetworkTorch(DIMS)
    assert sum(isinstance(m, nn.Linear) for m in model.net) == 3
    assert sum(isinstance(m, nn.ReLU) for m in model.net) == 2
    assert model(torch.randn(8, 6)).shape == (8, 1)


def test_shallow_and_deep_configs_both_work():
    for dims in ([6, 4, 1], [6, 20, 7, 5, 1]):
        assert DeepNetworkTorch(dims)(torch.randn(3, 6)).shape == (3, 1)


def test_he_init_scale_and_zero_biases():
    set_seed(0)
    model = DeepNetworkTorch([1000, 500, 1], init="he")
    assert model.linears[0].weight.std().item() == pytest.approx(np.sqrt(2 / 1000), rel=0.05)
    assert torch.count_nonzero(model.linears[0].bias) == 0


def test_unknown_init_raises():
    with pytest.raises(ValueError):
        DeepNetworkTorch(DIMS, init="zeros")


def test_forward_matches_numpy(toy):
    X, _ = toy
    ref = load_numpy_reference()
    params = ref.initialize_parameters(DIMS, "he", seed=1)
    AL, _ = ref.forward_propagation(X.T, params)

    model = DeepNetworkTorch(DIMS).double()
    model.load_numpy_params(params)
    probs = model.predict_proba(torch.as_tensor(X)).detach().numpy()
    np.testing.assert_allclose(probs.T, AL, atol=1e-10)


def test_gradients_match_numpy(toy):
    X, y = toy
    ref = load_numpy_reference()
    params = ref.initialize_parameters(DIMS, "he", seed=1)
    AL, caches = ref.forward_propagation(X.T, params)
    grads = ref.backward_propagation(AL, y.reshape(1, -1), caches)

    model = DeepNetworkTorch(DIMS).double()
    model.load_numpy_params(params)
    Xt, yt = to_tensors(X, y, dtype=torch.float64)
    nn.BCEWithLogitsLoss()(model(Xt), yt).backward()

    for l, layer in enumerate(model.linears, start=1):
        np.testing.assert_allclose(layer.weight.grad.numpy(), grads[f"dW{l}"], atol=1e-8)
        np.testing.assert_allclose(layer.bias.grad.numpy(), grads[f"db{l}"].ravel(), atol=1e-8)


def test_training_matches_numpy(toy):
    """Same init, same lr, full batch: the loss curves must coincide."""
    X, y = toy
    ref = load_numpy_reference()
    lr, epochs = 0.1, 50
    np_net = ref.DeepNetwork(DIMS, learning_rate=lr, num_iterations=epochs,
                             init="he", seed=1, log_every=1).fit(X.T, y.reshape(1, -1))

    model = DeepNetworkTorch(DIMS).double()
    model.load_numpy_params(ref.initialize_parameters(DIMS, "he", seed=1))
    Xt, yt = to_tensors(X, y, dtype=torch.float64)
    losses = train(model, Xt, yt, torch.optim.SGD(model.parameters(), lr=lr), epochs)

    np.testing.assert_allclose(losses, np_net.costs_, atol=1e-8)


def test_training_reduces_loss_on_nonlinear_problem():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(float)      # XOR-like, not linearly separable
    Xt, yt = to_tensors(X, y)
    set_seed(0)
    model = DeepNetworkTorch([2, 8, 8, 1])
    losses = train(model, Xt, yt, torch.optim.SGD(model.parameters(), lr=0.01), 500)
    assert losses[-1] < losses[0]
    assert ((model.predict_proba(Xt) > 0.5).double() == yt).double().mean() > 0.8
