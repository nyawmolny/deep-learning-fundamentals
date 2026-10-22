import numpy as np
import pytest
import torch
from optimizers import Adam, GradientDescent, Momentum, RMSprop

STEPS, LR, BETA = 50, 0.01, 0.9


def run_both(ours, make_torch_opt):
    rng = np.random.default_rng(0)
    w0, b0 = rng.normal(size=(3, 2)), rng.normal(size=(3, 1))
    grads = [{"dW1": rng.normal(size=(3, 2)), "db1": rng.normal(size=(3, 1))} for _ in range(STEPS)]

    params = {"W1": w0.copy(), "b1": b0.copy()}
    for g in grads:
        params = ours.step(params, g)

    w = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
    opt = make_torch_opt([w, b])
    for g in grads:
        w.grad = torch.tensor(g["dW1"])
        b.grad = torch.tensor(g["db1"])
        opt.step()
    return params, w.detach().numpy(), b.detach().numpy()


@pytest.mark.parametrize("ours, make_torch_opt", [
    (GradientDescent(LR), lambda p: torch.optim.SGD(p, lr=LR)),
    (Momentum(LR, BETA), lambda p: torch.optim.SGD(p, lr=LR * (1 - BETA), momentum=BETA)),
    (RMSprop(LR, BETA), lambda p: torch.optim.RMSprop(p, lr=LR, alpha=BETA, eps=1e-8)),
    (Adam(LR), lambda p: torch.optim.Adam(p, lr=LR, betas=(0.9, 0.999), eps=1e-8)),
], ids=["gd", "momentum", "rmsprop", "adam"])
def test_matches_torch_optim(ours, make_torch_opt):
    params, w, b = run_both(ours, make_torch_opt)
    np.testing.assert_allclose(params["W1"], w, atol=1e-10)
    np.testing.assert_allclose(params["b1"], b, atol=1e-10)
