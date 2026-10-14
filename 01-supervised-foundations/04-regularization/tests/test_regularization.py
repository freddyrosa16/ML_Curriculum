"""Tests for 04-regularization.

Gradient steps are checked against numerical (finite-difference) gradients.
"""
import numpy as np
import pytest

from helpers import load_task


def softmax(z):
    e = np.exp(z - z.max(axis=0))
    return e / e.sum(axis=0)


def make_net(sizes, m=20, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(sizes[0], m))
    labels = rng.integers(0, classes, m)
    Y = np.zeros((classes, m))
    Y[labels, np.arange(m)] = 1
    weights = {}
    for l in range(1, len(sizes)):
        weights[f"W{l}"] = rng.normal(size=(sizes[l], sizes[l - 1])) * np.sqrt(1 / sizes[l - 1])
        weights[f"b{l}"] = rng.normal(size=(sizes[l], 1)) * 0.1
    return X, Y, weights


def forward(X, weights, L, masks=None, keep_prob=1.0):
    """tanh hidden layers (optionally with fixed dropout masks), softmax output."""
    cache = {"A0": X}
    for l in range(1, L + 1):
        Z = weights[f"W{l}"] @ cache[f"A{l - 1}"] + weights[f"b{l}"]
        if l == L:
            cache[f"A{l}"] = softmax(Z)
        else:
            A = np.tanh(Z)
            if masks is not None:
                A = A * masks[f"D{l}"] / keep_prob
            cache[f"A{l}"] = A
    return cache


def ce(Y, A):
    return -np.sum(Y * np.log(A)) / Y.shape[1]


def numerical_grads(weights, f, eps=1e-6):
    grads = {}
    for key, value in weights.items():
        g = np.zeros_like(value)
        for idx in np.ndindex(value.shape):
            old = value[idx]
            value[idx] = old + eps
            plus = f()
            value[idx] = old - eps
            minus = f()
            value[idx] = old
            g[idx] = (plus - minus) / (2 * eps)
        grads[key] = g
    return grads


# 0 -----------------------------------------------------------------------------

def test_l2_reg_cost_numpy():
    f = load_task(__file__, "0-l2_reg_cost.py", "l2_reg_cost")
    _, _, weights = make_net([4, 5, 3])
    penalty = sum(np.sum(weights[f"W{l}"] ** 2) for l in (1, 2))
    assert np.isclose(f(0.7, 0.2, weights, 2, 20), 0.7 + 0.2 / 40 * penalty)
    w = {"W1": np.array([[1.0, 2.0], [3.0, 4.0]]), "b1": np.ones((2, 1)) * 100,
         "W2": np.array([[1.0, -1.0]]), "b2": np.ones((1, 1)) * 100}
    assert np.isclose(f(0.5, 0.1, w, 2, 10), 0.66), "penalize squared weights, not biases"


# 1 -----------------------------------------------------------------------------

def test_l2_reg_gradient_descent():
    f = load_task(__file__, "1-l2_reg_gradient_descent.py", "l2_reg_gradient_descent")
    X, Y, weights = make_net([4, 5, 4, 3], seed=1)
    lambtha, m, L = 0.3, X.shape[1], 3

    def cost():
        reg = sum(np.sum(weights[f"W{l}"] ** 2) for l in range(1, L + 1))
        return ce(Y, forward(X, weights, L)[f"A{L}"]) + lambtha / (2 * m) * reg

    grads = numerical_grads(weights, cost)
    old = {k: v.copy() for k, v in weights.items()}
    cache = forward(X, weights, L)
    assert f(Y, weights, cache, 0.1, lambtha, L) is None
    for k in old:
        assert np.allclose(weights[k], old[k] - 0.1 * grads[k], atol=1e-6), k


# 2-3 (PyTorch) -------------------------------------------------------------------

def test_l2_reg_cost_torch():
    torch = pytest.importorskip("torch")
    f = load_task(__file__, "2-l2_reg_cost.py", "l2_reg_cost")
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                                torch.nn.Sequential(torch.nn.Linear(8, 3)))
    cost = torch.tensor(1.25)
    out = f(cost, model, 0.05)
    expected = 1.25 + 0.05 * sum((m.weight ** 2).sum().item() for m in model.modules()
                                 if isinstance(m, torch.nn.Linear))
    assert torch.is_tensor(out)
    assert abs(out.item() - expected) < 1e-5
    out.backward()
    assert model[0].weight.grad is not None, "the result must be differentiable"
    assert model[0].bias.grad is None or torch.all(model[0].bias.grad == 0), \
        "biases shouldn't be penalized"


def test_l2_reg_optimizer():
    torch = pytest.importorskip("torch")
    f = load_task(__file__, "3-l2_reg_optimizer.py", "l2_reg_optimizer")
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    opt = f(model, 0.1, 0.01)
    assert isinstance(opt, torch.optim.SGD)
    seen = {}
    for group in opt.param_groups:
        assert group["lr"] == 0.1
        for p in group["params"]:
            seen[id(p)] = group["weight_decay"]
    params = list(model.parameters())
    assert len(seen) == len(params), "every parameter should be in the optimizer"
    for p in params:
        assert seen[id(p)] == (0.01 if p.ndim > 1 else 0.0)


def test_l2_reg_optimizer_shrinks_weights():
    torch = pytest.importorskip("torch")
    f = load_task(__file__, "3-l2_reg_optimizer.py", "l2_reg_optimizer")
    lin = torch.nn.Linear(2, 1)
    with torch.no_grad():
        lin.weight.fill_(1.0)
        lin.bias.fill_(1.0)
    opt = f(lin, 0.5, 0.2)
    opt.zero_grad()
    (lin.weight.sum() * 0 + lin.bias.sum() * 0).backward()  # zero loss gradient
    opt.step()
    assert torch.allclose(lin.weight, torch.full((1, 2), 0.9))  # 1 - 0.5 * 0.2
    assert torch.allclose(lin.bias, torch.ones(1))


# 4-5 -----------------------------------------------------------------------------

def test_dropout_forward_prop():
    f = load_task(__file__, "4-dropout_forward_prop.py", "dropout_forward_prop")
    X, _, weights = make_net([10, 200, 150, 3], m=50, seed=2)
    np.random.seed(0)
    cache = f(X, weights, 3, 0.8)
    assert set(cache) == {"A0", "A1", "A2", "A3", "D1", "D2"}
    for l in (1, 2):
        D = cache[f"D{l}"]
        assert D.shape == cache[f"A{l}"].shape
        assert set(np.unique(D)) <= {0, 1}
        assert abs(D.mean() - 0.8) < 0.02
        Z = weights[f"W{l}"] @ cache[f"A{l - 1}"] + weights[f"b{l}"]
        assert np.allclose(cache[f"A{l}"], np.tanh(Z) * D / 0.8)
    assert np.allclose(cache["A3"], softmax(weights["W3"] @ cache["A2"] + weights["b3"]))
    np.random.seed(1)
    other = f(X, weights, 3, 0.8)
    assert not np.array_equal(other["D1"], cache["D1"]), "masks should be random"


def test_dropout_gradient_descent():
    f = load_task(__file__, "5-dropout_gradient_descent.py", "dropout_gradient_descent")
    X, Y, weights = make_net([4, 6, 5, 3], seed=3)
    rng = np.random.default_rng(4)
    keep, L = 0.7, 3
    masks = {f"D{l}": (rng.random((n, X.shape[1])) < keep).astype(int)
             for l, n in ((1, 6), (2, 5))}

    def cost():
        return ce(Y, forward(X, weights, L, masks, keep)[f"A{L}"])

    grads = numerical_grads(weights, cost)
    old = {k: v.copy() for k, v in weights.items()}
    cache = forward(X, weights, L, masks, keep)
    cache.update(masks)
    assert f(Y, weights, cache, 0.2, keep, L) is None
    for k in old:
        assert np.allclose(weights[k], old[k] - 0.2 * grads[k], atol=1e-6), k


# 6 (PyTorch) ---------------------------------------------------------------------

def test_dropout_create_layer():
    torch = pytest.importorskip("torch")
    f = load_task(__file__, "6-dropout_create_layer.py", "dropout_create_layer")
    torch.manual_seed(0)
    layer = f(20, 500, torch.nn.Sigmoid(), 0.75)  # sigmoid output is never exactly 0
    lin = [m for m in layer.modules() if isinstance(m, torch.nn.Linear)]
    assert len(lin) == 1 and lin[0].in_features == 20 and lin[0].out_features == 500
    x = torch.randn(64, 20)
    layer.train()
    out = layer(x)
    assert out.shape == (64, 500)
    zero_frac = (out == 0).float().mean().item()
    assert abs(zero_frac - 0.25) < 0.02, "keep_prob is the probability of KEEPING a unit"
    layer.eval()
    clean = layer(x)
    assert (clean == 0).sum() == 0
    with torch.no_grad():
        expected = torch.sigmoid(lin[0](x))
    assert torch.allclose(clean, expected)
    kept = out != 0
    assert torch.allclose(out[kept], clean[kept] / 0.75, atol=1e-6), "use inverted dropout scaling"


# 7 -----------------------------------------------------------------------------

def test_early_stopping():
    f = load_task(__file__, "7-early_stopping.py", "early_stopping")
    assert f(1.0, 1.5, 0.1, 3, 2) == (False, 0)
    assert f(1.45, 1.5, 0.1, 3, 1) == (False, 2)
    assert f(1.45, 1.5, 0.1, 3, 2) == (True, 3)
    assert f(1.25, 1.5, 0.25, 1, 0) == (True, 1)  # an improvement of exactly threshold doesn't count
    assert f(2.0, 1.5, 0.0, 5, 0) == (False, 1)
