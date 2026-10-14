"""Tests for 02-optimization."""
import numpy as np
import pytest

from helpers import load_task


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[1.0, -5.0, 100.0], scale=[2.0, 0.5, 30.0], size=(200, 3))
    return X


def torch_or_skip():
    return pytest.importorskip("torch")


# 0-1 -------------------------------------------------------------------------

def test_normalization_constants(data):
    f = load_task(__file__, "0-norm_constants.py", "normalization_constants")
    m, s = f(data)
    assert m.shape == (3,) and s.shape == (3,)
    assert np.allclose(m, data.mean(axis=0))
    assert np.allclose(s, data.std(axis=0))


def test_normalize(data):
    f = load_task(__file__, "1-normalize.py", "normalize")
    m, s = data.mean(axis=0), data.std(axis=0)
    Xn = f(data, m, s)
    assert Xn.shape == data.shape
    assert np.allclose(Xn.mean(axis=0), 0)
    assert np.allclose(Xn.std(axis=0), 1)
    # works with constants computed on a different set (e.g. train stats on test data)
    assert np.allclose(f(np.array([[3.0, -5.0, 130.0]]), np.array([1.0, -5.0, 100.0]),
                         np.array([2.0, 1.0, 30.0])), [[1.0, 0.0, 1.0]])


# 2-3 -------------------------------------------------------------------------

def test_shuffle_data():
    f = load_task(__file__, "2-shuffle_data.py", "shuffle_data")
    X = np.arange(20).reshape(10, 2)
    Y = np.arange(10).reshape(10, 1) * 100
    np.random.seed(0)
    Xs, Ys = f(X, Y)
    np.random.seed(0)
    perm = np.random.permutation(10)
    assert np.array_equal(Xs, X[perm]) and np.array_equal(Ys, Y[perm])
    assert np.array_equal(X, np.arange(20).reshape(10, 2)), "don't modify the input"


def test_mini_batches():
    f = load_task(__file__, "3-mini_batch.py", "create_mini_batches")
    X = np.arange(46).reshape(23, 2)
    Y = np.arange(23).reshape(23, 1)
    np.random.seed(1)
    batches = f(X, Y, 5)
    assert isinstance(batches, list)
    assert [len(xb) for xb, _ in batches] == [5, 5, 5, 5, 3]
    for xb, yb in batches:
        assert xb.shape[1] == 2 and yb.shape[1] == 1
        assert np.array_equal(xb[:, 0] // 2, yb[:, 0]), "rows of X and Y got separated"
    allX = np.concatenate([xb for xb, _ in batches])
    np.random.seed(1)
    assert np.array_equal(allX, X[np.random.permutation(23)]), \
        "shuffle exactly once with shuffle_data, then slice in order"


def test_mini_batches_exact_division():
    f = load_task(__file__, "3-mini_batch.py", "create_mini_batches")
    X = np.zeros((12, 3))
    Y = np.zeros((12, 1))
    assert [len(xb) for xb, _ in f(X, Y, 4)] == [4, 4, 4]
    assert [len(xb) for xb, _ in f(X, Y, 32)] == [12]


# 4 ---------------------------------------------------------------------------

def test_moving_average():
    f = load_task(__file__, "4-moving_average.py", "moving_average")
    out = f([72, 78, 71, 68, 66, 69, 70], 0.9)
    assert isinstance(out, list) and len(out) == 7
    assert np.isclose(out[0], 72)  # bias correction makes the first value exact
    assert np.allclose(out[:3], [72.0, 75.15789473684211, 73.62361623616238])
    assert np.allclose(f([5, 5, 5, 5], 0.5), [5, 5, 5, 5])


# 5, 7, 9 ---------------------------------------------------------------------

def test_momentum():
    f = load_task(__file__, "5-momentum.py", "update_variables_momentum")
    var, grad, v = np.array([1.0, 2.0]), np.array([0.5, -1.0]), np.array([0.2, 0.1])
    new_var, new_v = f(0.1, 0.9, var, grad, v)
    assert np.allclose(new_v, 0.9 * v + 0.1 * grad)
    assert np.allclose(new_var, var - 0.1 * new_v)


def test_rmsprop():
    f = load_task(__file__, "7-RMSProp.py", "update_variables_RMSProp")
    var, grad, s = np.array([1.0, 2.0]), np.array([0.5, -1.0]), np.array([0.2, 0.1])
    new_var, new_s = f(0.1, 0.9, 1e-8, var, grad, s)
    assert np.allclose(new_s, 0.9 * s + 0.1 * grad ** 2)
    assert np.allclose(new_var, var - 0.1 * grad / (np.sqrt(new_s) + 1e-8))


def test_adam_first_step():
    f = load_task(__file__, "9-Adam.py", "update_variables_Adam")
    var, grad = np.array([1.0, 2.0]), np.array([0.5, -1.0])
    new_var, v, s = f(0.1, 0.9, 0.99, 1e-8, var, grad, np.zeros(2), np.zeros(2), 1)
    assert np.allclose(v, [0.05, -0.1])
    assert np.allclose(s, [0.0025, 0.01])
    # with bias correction, Adam's very first step has size ~alpha in every direction
    assert np.allclose(new_var, [0.9, 2.1])


def test_adam_later_step():
    f = load_task(__file__, "9-Adam.py", "update_variables_Adam")
    var, grad = np.array([1.0]), np.array([2.0])
    v, s = np.array([0.3]), np.array([0.04])
    new_var, new_v, new_s = f(0.01, 0.9, 0.999, 1e-7, var, grad, v, s, 3)
    assert np.allclose(new_v, 0.9 * 0.3 + 0.1 * 2.0)
    assert np.allclose(new_s, 0.999 * 0.04 + 0.001 * 4.0)
    v_hat = new_v / (1 - 0.9 ** 3)
    s_hat = new_s / (1 - 0.999 ** 3)
    assert np.allclose(new_var, 1.0 - 0.01 * v_hat / (np.sqrt(s_hat) + 1e-7))


# 6, 8, 10 (PyTorch) ----------------------------------------------------------

def _quadratic_steps(torch, opt, w, n):
    """Run n steps minimizing sum((w - 3)^2 * [1, 10]); return w history."""
    scale = torch.tensor([1.0, 10.0])
    out = []
    for _ in range(n):
        opt.zero_grad()
        loss = ((w - 3) ** 2 * scale).sum()
        loss.backward()
        opt.step()
        out.append(w.detach().clone().numpy())
    return out


def test_momentum_op():
    torch = torch_or_skip()
    f = load_task(__file__, "6-momentum.py", "create_momentum_op")
    w = torch.nn.Parameter(torch.zeros(2))
    opt = f([w], 0.01, 0.9)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["lr"] == 0.01
    assert opt.param_groups[0]["momentum"] == 0.9
    hist = _quadratic_steps(torch, opt, w, 200)
    assert np.allclose(hist[-1], [3, 3], atol=1e-2)


def test_rmsprop_op_matches_numpy():
    torch = torch_or_skip()
    f = load_task(__file__, "8-RMSProp.py", "create_RMSProp_op")
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    opt = f([w], 0.1, 0.9, 1e-8)
    assert isinstance(opt, torch.optim.RMSprop)
    assert opt.param_groups[0]["lr"] == 0.1
    var, s = np.array([1.0, 2.0]), np.zeros(2)
    for _ in range(3):
        opt.zero_grad()
        (w ** 2).sum().backward()
        grad = 2 * var
        opt.step()
        s = 0.9 * s + 0.1 * grad ** 2
        var = var - 0.1 * grad / (np.sqrt(s) + 1e-8)
        assert np.allclose(w.detach().numpy(), var, atol=1e-6)


def test_adam_op_matches_numpy():
    torch = torch_or_skip()
    f = load_task(__file__, "10-Adam.py", "create_Adam_op")
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    opt = f([w], 0.1, 0.9, 0.99, 1e-8)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["betas"] == (0.9, 0.99)
    assert opt.param_groups[0]["eps"] == 1e-8
    var, v, s = np.array([1.0, 2.0]), np.zeros(2), np.zeros(2)
    for t in range(1, 4):
        opt.zero_grad()
        (w ** 2).sum().backward()
        grad = 2 * var
        opt.step()
        v = 0.9 * v + 0.1 * grad
        s = 0.99 * s + 0.01 * grad ** 2
        var = var - 0.1 * (v / (1 - 0.9 ** t)) / (np.sqrt(s / (1 - 0.99 ** t)) + 1e-8)
        assert np.allclose(w.detach().numpy(), var, atol=1e-6)


# 11-12 -----------------------------------------------------------------------

def test_learning_rate_decay_numpy():
    f = load_task(__file__, "11-learning_rate_decay.py", "learning_rate_decay")
    got = [f(0.1, 1, step, 10) for step in (0, 9, 10, 19, 20, 25)]
    assert np.allclose(got, [0.1, 0.1, 0.05, 0.05, 0.1 / 3, 0.1 / 3])
    assert np.isclose(f(0.5, 0.5, 100, 10), 0.5 / 6)


def test_learning_rate_decay_torch():
    torch = torch_or_skip()
    f = load_task(__file__, "12-learning_rate_decay.py", "learning_rate_decay")
    w = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([w], lr=0.1)
    sched = f(opt, 1, 10)
    lrs = []
    for _ in range(31):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    expected = [0.1 / (1 + (k // 10)) for k in range(31)]
    assert np.allclose(lrs, expected)


# 13-14 -----------------------------------------------------------------------

def test_batch_norm():
    f = load_task(__file__, "13-batch_norm.py", "batch_norm")
    rng = np.random.default_rng(3)
    Z = rng.normal(5, 3, size=(50, 4))
    out = f(Z, np.ones((1, 4)), np.zeros((1, 4)), 1e-8)
    assert out.shape == (50, 4)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(out.std(axis=0), 1, atol=1e-6)
    gamma, beta = rng.uniform(1, 2, (1, 4)), rng.normal(size=(1, 4))
    out2 = f(Z, gamma, beta, 1e-8)
    assert np.allclose(out2, out * gamma + beta, atol=1e-6)
    # epsilon belongs inside the square root
    const = np.ones((3, 1)) * 7
    assert np.allclose(f(const, np.ones((1, 1)), np.zeros((1, 1)), 1e-8), 0)


def test_batch_norm_layer_structure():
    torch = torch_or_skip()
    f = load_task(__file__, "14-batch_norm.py", "create_batch_norm_layer")
    layer = f(4, 3, torch.nn.Tanh())
    assert isinstance(layer, torch.nn.Module)
    mods = [m for m in layer.modules() if not isinstance(m, torch.nn.Sequential)]
    kinds = [type(m) for m in mods]
    assert kinds == [torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.Tanh]
    linear, bn = mods[0], mods[1]
    assert linear.bias is None, "the linear layer should have no bias"
    assert linear.in_features == 4 and linear.out_features == 3
    assert bn.eps == 1e-7
    assert torch.all(bn.weight == 1) and torch.all(bn.bias == 0)


def test_batch_norm_layer_init_and_output():
    torch = torch_or_skip()
    f = load_task(__file__, "14-batch_norm.py", "create_batch_norm_layer")
    torch.manual_seed(0)
    layer = f(400, 300, torch.nn.Identity())
    linear = [m for m in layer.modules() if isinstance(m, torch.nn.Linear)][0]
    std = linear.weight.detach().std().item()
    assert abs(std - np.sqrt(2 / 700)) < 0.003, "use Xavier/Glorot *normal* init"
    layer.train()
    x = torch.randn(64, 400) * 5 + 2
    out = layer(x).detach().numpy()
    assert out.shape == (64, 300)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(out.std(axis=0), 1, atol=2e-2)
