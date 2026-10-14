"""Tests for 05-pytorch_basics (CPU only, small synthetic data)."""
import copy
import math

import numpy as np
import pytest

from helpers import load_task, project_dir

torch = pytest.importorskip("torch")
nn = torch.nn


@pytest.fixture
def blobs():
    """Three well-separated Gaussian blobs in 5 dimensions."""
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=3.0, size=(3, 5))
    y = rng.integers(0, 3, size=400)
    X = centers[y] + rng.normal(size=(400, 5))
    return X[:300], y[:300], X[300:], y[300:]


# 0 -----------------------------------------------------------------------------

def test_to_tensors():
    f = load_task(__file__, "0-to_tensors.py", "to_tensors")
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    Y = np.array([2, 0, 1], dtype=np.int32)
    Xt, Yt = f(X, Y)
    assert torch.is_tensor(Xt) and torch.is_tensor(Yt)
    assert Xt.dtype == torch.float32 and Yt.dtype == torch.int64
    assert Xt.shape == (3, 2) and Yt.shape == (3,)
    assert torch.equal(Xt, torch.tensor([[0.0, 1], [2, 3], [4, 5]]))
    assert Xt.device.type == "cpu"
    Xt2, _ = f(X, Y, device="cpu")
    assert Xt2.device.type == "cpu"


# 1 -----------------------------------------------------------------------------

def test_gradient_at():
    f = load_task(__file__, "1-autograd.py", "gradient_at")
    val, grad = f(lambda t: t ** 3 + 2 * t, 2.0)
    assert isinstance(val, float) and isinstance(grad, float)
    assert val == pytest.approx(12.0) and grad == pytest.approx(14.0)
    val, grad = f(lambda t: t ** 2 * torch.exp(t), 1.5)
    assert grad == pytest.approx((2 * 1.5 + 1.5 ** 2) * math.exp(1.5), rel=1e-9)
    assert f(torch.sin, 0.0) == pytest.approx((0.0, 1.0))
    assert f(lambda t: torch.log(t), 4) == pytest.approx((math.log(4), 0.25))


# 2 -----------------------------------------------------------------------------

def test_create_layer_structure():
    f = load_task(__file__, "2-create_layer.py", "create_layer")
    layer = f(8, 4, "tanh")
    mods = [type(m) for m in layer.modules() if not isinstance(m, nn.Sequential)]
    assert mods == [nn.Linear, nn.Tanh]
    assert [type(m) for m in f(8, 4, "relu").modules()
            if not isinstance(m, nn.Sequential)] == [nn.Linear, nn.ReLU]
    assert [type(m) for m in f(8, 4, "sigmoid").modules()
            if not isinstance(m, nn.Sequential)] == [nn.Linear, nn.Sigmoid]
    plain = f(8, 4)
    assert [type(m) for m in plain.modules() if not isinstance(m, nn.Sequential)] == [nn.Linear]
    with pytest.raises(ValueError):
        f(8, 4, "swish")


def test_create_layer_init_and_forward():
    f = load_task(__file__, "2-create_layer.py", "create_layer")
    torch.manual_seed(0)
    layer = f(500, 300, "tanh")
    lin = [m for m in layer.modules() if isinstance(m, nn.Linear)][0]
    bound = math.sqrt(6 / 800)
    w = lin.weight.detach()
    assert w.shape == (300, 500)
    assert w.abs().max() <= bound + 1e-6
    assert abs(w.std().item() - bound / math.sqrt(3)) < 0.002, "use Xavier *uniform*"
    assert torch.all(lin.bias == 0)
    x = torch.randn(7, 500)
    assert torch.allclose(layer(x), torch.tanh(x @ w.T))


# 3 -----------------------------------------------------------------------------

def test_network():
    Network = load_task(__file__, "3-forward_prop.py", "Network")
    torch.manual_seed(0)
    net = Network(6, [10, 8, 3], ["relu", "tanh", None])
    assert isinstance(net, nn.Module)
    assert net.nx == 6 and net.layer_sizes == [10, 8, 3]
    assert net.activations == ["relu", "tanh", None]
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(6, 10), (10, 8), (8, 3)]
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params == 6 * 10 + 10 + 10 * 8 + 8 + 8 * 3 + 3, \
        "are all layers registered as submodules?"
    x = torch.randn(4, 6)
    h = torch.relu(x @ linears[0].weight.T + linears[0].bias)
    h = torch.tanh(h @ linears[1].weight.T + linears[1].bias)
    out = h @ linears[2].weight.T + linears[2].bias
    assert torch.allclose(net(x), out, atol=1e-6)
    with pytest.raises(ValueError):
        Network(6, [10, 3], ["relu"])


# 4-5 ---------------------------------------------------------------------------

def test_calculate_accuracy():
    f = load_task(__file__, "4-calculate_accuracy.py", "calculate_accuracy")
    y = torch.tensor([0, 1, 2, 1])
    y_pred = torch.tensor([[2.0, 1, 0], [0, 3, 1], [1, 0, 0.5], [0, 0, 1]])
    acc = f(y, y_pred)
    assert torch.is_tensor(acc) and acc.dim() == 0 and acc.is_floating_point()
    assert acc.item() == pytest.approx(0.5)
    assert f(torch.tensor([1, 1]), torch.tensor([[0.0, 5], [-1, 2]])).item() == 1.0


def test_calculate_loss():
    f = load_task(__file__, "5-calculate_loss.py", "calculate_loss")
    torch.manual_seed(0)
    y_pred = torch.randn(10, 4, requires_grad=True)
    y = torch.randint(0, 4, (10,))
    loss = f(y, y_pred)
    assert loss.dim() == 0
    expected = -torch.log_softmax(y_pred, dim=1)[torch.arange(10), y].mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert y_pred.grad is not None


# 6 -----------------------------------------------------------------------------

def test_train_op_and_step():
    mod = load_task(__file__, "6-create_train_op.py")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))
    opt = mod.create_train_op(model, 0.2)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["lr"] == 0.2
    assert opt.param_groups[0]["momentum"] == 0
    x, y = torch.randn(16, 5), torch.randint(0, 3, (16,))
    ref = copy.deepcopy(model)
    ref_loss = nn.functional.cross_entropy(ref(x), y)
    ref_loss.backward()
    loss = mod.train_step(model, opt, x, y)
    assert isinstance(loss, float)
    assert loss == pytest.approx(ref_loss.item(), rel=1e-6)
    for p, q in zip(model.parameters(), ref.parameters()):
        assert torch.allclose(p, q - 0.2 * q.grad, atol=1e-6)
    # gradients must be cleared each step, not accumulated
    ref2 = copy.deepcopy(model)
    nn.functional.cross_entropy(ref2(x), y).backward()
    mod.train_step(model, opt, x, y)
    for p, q in zip(model.parameters(), ref2.parameters()):
        assert torch.allclose(p, q - 0.2 * q.grad, atol=1e-6), "did you zero the gradients?"


# 7-8 ---------------------------------------------------------------------------

def test_get_device():
    mod = load_task(__file__, "7-train.py")
    dev = mod.get_device()
    assert torch.device(dev).type in {"cpu", "cuda", "mps"}
    if torch.cuda.is_available():
        assert torch.device(dev).type == "cuda"


def test_train_and_evaluate(blobs, tmp_path, capsys):
    train = load_task(__file__, "7-train.py", "train")
    evaluate = load_task(__file__, "8-evaluate.py", "evaluate")
    X, y, Xv, yv = blobs
    path = str(tmp_path / "ckpt.pt")
    torch.manual_seed(0)
    out = train(X, y, Xv, yv, [16, 3], ["tanh", None], 0.1, 250, save_path=path, device="cpu")
    assert out == path
    lines = capsys.readouterr().out.splitlines()
    assert [l for l in lines if l.startswith("After")] == [
        "After 0 iterations:", "After 100 iterations:",
        "After 200 iterations:", "After 250 iterations:"]
    assert lines[1].startswith("\tTraining Cost: ")
    assert lines[2].startswith("\tTraining Accuracy: ")
    assert lines[3].startswith("\tValidation Cost: ")
    assert lines[4].startswith("\tValidation Accuracy: ")
    first_cost = float(lines[1].split(": ")[1])
    last_cost = float(lines[-4].split(": ")[1])
    assert last_cost < first_cost

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert set(ckpt) >= {"nx", "layer_sizes", "activations", "state_dict"}
    assert ckpt["nx"] == 5 and list(ckpt["layer_sizes"]) == [16, 3]

    y_pred, acc, loss = evaluate(Xv, yv, path)
    assert torch.is_tensor(y_pred) and y_pred.shape == (100, 3)
    assert isinstance(acc, float) and isinstance(loss, float)
    assert acc > 0.9
    assert acc == pytest.approx(float(lines[-1].split(": ")[1]), abs=1e-6)
    assert loss == pytest.approx(float(lines[-2].split(": ")[1]), rel=1e-5)


@pytest.mark.slow
def test_mnist(tmp_path):
    path = project_dir(__file__).parent / "01-classification" / "data" / "MNIST.npz"
    if not path.exists():
        pytest.skip("01-classification/data/MNIST.npz not found")
    train = load_task(__file__, "7-train.py", "train")
    evaluate = load_task(__file__, "8-evaluate.py", "evaluate")
    lib = np.load(path)
    X = lib["X_train"][:10000].reshape(10000, -1)
    Y = lib["Y_train"][:10000]
    Xv = lib["X_valid"][:2000].reshape(2000, -1)
    Yv = lib["Y_valid"][:2000]
    torch.manual_seed(0)
    ckpt = train(X, Y, Xv, Yv, [256, 256, 10], ["tanh", "tanh", None], 0.1, 300,
                 save_path=str(tmp_path / "mnist.pt"), device="cpu")
    _, acc, _ = evaluate(Xv, Yv, ckpt)
    assert acc > 0.85
