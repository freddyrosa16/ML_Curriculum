"""Tests for 02-vision/01-cnn.

PyTorch is used only as a trusted reference to compare your NumPy results against.
"""
import numpy as np
import pytest

from helpers import load_task

torch = pytest.importorskip("torch")
F = torch.nn.functional


def relu(z):
    return np.maximum(z, 0)


def identity(z):
    return z


def same_pad(size, k, s):
    return ((size - 1) * s + k - size) // 2


def torch_conv(A_prev, W, b, padding, stride):
    """Reference conv in channels-last layout, no activation."""
    kh, kw = W.shape[:2]
    if padding == "same":
        pad = (same_pad(A_prev.shape[1], kh, stride[0]),
               same_pad(A_prev.shape[2], kw, stride[1]))
    else:
        pad = (0, 0)
    x = torch.tensor(A_prev).permute(0, 3, 1, 2)
    w = torch.tensor(W).permute(3, 2, 0, 1)
    out = F.conv2d(x, w, torch.tensor(b.reshape(-1)), stride=stride, padding=pad)
    return out.permute(0, 2, 3, 1).numpy()


# ---------- Task 0 ----------
@pytest.mark.parametrize("padding,stride,kshape", [
    ("valid", (1, 1), (3, 3)),
    ("same", (1, 1), (3, 3)),
    ("same", (1, 1), (5, 5)),
    ("valid", (2, 2), (3, 3)),
    ("valid", (2, 1), (4, 2)),
])
def test_conv_forward_matches_reference(padding, stride, kshape):
    conv_forward = load_task(__file__, "0-conv_forward.py", "conv_forward")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 11, 10, 2))
    W = rng.standard_normal((*kshape, 2, 4))
    b = rng.standard_normal((1, 1, 1, 4))
    out = conv_forward(A, W, b, identity, padding=padding, stride=stride)
    expected = torch_conv(A, W, b, padding, stride)
    assert out.shape == expected.shape
    assert np.allclose(out, expected)


def test_conv_forward_applies_activation_and_same_keeps_size():
    conv_forward = load_task(__file__, "0-conv_forward.py", "conv_forward")
    rng = np.random.default_rng(1)
    A = rng.standard_normal((2, 28, 28, 1))
    W = rng.standard_normal((3, 3, 1, 2))
    b = rng.standard_normal((1, 1, 1, 2))
    out = conv_forward(A, W, b, relu)  # defaults: same padding, stride 1
    assert out.shape == (2, 28, 28, 2)
    assert np.allclose(out, relu(torch_conv(A, W, b, "same", (1, 1))))


def test_conv_forward_valid_shape():
    conv_forward = load_task(__file__, "0-conv_forward.py", "conv_forward")
    A = np.zeros((5, 28, 28, 1))
    W = np.ones((3, 3, 1, 2))
    b = np.zeros((1, 1, 1, 2))
    assert conv_forward(A, W, b, relu, padding="valid").shape == (5, 26, 26, 2)


# ---------- Task 1 ----------
@pytest.mark.parametrize("mode", ["max", "avg"])
@pytest.mark.parametrize("kernel,stride", [((2, 2), (2, 2)), ((3, 3), (1, 1)), ((3, 2), (2, 1))])
def test_pool_forward_matches_reference(mode, kernel, stride):
    pool_forward = load_task(__file__, "1-pool_forward.py", "pool_forward")
    rng = np.random.default_rng(2)
    A = rng.standard_normal((3, 9, 8, 2))
    out = pool_forward(A, kernel, stride=stride, mode=mode)
    x = torch.tensor(A).permute(0, 3, 1, 2)
    pool = F.max_pool2d if mode == "max" else F.avg_pool2d
    expected = pool(x, kernel, stride=stride).permute(0, 2, 3, 1).numpy()
    assert out.shape == expected.shape
    assert np.allclose(out, expected)


def test_pool_forward_default_is_max():
    pool_forward = load_task(__file__, "1-pool_forward.py", "pool_forward")
    A = np.arange(16, dtype=float).reshape(1, 4, 4, 1)
    out = pool_forward(A, (2, 2), stride=(2, 2))
    assert np.allclose(out[0, :, :, 0], [[5, 7], [13, 15]])


# ---------- Task 2 ----------
@pytest.mark.parametrize("padding,stride,kshape", [
    ("valid", (1, 1), (3, 3)),
    ("same", (1, 1), (3, 3)),
    ("same", (1, 1), (1, 1)),
    ("valid", (2, 2), (3, 3)),
    ("same", (2, 2), (3, 3)),
])
def test_conv_backward_matches_autograd(padding, stride, kshape):
    conv_backward = load_task(__file__, "2-conv_backward.py", "conv_backward")
    rng = np.random.default_rng(3)
    A = rng.standard_normal((2, 8, 7, 3))
    W = rng.standard_normal((*kshape, 3, 2))
    b = rng.standard_normal((1, 1, 1, 2))

    kh, kw = kshape
    pad = (same_pad(8, kh, stride[0]), same_pad(7, kw, stride[1])) if padding == "same" else (0, 0)
    x = torch.tensor(A).permute(0, 3, 1, 2).requires_grad_()
    w = torch.tensor(W).permute(3, 2, 0, 1).detach().requires_grad_()
    bt = torch.tensor(b.reshape(-1), requires_grad=True)
    Z = F.conv2d(x, w, bt, stride=stride, padding=pad)
    dZ = rng.standard_normal(tuple(Z.permute(0, 2, 3, 1).shape))
    Z.backward(torch.tensor(dZ).permute(0, 3, 1, 2))

    dA_prev, dW, db = conv_backward(dZ, A, W, b, padding=padding, stride=stride)
    assert dA_prev.shape == A.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape
    assert np.allclose(dA_prev, x.grad.permute(0, 2, 3, 1).numpy())
    assert np.allclose(dW, w.grad.permute(2, 3, 1, 0).numpy())
    assert np.allclose(db.reshape(-1), bt.grad.numpy())


# ---------- Task 3 ----------
@pytest.mark.parametrize("mode", ["max", "avg"])
@pytest.mark.parametrize("kernel,stride", [((2, 2), (2, 2)), ((3, 3), (1, 1)), ((3, 2), (2, 1))])
def test_pool_backward_matches_autograd(mode, kernel, stride):
    pool_backward = load_task(__file__, "3-pool_backward.py", "pool_backward")
    rng = np.random.default_rng(4)
    A = rng.standard_normal((2, 9, 8, 3))
    x = torch.tensor(A).permute(0, 3, 1, 2).requires_grad_()
    pool = F.max_pool2d if mode == "max" else F.avg_pool2d
    out = pool(x, kernel, stride=stride)
    dA = rng.standard_normal(tuple(out.permute(0, 2, 3, 1).shape))
    out.backward(torch.tensor(dA).permute(0, 3, 1, 2))

    dA_prev = pool_backward(dA, A, kernel, stride=stride, mode=mode)
    assert dA_prev.shape == A.shape
    assert np.allclose(dA_prev, x.grad.permute(0, 2, 3, 1).numpy())


# ---------- Task 4 ----------
def test_lenet5_shapes_and_params():
    lenet5 = load_task(__file__, "4-lenet5.py", "lenet5")
    torch.manual_seed(0)
    model = lenet5()
    assert isinstance(model, torch.nn.Module)
    out = model(torch.zeros(4, 1, 28, 28))
    assert tuple(out.shape) == (4, 10)
    assert sum(p.numel() for p in model.parameters()) == 61706


def test_lenet5_layers_and_init():
    lenet5 = load_task(__file__, "4-lenet5.py", "lenet5")
    torch.manual_seed(0)
    model = lenet5()
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert [c.out_channels for c in convs] == [6, 16]
    assert [c.kernel_size for c in convs] == [(5, 5), (5, 5)]
    assert [l.out_features for l in linears] == [120, 84, 10]
    assert any(isinstance(m, torch.nn.MaxPool2d) for m in model.modules())
    # He normal: std close to sqrt(2 / fan_in); biases zero
    fc1 = linears[0]
    expected_std = np.sqrt(2 / fc1.in_features)
    assert abs(fc1.weight.std().item() - expected_std) / expected_std < 0.1
    for m in convs + linears:
        assert torch.all(m.bias == 0)


# ---------- Task 5 ----------
def _toy_digits(n=600, seed=0):
    """10 classes; each class lights up a different 7x7 block of a 28x28 image."""
    rng = np.random.default_rng(seed)
    Y = rng.integers(0, 10, n)
    X = rng.normal(0, 0.3, (n, 1, 28, 28)).astype(np.float32)
    for i, y in enumerate(Y):
        r, c = divmod(int(y), 4)
        X[i, 0, r * 7:(r + 1) * 7, c * 7:(c + 1) * 7] += 1.0
    return X, Y


def test_train_lenet5_learns():
    lenet5 = load_task(__file__, "4-lenet5.py", "lenet5")
    train = load_task(__file__, "5-train_lenet5.py", "train_lenet5")
    torch.manual_seed(0)
    X, Y = _toy_digits()
    model = lenet5()
    history = train(model, X, Y, epochs=3, batch_size=32, device="cpu")
    assert set(history) >= {"loss", "accuracy"}
    assert len(history["loss"]) == 3 and len(history["accuracy"]) == 3
    assert all(isinstance(v, float) for v in history["loss"])
    assert history["loss"][-1] < history["loss"][0]
    assert 0.0 <= history["accuracy"][-1] <= 1.0
    assert history["accuracy"][-1] > 0.9
    # the model itself was trained (not a copy)
    with torch.no_grad():
        preds = model.cpu()(torch.tensor(X)).argmax(1).numpy()
    assert (preds == Y).mean() > 0.9


def test_train_lenet5_accepts_tensors_and_default_device():
    lenet5 = load_task(__file__, "4-lenet5.py", "lenet5")
    train = load_task(__file__, "5-train_lenet5.py", "train_lenet5")
    torch.manual_seed(0)
    X, Y = _toy_digits(n=64)
    history = train(lenet5(), torch.tensor(X), torch.tensor(Y), epochs=1)
    assert len(history["accuracy"]) == 1
