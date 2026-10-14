"""Tests for 06-pytorch_training (CPU only, small synthetic data)."""
import copy
import json

import numpy as np
import pytest

from helpers import load_task

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

TRAIN_FILES = [f"{i}-train.py" for i in range(4, 9)]


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(3, 6, generator=g) * 3
    y = torch.randint(0, 3, (260,), generator=g)
    X = centers[y] + torch.randn(260, 6, generator=g)
    return X[:200], y[:200], X[200:], y[200:]


def small_model(seed=0, keep=1.0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Dropout(1 - keep), nn.Linear(16, 3))


def kinds(model):
    return [type(m) for m in model.modules() if not isinstance(m, nn.Sequential)]


def train_fn(fname):
    return load_task(__file__, fname, "train_model")


# 0-1 ---------------------------------------------------------------------------

def test_build_model_sequential():
    build_model = load_task(__file__, "0-sequential.py", "build_model")
    model = build_model(10, [8, 6, 3], ["relu", "tanh", None], 0.75)
    assert isinstance(model, nn.Sequential)
    assert kinds(model) == [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.Tanh,
                            nn.Dropout, nn.Linear]
    drops = [m for m in model if isinstance(m, nn.Dropout)]
    assert all(abs(d.p - 0.25) < 1e-9 for d in drops)
    linears = [m for m in model if isinstance(m, nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(10, 8), (8, 6), (6, 3)]
    assert model.config == {"nx": 10, "layers": [8, 6, 3],
                            "activations": ["relu", "tanh", None], "keep_prob": 0.75}
    with pytest.raises(ValueError):
        build_model(10, [8, 3], ["relu"], 0.5)


def test_build_model_subclass():
    build_model = load_task(__file__, "1-input.py", "build_model")
    torch.manual_seed(0)
    model = build_model(10, [8, 6, 3], ["relu", "sigmoid", None], 0.5)
    assert isinstance(model, nn.Module) and not isinstance(model, nn.Sequential)
    params = list(model.parameters())
    assert [tuple(p.shape) for p in params] == [(8, 10), (8,), (6, 8), (6,), (3, 6), (3,)]
    x = torch.randn(5, 10)
    model.eval()
    W1, b1, W2, b2, W3, b3 = params
    expected = torch.sigmoid(torch.relu(x @ W1.T + b1) @ W2.T + b2) @ W3.T + b3
    assert torch.allclose(model(x), expected, atol=1e-6)
    model.train()
    torch.manual_seed(1)
    a = model(x)
    b = model(x)
    assert not torch.allclose(a, b), "dropout should be active in train mode"
    with pytest.raises(ValueError):
        build_model(10, [8, 3], ["relu"], 0.5)


# 2-3 ---------------------------------------------------------------------------

def test_optimize_model():
    optimize_model = load_task(__file__, "2-optimize.py", "optimize_model")
    model = small_model()
    opt, loss_fn = optimize_model(model, 0.002, 0.8, 0.95)
    assert isinstance(opt, torch.optim.Adam)
    g = opt.param_groups[0]
    assert g["lr"] == 0.002 and g["betas"] == (0.8, 0.95) and g["weight_decay"] == 0
    assert len(g["params"]) == len(list(model.parameters()))
    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    opt2, _ = optimize_model(model, 0.002, 0.8, 0.95, lambtha=0.01)
    assert opt2.param_groups[0]["weight_decay"] == 0.01


def test_one_hot():
    one_hot = load_task(__file__, "3-one_hot.py", "one_hot")
    out = one_hot([2, 0, 1], 4)
    assert out.dtype == torch.float32 and out.shape == (3, 4)
    assert torch.equal(out, torch.tensor([[0.0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]))
    assert one_hot(np.array([0, 3, 1])).shape == (3, 4)
    assert one_hot(torch.tensor([1, 1])).shape == (2, 2)


# 4-8: training loop --------------------------------------------------------------

@pytest.mark.parametrize("fname", TRAIN_FILES)
def test_train_learns(fname, data, capsys):
    train_model = train_fn(fname)
    X, y, _, _ = data
    model = small_model()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 32, 5)
    assert len(hist["loss"]) == 5 and len(hist["accuracy"]) == 5
    assert hist["loss"][-1] < hist["loss"][0]
    assert hist["accuracy"][-1] > 0.9
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 5
    assert lines[0].startswith("Epoch 1/5 - loss: ") and " - accuracy: " in lines[0]
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 32, 2, verbose=False)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("fname", TRAIN_FILES)
def test_train_epoch_metrics(fname, data):
    """With lr=0 the weights never change, so each epoch's average must equal
    the loss/accuracy of the model on the whole training set."""
    train_model = train_fn(fname)
    X, y, _, _ = data
    model = small_model()
    with torch.no_grad():
        logits = model(X)
    full_loss = F.cross_entropy(logits, y).item()
    full_acc = (logits.argmax(1) == y).float().mean().item()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 64, 2, verbose=False)
    assert hist["loss"][0] == pytest.approx(full_loss, rel=1e-5), \
        "weight each batch's loss by its size (the last batch is smaller)"
    assert hist["accuracy"][1] == pytest.approx(full_acc)


@pytest.mark.parametrize("fname", TRAIN_FILES)
def test_train_deterministic_without_shuffle(fname, data):
    train_model = train_fn(fname)
    X, y, _, _ = data
    hists = []
    for _ in range(2):
        model = small_model(seed=3)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        hists.append(train_model(model, opt, nn.CrossEntropyLoss(), X, y, 50, 3,
                                 verbose=False, shuffle=False))
    assert hists[0]["loss"] == hists[1]["loss"]


@pytest.mark.parametrize("fname", TRAIN_FILES[1:])
def test_train_validation(fname, data, capsys):
    train_model = train_fn(fname)
    X, y, Xv, yv = data
    model = small_model(keep=0.5)  # dropout must be OFF during validation
    with torch.no_grad():
        model.eval()
        logits = model(Xv)
    v_loss = F.cross_entropy(logits, yv).item()
    v_acc = (logits.argmax(1) == yv).float().mean().item()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 64, 2,
                       validation_data=(Xv, yv))
    assert hist["val_loss"] == pytest.approx([v_loss, v_loss], rel=1e-5)
    assert hist["val_accuracy"] == pytest.approx([v_acc, v_acc])
    line = capsys.readouterr().out.strip().splitlines()[0]
    assert " - val_loss: " in line and " - val_accuracy: " in line
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 64, 1, verbose=False)
    assert "val_loss" not in hist


@pytest.mark.parametrize("fname", TRAIN_FILES[2:])
@pytest.mark.parametrize("patience,expected_epochs", [(2, 3), (0, 2), (4, 5)])
def test_early_stopping(fname, patience, expected_epochs, data):
    """With lr=0 the validation loss never improves after epoch 1."""
    train_model = train_fn(fname)
    X, y, Xv, yv = data
    model = small_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    kw = dict(validation_data=(Xv, yv), early_stopping=True, patience=patience, verbose=False)
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 64, 10, **kw)
    assert len(hist["loss"]) == expected_epochs
    # no early stopping without validation data, or when switched off
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 64, 6, early_stopping=True,
                       patience=1, verbose=False)
    assert len(hist["loss"]) == 6
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 64, 6,
                       validation_data=(Xv, yv), patience=1, verbose=False)
    assert len(hist["loss"]) == 6


@pytest.mark.parametrize("fname", TRAIN_FILES[3:])
def test_learning_rate_decay(fname, data):
    train_model = train_fn(fname)
    X, y, Xv, yv = data
    model = small_model()
    opt = torch.optim.Adam(model.parameters(), lr=0.5)
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 64, 4,
                       validation_data=(Xv, yv), learning_rate_decay=True,
                       alpha=0.01, decay_rate=1, verbose=False)
    assert hist["lr"] == pytest.approx([0.01, 0.005, 0.01 / 3, 0.0025])
    model = small_model()
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 64, 3,
                       validation_data=(Xv, yv), verbose=False)
    assert hist["lr"] == pytest.approx([0.003] * 3), "no decay unless asked"
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 64, 3,
                       learning_rate_decay=True, alpha=0.01, verbose=False)
    assert hist["lr"] == pytest.approx([0.003] * 3), "decay only with validation data"


def test_save_best(data, tmp_path):
    train_model = train_fn("8-train.py")
    X, y, Xv, yv = data
    yv_noisy = yv.clone()
    yv_noisy[::3] = (yv_noisy[::3] + 1) % 3  # makes val loss bounce around
    model = small_model()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    path = tmp_path / "best.pt"
    hist = train_model(model, opt, nn.CrossEntropyLoss(), X, y, 32, 8,
                       validation_data=(Xv, yv_noisy), save_best=True,
                       filepath=str(path), verbose=False)
    assert path.exists()
    best = small_model()
    best.load_state_dict(torch.load(path, weights_only=True))
    best.eval()
    with torch.no_grad():
        loss = F.cross_entropy(best(Xv), yv_noisy).item()
    assert loss == pytest.approx(min(hist["val_loss"]), rel=1e-5)


# 9-11: saving --------------------------------------------------------------------

def test_save_load_model(tmp_path):
    mod = load_task(__file__, "9-model.py")
    model = small_model()
    path = str(tmp_path / "model.pt")
    mod.save_model(model, path)
    loaded = mod.load_model(path)
    assert isinstance(loaded, nn.Sequential) and loaded is not model
    x = torch.randn(4, 6)
    model.eval()
    loaded.eval()
    assert torch.allclose(model(x), loaded(x))


def test_save_load_weights(tmp_path):
    mod = load_task(__file__, "10-weights.py")
    model = small_model(seed=1)
    path = str(tmp_path / "w.pt")
    mod.save_weights(model, path)
    state = torch.load(path, weights_only=True)
    assert isinstance(state, dict) and set(state) == set(model.state_dict())
    other = small_model(seed=2)
    assert mod.load_weights(other, path) is None
    for a, b in zip(model.parameters(), other.parameters()):
        assert torch.equal(a, b)


def test_save_load_config(tmp_path):
    mod = load_task(__file__, "11-config.py")
    build_model = load_task(__file__, "0-sequential.py", "build_model")
    model = build_model(6, [5, 3], ["tanh", None], 0.9)
    path = str(tmp_path / "config.json")
    mod.save_config(model, path)
    with open(path) as f:
        assert json.load(f) == {"nx": 6, "layers": [5, 3],
                                "activations": ["tanh", None], "keep_prob": 0.9}
    fresh = mod.load_config(path)
    assert fresh.config == model.config
    assert kinds(fresh) == kinds(model)


# 12-13 ---------------------------------------------------------------------------

def test_test_model(data):
    evaluate = load_task(__file__, "12-test.py", "test_model")
    _, _, Xv, yv = data
    model = small_model(keep=0.5)
    model.train()
    loss, acc = evaluate(model, Xv, yv)
    assert isinstance(loss, float) and isinstance(acc, float)
    model.eval()
    with torch.no_grad():
        logits = model(Xv)
    assert loss == pytest.approx(F.cross_entropy(logits, yv).item(), rel=1e-5)
    assert acc == pytest.approx((logits.argmax(1) == yv).float().mean().item())


def test_predict(data):
    predict = load_task(__file__, "13-predict.py", "predict")
    _, _, Xv, _ = data
    model = small_model(keep=0.5)
    model.train()
    probs = predict(model, Xv[:7])
    assert probs.shape == (7, 3) and probs.device.type == "cpu"
    assert torch.allclose(probs.sum(dim=1), torch.ones(7))
    assert not probs.requires_grad
    model.eval()
    with torch.no_grad():
        expected = torch.softmax(model(Xv[:7]), dim=1)
    assert torch.allclose(probs, expected)
