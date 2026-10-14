"""Tests for 01-classification.

Gradient descent is checked against a *numerical* gradient (finite differences
of your own cost function), so the tests don't give away the backprop formulas.
"""
import sys

import numpy as np
import pytest

from helpers import load_task, project_dir

NEURON = [f"{i}-neuron.py" for i in range(8)]
NN = [f"{i}-neural_network.py" for i in range(8, 16)]
DEEP_BIN = [f"{i}-deep_neural_network.py" for i in range(16, 24)] + ["26-deep_neural_network.py"]
DEEP_MULTI = ["27-deep_neural_network.py", "28-deep_neural_network.py"]


def files_from(files, first):
    """Task files from task number `first` onward."""
    return [f for f in files if int(f.split("-")[0]) >= first]


def cls(filename):
    name = {"neuron": "Neuron", "neural_network": "NeuralNetwork",
            "deep_neural_network": "DeepNeuralNetwork"}[filename.split("-", 1)[1][:-3]]
    return load_task(__file__, filename, name)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cost(Y, A):
    m = Y.shape[1]
    return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 30)) * 0.5
    Y = (np.array([[1.0, -2.0, 0.5, 1.5]]) @ X > 0).astype(int)
    return X, Y


@pytest.fixture
def multi_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 30)) * 0.5
    labels = rng.integers(0, 3, size=30)
    Y = np.zeros((3, 30))
    Y[labels, np.arange(30)] = 1
    return X, Y


def numerical_grad(get, set_, f, eps=1e-6):
    """Central-difference gradient of f() with respect to a parameter."""
    base = np.array(get(), dtype=float)
    grad = np.zeros_like(base)
    for idx in np.ndindex(base.shape):
        for sign in (1, -1):
            p = base.copy()
            p[idx] += sign * eps
            set_(p)
            grad[idx] += sign * f()
    set_(base)
    return grad / (2 * eps)


# ---------------------------------------------------------------- constructors

@pytest.mark.parametrize("fname", NEURON)
def test_neuron_init(fname):
    Neuron = cls(fname)
    np.random.seed(0)
    n = Neuron(5)
    np.random.seed(0)
    assert n.W.shape == (1, 5)
    assert np.allclose(n.W, np.random.randn(1, 5))
    assert n.b == 0 and n.A == 0


@pytest.mark.parametrize("fname", NEURON)
def test_neuron_init_errors(fname):
    Neuron = cls(fname)
    with pytest.raises(TypeError, match="nx must be an integer"):
        Neuron(2.0)
    with pytest.raises(TypeError, match="nx must be an integer"):
        Neuron("3")
    with pytest.raises(ValueError, match="nx must be a positive integer"):
        Neuron(0)


def test_neuron_public_attributes():
    n = cls("0-neuron.py")(3)
    n.A = 10
    assert n.A == 10


@pytest.mark.parametrize("fname", files_from(NEURON, 1))
def test_neuron_private(fname):
    n = cls(fname)(3)
    for attr in ("W", "b", "A"):
        with pytest.raises(AttributeError):
            setattr(n, attr, 10)
    assert not hasattr(n, "__W")
    assert hasattr(n, "_Neuron__W")


@pytest.mark.parametrize("fname", NN)
def test_nn_init(fname):
    NeuralNetwork = cls(fname)
    np.random.seed(0)
    nn = NeuralNetwork(4, 3)
    np.random.seed(0)
    W1 = np.random.randn(3, 4)
    W2 = np.random.randn(1, 3)
    assert np.allclose(nn.W1, W1) and np.allclose(nn.W2, W2)
    assert nn.b1.shape == (3, 1) and np.all(nn.b1 == 0)
    assert nn.b2 == 0 and nn.A1 == 0 and nn.A2 == 0


@pytest.mark.parametrize("fname", NN)
def test_nn_init_errors(fname):
    NeuralNetwork = cls(fname)
    with pytest.raises(TypeError, match="nx must be an integer"):
        NeuralNetwork(1.5, 3)
    with pytest.raises(ValueError, match="nx must be a positive integer"):
        NeuralNetwork(0, 3)
    with pytest.raises(TypeError, match="nodes must be an integer"):
        NeuralNetwork(3, "3")
    with pytest.raises(ValueError, match="nodes must be a positive integer"):
        NeuralNetwork(3, -1)


@pytest.mark.parametrize("fname", files_from(NN, 9))
def test_nn_private(fname):
    nn = cls(fname)(3, 2)
    for attr in ("W1", "b1", "A1", "W2", "b2", "A2"):
        with pytest.raises(AttributeError):
            setattr(nn, attr, 1)


@pytest.mark.parametrize("fname", DEEP_BIN + DEEP_MULTI)
def test_deep_init(fname):
    Deep = cls(fname)
    np.random.seed(0)
    d = Deep(4, [5, 3, 1])
    np.random.seed(0)
    assert d.L == 3
    assert d.cache == {}
    assert list(d.weights) == ["W1", "b1", "W2", "b2", "W3", "b3"]
    sizes = [4, 5, 3, 1]
    for l in range(1, 4):
        expected = np.random.randn(sizes[l], sizes[l - 1]) * np.sqrt(2 / sizes[l - 1])
        assert np.allclose(d.weights[f"W{l}"], expected)
        assert d.weights[f"b{l}"].shape == (sizes[l], 1)
        assert np.all(d.weights[f"b{l}"] == 0)


@pytest.mark.parametrize("fname", DEEP_BIN + DEEP_MULTI)
def test_deep_init_errors(fname):
    Deep = cls(fname)
    with pytest.raises(TypeError, match="nx must be an integer"):
        Deep(2.5, [3])
    with pytest.raises(ValueError, match="nx must be a positive integer"):
        Deep(-1, [3])
    for bad in ([], [3, 0], [3, "2"], (3, 1), 5):
        with pytest.raises(TypeError, match="layers must be a list of positive integers"):
            Deep(3, bad)


@pytest.mark.parametrize("fname", files_from(DEEP_BIN, 17) + DEEP_MULTI)
def test_deep_private(fname):
    d = cls(fname)(3, [2, 1])
    for attr in ("L", "cache", "weights"):
        with pytest.raises(AttributeError):
            setattr(d, attr, 1)


# ----------------------------------------------------------- forward / cost

@pytest.mark.parametrize("fname", files_from(NEURON, 2))
def test_neuron_forward(fname, binary_data):
    X, _ = binary_data
    np.random.seed(1)
    n = cls(fname)(4)
    n._Neuron__b = 0.3
    A = n.forward_prop(X)
    assert A is n.A
    assert A.shape == (1, 30)
    assert np.allclose(A, sigmoid(n.W @ X + 0.3))


@pytest.mark.parametrize("fname", files_from(NN, 10))
def test_nn_forward(fname, binary_data):
    X, _ = binary_data
    np.random.seed(1)
    nn = cls(fname)(4, 3)
    A1, A2 = nn.forward_prop(X)
    assert A1 is nn.A1 and A2 is nn.A2
    assert np.allclose(A1, sigmoid(nn.W1 @ X + nn.b1))
    assert np.allclose(A2, sigmoid(nn.W2 @ A1 + nn.b2))


@pytest.mark.parametrize("fname", files_from(DEEP_BIN, 18))
def test_deep_forward(fname, binary_data):
    X, _ = binary_data
    np.random.seed(1)
    d = cls(fname)(4, [5, 3, 1])
    A, cache = d.forward_prop(X)
    assert cache is d.cache
    assert set(cache) == {"A0", "A1", "A2", "A3"}
    assert cache["A0"] is X
    prev = X
    for l in range(1, 4):
        prev = sigmoid(d.weights[f"W{l}"] @ prev + d.weights[f"b{l}"])
        assert np.allclose(cache[f"A{l}"], prev)
    assert np.allclose(A, prev) and A.shape == (1, 30)


@pytest.mark.parametrize("fname", files_from(NEURON, 3) + files_from(NN, 11)
                         + files_from(DEEP_BIN, 19))
def test_binary_cost(fname, binary_data):
    _, Y = binary_data
    obj = make(fname)
    A = np.random.default_rng(2).uniform(0.05, 0.95, size=(1, 30))
    assert np.isclose(obj.cost(Y, A), binary_cost(Y, A))


def make(fname, nx=4):
    if "deep" in fname:
        return cls(fname)(nx, [3, 1])
    if "neural_network" in fname:
        return cls(fname)(nx, 3)
    return cls(fname)(nx)


def output_of(obj, X):
    out = obj.forward_prop(X)
    if not isinstance(out, tuple):
        return out  # Neuron
    return out[0] if isinstance(out[1], dict) else out[1]


@pytest.mark.parametrize("fname", files_from(NEURON, 4) + files_from(NN, 12)
                         + files_from(DEEP_BIN, 20))
def test_binary_evaluate(fname, binary_data):
    X, Y = binary_data
    np.random.seed(3)
    obj = make(fname)
    pred, cost = obj.evaluate(X, Y)
    A = output_of(obj, X)
    assert pred.shape == (1, 30)
    assert np.issubdtype(pred.dtype, np.integer)
    assert np.array_equal(pred, (A >= 0.5).astype(int))
    assert np.isclose(cost, binary_cost(Y, A))


# ------------------------------------------------------------ gradient descent

@pytest.mark.parametrize("fname", files_from(NEURON, 5))
def test_neuron_gradient_descent(fname, binary_data):
    X, Y = binary_data
    np.random.seed(4)
    n = cls(fname)(4)
    f = lambda: n.cost(Y, n.forward_prop(X))  # noqa: E731
    grads = {}
    for attr in ("W", "b"):
        key = f"_Neuron__{attr}"
        grads[attr] = numerical_grad(lambda: getattr(n, key),
                                     lambda v, k=key: setattr(n, k, v), f)
    W0, b0 = n.W.copy(), float(n.b)
    A = n.forward_prop(X)
    assert n.gradient_descent(X, Y, A, 0.3) is None
    assert np.allclose(n.W, W0 - 0.3 * grads["W"], atol=1e-6)
    assert np.isclose(n.b, b0 - 0.3 * grads["b"], atol=1e-6)


@pytest.mark.parametrize("fname", files_from(NN, 13))
def test_nn_gradient_descent(fname, binary_data):
    X, Y = binary_data
    np.random.seed(5)
    nn = cls(fname)(4, 3)
    f = lambda: nn.cost(Y, nn.forward_prop(X)[1])  # noqa: E731
    old, grads = {}, {}
    for attr in ("W1", "b1", "W2", "b2"):
        key = f"_NeuralNetwork__{attr}"
        grads[attr] = numerical_grad(lambda k=key: getattr(nn, k),
                                     lambda v, k=key: setattr(nn, k, v), f)
        old[attr] = np.array(getattr(nn, attr), dtype=float)
    A1, A2 = nn.forward_prop(X)
    nn.gradient_descent(X, Y, A1, A2, 0.3)
    for attr in old:
        assert np.allclose(getattr(nn, attr), old[attr] - 0.3 * grads[attr], atol=1e-6), attr


def deep_grad_check(d, X, Y, alpha=0.3):
    f = lambda: d.cost(Y, d.forward_prop(X)[0])  # noqa: E731
    old, grads = {}, {}
    for key in list(d.weights):
        grads[key] = numerical_grad(lambda k=key: d.weights[k],
                                    lambda v, k=key: d.weights.__setitem__(k, v), f)
        old[key] = d.weights[key].copy()
    _, cache = d.forward_prop(X)
    d.gradient_descent(Y, cache, alpha)
    for key in old:
        assert np.allclose(d.weights[key], old[key] - alpha * grads[key], atol=1e-6), key


@pytest.mark.parametrize("fname", files_from(DEEP_BIN, 21))
def test_deep_gradient_descent(fname, binary_data):
    X, Y = binary_data
    np.random.seed(6)
    deep_grad_check(cls(fname)(4, [5, 3, 1]), X, Y)


# -------------------------------------------------------------------- training

TRAIN_FILES = (files_from(NEURON, 6) + files_from(NN, 14) + files_from(DEEP_BIN, 22))
UPGRADED = ["7-neuron.py", "15-neural_network.py"] + files_from(DEEP_BIN, 23)


def train(obj, fname, X, Y, **kw):
    if fname in UPGRADED:
        kw.setdefault("verbose", False)
        kw.setdefault("graph", False)
    return obj.train(X, Y, **kw)


@pytest.mark.parametrize("fname", TRAIN_FILES)
def test_train_errors(fname, binary_data):
    X, Y = binary_data
    obj = make(fname)
    with pytest.raises(TypeError, match="iterations must be an integer"):
        train(obj, fname, X, Y, iterations=10.0)
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        train(obj, fname, X, Y, iterations=0)
    with pytest.raises(TypeError, match="alpha must be a float"):
        train(obj, fname, X, Y, alpha=1)
    with pytest.raises(ValueError, match="alpha must be positive"):
        train(obj, fname, X, Y, alpha=-0.5)


@pytest.mark.parametrize("fname", TRAIN_FILES)
def test_train_matches_manual_steps(fname, binary_data):
    """train(n) should equal n rounds of your own forward_prop + gradient_descent."""
    X, Y = binary_data
    np.random.seed(7)
    a = make(fname)
    np.random.seed(7)
    b = make(fname)
    pred, cost = train(a, fname, X, Y, iterations=20, alpha=0.5)
    for _ in range(20):
        out = b.forward_prop(X)
        if "deep" in fname:
            b.gradient_descent(Y, out[1], 0.5)
        elif "neural_network" in fname:
            b.gradient_descent(X, Y, out[0], out[1], 0.5)
        else:
            b.gradient_descent(X, Y, out, 0.5)
    pred_b, cost_b = b.evaluate(X, Y)
    assert np.isclose(cost, cost_b)
    assert np.array_equal(pred, pred_b)


@pytest.mark.parametrize("fname", TRAIN_FILES)
def test_train_learns(fname, binary_data):
    X, Y = binary_data
    np.random.seed(8)
    obj = make(fname)
    _, start = obj.evaluate(X, Y)
    pred, end = train(obj, fname, X, Y, iterations=300, alpha=0.5)
    assert end < start
    assert np.mean(pred == Y) > 0.85


@pytest.mark.parametrize("fname", UPGRADED)
def test_train_verbose(fname, binary_data, capsys):
    X, Y = binary_data
    np.random.seed(9)
    obj = make(fname)
    _, final = obj.train(X, Y, iterations=25, alpha=0.5, verbose=True, graph=False, step=10)
    lines = capsys.readouterr().out.strip().splitlines()
    assert [l.split(":")[0] for l in lines] == [
        "Cost after 0 iterations", "Cost after 10 iterations",
        "Cost after 20 iterations", "Cost after 25 iterations"]
    assert np.isclose(float(lines[-1].split(":")[1]), final)


@pytest.mark.parametrize("fname", UPGRADED)
def test_train_step_errors(fname, binary_data):
    X, Y = binary_data
    obj = make(fname)
    with pytest.raises(TypeError, match="step must be an integer"):
        obj.train(X, Y, iterations=10, verbose=True, graph=False, step=2.0)
    with pytest.raises(ValueError, match="step must be positive and <= iterations"):
        obj.train(X, Y, iterations=10, verbose=True, graph=False, step=11)
    with pytest.raises(ValueError, match="step must be positive and <= iterations"):
        obj.train(X, Y, iterations=10, verbose=False, graph=True, step=0)
    # step is not checked when neither verbose nor graph is on
    obj.train(X, Y, iterations=10, verbose=False, graph=False, step=50)


@pytest.mark.parametrize("fname", UPGRADED)
def test_train_graph(fname, binary_data, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    X, Y = binary_data
    obj = make(fname)
    obj.train(X, Y, iterations=20, alpha=0.5, verbose=False, graph=True, step=5)
    ax = plt.gca()
    assert ax.get_xlabel() == "iteration" and ax.get_ylabel() == "cost"
    assert ax.get_title() == "Training Cost"
    plt.close("all")


# --------------------------------------------------------------------- one-hot

def test_one_hot_encode():
    one_hot_encode = load_task(__file__, "24-one_hot_encode.py", "one_hot_encode")
    Y = np.array([5, 0, 4, 1, 9, 2])
    oh = one_hot_encode(Y, 10)
    assert oh.shape == (10, 6)
    assert np.array_equal(oh.sum(axis=0), np.ones(6))
    assert np.array_equal(np.argmax(oh, axis=0), Y)
    assert one_hot_encode(Y, 9) is None
    assert one_hot_encode([1, 2], 3) is None
    assert one_hot_encode(Y, 10.0) is None
    assert one_hot_encode(np.array([], dtype=int), 3) is None


def test_one_hot_decode():
    one_hot_decode = load_task(__file__, "25-one_hot_decode.py", "one_hot_decode")
    oh = np.eye(4)[:, [3, 1, 0, 2, 2]]
    assert np.array_equal(one_hot_decode(oh), [3, 1, 0, 2, 2])
    assert one_hot_decode([[1, 0], [0, 1]]) is None
    assert one_hot_decode(np.array([1, 0, 0])) is None


# ----------------------------------------------------------------- save / load

@pytest.mark.parametrize("fname", ["26-deep_neural_network.py"] + DEEP_MULTI)
def test_save_load(fname, tmp_path, binary_data):
    module = load_task(__file__, fname)
    sys.modules[module.__name__] = module  # lets pickle find the class
    Deep = module.DeepNeuralNetwork
    np.random.seed(10)
    d = Deep(4, [3, 2])
    d.save(str(tmp_path / "net"))
    assert (tmp_path / "net.pkl").exists()
    d.save(str(tmp_path / "other.pkl"))
    assert (tmp_path / "other.pkl").exists()
    assert not (tmp_path / "other.pkl.pkl").exists()
    loaded = Deep.load(str(tmp_path / "net.pkl"))
    assert isinstance(loaded, Deep)
    assert loaded.L == 2
    for k in d.weights:
        assert np.array_equal(loaded.weights[k], d.weights[k])
    assert Deep.load(str(tmp_path / "missing.pkl")) is None


# ------------------------------------------------------------------ multiclass

def softmax(z):
    e = np.exp(z - z.max(axis=0))
    return e / e.sum(axis=0)


@pytest.mark.parametrize("fname", DEEP_MULTI)
def test_multiclass_forward_cost_evaluate(fname, multi_data):
    X, Y = multi_data
    np.random.seed(11)
    d = cls(fname)(4, [5, 3])
    A, cache = d.forward_prop(X)
    hidden = sigmoid(d.weights["W1"] @ X + d.weights["b1"])
    assert np.allclose(cache["A1"], hidden)
    assert np.allclose(A, softmax(d.weights["W2"] @ hidden + d.weights["b2"]))
    assert np.allclose(A.sum(axis=0), 1)
    assert np.isclose(d.cost(Y, A), -np.sum(Y * np.log(A)) / 30)
    pred, cost = d.evaluate(X, Y)
    assert pred.shape == (3, 30)
    assert np.array_equal(np.argmax(pred, axis=0), np.argmax(A, axis=0))
    assert np.array_equal(pred.sum(axis=0), np.ones(30))


@pytest.mark.parametrize("fname", DEEP_MULTI)
def test_multiclass_gradient_descent(fname, multi_data):
    X, Y = multi_data
    np.random.seed(12)
    deep_grad_check(cls(fname)(4, [5, 4, 3]), X, Y)


@pytest.mark.parametrize("fname", DEEP_MULTI)
def test_multiclass_train_learns(fname, multi_data):
    X, Y = multi_data
    np.random.seed(13)
    d = cls(fname)(4, [8, 3])
    _, start = d.evaluate(X, Y)
    _, end = d.train(X, Y, iterations=200, alpha=0.5, verbose=False, graph=False)
    assert end < start


def test_activation_option(multi_data):
    Deep = cls("28-deep_neural_network.py")
    assert Deep(4, [3]).activation == "sig"
    d = Deep(4, [5, 3], activation="tanh")
    assert d.activation == "tanh"
    with pytest.raises(AttributeError):
        d.activation = "sig"
    with pytest.raises(ValueError, match="activation must be 'sig' or 'tanh'"):
        Deep(4, [3], activation="relu")


def test_tanh_forward_and_gradient(multi_data):
    X, Y = multi_data
    Deep = cls("28-deep_neural_network.py")
    np.random.seed(14)
    d = Deep(4, [5, 4, 3], activation="tanh")
    _, cache = d.forward_prop(X)
    assert np.allclose(cache["A1"], np.tanh(d.weights["W1"] @ X + d.weights["b1"]))
    deep_grad_check(d, X, Y)


# ------------------------------------------------------------ real data (slow)

@pytest.mark.slow
def test_mnist_end_to_end():
    path = project_dir(__file__) / "data" / "MNIST.npz"
    if not path.exists():
        pytest.skip("data/MNIST.npz not found")
    Deep = cls("28-deep_neural_network.py")
    one_hot_encode = load_task(__file__, "24-one_hot_encode.py", "one_hot_encode")
    one_hot_decode = load_task(__file__, "25-one_hot_decode.py", "one_hot_decode")
    lib = np.load(path)
    X = lib["X_train"][:5000].reshape(5000, -1).T
    Y = lib["Y_train"][:5000]
    Xv = lib["X_valid"][:2000].reshape(2000, -1).T
    Yv = lib["Y_valid"][:2000]
    np.random.seed(0)
    d = Deep(784, [64, 10], activation="tanh")
    d.train(X, one_hot_encode(Y, 10), iterations=200, alpha=0.5,
            verbose=False, graph=False)
    pred, _ = d.evaluate(Xv, one_hot_encode(Yv, 10))
    assert np.mean(one_hot_decode(pred) == Yv) > 0.8
