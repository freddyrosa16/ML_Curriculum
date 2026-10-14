"""Tests for 00-math/04-autodiff. Run: pytest 00-math/04-autodiff

Gradients are checked against numerical finite differences:
    df/dx  ~  (f(x + h) - f(x - h)) / (2h)
"""
import math
import random
import sys

import pytest

from helpers import load_task

H = 1e-6


@pytest.fixture
def Value():
    return load_task(__file__, "engine.py", "Value")


@pytest.fixture
def nn():
    load_task(__file__, "engine.py")  # skip cleanly if engine.py is missing
    sys.modules.pop("engine", None)  # make nn.py import *this* project's engine
    return load_task(__file__, "nn.py")


def numeric_grads(Value, fn, xs):
    """Central-difference gradient of fn (built from Values) at the floats xs."""
    grads = []
    for i in range(len(xs)):
        up = [Value(x + (H if j == i else 0.0)) for j, x in enumerate(xs)]
        down = [Value(x - (H if j == i else 0.0)) for j, x in enumerate(xs)]
        grads.append((fn(*up).data - fn(*down).data) / (2 * H))
    return grads


def check_grads(Value, fn, xs, tol=1e-4):
    inputs = [Value(x) for x in xs]
    out = fn(*inputs)
    out.backward()
    expected = numeric_grads(Value, fn, xs)
    for v, g in zip(inputs, expected):
        assert v.grad == pytest.approx(g, rel=tol, abs=tol)


# ---------- 0. The Value object and the forward pass ----------

def test_0_value_basics(Value):
    a = Value(2.0)
    assert a.data == 2.0
    assert a.grad == 0.0
    assert isinstance(Value(3).data, float)
    assert "2.0" in repr(a)


def test_0_forward_ops(Value):
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == -1.0
    assert (a * b).data == -6.0
    assert (a - b).data == 5.0
    assert (-a).data == -2.0
    assert (a / b).data == pytest.approx(-2 / 3)
    assert (a ** 3).data == 8.0
    assert (a ** -1).data == 0.5


def test_0_mixing_with_plain_numbers(Value):
    a = Value(2.0)
    assert (a + 1).data == 3.0
    assert (1 + a).data == 3.0
    assert (a * 3).data == 6.0
    assert (3 * a).data == 6.0
    assert (a - 5).data == -3.0
    assert (5 - a).data == 3.0
    assert (a / 4).data == 0.5
    assert (4 / a).data == 2.0
    for result in (a + 1, 1 + a, 3 * a, 5 - a, 4 / a):
        assert isinstance(result, Value)


def test_0_nonlinearities_forward(Value):
    assert Value(0.5).tanh().data == pytest.approx(math.tanh(0.5))
    assert Value(1.5).exp().data == pytest.approx(math.exp(1.5))
    assert Value(-2.0).relu().data == 0.0
    assert Value(3.0).relu().data == 3.0


# ---------- 1. backward() for + and * ----------

def test_1_backward_add_mul(Value):
    a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
    e = a * b
    d = e + c
    L = d * f
    L.backward()
    assert L.grad == 1.0
    assert d.grad == -2.0
    assert f.grad == 4.0
    assert c.grad == -2.0
    assert e.grad == -2.0
    assert a.grad == 6.0
    assert b.grad == -4.0


def test_1_backward_deep_chain(Value):
    # x -> x*2 -> +1 -> *3 -> +x ... long chain, must be processed in topological order
    x = Value(1.5)
    y = x
    for _ in range(50):
        y = y * 1.01 + 0.1
    y.backward()
    assert x.grad == pytest.approx(1.01 ** 50)


# ---------- 2. Using a value more than once ----------

def test_2_reused_value_accumulates(Value):
    a = Value(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0

    a = Value(3.0)
    b = a * a
    b.backward()
    assert a.grad == 6.0


def test_2_diamond_graph(Value):
    # a feeds two branches that meet again: grads from both branches must add up
    a, b = Value(-2.0), Value(3.0)
    d = a * b
    e = a + b
    f = d * e
    f.backward()
    assert a.grad == pytest.approx(-3.0)  # e*b + d = 1*3 + (-6)
    assert b.grad == pytest.approx(-8.0)  # e*a + d = 1*(-2) + (-6)


# ---------- 3. Gradients for -, /, **, tanh, exp, relu ----------

def test_3_pow_and_div_grads(Value):
    check_grads(Value, lambda x: x ** 3, [1.7])
    check_grads(Value, lambda x: x ** -2, [0.8])
    check_grads(Value, lambda x, y: x / y, [1.3, -0.7])
    check_grads(Value, lambda x, y: 2 / x - y, [0.9, 4.0])


def test_3_tanh_exp_relu_grads(Value):
    check_grads(Value, lambda x: x.tanh(), [0.3])
    check_grads(Value, lambda x: x.exp(), [-1.2])
    check_grads(Value, lambda x: x.relu(), [2.5])
    check_grads(Value, lambda x: x.relu(), [-2.5])


def test_3_composite_expressions(Value):
    check_grads(Value, lambda x, y: (x * y + x ** 2).tanh() * 3 - y / x, [0.6, -1.1])
    check_grads(Value, lambda x: 1 / (1 + (-x).exp()), [0.4])  # sigmoid
    check_grads(Value, lambda x, y: (x * 2 - y).relu() + (x - y * 3).relu(), [1.0, 0.2])
    check_grads(Value, lambda x, y, z: ((x + y) * z).tanh() + (x * z).exp() / (1 + y ** 2),
                [0.5, -0.3, 0.8])


def test_3_random_polynomials(Value):
    rng = random.Random(0)
    for _ in range(10):
        c = [rng.uniform(-2, 2) for _ in range(4)]
        x0 = rng.uniform(-2, 2)
        check_grads(Value, lambda x: c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3, [x0])


# ---------- 4. Neuron, Layer, MLP ----------

def test_4_neuron(nn):
    random.seed(0)
    n = nn.Neuron(3)
    assert len(n.parameters()) == 4  # 3 weights + 1 bias
    out = n([1.0, -2.0, 0.5])
    assert -1.0 < out.data < 1.0  # tanh output


def test_4_layer(nn):
    random.seed(0)
    layer = nn.Layer(3, 4)
    outs = layer([1.0, 2.0, 3.0])
    assert len(outs) == 4
    assert len(layer.parameters()) == 4 * (3 + 1)
    single = nn.Layer(3, 1)([1.0, 2.0, 3.0])
    assert hasattr(single, "data"), "a layer with one neuron returns a single Value"


def test_4_mlp_shape_and_params(nn):
    random.seed(0)
    model = nn.MLP(3, [4, 4, 1])
    assert len(model.parameters()) == 4 * 4 + 4 * 5 + 1 * 5  # 41
    out = model([2.0, 3.0, -1.0])
    assert hasattr(out, "data") and -1.0 < out.data < 1.0


def test_4_mlp_grads_match_finite_differences(nn):
    random.seed(1)
    model = nn.MLP(2, [3, 1])
    x, y = [0.7, -1.2], 0.5

    def loss():
        return (model(x) - y) ** 2

    L = loss()
    model.zero_grad()
    L.backward()
    for p in model.parameters():
        analytic = p.grad
        old = p.data
        p.data = old + H
        up = loss().data
        p.data = old - H
        down = loss().data
        p.data = old
        assert analytic == pytest.approx((up - down) / (2 * H), rel=1e-4, abs=1e-6)


def test_4_zero_grad(nn):
    random.seed(0)
    model = nn.MLP(2, [2, 1])
    model([1.0, 2.0]).backward()
    assert any(p.grad != 0 for p in model.parameters())
    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())


# ---------- 5. Train it ----------

XS = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
YS = [1.0, -1.0, -1.0, 1.0]


def test_5_train_reduces_loss(nn):
    random.seed(0)
    model = nn.MLP(3, [4, 4, 1])
    losses = nn.train(model, XS, YS, steps=100, lr=0.05)
    assert len(losses) == 100
    assert all(isinstance(v, float) for v in losses)
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.05


def test_5_trained_model_fits_data(nn):
    random.seed(3)
    model = nn.MLP(3, [4, 4, 1])
    nn.train(model, XS, YS, steps=200, lr=0.05)
    preds = [model(x).data for x in XS]
    for p, y in zip(preds, YS):
        assert p == pytest.approx(y, abs=0.2)
