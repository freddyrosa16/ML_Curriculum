"""Tests for 02-vision/07-neural_style_transfer.

Non-slow tests build VGG19 with weights=None (random weights, no download)
and use tiny images (max_dim=48) so everything runs on a CPU in seconds.
"""
import pathlib
import re

import numpy as np
import pytest

from helpers import load_task

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def get_nst(*methods):
    NST = load_task(__file__, "neural_style.py", "NST")
    for m in methods:
        if not hasattr(NST, m):
            pytest.skip(f"NST.{m} not written yet")
    return NST


def images(seed=0):
    rng = np.random.default_rng(seed)
    style = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    content = rng.integers(0, 256, (60, 45, 3), dtype=np.uint8)
    return style, content


FULL = ("scale_image", "load_model", "gram_matrix", "get_features", "generate_features")


@pytest.fixture(scope="module")
def nst():
    NST = get_nst(*FULL)
    torch.manual_seed(0)
    style, content = images()
    return NST(style, content, weights=None, max_dim=48, device="cpu")


# ---------- Task 0: init + scale_image ----------
def test_scale_image():
    NST = get_nst("scale_image")
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    out = NST.scale_image(img)
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (1, 3, 256, 512)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)
    out = NST.scale_image(np.zeros((300, 150, 3), dtype=np.uint8), max_dim=64)
    assert tuple(out.shape) == (1, 3, 64, 32)
    rng = np.random.default_rng(0)
    out = NST.scale_image(rng.integers(0, 256, (37, 81, 3), dtype=np.uint8), max_dim=50)
    assert out.shape[3] == 50 and out.min() >= 0 and out.max() <= 1


def test_scale_image_errors():
    NST = get_nst("scale_image")
    msg = re.escape("image must be a numpy.ndarray with shape (h, w, 3)")
    with pytest.raises(TypeError, match=msg):
        NST.scale_image(np.zeros((10, 10, 4)))
    with pytest.raises(TypeError, match=msg):
        NST.scale_image([[1, 2, 3]])


def test_init_stores_attributes(nst):
    assert nst.alpha == 1e4 and nst.beta == 1 and nst.var == 10
    assert tuple(nst.style_image.shape) == (1, 3, 32, 48)
    assert tuple(nst.content_image.shape) == (1, 3, 48, 36)
    assert list(type(nst).style_layers) == [1, 6, 11, 20, 29]
    assert type(nst).content_layer == 31


def test_init_validation():
    NST = get_nst("scale_image")
    style, content = images()
    kw = dict(weights=None, max_dim=32, device="cpu")
    with pytest.raises(TypeError, match=re.escape("style_image must be a numpy.ndarray with shape (h, w, 3)")):
        NST(style[..., :2], content, **kw)
    with pytest.raises(TypeError, match=re.escape("content_image must be a numpy.ndarray with shape (h, w, 3)")):
        NST(style, "not an image", **kw)
    with pytest.raises(TypeError, match="alpha must be a non-negative number"):
        NST(style, content, alpha=-1, **kw)
    with pytest.raises(TypeError, match="beta must be a non-negative number"):
        NST(style, content, beta="1", **kw)
    with pytest.raises(TypeError, match="var must be a non-negative number"):
        NST(style, content, var=-0.5, **kw)


# ---------- Task 1: load_model ----------
def test_load_model(nst):
    model = nst.model
    assert isinstance(model, torch.nn.Module)
    assert len(model) == 32
    assert not any(isinstance(m, torch.nn.MaxPool2d) for m in model.modules())
    assert sum(isinstance(m, torch.nn.AvgPool2d) for m in model.modules()) == 4
    assert all(not p.requires_grad for p in model.parameters())
    assert not model.training


# ---------- Task 2: gram_matrix ----------
def test_gram_matrix_hand_example():
    NST = get_nst("gram_matrix")
    x = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])     # (1, 2, 1, 2)
    g = NST.gram_matrix(x)
    assert torch.allclose(g, torch.tensor([[[2.5, 5.5], [5.5, 12.5]]]))


def test_gram_matrix_shape_symmetry_and_errors():
    NST = get_nst("gram_matrix")
    torch.manual_seed(0)
    g = NST.gram_matrix(torch.randn(1, 7, 5, 6))
    assert tuple(g.shape) == (1, 7, 7)
    assert torch.allclose(g, g.transpose(1, 2), atol=1e-6)
    msg = "input_layer must be a tensor of rank 4"
    with pytest.raises(TypeError, match=msg):
        NST.gram_matrix(torch.randn(7, 5, 6))
    with pytest.raises(TypeError, match=msg):
        NST.gram_matrix(np.zeros((1, 2, 3, 4)))


# ---------- Task 3: get_features + generate_features ----------
def test_get_features_shapes(nst):
    style_outputs, content_output = nst.get_features(nst.content_image)
    assert isinstance(style_outputs, list) and len(style_outputs) == 5
    assert [s.shape[1] for s in style_outputs] == [64, 128, 256, 512, 512]
    assert tuple(style_outputs[0].shape[2:]) == (48, 36)
    assert tuple(style_outputs[1].shape[2:]) == (24, 18)
    assert tuple(content_output.shape) == (1, 512, 3, 2)


def test_get_features_normalizes_input(nst):
    # the ImageNet-mean colour becomes all zeros after normalization,
    # so the first conv layer outputs just its bias (then ReLU)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    style_outputs, _ = nst.get_features(mean.clone())
    bias = nst.model[0].bias.detach().clamp(min=0).view(1, -1, 1, 1)
    assert torch.allclose(style_outputs[0], bias.expand_as(style_outputs[0]), atol=1e-5)


def test_generate_features(nst):
    assert len(nst.gram_style_features) == 5
    assert [tuple(g.shape) for g in nst.gram_style_features] == [
        (1, 64, 64), (1, 128, 128), (1, 256, 256), (1, 512, 512), (1, 512, 512)]
    assert tuple(nst.content_feature.shape) == (1, 512, 3, 2)
    assert not nst.content_feature.requires_grad


# ---------- Task 4: layer_style_cost ----------
def test_layer_style_cost(nst):
    if not hasattr(nst, "layer_style_cost"):
        pytest.skip("NST.layer_style_cost not written yet")
    x = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    cost = nst.layer_style_cost(x, torch.zeros(1, 2, 2))
    assert torch.isclose(cost, torch.tensor(55.75))
    g = type(nst).gram_matrix(x)
    assert torch.isclose(nst.layer_style_cost(x, g), torch.tensor(0.0))
    with pytest.raises(TypeError, match="style_output must be a tensor of rank 4"):
        nst.layer_style_cost(x[0], g)
    with pytest.raises(TypeError, match=re.escape("gram_target must be a tensor of shape [1, 2, 2]")):
        nst.layer_style_cost(x, torch.zeros(2, 2))


# ---------- Task 5: style_cost ----------
def test_style_cost(nst):
    if not hasattr(nst, "style_cost"):
        pytest.skip("NST.style_cost not written yet")
    style_outputs, _ = nst.get_features(nst.style_image)
    assert nst.style_cost(style_outputs).item() == pytest.approx(0.0, abs=1e-6)
    other, _ = nst.get_features(nst.content_image)
    expected = sum(nst.layer_style_cost(s, g) for s, g in zip(other, nst.gram_style_features)) / 5
    assert torch.isclose(nst.style_cost(other), expected, rtol=1e-5)
    with pytest.raises(TypeError, match="style_outputs must be a list with a length of 5"):
        nst.style_cost(other[:4])


# ---------- Task 6: content_cost ----------
def test_content_cost(nst):
    if not hasattr(nst, "content_cost"):
        pytest.skip("NST.content_cost not written yet")
    _, content_output = nst.get_features(nst.content_image)
    assert nst.content_cost(content_output).item() == pytest.approx(0.0, abs=1e-6)
    shifted = nst.content_feature + 2.0
    assert torch.isclose(nst.content_cost(shifted), torch.tensor(4.0))
    with pytest.raises(TypeError, match=re.escape("content_output must be a tensor of shape (1, 512, 3, 2)")):
        nst.content_cost(torch.zeros(1, 512, 2, 2))


# ---------- Task 7: variational_cost ----------
def test_variational_cost():
    NST = get_nst("variational_cost")
    img = torch.zeros(1, 3, 2, 2)
    img[0, 0] = torch.tensor([[0.0, 1.0], [2.0, 4.0]])
    # vertical |2-0| + |4-1| = 5, horizontal |1-0| + |4-2| = 3
    assert torch.isclose(NST.variational_cost(img), torch.tensor(8.0))
    assert NST.variational_cost(torch.ones(1, 3, 5, 5)).item() == 0.0
    with pytest.raises(TypeError, match="generated_image must be a tensor of rank 4"):
        NST.variational_cost(torch.zeros(3, 2, 2))


# ---------- Task 8: total_cost ----------
def test_total_cost(nst):
    if not hasattr(nst, "total_cost"):
        pytest.skip("NST.total_cost not written yet")
    J, J_content, J_style, J_var = nst.total_cost(nst.content_image)
    assert J_content.item() == pytest.approx(0.0, abs=1e-6)
    expected = nst.alpha * J_content + nst.beta * J_style + nst.var * J_var
    assert torch.isclose(J, expected, rtol=1e-5)
    assert torch.isclose(J_var, type(nst).variational_cost(nst.content_image))
    with pytest.raises(TypeError, match=re.escape("generated_image must be a tensor of shape (1, 3, 48, 36)")):
        nst.total_cost(nst.style_image)


# ---------- Task 9: compute_grads ----------
def test_compute_grads(nst):
    if not hasattr(nst, "compute_grads"):
        pytest.skip("NST.compute_grads not written yet")
    torch.manual_seed(1)
    image = torch.rand_like(nst.content_image)
    grad, J, J_content, J_style, J_var = nst.compute_grads(image)
    assert grad.shape == image.shape
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    J2 = nst.total_cost(image)[0]
    assert torch.isclose(J, J2.detach(), rtol=1e-4)
    # gradient check along a random direction
    d = torch.randn_like(image)
    eps = 1e-3
    with torch.no_grad():
        plus = nst.total_cost(image + eps * d)[0]
        minus = nst.total_cost(image - eps * d)[0]
    numeric = (plus - minus) / (2 * eps)
    assert torch.isclose((grad * d).sum(), numeric, rtol=0.05)
    with pytest.raises(TypeError, match="generated_image must be a tensor of shape"):
        nst.compute_grads(torch.zeros(1, 3, 4, 4))


# ---------- Task 10: generate_image ----------
def test_generate_image_runs_and_improves(nst, capsys):
    if not hasattr(nst, "generate_image"):
        pytest.skip("NST.generate_image not written yet")
    J0 = nst.total_cost(nst.content_image)[0].item()
    image, cost = nst.generate_image(iterations=6, step=3, lr=0.01)
    assert isinstance(image, np.ndarray) and image.shape == (48, 36, 3)
    assert image.min() >= 0 and image.max() <= 1
    assert isinstance(cost, float) and cost <= J0 + 1e-6
    printed = capsys.readouterr().out
    assert "Cost at iteration 0" in printed and "Cost at iteration 6" in printed


def test_generate_image_validation(nst):
    if not hasattr(nst, "generate_image"):
        pytest.skip("NST.generate_image not written yet")
    cases = [
        (dict(iterations=1.5), TypeError, "iterations must be an integer"),
        (dict(iterations=0), ValueError, "iterations must be positive"),
        (dict(iterations=5, step=2.0), TypeError, "step must be an integer"),
        (dict(iterations=5, step=5), ValueError, "step must be positive and less than iterations"),
        (dict(iterations=5, lr="big"), TypeError, "lr must be a number"),
        (dict(iterations=5, lr=0), ValueError, "lr must be positive"),
        (dict(iterations=5, beta1=1), TypeError, "beta1 must be a float"),
        (dict(iterations=5, beta1=1.5), ValueError, re.escape("beta1 must be in the range [0, 1]")),
        (dict(iterations=5, beta2=0), TypeError, "beta2 must be a float"),
        (dict(iterations=5, beta2=-0.1), ValueError, re.escape("beta2 must be in the range [0, 1]")),
    ]
    for kwargs, err, msg in cases:
        with pytest.raises(err, match=msg):
            nst.generate_image(**kwargs)


# ---------- Slow: the real thing ----------
@pytest.mark.slow
def test_style_transfer_with_pretrained_vgg():
    """Downloads VGG19 ImageNet weights (~550 MB) the first time."""
    NST = get_nst(*FULL, "generate_image")
    from PIL import Image
    style = np.array(Image.open(DATA / "starry_night.jpg").convert("RGB"))
    content = np.array(Image.open(DATA / "golden_gate.jpg").convert("RGB"))
    nst = NST(style, content, max_dim=128)
    J0 = nst.total_cost(nst.content_image)[0].item()
    image, cost = nst.generate_image(iterations=100, lr=0.02)
    assert image.shape[2] == 3 and max(image.shape[:2]) == 128
    assert cost < 0.5 * J0
