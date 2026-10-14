"""Tests for 00-math/06-convolutions_and_pooling.
Run: pytest 00-math/06-convolutions_and_pooling

Reference results come from scipy.signal.correlate2d (cross-correlation,
which is what deep learning calls "convolution": the kernel is not flipped).
"""
import ast

import numpy as np
import pytest
from scipy.signal import correlate2d

from helpers import load_task, project_dir


def assert_max_loops(filename, n):
    tree = ast.parse((project_dir(__file__) / filename).read_text())
    loops = [x for x in ast.walk(tree) if isinstance(x, (ast.For, ast.While, ast.comprehension))]
    assert len(loops) <= n, f"{filename}: at most {n} for loops (found {len(loops)} loops)"


def ref_gray(images, kernel, ph=0, pw=0, sh=1, sw=1):
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))
    return np.stack([correlate2d(img, kernel, mode="valid")[::sh, ::sw] for img in padded])


def ref_channels(images, kernel, ph=0, pw=0, sh=1, sw=1):
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    out = []
    for img in padded:
        total = sum(correlate2d(img[..., c], kernel[..., c], mode="valid")
                    for c in range(img.shape[-1]))
        out.append(total[::sh, ::sw])
    return np.stack(out)


@pytest.fixture
def gray():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (5, 28, 28)).astype(float)


@pytest.fixture
def rgb():
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (4, 32, 32, 3)).astype(float)


EDGE = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])


# ---------- 0. Valid Convolution ----------

def test_0_valid(gray):
    f = load_task(__file__, "0-convolve_grayscale_valid.py", "convolve_grayscale_valid")
    out = f(gray, EDGE)
    assert out.shape == (5, 26, 26)
    assert np.allclose(out, ref_gray(gray, EDGE))
    k = np.arange(10).reshape(2, 5)  # non-square kernel
    assert np.allclose(f(gray, k), ref_gray(gray, k))
    assert_max_loops("0-convolve_grayscale_valid.py", 2)


def test_0_valid_tiny_example():
    f = load_task(__file__, "0-convolve_grayscale_valid.py", "convolve_grayscale_valid")
    img = np.arange(16, dtype=float).reshape(1, 4, 4)
    out = f(img, np.ones((2, 2)))
    assert np.allclose(out[0], [[10, 14, 18], [26, 30, 34], [42, 46, 50]])


# ---------- 1. Same Convolution ----------

def test_1_same(gray):
    f = load_task(__file__, "1-convolve_grayscale_same.py", "convolve_grayscale_same")
    out = f(gray, EDGE)
    assert out.shape == (5, 28, 28)
    assert np.allclose(out, ref_gray(gray, EDGE, 1, 1))
    k = np.ones((5, 3))
    assert np.allclose(f(gray, k), ref_gray(gray, k, 2, 1))
    assert_max_loops("1-convolve_grayscale_same.py", 2)


# ---------- 2. Convolution with Padding ----------

def test_2_padding(gray):
    f = load_task(__file__, "2-convolve_grayscale_padding.py", "convolve_grayscale_padding")
    out = f(gray, EDGE, (2, 4))
    assert out.shape == (5, 30, 34)
    assert np.allclose(out, ref_gray(gray, EDGE, 2, 4))
    assert np.allclose(f(gray, EDGE, (0, 0)), ref_gray(gray, EDGE))
    assert_max_loops("2-convolve_grayscale_padding.py", 2)


# ---------- 3. Strided Convolution ----------

def test_3_strided(gray):
    f = load_task(__file__, "3-convolve_grayscale.py", "convolve_grayscale")
    out = f(gray, EDGE, padding="valid", stride=(2, 2))
    assert out.shape == (5, 13, 13)
    assert np.allclose(out, ref_gray(gray, EDGE, sh=2, sw=2))
    out = f(gray, EDGE)  # defaults: same, stride 1
    assert out.shape == (5, 28, 28)
    assert np.allclose(out, ref_gray(gray, EDGE, 1, 1))
    out = f(gray, EDGE, padding=(2, 1), stride=(3, 2))
    assert np.allclose(out, ref_gray(gray, EDGE, 2, 1, 3, 2))
    out = f(gray, np.ones((5, 5)), padding="same", stride=(2, 3))
    assert np.allclose(out, ref_gray(gray, np.ones((5, 5)), 2, 2, 2, 3))
    assert_max_loops("3-convolve_grayscale.py", 2)


# ---------- 4. Convolution with Channels ----------

def test_4_channels(rgb):
    f = load_task(__file__, "4-convolve_channels.py", "convolve_channels")
    k = np.random.default_rng(2).normal(size=(3, 3, 3))
    out = f(rgb, k, padding="valid")
    assert out.shape == (4, 30, 30)
    assert np.allclose(out, ref_channels(rgb, k))
    out = f(rgb, k)
    assert out.shape == (4, 32, 32)
    assert np.allclose(out, ref_channels(rgb, k, 1, 1))
    out = f(rgb, k, padding=(1, 2), stride=(2, 3))
    assert np.allclose(out, ref_channels(rgb, k, 1, 2, 2, 3))
    assert_max_loops("4-convolve_channels.py", 2)


# ---------- 5. Multiple Kernels ----------

def test_5_multiple_kernels(rgb):
    f = load_task(__file__, "5-convolve.py", "convolve")
    ks = np.random.default_rng(3).normal(size=(3, 3, 3, 4))
    out = f(rgb, ks, padding="valid")
    assert out.shape == (4, 30, 30, 4)
    for i in range(4):
        assert np.allclose(out[..., i], ref_channels(rgb, ks[..., i]))
    out = f(rgb, ks, padding="same", stride=(2, 2))
    assert out.shape == (4, 16, 16, 4)
    for i in range(4):
        assert np.allclose(out[..., i], ref_channels(rgb, ks[..., i], 1, 1, 2, 2))
    assert_max_loops("5-convolve.py", 3)


# ---------- 6. Pooling ----------

def test_6_pool_tiny_example():
    f = load_task(__file__, "6-pool.py", "pool")
    img = np.arange(16, dtype=float).reshape(1, 4, 4, 1)
    img = np.concatenate([img, -img], axis=3)  # 2 channels
    mx = f(img, (2, 2), (2, 2), mode="max")
    avg = f(img, (2, 2), (2, 2), mode="avg")
    assert mx.shape == (1, 2, 2, 2)
    assert np.allclose(mx[0, ..., 0], [[5, 7], [13, 15]])
    assert np.allclose(mx[0, ..., 1], [[0, -2], [-8, -10]])
    assert np.allclose(avg[0, ..., 0], [[2.5, 4.5], [10.5, 12.5]])
    assert np.allclose(f(img, (2, 2), (2, 2)), mx), "default mode is max"


def test_6_pool_overlapping_windows():
    f = load_task(__file__, "6-pool.py", "pool")
    img = np.array([[1, 5, 2], [7, 3, 9], [4, 8, 6]], dtype=float).reshape(1, 3, 3, 1)
    out = f(img, (2, 2), (1, 1), mode="max")
    assert np.allclose(out[0, ..., 0], [[7, 9], [8, 9]])
    out = f(img, (3, 2), (1, 1), mode="avg")
    assert np.allclose(out[0, ..., 0], [[(1 + 5 + 7 + 3 + 4 + 8) / 6, (5 + 2 + 3 + 9 + 8 + 6) / 6]])


def test_6_pool_shapes(rgb):
    f = load_task(__file__, "6-pool.py", "pool")
    assert f(rgb, (2, 2), (2, 2), mode="avg").shape == (4, 16, 16, 3)
    assert f(rgb, (3, 3), (2, 2)).shape == (4, 15, 15, 3)
    out = f(rgb, (2, 2), (2, 2), mode="max")
    assert np.all(out >= f(rgb, (2, 2), (2, 2), mode="avg"))
    assert_max_loops("6-pool.py", 2)
