"""Tests for 00-math/08-multivariate_prob. Run: pytest 00-math/08-multivariate_prob

numpy.cov / np.corrcoef / scipy are used here only as references;
your code may not call numpy.cov.
"""
import numpy as np
import pytest
from scipy import stats

from helpers import load_task, project_dir


def assert_no_np_cov(filename):
    src = (project_dir(__file__) / filename).read_text()
    assert ".cov(" not in src, f"{filename}: numpy.cov is not allowed, compute it yourself"


def sample(seed=0, n=100_000):
    rng = np.random.default_rng(seed)
    mean = [12, 30, 10]
    cov = [[36, -30, 15], [-30, 100, -20], [15, -20, 25]]
    return rng.multivariate_normal(mean, cov, n)


# ---------- 0. Mean and Covariance ----------

def test_0_mean_cov():
    mean_cov = load_task(__file__, "0-mean_cov.py", "mean_cov")
    X = sample(n=1000)
    mean, cov = mean_cov(X)
    assert mean.shape == (1, 3)
    assert cov.shape == (3, 3)
    assert np.allclose(mean, X.mean(axis=0))
    assert np.allclose(cov, np.cov(X, rowvar=False))  # sample covariance: divide by n - 1
    assert_no_np_cov("0-mean_cov.py")


def test_0_mean_cov_close_to_truth():
    mean_cov = load_task(__file__, "0-mean_cov.py", "mean_cov")
    mean, cov = mean_cov(sample())
    assert np.allclose(mean, [[12, 30, 10]], atol=0.2)
    assert np.allclose(cov, [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], atol=2)


def test_0_mean_cov_errors():
    mean_cov = load_task(__file__, "0-mean_cov.py", "mean_cov")
    for bad in ([[1, 2], [3, 4]], np.array([1, 2, 3]), np.zeros((2, 2, 2))):
        with pytest.raises(TypeError, match="X must be a 2D numpy.ndarray"):
            mean_cov(bad)
    with pytest.raises(ValueError, match="X must contain multiple data points"):
        mean_cov(np.array([[1.0, 2.0, 3.0]]))


# ---------- 1. Correlation ----------

def test_1_correlation():
    correlation = load_task(__file__, "1-correlation.py", "correlation")
    C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    expected = np.array([[1, -0.5, 0.5], [-0.5, 1, -0.4], [0.5, -0.4, 1]])
    out = correlation(C)
    assert out.shape == (3, 3)
    assert np.allclose(out, expected)
    X = sample(1, 500)
    assert np.allclose(correlation(np.cov(X, rowvar=False)), np.corrcoef(X, rowvar=False))


def test_1_correlation_errors():
    correlation = load_task(__file__, "1-correlation.py", "correlation")
    with pytest.raises(TypeError, match="C must be a numpy.ndarray"):
        correlation([[1, 0], [0, 1]])
    for bad in (np.zeros((2, 3)), np.zeros(3), np.zeros((2, 2, 2))):
        with pytest.raises(ValueError, match="C must be a 2D square matrix"):
            correlation(bad)


# ---------- 2. Initialize ----------

def test_2_multinormal_init():
    MultiNormal = load_task(__file__, "multinormal.py", "MultiNormal")
    data = sample(n=1000).T  # shape (d, n)
    mn = MultiNormal(data)
    assert mn.mean.shape == (3, 1)
    assert mn.cov.shape == (3, 3)
    assert np.allclose(mn.mean[:, 0], data.mean(axis=1))
    assert np.allclose(mn.cov, np.cov(data))
    assert_no_np_cov("multinormal.py")


def test_2_multinormal_one_dimension():
    MultiNormal = load_task(__file__, "multinormal.py", "MultiNormal")
    data = np.random.default_rng(0).normal(5, 2, (1, 200))
    mn = MultiNormal(data)
    assert mn.mean.shape == (1, 1) and mn.cov.shape == (1, 1)
    assert mn.cov[0, 0] == pytest.approx(np.var(data, ddof=1))


def test_2_multinormal_errors():
    MultiNormal = load_task(__file__, "multinormal.py", "MultiNormal")
    for bad in ([[1, 2], [3, 4]], np.array([1, 2, 3])):
        with pytest.raises(TypeError, match="data must be a 2D numpy.ndarray"):
            MultiNormal(bad)
    with pytest.raises(ValueError, match="data must contain multiple data points"):
        MultiNormal(np.array([[1.0], [2.0], [3.0]]))


# ---------- 3. PDF ----------

def test_3_multinormal_pdf():
    MultiNormal = load_task(__file__, "multinormal.py", "MultiNormal")
    data = sample(2, 1000).T
    mn = MultiNormal(data)
    ref = stats.multivariate_normal(mn.mean[:, 0], mn.cov)
    rng = np.random.default_rng(3)
    for _ in range(5):
        x = (mn.mean[:, 0] + rng.normal(0, 3, 3)).reshape(3, 1)
        out = mn.pdf(x)
        assert np.ndim(out) == 0, "return a single number"
        assert float(out) == pytest.approx(ref.pdf(x[:, 0]), rel=1e-6)


def test_3_multinormal_pdf_errors():
    MultiNormal = load_task(__file__, "multinormal.py", "MultiNormal")
    mn = MultiNormal(sample(n=500).T)
    with pytest.raises(TypeError, match="x must be a numpy.ndarray"):
        mn.pdf([[1], [2], [3]])
    for bad in (np.zeros((3,)), np.zeros((2, 1)), np.zeros((3, 2)), np.zeros((1, 3))):
        with pytest.raises(ValueError, match=r"x must have the shape \(3, 1\)"):
            mn.pdf(bad)
