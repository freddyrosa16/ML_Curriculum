"""Tests for 05-unsupervised/02-clustering."""
import matplotlib

matplotlib.use("Agg")  # never open plot windows during tests

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import multivariate_normal  # noqa: E402

from helpers import load_task  # noqa: E402

CENTERS = np.array([[30, 40], [10, 25], [40, 20], [60, 30], [20, 70]], dtype=float)


def five_blobs(seed=0):
    """250 points: 50 around each of 5 well-separated centres (std 4)."""
    np.random.seed(seed)
    X = np.concatenate([np.random.multivariate_normal(c, 16 * np.eye(2), size=50) for c in CENTERS])
    np.random.shuffle(X)
    return X


def three_blobs(seed=1):
    """300 points around 3 very well-separated centres, with different spreads."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
    covs = [np.array([[2.0, 0.5], [0.5, 1.0]]), np.eye(2) * 1.5, np.array([[1.0, -0.3], [-0.3, 2.0]])]
    X = np.concatenate([rng.multivariate_normal(c, s, size=100) for c, s in zip(centers, covs)])
    return X, centers


def matched_distance(found, true):
    """Largest distance from a true centre to its closest found centre."""
    d = np.linalg.norm(true[:, None] - found[None], axis=-1)
    return d.min(axis=1).max()


# ---------- Task 0: initialize ----------

def test_initialize_shape_bounds_and_uniform():
    initialize = load_task(__file__, "0-initialize.py", "initialize")
    X = five_blobs()
    np.random.seed(3)
    C = initialize(X, 5)
    assert C.shape == (5, 2)
    assert np.all(C >= X.min(axis=0)) and np.all(C <= X.max(axis=0))
    np.random.seed(3)
    expected = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(5, 2))
    assert np.allclose(C, expected)


@pytest.mark.parametrize("args", [("not an array", 3), (np.zeros(5), 2), (None, 2)])
def test_initialize_bad_X(args):
    initialize = load_task(__file__, "0-initialize.py", "initialize")
    assert initialize(*args) is None


@pytest.mark.parametrize("k", [0, -1, 2.5, "3"])
def test_initialize_bad_k(k):
    initialize = load_task(__file__, "0-initialize.py", "initialize")
    assert initialize(five_blobs(), k) is None


# ---------- Task 1: k-means ----------

def test_kmeans_converged_solution_is_consistent():
    kmeans = load_task(__file__, "1-kmeans.py", "kmeans")
    X = five_blobs()
    np.random.seed(0)
    C, clss = kmeans(X, 5)
    assert C.shape == (5, 2) and clss.shape == (250,)
    # every point is assigned to its nearest centroid
    d = np.linalg.norm(X[:, None] - C[None], axis=-1)
    assert np.array_equal(clss, np.argmin(d, axis=1))
    # every non-empty cluster's centroid is the mean of its points
    for i in range(5):
        if np.any(clss == i):
            assert np.allclose(C[i], X[clss == i].mean(axis=0))


def test_kmeans_finds_the_blobs():
    kmeans = load_task(__file__, "1-kmeans.py", "kmeans")
    X = five_blobs()
    best = np.inf
    for seed in range(5):  # k-means can get stuck; at least one start should work
        np.random.seed(seed)
        C, _ = kmeans(X, 5)
        best = min(best, matched_distance(C, CENTERS))
    assert best < 2.0


def test_kmeans_bad_inputs():
    kmeans = load_task(__file__, "1-kmeans.py", "kmeans")
    X = five_blobs()
    assert kmeans(X, 0) == (None, None)
    assert kmeans(X, 3, 0) == (None, None)
    assert kmeans(X, 3, 1.5) == (None, None)
    assert kmeans(np.zeros(10), 3) == (None, None)


# ---------- Task 2: variance ----------

def test_variance_matches_brute_force():
    variance = load_task(__file__, "2-variance.py", "variance")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    C = rng.normal(size=(4, 3))
    brute = sum(min(np.sum((x - c) ** 2) for c in C) for x in X)
    assert np.isclose(variance(X, C), brute)


def test_variance_bad_inputs():
    variance = load_task(__file__, "2-variance.py", "variance")
    X = np.ones((10, 2))
    assert variance(X, np.ones((3, 3))) is None
    assert variance(np.ones(10), np.ones((3, 2))) is None
    assert variance(X, "C") is None


# ---------- Task 3: optimum k ----------

def test_optimum_k():
    optimum_k = load_task(__file__, "3-optimum.py", "optimum_k")
    variance = load_task(__file__, "2-variance.py", "variance")
    X = five_blobs()
    np.random.seed(0)
    results, d_vars = optimum_k(X, kmax=6)
    assert len(results) == 6 and len(d_vars) == 6
    for k, (C, clss) in enumerate(results, start=1):
        assert C.shape == (k, 2) and clss.shape == (250,)
    base = variance(X, results[0][0])
    for (C, _), dv in zip(results, d_vars):
        assert np.isclose(dv, base - variance(X, C))
    assert d_vars[0] == 0
    assert d_vars[4] > d_vars[1] > 0


def test_optimum_k_bad_inputs():
    optimum_k = load_task(__file__, "3-optimum.py", "optimum_k")
    X = five_blobs()
    assert optimum_k(X, kmin=5, kmax=5) == (None, None)
    assert optimum_k(X, kmin=0, kmax=5) == (None, None)
    assert optimum_k(X, kmin=1, kmax=3, iterations=0) == (None, None)


# ---------- Task 4: GMM initialize ----------

def test_gmm_initialize():
    initialize = load_task(__file__, "4-initialize.py", "initialize")
    kmeans = load_task(__file__, "1-kmeans.py", "kmeans")
    X = five_blobs()
    np.random.seed(7)
    pi, m, S = initialize(X, 5)
    np.random.seed(7)
    C, _ = kmeans(X, 5)
    assert pi.shape == (5,) and np.allclose(pi, 0.2)
    assert m.shape == (5, 2) and np.allclose(m, C)
    assert S.shape == (5, 2, 2) and np.allclose(S, np.eye(2))


def test_gmm_initialize_bad_inputs():
    initialize = load_task(__file__, "4-initialize.py", "initialize")
    assert initialize(np.ones(5), 2) == (None, None, None)
    assert initialize(five_blobs(), 0) == (None, None, None)


# ---------- Task 5: pdf ----------

def test_pdf_matches_scipy():
    pdf = load_task(__file__, "5-pdf.py", "pdf")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 3))
    m = np.array([0.5, -0.2, 0.1])
    A = rng.normal(size=(3, 3))
    S = A @ A.T + np.eye(3)
    P = pdf(X, m, S)
    assert P.shape == (30,)
    assert np.allclose(P, multivariate_normal(m, S).pdf(X))


def test_pdf_floor():
    pdf = load_task(__file__, "5-pdf.py", "pdf")
    P = pdf(np.array([[1000.0, 1000.0]]), np.zeros(2), np.eye(2))
    assert P[0] == pytest.approx(1e-300)


def test_pdf_bad_inputs():
    pdf = load_task(__file__, "5-pdf.py", "pdf")
    X = np.ones((4, 2))
    assert pdf(X, np.zeros(3), np.eye(2)) is None
    assert pdf(X, np.zeros(2), np.eye(3)) is None
    assert pdf(np.ones(4), np.zeros(2), np.eye(2)) is None


# ---------- Task 6: expectation ----------

def test_expectation():
    expectation = load_task(__file__, "6-expectation.py", "expectation")
    X, centers = three_blobs()
    pi = np.array([0.2, 0.5, 0.3])
    S = np.array([np.eye(2) * 2, np.eye(2), np.eye(2) * 3])
    g, ll = expectation(X, pi, centers, S)
    assert g.shape == (3, 300)
    assert np.allclose(g.sum(axis=0), 1)
    weighted = np.array([p * multivariate_normal(c, s).pdf(X) for p, c, s in zip(pi, centers, S)])
    assert np.allclose(g, weighted / weighted.sum(axis=0))
    assert np.isclose(ll, np.sum(np.log(weighted.sum(axis=0))))


def test_expectation_bad_inputs():
    expectation = load_task(__file__, "6-expectation.py", "expectation")
    X, centers = three_blobs()
    S = np.array([np.eye(2)] * 3)
    assert expectation(X, np.array([0.5, 0.5, 0.5]), centers, S) == (None, None)  # pi doesn't sum to 1
    assert expectation(X, np.ones(3) / 3, centers[:2], S) == (None, None)


# ---------- Task 7: maximization ----------

def test_maximization_hard_assignments():
    maximization = load_task(__file__, "7-maximization.py", "maximization")
    X, _ = three_blobs()
    labels = np.repeat(np.arange(3), 100)
    g = np.eye(3)[labels].T  # one-hot, shape (3, 300)
    pi, m, S = maximization(X, g)
    assert pi.shape == (3,) and m.shape == (3, 2) and S.shape == (3, 2, 2)
    assert np.allclose(pi, 1 / 3)
    for i in range(3):
        pts = X[labels == i]
        assert np.allclose(m[i], pts.mean(axis=0))
        assert np.allclose(S[i], np.cov(pts.T, bias=True))


def test_maximization_soft_assignments():
    maximization = load_task(__file__, "7-maximization.py", "maximization")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 2))
    g = rng.random((2, 50))
    g /= g.sum(axis=0)
    pi, m, S = maximization(X, g)
    assert np.allclose(pi, g.mean(axis=1))
    for i in range(2):
        assert np.allclose(m[i], np.average(X, axis=0, weights=g[i]))
        assert np.allclose(S[i], np.cov(X.T, aweights=g[i], bias=True))


def test_maximization_bad_inputs():
    maximization = load_task(__file__, "7-maximization.py", "maximization")
    X = np.ones((10, 2))
    assert maximization(X, np.ones((2, 10))) == (None, None, None)  # columns don't sum to 1
    assert maximization(X, np.ones((2, 9)) / 2) == (None, None, None)


# ---------- Task 8: EM ----------

def test_em_recovers_mixture(capsys):
    expectation_maximization = load_task(__file__, "8-EM.py", "expectation_maximization")
    X, centers = three_blobs()
    best = None
    for seed in range(3):
        np.random.seed(seed)
        out = expectation_maximization(X, 3, iterations=200, verbose=(seed == 0))
        if best is None or out[4] > best[4]:
            best = out
    pi, m, S, g, ll = best
    assert pi.shape == (3,) and m.shape == (3, 2) and S.shape == (3, 2, 2) and g.shape == (3, 300)
    assert np.isclose(pi.sum(), 1)
    assert np.allclose(np.sort(pi), 1 / 3, atol=0.02)
    assert matched_distance(m, centers) < 0.5
    assert np.isfinite(ll)
    printed = capsys.readouterr().out
    assert "Log Likelihood after 0 iterations:" in printed


def test_em_bad_inputs():
    expectation_maximization = load_task(__file__, "8-EM.py", "expectation_maximization")
    X, _ = three_blobs()
    bad = (None, None, None, None, None)
    assert expectation_maximization(X, 0) == bad
    assert expectation_maximization(X, 2, iterations=0) == bad
    assert expectation_maximization(X, 2, tol=-1.0) == bad
    assert expectation_maximization(X, 2, verbose="yes") == bad


# ---------- Task 9: BIC ----------

def test_bic_picks_three():
    BIC = load_task(__file__, "9-BIC.py", "BIC")
    X, _ = three_blobs()
    np.random.seed(0)
    best_k, best_result, ll, b = BIC(X, kmin=1, kmax=5)
    assert best_k == 3
    pi, m, S = best_result
    assert pi.shape == (3,) and m.shape == (3, 2) and S.shape == (3, 2, 2)
    assert ll.shape == (5,) and b.shape == (5,)
    n, d = X.shape
    p = np.array([k * d + k * d * (d + 1) / 2 + (k - 1) for k in range(1, 6)])
    assert np.allclose(b, p * np.log(n) - 2 * ll)


def test_bic_respects_kmin():
    BIC = load_task(__file__, "9-BIC.py", "BIC")
    X, _ = three_blobs()
    np.random.seed(0)
    best_k, best_result, ll, b = BIC(X, kmin=2, kmax=4)
    assert best_k == 3
    assert ll.shape == (3,) and best_result[0].shape == (3,)


def test_bic_bad_inputs():
    BIC = load_task(__file__, "9-BIC.py", "BIC")
    X, _ = three_blobs()
    bad = (None, None, None, None)
    assert BIC(X, kmin=3, kmax=3) == bad
    assert BIC(X, kmin=0, kmax=3) == bad
    assert BIC(np.ones(5), kmin=1, kmax=2) == bad


# ---------- Task 10: k-means with scikit-learn ----------

def test_sklearn_kmeans():
    kmeans = load_task(__file__, "10-kmeans.py", "kmeans")
    X = five_blobs()
    C, clss = kmeans(X, 5)
    assert C.shape == (5, 2) and clss.shape == (250,)
    assert matched_distance(C, CENTERS) < 2.0


# ---------- Task 11: GMM with scikit-learn ----------

def test_sklearn_gmm():
    gmm = load_task(__file__, "11-gmm.py", "gmm")
    X, centers = three_blobs()
    pi, m, S, clss, bic = gmm(X, 3)
    assert pi.shape == (3,) and m.shape == (3, 2) and S.shape == (3, 2, 2) and clss.shape == (300,)
    assert np.isclose(pi.sum(), 1)
    assert matched_distance(m, centers) < 0.5
    assert np.isscalar(bic) or np.ndim(bic) == 0


# ---------- Task 12: agglomerative clustering ----------

def test_agglomerative(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    agglomerative = load_task(__file__, "12-agglomerative.py", "agglomerative")
    X, _ = three_blobs()
    clss = agglomerative(X, 60)
    assert clss.shape == (300,)
    assert len(np.unique(clss)) == 3
    # points from the same true blob share a label
    for i in range(3):
        assert len(np.unique(clss[i * 100:(i + 1) * 100])) == 1
    plt.close("all")
