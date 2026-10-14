"""Tests for 05-unsupervised/01-dimensionality_reduction."""
import numpy as np
import pytest

from helpers import load_task


def _six_column_data():
    """50 points in 6-D that really live on a 3-D subspace."""
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(3, 50))
    X = np.array([a, b, c, 2 * a, -5 * b, 10 * c]).T
    return X - X.mean(axis=0)


def _sorted_eigvals(X):
    """Eigenvalues of X^T X, largest first (used to check captured variance)."""
    return np.sort(np.linalg.eigvalsh(X.T @ X))[::-1]


# ---------- Task 0: PCA (keep a fraction of the variance) ----------

def test_pca_var_shape_and_orthonormal():
    pca = load_task(__file__, "0-pca.py", "pca")
    X = _six_column_data()
    W = pca(X)
    assert W.shape == (6, 2)
    assert np.allclose(W.T @ W, np.eye(W.shape[1]), atol=1e-8)


def test_pca_var_keeps_requested_variance():
    pca = load_task(__file__, "0-pca.py", "pca")
    X = _six_column_data()
    ev = _sorted_eigvals(X)
    for var in (0.5, 0.95, 0.999):
        W = pca(X, var)
        nd = W.shape[1]
        kept = ev[:nd].sum() / ev.sum()
        assert kept >= var - 1e-9
        # and it is the smallest such number of components
        if nd > 1:
            assert ev[:nd - 1].sum() / ev.sum() < var
        # W spans the top principal directions: projected variance equals the top eigenvalues
        assert np.isclose(np.sum((X @ W) ** 2), ev[:nd].sum())


def test_pca_var_reconstruction_full_rank():
    pca = load_task(__file__, "0-pca.py", "pca")
    X = _six_column_data()
    W = pca(X, 0.9999999)
    assert W.shape[1] == 3
    X_rec = X @ W @ W.T
    assert np.allclose(X, X_rec, atol=1e-8)


# ---------- Task 1: PCA v2 (fixed number of dimensions) ----------

def test_pca_ndim_shape_and_variance():
    pca = load_task(__file__, "1-pca.py", "pca")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 10)) @ rng.normal(size=(10, 10)) + 5.0  # not centered
    T = pca(X, 4)
    assert T.shape == (100, 4)
    Xc = X - X.mean(axis=0)
    ev = _sorted_eigvals(Xc)
    # each column carries one principal variance, in decreasing order
    col_var = np.sum(T ** 2, axis=0)
    assert np.allclose(col_var, ev[:4], rtol=1e-6)
    # columns are uncorrelated
    G = T.T @ T
    assert np.allclose(G - np.diag(np.diag(G)), 0, atol=1e-6)
    # T is centered even though X was not
    assert np.allclose(T.mean(axis=0), 0, atol=1e-8)


# ---------- Task 2: P_init ----------

def test_P_init():
    P_init = load_task(__file__, "2-P_init.py", "P_init")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(7, 3))
    D, P, betas, H = P_init(X, 5.0)
    brute = np.array([[np.sum((x - y) ** 2) for y in X] for x in X])
    assert D.shape == (7, 7) and np.allclose(D, brute)
    assert np.allclose(np.diag(D), 0)
    assert P.shape == (7, 7) and np.all(P == 0)
    assert betas.shape == (7, 1) and np.all(betas == 1)
    assert np.isclose(H, np.log2(5.0))


# ---------- Task 3: entropy ----------

def test_HP_basic():
    HP = load_task(__file__, "3-entropy.py", "HP")
    Di = np.array([1.0, 2.0, 4.0, 8.0])
    Hi, Pi = HP(Di, 0.5)
    expected = np.exp(-0.5 * Di) / np.exp(-0.5 * Di).sum()
    assert Pi.shape == (4,)
    assert np.allclose(Pi, expected)
    assert np.isclose(Hi, -np.sum(expected * np.log2(expected)))


def test_HP_beta_zero_is_uniform():
    HP = load_task(__file__, "3-entropy.py", "HP")
    Hi, Pi = HP(np.array([1.0, 5.0, 9.0, 2.0, 3.0, 7.0, 4.0, 6.0]), 0.0)
    assert np.allclose(Pi, 1 / 8)
    assert np.isclose(Hi, 3.0)  # log2(8)


# ---------- Task 4: P affinities ----------

def test_P_affinities_values():
    P_affinities = load_task(__file__, "4-P_affinities.py", "P_affinities")
    X = np.array([[0., 0], [1, 0], [0, 2], [3, 3], [4, 1], [-2, 1]])
    P = P_affinities(X, perplexity=2.0)
    assert P.shape == (6, 6)
    assert np.allclose(P, P.T)
    assert np.isclose(P.sum(), 1.0)
    assert np.allclose(np.diag(P), 0)
    assert np.isclose(P[0, 1], 0.12947914, rtol=1e-3)
    assert np.isclose(P[3, 4], 0.12918384, rtol=1e-3)
    assert np.isclose(P[2, 5], 0.05112685, rtol=1e-3)


def test_P_affinities_closer_means_bigger():
    P_affinities = load_task(__file__, "4-P_affinities.py", "P_affinities")
    X = np.arange(20, dtype=float).reshape(-1, 1)
    P = P_affinities(X, perplexity=5.0)
    assert P[5, 6] > P[5, 8] > P[5, 12]


# ---------- Task 5: Q affinities ----------

def test_Q_affinities():
    Q_affinities = load_task(__file__, "5-Q_affinities.py", "Q_affinities")
    Y = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    Q, num = Q_affinities(Y)
    assert Q.shape == (3, 3) and num.shape == (3, 3)
    assert np.allclose(np.diag(num), 0)
    assert np.isclose(num[0, 1], 1 / 26)
    assert np.isclose(num[0, 2], 1 / 2)
    assert np.isclose(num[1, 2], 1 / 19)
    assert np.allclose(num, num.T)
    assert np.isclose(Q.sum(), 1.0)
    assert np.allclose(Q, num / num.sum())


# ---------- Task 6: gradients ----------

def test_grads_match_numerical_gradient():
    grads = load_task(__file__, "6-grads.py", "grads")
    Q_affinities = load_task(__file__, "5-Q_affinities.py", "Q_affinities")
    cost = load_task(__file__, "7-cost.py", "cost")
    rng = np.random.default_rng(3)
    P = rng.random((6, 6))
    P = P + P.T
    np.fill_diagonal(P, 0)
    P /= P.sum()
    Y = rng.normal(size=(6, 2))
    dY, Q = grads(Y, P)
    assert dY.shape == (6, 2)
    assert np.allclose(Q, Q_affinities(Y)[0])
    eps = 1e-6
    num = np.zeros_like(Y)
    for i in range(6):
        for j in range(2):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[i, j] += eps
            Ym[i, j] -= eps
            num[i, j] = (cost(P, Q_affinities(Yp)[0]) - cost(P, Q_affinities(Ym)[0])) / (2 * eps)
    # the task leaves out the constant factor 4 from the paper
    assert np.allclose(4 * dY, num, atol=1e-5)


# ---------- Task 7: cost ----------

def test_cost():
    cost = load_task(__file__, "7-cost.py", "cost")
    P = np.array([[0, 0.25], [0.25, 0.0]]) * 2  # sums to 1 with zero diagonal
    assert np.isclose(cost(P, P), 0.0, atol=1e-10)
    P = np.array([[0.0, 0.3, 0.2], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]])
    Q = np.array([[0.0, 0.1, 0.4], [0.1, 0.0, 0.0], [0.4, 0.0, 0.0]])
    expected = 2 * (0.3 * np.log(3) + 0.2 * np.log(0.5))
    C = cost(P, Q)
    assert np.isfinite(C)
    assert np.isclose(C, expected)


# ---------- Task 8: t-SNE ----------

def test_tsne_separates_clusters(capsys):
    tsne = load_task(__file__, "8-tsne.py", "tsne")
    rng = np.random.default_rng(4)
    centers = rng.normal(scale=10, size=(3, 10))
    X = np.concatenate([c + rng.normal(size=(20, 10)) for c in centers])
    labels = np.repeat(np.arange(3), 20)
    np.random.seed(0)
    Y = tsne(X, ndims=2, idims=5, perplexity=10.0, iterations=300, lr=100)
    assert Y.shape == (60, 2)
    assert np.all(np.isfinite(Y))
    assert np.allclose(Y.mean(axis=0), 0, atol=1e-6)
    out = capsys.readouterr().out
    assert "Cost at iteration 100" in out and "Cost at iteration 300" in out
    # each point's nearest neighbour in the embedding is from its own cluster
    D = np.sum((Y[:, None] - Y[None]) ** 2, axis=-1)
    np.fill_diagonal(D, np.inf)
    assert np.mean(labels[np.argmin(D, axis=1)] == labels) > 0.95
