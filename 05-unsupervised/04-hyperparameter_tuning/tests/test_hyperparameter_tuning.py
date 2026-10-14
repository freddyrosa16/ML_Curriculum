"""Tests for 05-unsupervised/04-hyperparameter_tuning."""
import numpy as np
import pytest

from helpers import load_task


def f(x):
    """The 'black box' function used throughout the project."""
    return np.sin(5 * x) + 2 * np.sin(-2 * x)


def init_points(n=2, seed=0):
    np.random.seed(seed)
    X = np.random.uniform(-np.pi, 2 * np.pi, (n, 1))
    return X, f(X)


def rbf(a, b, l, sigma_f):
    return sigma_f ** 2 * np.exp(-((a - b) ** 2) / (2 * l ** 2))


# ---------- Task 0: GP init + kernel ----------

def test_gp_init_and_kernel():
    GP = load_task(__file__, "0-gp.py", "GaussianProcess")
    X, Y = init_points(3)
    gp = GP(X, Y, l=0.6, sigma_f=2)
    assert gp.X is X and gp.Y is Y
    assert gp.l == 0.6 and gp.sigma_f == 2
    expected = np.array([[rbf(a, b, 0.6, 2) for b in X[:, 0]] for a in X[:, 0]])
    assert gp.K.shape == (3, 3)
    assert np.allclose(gp.K, expected)
    Z = np.array([[0.0], [1.0]])
    K = gp.kernel(X, Z)
    assert K.shape == (3, 2)
    assert np.allclose(K, [[rbf(a, b, 0.6, 2) for b in Z[:, 0]] for a in X[:, 0]])


def test_gp_defaults():
    GP = load_task(__file__, "0-gp.py", "GaussianProcess")
    X, Y = init_points()
    gp = GP(X, Y)
    assert gp.l == 1 and gp.sigma_f == 1
    assert np.allclose(np.diag(gp.K), 1)


# ---------- Task 1: predict ----------

def test_predict_single_point_closed_form():
    GP = load_task(__file__, "1-gp.py", "GaussianProcess")
    x0, y0 = np.array([[0.5]]), np.array([[1.5]])
    gp = GP(x0, y0, l=0.8, sigma_f=1.5)
    X_s = np.linspace(-2, 3, 7).reshape(-1, 1)
    mu, var = gp.predict(X_s)
    assert mu.shape == (7,) and var.shape == (7,)
    k = rbf(X_s[:, 0], 0.5, 0.8, 1.5)
    assert np.allclose(mu, k / 1.5 ** 2 * 1.5)
    assert np.allclose(var, 1.5 ** 2 - k ** 2 / 1.5 ** 2)


def test_predict_interpolates_training_points():
    GP = load_task(__file__, "1-gp.py", "GaussianProcess")
    X, Y = init_points(4, seed=1)
    gp = GP(X, Y, l=0.6, sigma_f=2)
    mu, var = gp.predict(X)
    assert np.allclose(mu, Y[:, 0], atol=1e-6)
    assert np.allclose(var, 0, atol=1e-6)
    # far from the data: back to the prior (mean 0, variance sigma_f^2)
    mu_far, var_far = gp.predict(np.array([[100.0]]))
    assert np.allclose(mu_far, 0, atol=1e-6) and np.allclose(var_far, 4)


# ---------- Task 2: update ----------

def test_update():
    GP = load_task(__file__, "2-gp.py", "GaussianProcess")
    X, Y = init_points()
    gp = GP(X, Y, l=0.6, sigma_f=2)
    x_new = np.array([1.0])
    gp.update(x_new, f(x_new))
    assert gp.X.shape == (3, 1) and gp.Y.shape == (3, 1)
    assert np.isclose(gp.X[-1, 0], 1.0) and np.isclose(gp.Y[-1, 0], f(1.0))
    assert gp.K.shape == (3, 3)
    assert np.allclose(gp.K, gp.kernel(gp.X, gp.X))
    mu, var = gp.predict(np.array([[1.0]]))
    assert np.isclose(mu[0], f(1.0), atol=1e-6)


# ---------- Task 3: Bayesian optimization init ----------

def test_bo_init():
    BO = load_task(__file__, "3-bayes_opt.py", "BayesianOptimization")
    X, Y = init_points()
    bo = BO(f, X, Y, (-np.pi, 2 * np.pi), 50, l=2, sigma_f=3, xsi=0.05)
    assert bo.f is f
    assert type(bo.gp).__name__ == "GaussianProcess"
    assert bo.gp.X is X and bo.gp.Y is Y
    assert bo.gp.l == 2 and bo.gp.sigma_f == 3
    assert bo.X_s.shape == (50, 1)
    assert np.allclose(bo.X_s[:, 0], np.linspace(-np.pi, 2 * np.pi, 50))
    assert bo.xsi == 0.05 and bo.minimize is True
    bo2 = BO(f, X, Y, (0, 1), 10)
    assert bo2.xsi == 0.01 and bo2.gp.l == 1 and bo2.gp.sigma_f == 1


# ---------- Task 4: acquisition (Expected Improvement) ----------

def monte_carlo_ei(mu, var, best, xsi, minimize, n=400_000, seed=0):
    """Estimate E[max(improvement, 0)] by sampling from N(mu, var)."""
    rng = np.random.default_rng(seed)
    samples = mu[:, None] + np.sqrt(np.maximum(var, 0))[:, None] * rng.standard_normal((len(mu), n))
    imp = (best - samples - xsi) if minimize else (samples - best - xsi)
    return np.maximum(imp, 0).mean(axis=1)


@pytest.mark.parametrize("minimize", [True, False])
def test_acquisition_matches_monte_carlo(minimize):
    BO = load_task(__file__, "4-bayes_opt.py", "BayesianOptimization")
    X, Y = init_points(3)
    bo = BO(f, X, Y, (-np.pi, 2 * np.pi), 12, l=0.6, sigma_f=2, xsi=0.05, minimize=minimize)
    X_next, EI = bo.acquisition()
    assert X_next.shape == (1,) and EI.shape == (12,)
    assert np.all(EI >= -1e-12)
    mu, var = bo.gp.predict(bo.X_s)
    best = Y.min() if minimize else Y.max()
    expected = monte_carlo_ei(mu, var, best, 0.05, minimize)
    assert np.allclose(EI, expected, atol=0.01)
    assert np.allclose(X_next, bo.X_s[np.argmax(EI)])


# ---------- Task 5: optimize ----------

def test_optimize_finds_minimum():
    BO = load_task(__file__, "5-bayes_opt.py", "BayesianOptimization")
    X, Y = init_points()
    bo = BO(f, X, Y, (-np.pi, 2 * np.pi), 50, l=0.6, sigma_f=2)
    X_opt, Y_opt = bo.optimize(50)
    assert X_opt.shape == (1,) and Y_opt.shape == (1,)
    assert np.isclose(Y_opt[0], bo.gp.Y.min())
    assert np.isclose(f(X_opt[0]), Y_opt[0])
    grid_min = f(bo.X_s).min()
    assert Y_opt[0] <= grid_min + 1e-6
    # stopped early instead of re-sampling the same point
    sampled = bo.gp.X[2:, 0]
    assert len(np.unique(np.round(sampled, 8))) == len(sampled)
    assert len(sampled) < 50


def test_optimize_maximize():
    BO = load_task(__file__, "5-bayes_opt.py", "BayesianOptimization")
    X, Y = init_points()
    bo = BO(f, X, Y, (-np.pi, 2 * np.pi), 50, l=0.6, sigma_f=2, minimize=False)
    X_opt, Y_opt = bo.optimize(50)
    assert np.isclose(Y_opt[0], bo.gp.Y.max())
    assert Y_opt[0] >= f(bo.X_s).max() - 1e-6


# ---------- Task 6: Optuna (slow, optional) ----------

@pytest.mark.slow
def test_optuna_study(tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("sklearn")
    run_study = load_task(__file__, "6-optuna.py", "run_study")
    report = tmp_path / "bayes_opt.txt"
    study = run_study(n_trials=6, seed=0, report_path=str(report))
    assert len(study.trials) == 6
    assert study.direction.name == "MAXIMIZE"
    assert len(study.best_params) >= 5
    assert 0.85 < study.best_value <= 1.0
    assert report.exists()
    assert str(round(study.best_value, 3))[:4] in report.read_text()
