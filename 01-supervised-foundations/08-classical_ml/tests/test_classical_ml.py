"""Tests for 08-classical_ml."""
import numpy as np
import pytest

from helpers import load_task

sklearn = pytest.importorskip("sklearn")
from sklearn.base import clone  # noqa: E402
from sklearn.datasets import (load_breast_cancer, load_diabetes, make_circles,  # noqa: E402
                              make_moons)
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402


def xgboost_or_skip():
    try:
        import xgboost  # noqa: F401
    except Exception as exc:  # ImportError, or XGBoostError when libomp is missing
        pytest.skip(f"xgboost not usable here: {exc.__class__.__name__}")
    return xgboost


@pytest.fixture
def cancer():
    return load_breast_cancer(return_X_y=True)


# 0-1 -------------------------------------------------------------------------

def test_split_data(cancer):
    split_data = load_task(__file__, "0-split.py", "split_data")
    X, y = cancer
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    assert X_tr.shape == (455, 30) and X_te.shape == (114, 30)
    assert len(y_tr) == 455 and len(y_te) == 114
    assert abs(y_tr.mean() - y_te.mean()) < 0.01, "use a stratified split"
    again = split_data(X, y)
    assert np.array_equal(again[1], X_te), "use the seed so the split is reproducible"
    other = split_data(X, y, test_size=0.5, seed=1)
    assert len(other[1]) == 285 or len(other[1]) == 284


def test_cv_score(cancer):
    cv_score = load_task(__file__, "1-cross_validate.py", "cv_score")
    X, y = cancer
    model = LogisticRegression(max_iter=5000)
    mean, std = cv_score(model, X, y, k=5, seed=0)
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(clone(model), X, y, cv=folds)
    assert float(mean) == pytest.approx(scores.mean())
    assert float(std) == pytest.approx(scores.std())
    mean3, _ = cv_score(model, X, y, k=3, seed=2)
    scores3 = cross_val_score(clone(model), X, y,
                              cv=StratifiedKFold(3, shuffle=True, random_state=2))
    assert float(mean3) == pytest.approx(scores3.mean())


# 2-4 -------------------------------------------------------------------------

def test_normal_equation():
    normal_equation = load_task(__file__, "2-normal_equation.py", "normal_equation")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    true = np.array([4.0, 1.0, -2.0, 0.5])
    y = true[0] + X @ true[1:]
    assert np.allclose(normal_equation(X, y), true)
    Xd, yd = load_diabetes(return_X_y=True)
    theta = normal_equation(Xd, yd)
    ref = LinearRegression().fit(Xd, yd)
    assert theta.shape == (11,)
    assert np.isclose(theta[0], ref.intercept_)
    assert np.allclose(theta[1:], ref.coef_)


def test_linear_regression():
    mod = load_task(__file__, "3-linear_regression.py")
    X, y = load_diabetes(return_X_y=True)
    model = mod.fit_linear_regression(X, y)
    assert isinstance(model, LinearRegression)
    assert np.isclose(model.intercept_, 152.13348416289597)
    err = mod.rmse(model, X, y)
    assert isinstance(err, float)
    assert err == pytest.approx(np.sqrt(np.mean((model.predict(X) - y) ** 2)))


def test_logistic_regression(cancer):
    f = load_task(__file__, "4-logistic_regression.py", "fit_logistic_regression")
    X, y = cancer
    clf = f(X, y)
    assert isinstance(clf, LogisticRegression)
    assert clf.C == 1.0
    assert clf.score(X, y) > 0.94
    strong = f(X, y, C=0.001)
    assert strong.C == 0.001
    assert np.abs(strong.coef_).sum() < np.abs(clf.coef_).sum(), \
        "smaller C means stronger regularization"


# 5-6 -------------------------------------------------------------------------

def test_make_scaled_model(cancer):
    make_scaled_model = load_task(__file__, "5-pipeline.py", "make_scaled_model")
    X, y = cancer
    pipe = make_scaled_model(SVC())
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["scaler", "model"]
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["model"], SVC)
    assert not hasattr(pipe.named_steps["scaler"], "mean_"), "return it unfitted"
    pipe.fit(X[:400], y[:400])
    assert np.allclose(pipe.named_steps["scaler"].mean_, X[:400].mean(axis=0))
    assert pipe.score(X[400:], y[400:]) > SVC().fit(X[:400], y[:400]).score(X[400:], y[400:])


def test_compare_kernels():
    compare_kernels = load_task(__file__, "6-svm.py", "compare_kernels")
    X, y = make_circles(400, noise=0.1, factor=0.4, random_state=0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(400)
    tr, te = idx[:320], idx[320:]
    res = compare_kernels(X[tr], y[tr], X[te], y[te])
    assert set(res) == {"linear", "poly", "rbf"}
    assert res["rbf"] > 0.95
    assert res["linear"] < 0.7
    only = compare_kernels(X[tr], y[tr], X[te], y[te], kernels=("rbf",))
    assert set(only) == {"rbf"}


# 7 ---------------------------------------------------------------------------

@pytest.fixture
def moons():
    X, y = make_moons(400, noise=0.25, random_state=0)
    return X[:320], y[:320], X[320:], y[320:]


def test_hist_gb(moons):
    from sklearn.ensemble import HistGradientBoostingClassifier
    mod = load_task(__file__, "7-gradient_boosting.py")
    X_tr, y_tr, X_te, y_te = moons
    model = mod.fit_hist_gb(X_tr, y_tr, seed=0)
    assert isinstance(model, HistGradientBoostingClassifier)
    assert model.random_state == 0
    assert model.score(X_te, y_te) > 0.85


def test_xgboost(moons):
    xgboost = xgboost_or_skip()
    mod = load_task(__file__, "7-gradient_boosting.py")
    X_tr, y_tr, X_te, y_te = moons
    model = mod.fit_xgboost(X_tr, y_tr, seed=0)
    assert isinstance(model, xgboost.XGBClassifier)
    params = model.get_params()
    assert params["n_estimators"] == 200 and params["max_depth"] == 4
    assert params["learning_rate"] == pytest.approx(0.1)
    assert model.score(X_te, y_te) > 0.85


# 8-9 -------------------------------------------------------------------------

def test_tune_svm(moons):
    from sklearn.model_selection import GridSearchCV
    tune_svm = load_task(__file__, "8-grid_search.py", "tune_svm")
    X_tr, y_tr, X_te, y_te = moons
    search = tune_svm(X_tr, y_tr, seed=0)
    assert isinstance(search, GridSearchCV)
    assert hasattr(search, "best_estimator_"), "return the fitted search"
    assert len(search.cv_results_["params"]) == 16
    tried_C = sorted({p["model__C"] for p in search.cv_results_["params"]})
    assert tried_C == [0.1, 1, 10, 100]
    assert search.best_params_["model__C"] in (0.1, 1, 10, 100)
    assert search.n_splits_ == 5
    assert search.best_score_ > 0.85
    assert search.score(X_te, y_te) > 0.85


def test_compare_models():
    compare_models = load_task(__file__, "9-compare_models.py", "compare_models")
    results = compare_models(seed=0)
    required = {"logistic_regression", "svm_rbf", "random_forest", "hist_gradient_boosting"}
    assert required <= set(results)
    assert set(results) <= required | {"xgboost"}
    for name, r in results.items():
        assert set(r) >= {"cv_mean", "cv_std", "test_accuracy"}
        assert 0.9 < float(r["cv_mean"]) <= 1.0, name
        assert 0.9 < float(r["test_accuracy"]) <= 1.0, name
        assert 0 <= float(r["cv_std"]) < 0.1
    assert results["logistic_regression"]["test_accuracy"] == pytest.approx(0.9824561403508771)
