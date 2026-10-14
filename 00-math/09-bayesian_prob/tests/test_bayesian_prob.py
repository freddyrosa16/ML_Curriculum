"""Tests for 00-math/09-bayesian_prob. Run: pytest 00-math/09-bayesian_prob

scipy.stats is used here only as the reference.
"""
import numpy as np
import pytest
from scipy import stats

from helpers import load_task

P = np.linspace(0, 1, 11)
PR = np.ones(11) / 11
X, N = 26, 130


def check_common_errors(fn, with_prior):
    args = (P, PR) if with_prior else (P,)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        fn(1, 0, *args)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        fn(1, 3.0, *args)
    with pytest.raises(ValueError, match="x must be an integer that is greater than or equal to 0"):
        fn(-1, 10, *args)
    with pytest.raises(ValueError, match="x must be an integer that is greater than or equal to 0"):
        fn(2.5, 10, *args)
    with pytest.raises(ValueError, match="x cannot be greater than n"):
        fn(11, 10, *args)
    bad_P = (P.reshape(1, -1), PR) if with_prior else (P.reshape(1, -1),)
    with pytest.raises(TypeError, match="P must be a 1D numpy.ndarray"):
        fn(1, 10, *bad_P)
    out_of_range = (np.array([0.1, 1.2]), np.array([0.5, 0.5])) if with_prior else (np.array([0.1, 1.2]),)
    with pytest.raises(ValueError, match=r"All values in P must be in the range \[0, 1\]"):
        fn(1, 10, *out_of_range)


def check_prior_errors(fn):
    with pytest.raises(TypeError, match="Pr must be a numpy.ndarray with the same shape as P"):
        fn(X, N, P, list(PR))
    with pytest.raises(TypeError, match="Pr must be a numpy.ndarray with the same shape as P"):
        fn(X, N, P, PR[:5])
    with pytest.raises(ValueError, match=r"All values in Pr must be in the range \[0, 1\]"):
        fn(X, N, np.array([0.2, 0.8]), np.array([1.5, -0.5]))
    with pytest.raises(ValueError, match="Pr must sum to 1"):
        fn(X, N, np.array([0.2, 0.8]), np.array([0.2, 0.2]))


# ---------- 0. Likelihood ----------

def test_0_likelihood():
    likelihood = load_task(__file__, "0-likelihood.py", "likelihood")
    out = likelihood(X, N, P)
    assert out.shape == (11,)
    assert np.allclose(out, stats.binom.pmf(X, N, P))
    assert out[2] == pytest.approx(0.0872, abs=1e-4)
    assert np.allclose(likelihood(0, 5, np.array([0.0, 0.5, 1.0])), [1.0, 1 / 32, 0.0])


def test_0_likelihood_errors():
    likelihood = load_task(__file__, "0-likelihood.py", "likelihood")
    check_common_errors(likelihood, with_prior=False)


# ---------- 1. Intersection ----------

def test_1_intersection():
    intersection = load_task(__file__, "1-intersection.py", "intersection")
    out = intersection(X, N, P, PR)
    assert out.shape == (11,)
    assert np.allclose(out, stats.binom.pmf(X, N, P) * PR)
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    assert np.allclose(intersection(3, 7, p, prior), stats.binom.pmf(3, 7, p) * prior)
    assert np.allclose(intersection(0, 7, p, prior), stats.binom.pmf(0, 7, p) * prior), "x = 0 is allowed"


def test_1_intersection_errors():
    intersection = load_task(__file__, "1-intersection.py", "intersection")
    check_common_errors(intersection, with_prior=True)
    check_prior_errors(intersection)


# ---------- 2. Marginal Probability ----------

def test_2_marginal():
    marginal = load_task(__file__, "2-marginal.py", "marginal")
    out = marginal(X, N, P, PR)
    assert np.ndim(out) == 0
    assert out == pytest.approx(np.sum(stats.binom.pmf(X, N, P) * PR))
    assert out == pytest.approx(0.008229580791426582, rel=1e-6)


def test_2_marginal_errors():
    marginal = load_task(__file__, "2-marginal.py", "marginal")
    check_common_errors(marginal, with_prior=True)
    check_prior_errors(marginal)


# ---------- 3. Posterior ----------

def test_3_posterior():
    posterior = load_task(__file__, "3-posterior.py", "posterior")
    out = posterior(X, N, P, PR)
    joint = stats.binom.pmf(X, N, P) * PR
    assert out.shape == (11,)
    assert np.allclose(out, joint / joint.sum())
    assert out.sum() == pytest.approx(1.0)
    assert np.argmax(out) == 2  # 26/130 = 0.2 is the most likely p


def test_3_posterior_errors():
    posterior = load_task(__file__, "3-posterior.py", "posterior")
    check_common_errors(posterior, with_prior=True)
    check_prior_errors(posterior)


# ---------- 100. Continuous Posterior (bonus) ----------

def test_100_continuous_posterior():
    posterior = load_task(__file__, "100-continuous.py", "posterior")
    out = posterior(X, N, 0.17, 0.23)
    assert out == pytest.approx(0.6098093274896035, rel=1e-6)
    beta = stats.beta(X + 1, N - X + 1)
    for p1, p2 in ((0.0, 1.0), (0.1, 0.15), (0.2, 0.5)):
        assert posterior(X, N, p1, p2) == pytest.approx(beta.cdf(p2) - beta.cdf(p1), rel=1e-6, abs=1e-12)


def test_100_continuous_posterior_errors():
    posterior = load_task(__file__, "100-continuous.py", "posterior")
    with pytest.raises(ValueError, match="n must be a positive integer"):
        posterior(1, 0, 0.1, 0.2)
    with pytest.raises(ValueError, match="x must be an integer that is greater than or equal to 0"):
        posterior(-1, 10, 0.1, 0.2)
    with pytest.raises(ValueError, match="x cannot be greater than n"):
        posterior(11, 10, 0.1, 0.2)
    with pytest.raises(ValueError, match=r"p1 must be a float in the range \[0, 1\]"):
        posterior(1, 10, 1.5, 0.2)
    with pytest.raises(ValueError, match=r"p2 must be a float in the range \[0, 1\]"):
        posterior(1, 10, 0.1, 1)
    with pytest.raises(ValueError, match="p2 must be greater than p1"):
        posterior(1, 10, 0.5, 0.2)
