"""Tests for 00-math/05-probability. Run: pytest 00-math/05-probability

Your classes use no imports; the tests use scipy.stats as the reference.
"""
import ast

import numpy as np
import pytest
from scipy import stats

from helpers import load_task, project_dir


def assert_no_imports(filename):
    tree = ast.parse((project_dir(__file__) / filename).read_text())
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)), \
        f"{filename}: no imports allowed, build it from scratch"


def poisson_data():
    np.random.seed(0)
    return np.random.poisson(5., 100).tolist()


def exponential_data():
    np.random.seed(0)
    return np.random.exponential(0.5, 100).tolist()


def normal_data():
    np.random.seed(0)
    return np.random.normal(70, 10, 100).tolist()


def binomial_data():
    np.random.seed(0)
    return np.random.binomial(50, 0.6, 100).tolist()


@pytest.fixture
def Poisson():
    cls = load_task(__file__, "poisson.py", "Poisson")
    assert_no_imports("poisson.py")
    return cls


@pytest.fixture
def Exponential():
    cls = load_task(__file__, "exponential.py", "Exponential")
    assert_no_imports("exponential.py")
    return cls


@pytest.fixture
def Normal():
    cls = load_task(__file__, "normal.py", "Normal")
    assert_no_imports("normal.py")
    return cls


@pytest.fixture
def Binomial():
    cls = load_task(__file__, "binomial.py", "Binomial")
    assert_no_imports("binomial.py")
    return cls


# ---------- 0. Initialize Poisson ----------

def test_0_poisson_init(Poisson):
    p1 = Poisson(poisson_data())
    assert p1.lambtha == pytest.approx(4.84)
    p2 = Poisson(lambtha=5)
    assert p2.lambtha == 5.0 and isinstance(p2.lambtha, float)
    assert Poisson().lambtha == 1.0


def test_0_poisson_errors(Poisson):
    with pytest.raises(ValueError, match="lambtha must be a positive value"):
        Poisson(lambtha=0)
    with pytest.raises(ValueError, match="lambtha must be a positive value"):
        Poisson(lambtha=-2)
    with pytest.raises(TypeError, match="data must be a list"):
        Poisson(data=(1, 2, 3))
    with pytest.raises(ValueError, match="data must contain multiple values"):
        Poisson(data=[3])


# ---------- 1. Poisson PMF ----------

def test_1_poisson_pmf(Poisson):
    for lam in (5.0, 4.84, 0.7):
        p = Poisson(lambtha=lam)
        for k in range(0, 16):
            assert p.pmf(k) == pytest.approx(stats.poisson.pmf(k, lam), rel=1e-6)


def test_1_poisson_pmf_edge_cases(Poisson):
    p = Poisson(lambtha=5)
    assert p.pmf(9.7) == pytest.approx(p.pmf(9)), "non-integer k is converted to int"
    assert p.pmf(-1) == 0


# ---------- 2. Poisson CDF ----------

def test_2_poisson_cdf(Poisson):
    for lam in (5.0, 4.84):
        p = Poisson(lambtha=lam)
        for k in range(0, 16):
            assert p.cdf(k) == pytest.approx(stats.poisson.cdf(k, lam), rel=1e-6)
    p = Poisson(lambtha=5)
    assert p.cdf(9.5) == pytest.approx(p.cdf(9))
    assert p.cdf(-3) == 0


# ---------- 3. Initialize Exponential ----------

def test_3_exponential_init(Exponential):
    e1 = Exponential(exponential_data())
    assert e1.lambtha == pytest.approx(2.1771114730906937)
    e2 = Exponential(lambtha=2)
    assert e2.lambtha == 2.0 and isinstance(e2.lambtha, float)


def test_3_exponential_errors(Exponential):
    with pytest.raises(ValueError, match="lambtha must be a positive value"):
        Exponential(lambtha=0)
    with pytest.raises(TypeError, match="data must be a list"):
        Exponential(data="1, 2")
    with pytest.raises(ValueError, match="data must contain multiple values"):
        Exponential(data=[1.0])


# ---------- 4. Exponential PDF ----------

def test_4_exponential_pdf(Exponential):
    for lam in (2.0, 0.5):
        e = Exponential(lambtha=lam)
        for x in (0.0, 0.1, 1.0, 2.5, 7.0):
            assert e.pdf(x) == pytest.approx(stats.expon.pdf(x, scale=1 / lam), rel=1e-6)
    assert Exponential(lambtha=2).pdf(-1) == 0


# ---------- 5. Exponential CDF ----------

def test_5_exponential_cdf(Exponential):
    for lam in (2.0, 0.5):
        e = Exponential(lambtha=lam)
        for x in (0.0, 0.1, 1.0, 2.5, 7.0):
            assert e.cdf(x) == pytest.approx(stats.expon.cdf(x, scale=1 / lam), rel=1e-6, abs=1e-12)
    assert Exponential(lambtha=2).cdf(-1) == 0


# ---------- 6. Initialize Normal ----------

def test_6_normal_init(Normal):
    n1 = Normal(normal_data())
    assert n1.mean == pytest.approx(70.59808015534485)
    assert n1.stddev == pytest.approx(10.078822447165797)  # population stddev (divide by n)
    n2 = Normal(mean=70, stddev=10)
    assert n2.mean == 70.0 and n2.stddev == 10.0
    assert isinstance(n2.mean, float) and isinstance(n2.stddev, float)


def test_6_normal_errors(Normal):
    with pytest.raises(ValueError, match="stddev must be a positive value"):
        Normal(stddev=0)
    with pytest.raises(ValueError, match="stddev must be a positive value"):
        Normal(stddev=-1)
    with pytest.raises(TypeError, match="data must be a list"):
        Normal(data=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="data must contain multiple values"):
        Normal(data=[1.0])


# ---------- 7. Normalize Normal ----------

def test_7_z_score_and_x_value(Normal):
    n = Normal(mean=70, stddev=10)
    assert n.z_score(90) == pytest.approx(2.0)
    assert n.x_value(2) == pytest.approx(90.0)
    assert n.x_value(n.z_score(63.3)) == pytest.approx(63.3)
    n1 = Normal(normal_data())
    assert n1.z_score(90) == pytest.approx(1.9250185174272068)
    assert n1.x_value(2) == pytest.approx(90.75572504967644)


# ---------- 8. Normal PDF ----------

def test_8_normal_pdf(Normal):
    n = Normal(mean=70, stddev=10)
    for x in (40, 55, 70, 81.5, 90, 110):
        assert n.pdf(x) == pytest.approx(stats.norm.pdf(x, 70, 10), rel=1e-6)


# ---------- 9. Normal CDF ----------

def test_9_normal_cdf_near_mean(Normal):
    # Holberton's version approximates erf with a short series, which is accurate
    # near the mean; a better approximation is welcome too.
    n = Normal(mean=70, stddev=10)
    for x in np.linspace(60, 80, 9):
        assert n.cdf(x) == pytest.approx(stats.norm.cdf(x, 70, 10), abs=1e-3)
    assert n.cdf(70) == pytest.approx(0.5)


def test_9_normal_cdf_is_increasing(Normal):
    n = Normal(mean=0, stddev=1)
    values = [n.cdf(x) for x in np.linspace(-1, 1, 21)]
    assert all(a < b for a, b in zip(values, values[1:]))


# ---------- 10. Initialize Binomial ----------

def test_10_binomial_init(Binomial):
    b1 = Binomial(binomial_data())
    assert b1.n == 50 and isinstance(b1.n, int)
    assert b1.p == pytest.approx(0.606)
    b2 = Binomial(n=50, p=0.6)
    assert b2.n == 50 and b2.p == 0.6
    assert isinstance(b2.p, float)


def test_10_binomial_errors(Binomial):
    with pytest.raises(ValueError, match="n must be a positive value"):
        Binomial(n=0)
    for bad_p in (0, 1, -0.2, 1.5):
        with pytest.raises(ValueError, match="p must be greater than 0 and less than 1"):
            Binomial(n=10, p=bad_p)
    with pytest.raises(TypeError, match="data must be a list"):
        Binomial(data={1, 2})
    with pytest.raises(ValueError, match="data must contain multiple values"):
        Binomial(data=[1])


# ---------- 11. Binomial PMF ----------

def test_11_binomial_pmf(Binomial):
    b = Binomial(n=50, p=0.6)
    for k in (0, 1, 10, 25, 30, 42, 50):
        assert b.pmf(k) == pytest.approx(stats.binom.pmf(k, 50, 0.6), rel=1e-6, abs=1e-30)
    assert b.pmf(30.9) == pytest.approx(b.pmf(30))
    assert b.pmf(-1) == 0
    assert b.pmf(51) == 0
    b1 = Binomial(binomial_data())
    assert b1.pmf(30) == pytest.approx(0.11412829839570347, rel=1e-6)


# ---------- 12. Binomial CDF ----------

def test_12_binomial_cdf(Binomial):
    b = Binomial(n=50, p=0.6)
    for k in (0, 10, 25, 30, 42):
        assert b.cdf(k) == pytest.approx(stats.binom.cdf(k, 50, 0.6), rel=1e-6, abs=1e-30)
    assert b.cdf(-1) == 0
    assert b.cdf(50) == pytest.approx(1.0)
    assert b.cdf(60) == pytest.approx(1.0), "every outcome is <= 60, so the CDF is 1"
    b1 = Binomial(binomial_data())
    assert b1.cdf(30) == pytest.approx(0.5189392017296368, rel=1e-6)
