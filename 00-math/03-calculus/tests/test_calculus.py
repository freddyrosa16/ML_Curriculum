"""Tests for 00-math/03-calculus. Run: pytest 00-math/03-calculus

Short-answer questions: put a single number in each answer file
(e.g. 0-sigma_is_for_sum.txt). Decimals or fractions like 3/4 are fine.
The expected answers are stored hashed (rounded to 2 decimals) so this
file doesn't spoil them.
"""
import ast
import hashlib
from fractions import Fraction

import pytest

from helpers import load_task, project_dir

ANSWER_HASHES = {
    "0-sigma_is_for_sum": "3d03d318fa79a54f7596d2794f0e8bf320a5eb4d4384c024264c2e734256a63c",
    "1-seegma": "35e594c4c74652c95aa3bd361e9854efdeb209508f3d50d2f43b315319b8001e",
    "2-pi_is_for_product": "5e0de86a2a25df2c147714d064da7b24926eeff47c9142fe1cae1a71c7eb0cab",
    "3-pee": "561b2814d3c09e62a92442c946307918f7f63f833c84876c08bd4c406767e53b",
    "4-hello_derivatives": "59595eca9f5351670d1171a6f8f959e386f5db7291a2549ef8826ee0b5a64595",
    "5-log_on_fire": "9f770654da26ea8d83cecc52f971d4411c85522f9d5cb8c6042642ba6552a03b",
    "6-voltaire": "f88afaea4207e3bdbf9381c1649835a1780019d246fe640dd86f1abe706462c0",
    "7-partial_truths": "9f770654da26ea8d83cecc52f971d4411c85522f9d5cb8c6042642ba6552a03b",
    "8-all-together": "ba3109ff622b04cfc3afa71bbed22f7033ba85d9192156094e9afd69504212d4",
    "11-integral": "8c9d9d6d5ec9619db9289351778798376a2475205c101c71c09610fdf3fd313b",
    "12-integral": "f88afaea4207e3bdbf9381c1649835a1780019d246fe640dd86f1abe706462c0",
    "13-definite": "4cc019de7a747650855142af31a90fa652d0e00ee24e804acf171b4c82d675e9",
    "14-definite": "cf9dcf6da8a82be1335c398a4005def7ee3a53d4698c59dbc6b2b14e72d1263c",
    "15-definite": "a8994a9020c4c3c2ad52f5ea86644fea733f40ee98af95a05a5bd864a2875f67",
    "16-double": "0ce69db87d55e8c4d9bb9b2856a97837da4a3965512288b685f2aef48b569aff",
}


def _hash_number(value):
    rounded = round(float(value), 2) + 0.0  # + 0.0 turns -0.0 into 0.0
    return hashlib.sha256(f"{rounded:.2f}".encode()).hexdigest()


@pytest.mark.parametrize("name", list(ANSWER_HASHES))
def test_short_answer(name):
    path = project_dir(__file__) / f"{name}.txt"
    if not path.exists():
        pytest.skip(f"{name}.txt not written yet")
    text = path.read_text().strip()
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        pytest.fail(f"{name}.txt should hold one plain number (like 2.5 or 5/2), got {text!r}")
    assert _hash_number(value) == ANSWER_HASHES[name], f"{name}: {text} is not right yet"


# ---------- 9. Our life is the sum total of all the decisions we make every day ----------

def test_9_summation_i_squared():
    f = load_task(__file__, "9-sum_total.py", "summation_i_squared")
    assert f(5) == 55
    assert f(1) == 1
    assert f(100) == 338350
    assert f(10**6) == 333333833333500000
    assert isinstance(f(5), int)


def test_9_invalid_input():
    f = load_task(__file__, "9-sum_total.py", "summation_i_squared")
    assert f(0) is None
    assert f(-3) is None
    assert f("5") is None
    assert f(2.5) is None


def test_9_no_loops():
    load_task(__file__, "9-sum_total.py")
    tree = ast.parse((project_dir(__file__) / "9-sum_total.py").read_text())
    loops = (ast.For, ast.While, ast.comprehension)
    assert not any(isinstance(n, loops) for n in ast.walk(tree)), "no loops: find the closed-form formula"
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "sum" not in calls, "no built-in sum either"


# ---------- 10. Derive happiness in oneself from a good day's work ----------

def test_10_poly_derivative():
    f = load_task(__file__, "10-matisse.py", "poly_derivative")
    assert f([5, 3, 0, 1]) == [3, 0, 3]
    assert f([1, 2, 3]) == [2, 6]
    assert f([0, 0, 0, 4]) == [0, 0, 12]


def test_10_constant_and_invalid():
    f = load_task(__file__, "10-matisse.py", "poly_derivative")
    assert f([5]) == [0]
    assert f([7, 0, 0]) == [0], "if the derivative is 0, return [0]"
    assert f([]) is None
    assert f("not a list") is None
    assert f(None) is None


# ---------- 17. Integrate ----------

def test_17_poly_integral():
    f = load_task(__file__, "17-integrate.py", "poly_integral")
    assert f([5, 3, 0, 1]) == [0, 5, 1.5, 0, 0.25]
    assert f([5, 3, 0, 1], C=2) == [2, 5, 1.5, 0, 0.25]
    assert f([1, 1]) == [0, 1, 0.5]


def test_17_whole_numbers_are_ints():
    f = load_task(__file__, "17-integrate.py", "poly_integral")
    out = f([4, 6])
    assert out == [0, 4, 3]
    assert all(isinstance(c, int) for c in out)


def test_17_smallest_list():
    f = load_task(__file__, "17-integrate.py", "poly_integral")
    assert f([0]) == [0]
    assert f([0], C=3) == [3]
    assert f([2, 0, 0]) == [0, 2]


def test_17_invalid():
    f = load_task(__file__, "17-integrate.py", "poly_integral")
    assert f([]) is None
    assert f("abc") is None
    assert f([1, 2], C=1.5) is None
    assert f([1, 2], C="0") is None
