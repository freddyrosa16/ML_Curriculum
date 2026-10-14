"""Tests for 00-math/07-advanced_linear_algebra.
Run: pytest 00-math/07-advanced_linear_algebra

Tasks 0-4 must be plain Python; the tests use numpy.linalg as the reference.
"""
import ast

import numpy as np
import pytest

from helpers import load_task, project_dir

M3 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
M4 = [[2, -1, 0, 3], [1, 4, -2, 0], [0, 5, 1, -1], [3, 0, 2, 2]]


def random_int_matrices(seed=0, count=6, max_n=5):
    rng = np.random.default_rng(seed)
    return [rng.integers(-6, 7, (n, n)).tolist() for n in rng.integers(2, max_n + 1, count)]


def assert_no_imports(filename):
    tree = ast.parse((project_dir(__file__) / filename).read_text())
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.Import, ast.ImportFrom)), \
            f"{filename}: no imports (reusing your earlier tasks with __import__ is fine)"


def sign_pattern(n):
    return np.array([[(-1) ** (i + j) for j in range(n)] for i in range(n)])


def check_list_of_lists_errors(fn):
    for bad in (5, "matrix", [1, 2, 3], [[1, 2], 3]):
        with pytest.raises(TypeError, match="matrix must be a list of lists"):
            fn(bad)


def check_square_errors(fn, msg="matrix must be a non-empty square matrix"):
    for bad in ([[1, 2, 3], [4, 5, 6]], [[1, 2]], [[]]):
        with pytest.raises(ValueError, match=msg):
            fn(bad)


# ---------- 0. Determinant ----------

def test_0_determinant_examples():
    det = load_task(__file__, "0-determinant.py", "determinant")
    assert det([[]]) == 1
    assert det([[5]]) == 5
    assert det([[1, 2], [3, 4]]) == -2
    assert det([[1, 1], [1, 1]]) == 0
    assert det(M3) == 192
    assert_no_imports("0-determinant.py")


def test_0_determinant_matches_numpy():
    det = load_task(__file__, "0-determinant.py", "determinant")
    for m in random_int_matrices() + [M4]:
        assert det(m) == pytest.approx(np.linalg.det(m), abs=1e-6)


def test_0_determinant_errors():
    det = load_task(__file__, "0-determinant.py", "determinant")
    check_list_of_lists_errors(det)
    with pytest.raises(TypeError, match="matrix must be a list of lists"):
        det([])
    for bad in ([[1, 2, 3], [4, 5, 6]], [[1, 2]], [[1, 2], [3]]):
        with pytest.raises(ValueError, match="matrix must be a square matrix"):
            det(bad)


# ---------- 1. Minor ----------

def test_1_minor():
    minor = load_task(__file__, "1-minor.py", "minor")
    assert minor([[5]]) == [[1]]
    assert minor([[1, 2], [3, 4]]) == [[4, 3], [2, 1]]
    assert minor([[1, 1], [1, 1]]) == [[1, 1], [1, 1]]
    assert minor(M3) == [[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
    for m in random_int_matrices(1) + [M4]:
        a = np.array(m, dtype=float)
        expected = np.linalg.det(a) * np.linalg.inv(a).T * sign_pattern(len(m)) \
            if abs(np.linalg.det(a)) > 1e-9 else None
        if expected is not None:
            assert np.allclose(minor(m), expected)
    assert_no_imports("1-minor.py")


def test_1_minor_errors():
    minor = load_task(__file__, "1-minor.py", "minor")
    check_list_of_lists_errors(minor)
    check_square_errors(minor)


# ---------- 2. Cofactor ----------

def test_2_cofactor():
    cofactor = load_task(__file__, "2-cofactor.py", "cofactor")
    assert cofactor([[5]]) == [[1]]
    assert cofactor([[1, 2], [3, 4]]) == [[4, -3], [-2, 1]]
    assert cofactor(M3) == [[-12, 36, 0], [-10, -34, 32], [47, -13, -16]]
    for m in random_int_matrices(2) + [M4]:
        a = np.array(m, dtype=float)
        if abs(np.linalg.det(a)) > 1e-9:
            assert np.allclose(cofactor(m), np.linalg.det(a) * np.linalg.inv(a).T)
    assert_no_imports("2-cofactor.py")


def test_2_cofactor_errors():
    cofactor = load_task(__file__, "2-cofactor.py", "cofactor")
    check_list_of_lists_errors(cofactor)
    check_square_errors(cofactor)


# ---------- 3. Adjugate ----------

def test_3_adjugate():
    adjugate = load_task(__file__, "3-adjugate.py", "adjugate")
    assert adjugate([[5]]) == [[1]]
    assert adjugate([[1, 2], [3, 4]]) == [[4, -2], [-3, 1]]
    assert adjugate(M3) == [[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
    for m in random_int_matrices(3) + [M4]:
        a = np.array(m, dtype=float)
        if abs(np.linalg.det(a)) > 1e-9:
            assert np.allclose(adjugate(m), np.linalg.det(a) * np.linalg.inv(a))
    assert_no_imports("3-adjugate.py")


def test_3_adjugate_errors():
    adjugate = load_task(__file__, "3-adjugate.py", "adjugate")
    check_list_of_lists_errors(adjugate)
    check_square_errors(adjugate)


# ---------- 4. Inverse ----------

def test_4_inverse():
    inverse = load_task(__file__, "4-inverse.py", "inverse")
    assert np.allclose(inverse([[5]]), [[0.2]])
    assert np.allclose(inverse([[1, 2], [3, 4]]), [[-2.0, 1.0], [1.5, -0.5]])
    assert inverse([[1, 1], [1, 1]]) is None
    assert np.allclose(inverse(M3), np.linalg.inv(M3))
    for m in random_int_matrices(4) + [M4]:
        a = np.array(m, dtype=float)
        if abs(np.linalg.det(a)) > 1e-9:
            assert np.allclose(inverse(m), np.linalg.inv(a))
    assert isinstance(inverse(M3), list)
    assert_no_imports("4-inverse.py")


def test_4_inverse_errors():
    inverse = load_task(__file__, "4-inverse.py", "inverse")
    check_list_of_lists_errors(inverse)
    check_square_errors(inverse)


# ---------- 5. Definiteness ----------

def test_5_definiteness():
    definiteness = load_task(__file__, "5-definiteness.py", "definiteness")
    assert definiteness(np.array([[5, 1], [1, 1]])) == "Positive definite"
    assert definiteness(np.array([[2, 4], [4, 8]])) == "Positive semi-definite"
    assert definiteness(np.array([[-1, 1], [1, -1]])) == "Negative semi-definite"
    assert definiteness(np.array([[-2, 4], [4, -9]])) == "Negative definite"
    assert definiteness(np.array([[1, 2], [2, 1]])) == "Indefinite"
    assert definiteness(np.array([[2, 1, 1], [1, 2, -1], [1, -1, 2]])) == "Positive semi-definite"
    assert definiteness(np.array([[-1, 2, 0], [2, -5, 2], [0, 2, -6]])) == "Negative definite"


def test_5_definiteness_invalid():
    definiteness = load_task(__file__, "5-definiteness.py", "definiteness")
    assert definiteness(np.array([[1, 2], [1, 2]])) is None, "not symmetric"
    assert definiteness(np.array([[1, 2, 3], [4, 5, 6]])) is None, "not square"
    assert definiteness(np.array([1, 2, 3])) is None, "not 2D"
    with pytest.raises(TypeError, match="matrix must be a numpy.ndarray"):
        definiteness([[1, 0], [0, 1]])
