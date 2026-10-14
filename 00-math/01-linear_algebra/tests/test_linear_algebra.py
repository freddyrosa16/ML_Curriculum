"""Tests for 00-math/01-linear_algebra. Run: pytest 00-math/01-linear_algebra"""
import ast

import numpy as np
import pytest

from helpers import load_task, project_dir


# ---------- small helpers for the "no loops / no if" rules ----------

def _tree(filename):
    path = project_dir(__file__) / filename
    if not path.exists():
        pytest.skip(f"{filename} not written yet")
    return ast.parse(path.read_text())


def _count(filename, *node_types):
    return sum(isinstance(n, node_types) for n in ast.walk(_tree(filename)))


LOOPS = (ast.For, ast.While, ast.comprehension)
CONDITIONALS = (ast.If, ast.IfExp)


def assert_no_loops_or_ifs(filename):
    assert _count(filename, *LOOPS) == 0, f"{filename}: no loops allowed"
    assert _count(filename, *CONDITIONALS) == 0, f"{filename}: no if allowed"


# ---------- 0. Slice Me Up ----------

def test_0_slice_me_up():
    mod = load_task(__file__, "0-slice_me_up.py")
    assert mod.arr1 == [9, 8]
    assert mod.arr2 == [9, 4, 1, 0, 3]
    assert mod.arr3 == [8, 2, 3, 9, 4]
    assert_no_loops_or_ifs("0-slice_me_up.py")


# ---------- 1. Trim Me Down ----------

def test_1_trim_me_down():
    mod = load_task(__file__, "1-trim_me_down.py")
    assert mod.the_middle == [[9, 4], [7, 3], [4, 6]]
    assert _count("1-trim_me_down.py", ast.For) <= 1, "only one for loop"
    assert _count("1-trim_me_down.py", ast.While, ast.comprehension) == 0
    assert _count("1-trim_me_down.py", *CONDITIONALS) == 0


# ---------- 2. Size Me Please ----------

def test_2_matrix_shape():
    matrix_shape = load_task(__file__, "2-size_me_please.py", "matrix_shape")
    assert list(matrix_shape([[1, 2], [3, 4]])) == [2, 2]
    mat = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
           [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
    assert list(matrix_shape(mat)) == [2, 3, 5]
    assert list(matrix_shape([1, 2, 3])) == [3]


# ---------- 3. Flip Me Over ----------

def test_3_matrix_transpose():
    matrix_transpose = load_task(__file__, "3-flip_me_over.py", "matrix_transpose")
    mat = [[1, 2, 3], [4, 5, 6]]
    out = matrix_transpose(mat)
    assert out == [[1, 4], [2, 5], [3, 6]]
    assert mat == [[1, 2, 3], [4, 5, 6]], "don't change the input"
    assert matrix_transpose([[7]]) == [[7]]


# ---------- 4. Line Up ----------

def test_4_add_arrays():
    add_arrays = load_task(__file__, "4-line_up.py", "add_arrays")
    a, b = [1, 2, 3, 4], [5, 6, 7, 8]
    out = add_arrays(a, b)
    assert out == [6, 8, 10, 12]
    assert out is not a and out is not b, "return a new list"
    assert add_arrays([1, 2], [1, 2, 3]) is None


# ---------- 5. Across The Planes ----------

def test_5_add_matrices2D():
    add_matrices2D = load_task(__file__, "5-across_the_planes.py", "add_matrices2D")
    assert add_matrices2D([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[6, 8], [10, 12]]
    assert add_matrices2D([[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4]]) is None
    assert add_matrices2D([[1, 2]], [[1, 2], [3, 4]]) is None


# ---------- 6. Howdy Partner ----------

def test_6_cat_arrays():
    cat_arrays = load_task(__file__, "6-howdy_partner.py", "cat_arrays")
    a, b = [1, 2, 3, 4, 5], [6, 7, 8]
    out = cat_arrays(a, b)
    assert out == [1, 2, 3, 4, 5, 6, 7, 8]
    assert a == [1, 2, 3, 4, 5], "don't change the input"


# ---------- 7. Gettin' Cozy ----------

def test_7_cat_matrices2D():
    cat_matrices2D = load_task(__file__, "7-gettin_cozy.py", "cat_matrices2D")
    m1, m2, m3 = [[1, 2], [3, 4]], [[5, 6]], [[7], [8]]
    assert cat_matrices2D(m1, m2) == [[1, 2], [3, 4], [5, 6]]
    assert cat_matrices2D(m1, m3, axis=1) == [[1, 2, 7], [3, 4, 8]]
    assert cat_matrices2D(m1, m2, axis=1) is None
    assert cat_matrices2D(m1, m3, axis=0) is None


def test_7_cat_matrices2D_returns_new_matrix():
    cat_matrices2D = load_task(__file__, "7-gettin_cozy.py", "cat_matrices2D")
    m1, m2 = [[1, 2], [3, 4]], [[5, 6]]
    out = cat_matrices2D(m1, m2)
    out[0][0] = 999
    assert m1 == [[1, 2], [3, 4]], "changing the result must not change the input"


# ---------- 8. Ridin' Bareback ----------

def test_8_mat_mul():
    mat_mul = load_task(__file__, "8-ridin_bareback.py", "mat_mul")
    m1 = [[1, 2], [3, 4], [5, 6]]
    m2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert mat_mul(m1, m2) == [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]
    assert mat_mul(m2, m1) is None


def test_8_mat_mul_random_matches_numpy():
    mat_mul = load_task(__file__, "8-ridin_bareback.py", "mat_mul")
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 5, (4, 3))
    b = rng.integers(-5, 5, (3, 5))
    assert np.array_equal(np.array(mat_mul(a.tolist(), b.tolist())), a @ b)


# ---------- 9. Let The Butcher Slice It ----------

def test_9_butcher():
    mod = load_task(__file__, "9-let_the_butcher_slice_it.py")
    assert np.array_equal(mod.mat1, [[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]])
    assert np.array_equal(mod.mat2, [[3, 4], [9, 10], [15, 16], [21, 22]])
    assert np.array_equal(mod.mat3, [[10, 11, 12], [16, 17, 18], [22, 23, 24]])
    assert_no_loops_or_ifs("9-let_the_butcher_slice_it.py")


# ---------- 10. I'll Use My Scale ----------

def test_10_np_shape():
    np_shape = load_task(__file__, "10-ill_use_my_scale.py", "np_shape")
    assert np_shape(np.array([1, 2, 3, 4, 5, 6])) == (6,)
    assert np_shape(np.array([])) == (0,)
    assert np_shape(np.zeros((2, 2, 5))) == (2, 2, 5)
    assert isinstance(np_shape(np.zeros((2, 3))), tuple)
    assert_no_loops_or_ifs("10-ill_use_my_scale.py")
    assert _count("10-ill_use_my_scale.py", ast.Try) == 0


# ---------- 11. The Western Exchange ----------

def test_11_np_transpose():
    np_transpose = load_task(__file__, "11-the_western_exchange.py", "np_transpose")
    out = np_transpose(np.array([[1, 2, 3], [4, 5, 6]]))
    assert np.array_equal(out, [[1, 4], [2, 5], [3, 6]])
    assert np_transpose(np.zeros((2, 3, 4))).shape == (4, 3, 2)
    assert np.array_equal(np_transpose(np.array([1, 2, 3])), [1, 2, 3])
    assert_no_loops_or_ifs("11-the_western_exchange.py")


# ---------- 12. Bracin' The Elements ----------

def test_12_np_elementwise():
    np_elementwise = load_task(__file__, "12-bracin_the_elements.py", "np_elementwise")
    a = np.array([[11, 22, 33], [44, 55, 66]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    add, sub, mul, div = np_elementwise(a, b)
    assert np.array_equal(add, [[12, 24, 36], [48, 60, 72]])
    assert np.array_equal(sub, [[10, 20, 30], [40, 50, 60]])
    assert np.array_equal(mul, [[11, 44, 99], [176, 275, 396]])
    assert np.allclose(div, 11)
    add, sub, mul, div = np_elementwise(a, 2)
    assert np.array_equal(mul, a * 2) and np.allclose(div, a / 2)
    assert_no_loops_or_ifs("12-bracin_the_elements.py")


# ---------- 13. Cat's Got Your Tongue ----------

def test_13_np_cat():
    np_cat = load_task(__file__, "13-cats_got_your_tongue.py", "np_cat")
    a = np.array([[11, 22, 33], [44, 55, 66]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    c = np.array([[7], [8]])
    assert np.array_equal(np_cat(a, b), [[11, 22, 33], [44, 55, 66], [1, 2, 3], [4, 5, 6]])
    assert np.array_equal(np_cat(a, b, axis=1), [[11, 22, 33, 1, 2, 3], [44, 55, 66, 4, 5, 6]])
    assert np.array_equal(np_cat(a, c, axis=1), [[11, 22, 33, 7], [44, 55, 66, 8]])
    assert_no_loops_or_ifs("13-cats_got_your_tongue.py")


# ---------- 14. Saddle Up ----------

def test_14_np_matmul():
    np_matmul = load_task(__file__, "14-saddle_up.py", "np_matmul")
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert np.array_equal(np_matmul(a, b), [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
    assert np.array_equal(np_matmul(np.array([[2]]), np.array([[3]])), [[6]])
    assert_no_loops_or_ifs("14-saddle_up.py")


# ---------- 100. Slice Like A Ninja (bonus) ----------

def test_100_np_slice():
    np_slice = load_task(__file__, "100-slice_like_a_ninja.py", "np_slice")
    m1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    assert np.array_equal(np_slice(m1, axes={1: (1, 3)}), [[2, 3], [7, 8]])
    assert np.array_equal(m1, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), "don't change the input"
    m2 = np.arange(1, 31).reshape(3, 2, 5)
    out = np_slice(m2, axes={0: (2,), 2: (None, None, -2)})
    assert np.array_equal(out, [[[5, 3, 1], [10, 8, 6]], [[15, 13, 11], [20, 18, 16]]])
    assert np.array_equal(np_slice(m2), m2)


# ---------- 101. The Whole Barn (bonus) ----------

def test_101_add_matrices():
    add_matrices = load_task(__file__, "101-the_whole_barn.py", "add_matrices")
    assert add_matrices([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[6, 8], [10, 12]]
    a = np.arange(24).reshape(2, 3, 2, 2)
    b = np.arange(100, 124).reshape(2, 3, 2, 2)
    assert add_matrices(a.tolist(), b.tolist()) == (a + b).tolist()
    assert add_matrices(a.tolist(), np.zeros((2, 2, 2, 2)).tolist()) is None
    assert add_matrices([[1, 2]], [1, 2]) is None


# ---------- 102. Squashed Like Sardines (bonus) ----------

def test_102_cat_matrices():
    cat_matrices = load_task(__file__, "102-squashed_like_sardines.py", "cat_matrices")
    assert cat_matrices([1, 2, 3], [4, 5, 6]) == [1, 2, 3, 4, 5, 6]
    assert cat_matrices([[1, 2], [3, 4]], [[5, 6]]) == [[1, 2], [3, 4], [5, 6]]
    assert cat_matrices([[1, 2], [3, 4]], [[5], [6]], axis=1) == [[1, 2, 5], [3, 4, 6]]
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(100, 112).reshape(2, 3, 2)
    assert cat_matrices(a.tolist(), b.tolist(), axis=2) == np.concatenate((a, b), axis=2).tolist()
    assert cat_matrices(a.tolist(), b.tolist(), axis=0) is None
    assert cat_matrices([[1, 2]], [1, 2]) is None
