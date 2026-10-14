"""Tests for 03-error_analysis."""
import hashlib

import numpy as np
import pytest

from helpers import load_task, project_dir

CONF = np.array([[50, 3, 2], [5, 40, 5], [0, 10, 85]], dtype=float)


def counts(c):
    tp = np.diag(c)
    fn = c.sum(axis=1) - tp
    fp = c.sum(axis=0) - tp
    tn = c.sum() - tp - fn - fp
    return tp, fp, fn, tn


@pytest.fixture
def random_conf():
    rng = np.random.default_rng(0)
    return rng.integers(1, 100, size=(6, 6)).astype(float)


def test_create_confusion():
    f = load_task(__file__, "0-create_confusion.py", "create_confusion_matrix")
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    labels, logits = np.eye(4)[y], np.eye(4)[p]
    c = f(labels, logits)
    assert c.shape == (4, 4)
    expected = np.zeros((4, 4))
    np.add.at(expected, (y, p), 1)
    assert np.array_equal(c, expected)
    assert c.sum() == 500


def test_create_confusion_orientation():
    f = load_task(__file__, "0-create_confusion.py", "create_confusion_matrix")
    labels = np.eye(3)[[0, 0, 0]]
    logits = np.eye(3)[[1, 1, 2]]
    c = f(labels, logits)
    assert c[0, 1] == 2 and c[0, 2] == 1, "rows are true labels, columns are predictions"


@pytest.mark.parametrize("conf", ["fixed", "random"])
def test_sensitivity(conf, random_conf):
    c = CONF if conf == "fixed" else random_conf
    f = load_task(__file__, "1-sensitivity.py", "sensitivity")
    tp, fp, fn, tn = counts(c)
    out = f(c)
    assert out.shape == (c.shape[0],)
    assert np.allclose(out, tp / (tp + fn))


@pytest.mark.parametrize("conf", ["fixed", "random"])
def test_precision(conf, random_conf):
    c = CONF if conf == "fixed" else random_conf
    f = load_task(__file__, "2-precision.py", "precision")
    tp, fp, fn, tn = counts(c)
    assert np.allclose(f(c), tp / (tp + fp))


@pytest.mark.parametrize("conf", ["fixed", "random"])
def test_specificity(conf, random_conf):
    c = CONF if conf == "fixed" else random_conf
    f = load_task(__file__, "3-specificity.py", "specificity")
    tp, fp, fn, tn = counts(c)
    assert np.allclose(f(c), tn / (tn + fp))


@pytest.mark.parametrize("conf", ["fixed", "random"])
def test_f1(conf, random_conf):
    c = CONF if conf == "fixed" else random_conf
    f = load_task(__file__, "4-f1_score.py", "f1_score")
    tp, fp, fn, tn = counts(c)
    p, r = tp / (tp + fp), tp / (tp + fn)
    assert np.allclose(f(c), 2 * p * r / (p + r))


# --- short answers: compared by fingerprint so the answers aren't spoiled ---

def _fingerprint(line):
    clean = line.strip().upper().replace(" ", "")
    return hashlib.sha256(("ml-curriculum:" + clean).encode()).hexdigest()[:16]


def _read_answer(name):
    path = project_dir(__file__) / name
    if not path.exists():
        pytest.skip(f"{name} not written yet")
    return [l for l in path.read_text().splitlines() if l.strip()]


ACCEPTED_5 = [
    {"a25a4fe1a871006d"},
    {"4d1530a70a2b43d2"},
    {"43093df1d726ecc2", "ad70ed2c160525b4"},
    {"86b8152bcf9ded04"},
]


def test_error_handling():
    lines = _read_answer("5-error_handling")
    assert len(lines) == 4, "write exactly one line per scenario"
    for i, (line, ok) in enumerate(zip(lines, ACCEPTED_5), start=1):
        letters = line.strip().replace(" ", "").split(",")
        assert letters == sorted(letters), f"scenario {i}: list letters alphabetically"
        assert _fingerprint(line) in ok, f"scenario {i} isn't right yet - rethink it"


def test_compare_and_contrast():
    lines = _read_answer("6-compare_and_contrast")
    assert len(lines) == 1, "write a single letter"
    assert _fingerprint(lines[0]) == "30943acdf612fde4", "not quite - compare the two gaps"
