"""Tests for 05-unsupervised/03-hmm."""
import itertools

import numpy as np
import pytest

from helpers import load_task

P4 = np.array([[0.25, 0.2, 0.25, 0.3],
               [0.2, 0.3, 0.2, 0.3],
               [0.25, 0.25, 0.4, 0.1],
               [0.3, 0.3, 0.1, 0.3]])

# A small HMM used by the forward / backward / viterbi tests
E_SMALL = np.array([[0.7, 0.2, 0.1],
                    [0.1, 0.3, 0.6]])
T_SMALL = np.array([[0.8, 0.2],
                    [0.3, 0.7]])
I_SMALL = np.array([[0.6], [0.4]])
OBS_SMALL = np.array([0, 2, 1, 2, 2])


def joint_prob(path, obs, E, T, I):
    """Probability of a hidden path AND the observations together."""
    p = I[path[0], 0] * E[path[0], obs[0]]
    for t in range(1, len(obs)):
        p *= T[path[t - 1], path[t]] * E[path[t], obs[t]]
    return p


def all_paths(n_states, length):
    return itertools.product(range(n_states), repeat=length)


def sample_hmm(seed, T, E, I, length):
    rng = np.random.default_rng(seed)
    h = [rng.choice(len(I), p=I.ravel())]
    for _ in range(length - 1):
        h.append(rng.choice(len(I), p=T[h[-1]]))
    return np.array([rng.choice(E.shape[1], p=E[s]) for s in h])


# ---------- Task 0: Markov chain ----------

def test_markov_chain_matches_matrix_power():
    markov_chain = load_task(__file__, "0-markov_chain.py", "markov_chain")
    s = np.array([[1.0, 0, 0, 0]])
    for t in (1, 2, 5):
        out = markov_chain(P4, s, t)
        assert out.shape == (1, 4)
        assert np.allclose(out, s @ np.linalg.matrix_power(P4, t))


def test_markov_chain_long_run_is_stationary():
    markov_chain = load_task(__file__, "0-markov_chain.py", "markov_chain")
    out = markov_chain(P4, np.array([[1.0, 0, 0, 0]]), 300)
    assert np.allclose(out @ P4, out)
    assert np.isclose(out.sum(), 1)


def test_markov_chain_bad_inputs():
    markov_chain = load_task(__file__, "0-markov_chain.py", "markov_chain")
    s = np.array([[1.0, 0, 0, 0]])
    assert markov_chain(P4[:3], s) is None              # not square
    assert markov_chain(P4, np.array([1.0, 0, 0, 0])) is None  # s is 1-D
    assert markov_chain(P4, s, 0) is None
    assert markov_chain("P", s) is None


# ---------- Task 1: regular chains ----------

def test_regular_two_state():
    regular = load_task(__file__, "1-regular.py", "regular")
    out = regular(np.array([[0.6, 0.4], [0.3, 0.7]]))
    assert out.shape == (1, 2)
    assert np.allclose(out, [[3 / 7, 4 / 7]])


def test_regular_four_state_is_stationary():
    regular = load_task(__file__, "1-regular.py", "regular")
    out = regular(P4)
    assert out.shape == (1, 4)
    assert np.isclose(out.sum(), 1)
    assert np.allclose(out @ P4, out)


def test_regular_with_zero_entries():
    """Regular even though P itself has zeros: a higher power is all positive."""
    regular = load_task(__file__, "1-regular.py", "regular")
    assert np.allclose(regular(np.array([[0.0, 1.0], [0.5, 0.5]])), [[1 / 3, 2 / 3]])
    P = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.5]])
    assert np.allclose(regular(P), [[0.25, 0.25, 0.5]])


def test_regular_rejects_non_regular():
    regular = load_task(__file__, "1-regular.py", "regular")
    assert regular(np.eye(2)) is None
    blocks = np.array([[0.8, 0.2, 0, 0, 0],
                       [0.25, 0.75, 0, 0, 0],
                       [0, 0, 0.5, 0.2, 0.3],
                       [0, 0, 0.3, 0.5, .2],
                       [0, 0, 0.2, 0.3, 0.5]])
    assert regular(blocks) is None
    assert regular(np.array([[0.0, 1.0], [1.0, 0.0]])) is None  # periodic: never all positive


def test_regular_bad_inputs():
    regular = load_task(__file__, "1-regular.py", "regular")
    assert regular(np.array([[1.0, 0.25], [0.25, 0.75]])) is None  # row doesn't sum to 1
    assert regular(np.ones((2, 3)) / 3) is None
    assert regular("P") is None


# ---------- Task 2: absorbing chains ----------

@pytest.mark.parametrize("P, expected", [
    (np.eye(2), True),
    (np.array([[0.6, 0.4], [0.3, 0.7]]), False),
    (P4, False),
    (np.array([[1, 0, 0, 0, 0],
               [0.25, 0.75, 0, 0, 0],
               [0, 0, 0.5, 0.2, 0.3],
               [0, 0, 0.3, 0.5, .2],
               [0, 0, 0.2, 0.3, 0.5]]), False),
    (np.array([[1, 0, 0, 0, 0],
               [0.25, 0.75, 0, 0, 0],
               [0, 0.1, 0.5, 0.2, 0.2],
               [0, 0.1, 0.2, 0.5, .2],
               [0, 0.1, 0.2, 0.2, 0.5]]), True),
    (np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0.5, 0.5],
               [0, 0.5, 0.5, 0]]), True),
    # the absorbing state doesn't have to come first
    (np.array([[0.5, 0.5, 0], [0, 1, 0], [0, 0.5, 0.5]]), True),
    # has an absorbing state, but states 1 and 2 bounce between each other forever
    (np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), False),
])
def test_absorbing(P, expected):
    absorbing = load_task(__file__, "2-absorbing.py", "absorbing")
    assert absorbing(np.array(P, dtype=float)) == expected


def test_absorbing_bad_inputs():
    absorbing = load_task(__file__, "2-absorbing.py", "absorbing")
    assert absorbing(np.ones((2, 3)) / 3) is None
    assert absorbing("P") is None


# ---------- Task 3: forward ----------

def test_forward_matches_brute_force():
    forward = load_task(__file__, "3-forward.py", "forward")
    P, F = forward(OBS_SMALL, E_SMALL, T_SMALL, I_SMALL)
    brute = sum(joint_prob(p, OBS_SMALL, E_SMALL, T_SMALL, I_SMALL) for p in all_paths(2, 5))
    assert F.shape == (2, 5)
    assert np.isclose(P, brute)
    assert np.allclose(F[:, 0], I_SMALL[:, 0] * E_SMALL[:, OBS_SMALL[0]])
    assert np.isclose(F[:, -1].sum(), P)


def test_forward_bad_inputs():
    forward = load_task(__file__, "3-forward.py", "forward")
    assert forward(OBS_SMALL.reshape(1, -1), E_SMALL, T_SMALL, I_SMALL) == (None, None)
    assert forward(OBS_SMALL, E_SMALL, T_SMALL, I_SMALL.ravel()) == (None, None)
    assert forward(OBS_SMALL, E_SMALL, np.eye(3), I_SMALL) == (None, None)


# ---------- Task 4: Viterbi ----------

def test_viterbi_matches_brute_force():
    viterbi = load_task(__file__, "4-viterbi.py", "viterbi")
    path, P = viterbi(OBS_SMALL, E_SMALL, T_SMALL, I_SMALL)
    best = max(all_paths(2, 5), key=lambda p: joint_prob(p, OBS_SMALL, E_SMALL, T_SMALL, I_SMALL))
    assert isinstance(path, list) and len(path) == 5
    assert [int(s) for s in path] == list(best)
    assert np.isclose(P, joint_prob(best, OBS_SMALL, E_SMALL, T_SMALL, I_SMALL))


def test_viterbi_three_states():
    viterbi = load_task(__file__, "4-viterbi.py", "viterbi")
    E = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    T = np.array([[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]])
    I = np.array([[0.5], [0.3], [0.2]])
    obs = np.array([2, 0, 1, 1, 2, 0])
    path, P = viterbi(obs, E, T, I)
    best = max(all_paths(3, 6), key=lambda p: joint_prob(p, obs, E, T, I))
    assert [int(s) for s in path] == list(best)
    assert np.isclose(P, joint_prob(best, obs, E, T, I))


def test_viterbi_bad_inputs():
    viterbi = load_task(__file__, "4-viterbi.py", "viterbi")
    assert viterbi(OBS_SMALL, E_SMALL, T_SMALL, I_SMALL.ravel()) == (None, None)


# ---------- Task 5: backward ----------

def test_backward_matches_brute_force():
    backward = load_task(__file__, "5-backward.py", "backward")
    P, B = backward(OBS_SMALL, E_SMALL, T_SMALL, I_SMALL)
    brute = sum(joint_prob(p, OBS_SMALL, E_SMALL, T_SMALL, I_SMALL) for p in all_paths(2, 5))
    assert B.shape == (2, 5)
    assert np.allclose(B[:, -1], 1)
    assert np.isclose(P, brute)


def test_forward_backward_agree_at_every_step():
    forward = load_task(__file__, "3-forward.py", "forward")
    backward = load_task(__file__, "5-backward.py", "backward")
    obs = sample_hmm(0, T_SMALL, E_SMALL, I_SMALL, 30)
    Pf, F = forward(obs, E_SMALL, T_SMALL, I_SMALL)
    Pb, B = backward(obs, E_SMALL, T_SMALL, I_SMALL)
    assert np.isclose(Pf, Pb)
    # sum_i F[i, t] * B[i, t] is the likelihood for every t
    assert np.allclose((F * B).sum(axis=0), Pf)


def test_backward_bad_inputs():
    backward = load_task(__file__, "5-backward.py", "backward")
    assert backward(OBS_SMALL.reshape(1, -1), E_SMALL, T_SMALL, I_SMALL) == (None, None)


# ---------- Task 6: Baum-Welch ----------

def test_baum_welch_improves_likelihood():
    baum_welch = load_task(__file__, "6-baum_welch.py", "baum_welch")
    forward = load_task(__file__, "3-forward.py", "forward")
    E_true = np.array([[0.90, 0.10, 0.00], [0.40, 0.50, 0.10]])
    T_true = np.array([[0.60, 0.4], [0.30, 0.70]])
    I = np.array([[0.5], [0.5]])
    obs = sample_hmm(1, T_true, E_true, I, 120)
    rng = np.random.default_rng(2)
    T0 = np.array([[0.5, 0.5], [0.5, 0.5]])
    E0 = rng.random((2, 3))
    E0 /= E0.sum(axis=1, keepdims=True)
    T, E = baum_welch(obs, T0.copy(), E0.copy(), I, iterations=50)
    assert T.shape == (2, 2) and E.shape == (2, 3)
    assert np.allclose(T.sum(axis=1), 1) and np.allclose(E.sum(axis=1), 1)
    assert np.all(T >= 0) and np.all(E >= 0)
    before, _ = forward(obs, E0, T0, I)
    after, _ = forward(obs, E, T, I)
    assert after > before * 10  # a real improvement, not just noise


def test_baum_welch_bad_inputs():
    baum_welch = load_task(__file__, "6-baum_welch.py", "baum_welch")
    assert baum_welch(OBS_SMALL.reshape(1, -1), T_SMALL, E_SMALL, I_SMALL) == (None, None)
    assert baum_welch(OBS_SMALL, T_SMALL, E_SMALL, I_SMALL.ravel()) == (None, None)
