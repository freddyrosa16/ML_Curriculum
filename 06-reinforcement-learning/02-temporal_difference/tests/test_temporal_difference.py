"""Tests for 06-reinforcement-learning/02-temporal_difference."""
import random

import numpy as np
import pytest

from helpers import load_task

gym = pytest.importorskip("gymnasium")

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
GAMMA = 0.99


def _seed(env, seed=0):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)


def _tiny_env():
    """3x3 deterministic lake: S F F / F H H / F F G."""
    env = gym.make("FrozenLake-v1", desc=["SFF", "FHH", "FFG"], is_slippery=False)
    _seed(env)
    return env


# Fixed policy on the tiny lake: down, down, right, right reaches G in 4 steps.
TINY_POLICY = {0: DOWN, 3: DOWN, 6: RIGHT, 7: RIGHT}


def _tiny_policy(s):
    return TINY_POLICY.get(s, RIGHT)


# True values under that policy: reward 1 only on the last step.
TINY_TRUE = {0: GAMMA ** 3, 3: GAMMA ** 2, 6: GAMMA, 7: 1.0}


def _lake8():
    env = gym.make("FrozenLake8x8-v1")
    _seed(env)
    desc = env.unwrapped.desc

    def policy(s):
        # the same random "mostly right/down, avoid holes" policy as the original task
        p = np.random.uniform()
        r, c = s // 8, s % 8
        if p > 0.5:
            if c != 7 and desc[r, c + 1] != b"H":
                return RIGHT
            if r != 7 and desc[r + 1, c] != b"H":
                return DOWN
            if r != 0 and desc[r - 1, c] != b"H":
                return UP
            return LEFT
        if r != 7 and desc[r + 1, c] != b"H":
            return DOWN
        if c != 7 and desc[r, c + 1] != b"H":
            return RIGHT
        if c != 0 and desc[r, c - 1] != b"H":
            return LEFT
        return UP

    V = np.where(desc == b"H", -1, 1).reshape(64).astype("float64")
    return env, policy, V, desc


def _check_tiny_values(V):
    for s, v in TINY_TRUE.items():
        assert V[s] == pytest.approx(v, abs=1e-3), f"V[{s}]"
    # states the policy never visits keep their initial value
    for s in (1, 2, 4, 5, 8):
        assert V[s] == 0


# ---------- 0. Monte Carlo ----------

def test_monte_carlo_converges_on_tiny_lake():
    monte_carlo = load_task(__file__, "0-monte_carlo.py", "monte_carlo")
    V = monte_carlo(_tiny_env(), np.zeros(9), _tiny_policy, episodes=300)
    assert V.shape == (9,)
    _check_tiny_values(V)


def test_monte_carlo_8x8_shape_and_terminal_states():
    monte_carlo = load_task(__file__, "0-monte_carlo.py", "monte_carlo")
    env, policy, V, desc = _lake8()
    V0 = V.copy()
    V = monte_carlo(env, V, policy, episodes=300)
    assert V.shape == (64,)
    holes = (desc == b"H").reshape(64)
    assert np.all(V[holes] == -1)      # never a "current state", never updated
    assert V[63] == 1                  # goal is terminal too
    assert V[0] != V0[0]               # the start state did get updated
    assert np.all(np.abs(V) <= 1 + 1e-9)


# ---------- 1. TD(lambda) ----------

@pytest.mark.parametrize("lambtha", [0.0, 0.5, 0.9])
def test_td_lambtha_converges_on_tiny_lake(lambtha):
    td_lambtha = load_task(__file__, "1-td_lambtha.py", "td_lambtha")
    V = td_lambtha(_tiny_env(), np.zeros(9), _tiny_policy, lambtha, episodes=500)
    assert V.shape == (9,)
    _check_tiny_values(V)


def test_td_lambtha_8x8_shape_and_terminal_states():
    td_lambtha = load_task(__file__, "1-td_lambtha.py", "td_lambtha")
    env, policy, V, desc = _lake8()
    V0 = V.copy()
    V = td_lambtha(env, V, policy, 0.9, episodes=300)
    assert V.shape == (64,)
    holes = (desc == b"H").reshape(64)
    assert np.all(V[holes] == -1)
    assert V[63] == 1
    assert V[0] != V0[0]


# ---------- 2. SARSA(lambda) ----------

def test_sarsa_lambtha_learns_tiny_lake():
    sarsa_lambtha = load_task(__file__, "2-sarsa_lambtha.py", "sarsa_lambtha")
    env = _tiny_env()
    Q = sarsa_lambtha(env, np.zeros((9, 4)), 0.9, episodes=1000)
    assert Q.shape == (9, 4)
    state, _ = env.reset()
    reward = 0
    for _ in range(20):
        state, reward, terminated, truncated, _ = env.step(int(np.argmax(Q[state])))
        if terminated or truncated:
            break
    assert state == 8 and reward == 1, "greedy policy from Q should reach the goal"


def test_sarsa_lambtha_8x8_runs():
    sarsa_lambtha = load_task(__file__, "2-sarsa_lambtha.py", "sarsa_lambtha")
    env = gym.make("FrozenLake8x8-v1")
    _seed(env)
    Q0 = np.random.uniform(size=(64, 4))
    Q = sarsa_lambtha(env, Q0.copy(), 0.9, episodes=200)
    assert Q.shape == (64, 4)
    assert np.all(np.isfinite(Q))
    assert not np.allclose(Q, Q0)
