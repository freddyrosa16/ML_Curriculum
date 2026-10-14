"""Tests for 06-reinforcement-learning/04-policy_gradients."""
import inspect
import random

import numpy as np
import pytest

from helpers import load_task

WEIGHT = np.array([
    [4.17022005e-01, 7.20324493e-01],
    [1.14374817e-04, 3.02332573e-01],
    [1.46755891e-01, 9.23385948e-02],
    [1.86260211e-01, 3.45560727e-01],
])
STATE = np.array([[-0.04428214, 0.01636746, 0.01196594, -0.03095031]])


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ---------- 0. policy ----------

def test_policy_reference_values():
    policy = load_task(__file__, "policy_gradient.py", "policy")
    res = policy(STATE, WEIGHT)
    assert res.shape == (1, 2)
    assert np.allclose(res, [[0.50351642, 0.49648358]])


def test_policy_is_softmax_of_state_times_weight():
    policy = load_task(__file__, "policy_gradient.py", "policy")
    rng = np.random.default_rng(0)
    for _ in range(5):
        s, w = rng.normal(size=(1, 4)), rng.normal(size=(4, 3))
        res = policy(s, w)
        assert np.allclose(res.sum(), 1)
        assert np.all(res > 0)
        assert np.allclose(res.ravel(), _softmax((s @ w).ravel()))


def test_policy_is_numerically_stable():
    policy = load_task(__file__, "policy_gradient.py", "policy")
    res = policy(np.array([[1000.0, 0, 0, 0]]), WEIGHT)
    assert np.all(np.isfinite(res))
    assert np.allclose(res.sum(), 1)


# ---------- 1. policy_gradient ----------

def test_policy_gradient_shapes_and_action():
    policy_gradient = load_task(__file__, "policy_gradient.py", "policy_gradient")
    np.random.seed(0)
    action, grad = policy_gradient(STATE[0], WEIGHT)
    assert int(action) in (0, 1)
    assert grad.shape == (4, 2)


def test_policy_gradient_is_grad_of_log_prob():
    policy_gradient = load_task(__file__, "policy_gradient.py", "policy_gradient")
    np.random.seed(1)
    rng = np.random.default_rng(1)
    for _ in range(10):
        state, weight = rng.normal(size=4), rng.normal(size=(4, 2))
        action, grad = policy_gradient(state, weight)
        p = _softmax(state @ weight)
        onehot = np.eye(2)[int(action)]
        # d log pi(a|s) / dW = s (outer) (onehot(a) - pi(.|s))
        assert np.allclose(grad, np.outer(state, onehot - p))


def test_policy_gradient_samples_from_the_policy():
    policy_gradient = load_task(__file__, "policy_gradient.py", "policy_gradient")
    np.random.seed(2)
    state = np.array([1.0, 0, 0, 0])
    weight = np.array([[np.log(3), 0.0], [0, 0], [0, 0], [0, 0]])  # pi = [0.75, 0.25]
    actions = [int(policy_gradient(state, weight)[0]) for _ in range(4000)]
    assert np.mean(np.array(actions) == 0) == pytest.approx(0.75, abs=0.03)


# ---------- 2. train ----------

def _cartpole():
    gym = pytest.importorskip("gymnasium")
    env = gym.make("CartPole-v1")
    env.reset(seed=0)
    np.random.seed(0)
    random.seed(0)
    return env


def test_train_returns_one_score_per_episode_and_prints(capsys):
    train = load_task(__file__, "train.py", "train")
    scores = train(_cartpole(), 30)
    assert len(scores) == 30
    assert all(1 <= s <= 500 for s in scores)
    out = capsys.readouterr().out
    assert "Episode: 0 Score: " in out
    assert "Episode: 29 Score: " in out


def test_train_score_is_sum_of_rewards(capsys):
    train = load_task(__file__, "train.py", "train")
    scores = train(_cartpole(), 5)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("Episode:")]
    printed = [float(l.split("Score:")[1]) for l in lines]
    assert np.allclose(printed, scores)


@pytest.mark.slow
def test_train_improves_over_time():
    train = load_task(__file__, "train.py", "train")
    scores = np.array(train(_cartpole(), 10000))
    assert scores[-500:].mean() > 2 * scores[:500].mean()
    assert scores[-1000:].mean() > 150


# ---------- 3. show_result ----------

def test_train_has_show_result_flag():
    train = load_task(__file__, "train.py", "train")
    params = inspect.signature(train).parameters
    assert "show_result" in params
    assert params["show_result"].default is False
