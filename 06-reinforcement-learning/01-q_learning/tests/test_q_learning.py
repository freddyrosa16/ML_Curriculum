"""Tests for 06-reinforcement-learning/01-q_learning."""
import numpy as np
import pytest

from helpers import load_task

pytest.importorskip("gymnasium")

DESC_3X3 = [["S", "F", "F"], ["F", "H", "H"], ["F", "F", "G"]]


def _env(**kwargs):
    load_frozen_lake = load_task(__file__, "0-load_env.py", "load_frozen_lake")
    return load_frozen_lake(**kwargs)


# ---------- 0. load_frozen_lake ----------

def test_load_default_is_random_8x8_not_slippery():
    env = _env()
    assert env.unwrapped.desc.shape == (8, 8)
    # not slippery: each action has exactly one outcome
    assert len(env.unwrapped.P[0][0]) == 1


def test_load_slippery_has_three_outcomes():
    env = _env(is_slippery=True)
    assert len(env.unwrapped.P[0][0]) == 3


def test_load_custom_desc():
    env = _env(desc=DESC_3X3)
    expected = np.array(DESC_3X3, dtype="c")
    assert np.array_equal(env.unwrapped.desc, expected)


def test_load_map_name_4x4():
    env = _env(map_name="4x4")
    assert env.unwrapped.desc.shape == (4, 4)
    assert env.unwrapped.desc[0, 0] == b"S"
    assert env.unwrapped.desc[3, 3] == b"G"


def test_load_render_mode_is_ansi():
    env = _env(desc=DESC_3X3)
    env.reset(seed=0)
    out = env.render()
    assert isinstance(out, str)
    assert "S" in out and "G" in out


# ---------- 1. q_init ----------

@pytest.mark.parametrize("kwargs, shape", [
    ({}, (64, 4)),
    ({"is_slippery": True}, (64, 4)),
    ({"desc": DESC_3X3}, (9, 4)),
    ({"map_name": "4x4"}, (16, 4)),
])
def test_q_init_shape_and_zeros(kwargs, shape):
    q_init = load_task(__file__, "1-q_init.py", "q_init")
    Q = q_init(_env(**kwargs))
    assert isinstance(Q, np.ndarray)
    assert Q.shape == shape
    assert np.all(Q == 0)


# ---------- 2. epsilon_greedy ----------

def _q7():
    Q = np.zeros((9, 4))
    Q[7] = np.array([0.5, 0.7, 1, -1])
    return Q


def test_epsilon_greedy_matches_reference_seeds():
    epsilon_greedy = load_task(__file__, "2-epsilon_greedy.py", "epsilon_greedy")
    Q = _q7()
    np.random.seed(0)
    assert epsilon_greedy(Q, 7, 0.5) == 2   # exploit
    np.random.seed(1)
    assert epsilon_greedy(Q, 7, 0.5) == 0   # explore


def test_epsilon_zero_always_exploits():
    epsilon_greedy = load_task(__file__, "2-epsilon_greedy.py", "epsilon_greedy")
    Q = _q7()
    np.random.seed(3)
    assert all(epsilon_greedy(Q, 7, 0.0) == 2 for _ in range(50))


def test_epsilon_one_explores_every_action():
    epsilon_greedy = load_task(__file__, "2-epsilon_greedy.py", "epsilon_greedy")
    Q = _q7()
    np.random.seed(4)
    actions = {int(epsilon_greedy(Q, 7, 1.0)) for _ in range(200)}
    assert actions == {0, 1, 2, 3}


# ---------- 3. train ----------

@pytest.fixture(scope="module")
def trained():
    q_init = load_task(__file__, "1-q_init.py", "q_init")
    train = load_task(__file__, "3-q_learning.py", "train")
    np.random.seed(0)
    env = _env(desc=DESC_3X3)
    env.reset(seed=0)
    Q = q_init(env)
    Q, total_rewards = train(env, Q)
    return env, Q, total_rewards


def test_train_returns_q_and_one_reward_per_episode(trained):
    _, Q, total_rewards = trained
    assert Q.shape == (9, 4)
    assert len(total_rewards) == 5000  # default episodes


def test_train_holes_give_minus_one(trained):
    _, _, total_rewards = trained
    assert -1 in [round(float(r)) for r in total_rewards]


def test_train_learns_to_reach_goal(trained):
    _, _, total_rewards = trained
    assert np.mean(total_rewards[-1000:]) > 0.7


def test_train_greedy_policy_reaches_goal(trained):
    env, Q, _ = trained
    state, _ = env.reset(seed=0)
    for _ in range(20):
        state, reward, terminated, truncated, _ = env.step(int(np.argmax(Q[state])))
        if terminated or truncated:
            break
    assert state == 8 and reward == 1


# ---------- 4. play ----------

def test_play_reaches_goal_and_renders_every_step(trained):
    play = load_task(__file__, "4-play.py", "play")
    env, Q, _ = trained
    total, outputs = play(env, Q)
    assert total == 1
    # optimal path in the 3x3 map is 4 moves: initial board + 4 boards
    assert len(outputs) == 5
    assert all(isinstance(o, str) for o in outputs)


def test_play_respects_max_steps():
    play = load_task(__file__, "4-play.py", "play")
    env = _env(desc=DESC_3X3)
    Q = np.zeros((9, 4))
    Q[:, 0] = 1  # always LEFT: never reaches the goal
    total, outputs = play(env, Q, max_steps=7)
    assert total == 0
    assert len(outputs) == 8
