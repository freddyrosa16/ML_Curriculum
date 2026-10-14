"""Tests for 06-reinforcement-learning/03-deep_q_learning."""
import copy

import numpy as np
import pytest

from helpers import load_task

torch = pytest.importorskip("torch")
nn = torch.nn


class FixedQ(nn.Module):
    """A 'network' that returns preset Q-values, for checking targets by hand."""

    def __init__(self, table):
        super().__init__()
        self.table = torch.tensor(table, dtype=torch.float32)

    def forward(self, x):
        # state i (stored in column 0) -> row i of the table
        return self.table[x[:, 0].long()]


# ---------- 0. ReplayBuffer ----------

def _filled_buffer(capacity=100, n=10, seed=0):
    ReplayBuffer = load_task(__file__, "0-replay_buffer.py", "ReplayBuffer")
    buf = ReplayBuffer(capacity, seed=seed)
    for i in range(n):
        buf.push(np.full(4, i, dtype=np.float32), i % 2, float(i),
                 np.full(4, i + 1, dtype=np.float32), i == n - 1)
    return buf


def test_buffer_len_grows_then_caps():
    buf = _filled_buffer(capacity=5, n=3)
    assert len(buf) == 3
    buf = _filled_buffer(capacity=5, n=12)
    assert len(buf) == 5


def test_buffer_overwrites_oldest():
    buf = _filled_buffer(capacity=5, n=12)
    s, a, r, s2, d = buf.sample(5)
    # only the 5 most recent transitions (rewards 7..11) remain
    assert sorted(r.tolist()) == [7.0, 8.0, 9.0, 10.0, 11.0]


def test_buffer_sample_shapes_and_dtypes():
    buf = _filled_buffer(n=20)
    s, a, r, s2, d = buf.sample(8)
    assert s.shape == (8, 4) and s2.shape == (8, 4)
    assert a.shape == (8,) and r.shape == (8,) and d.shape == (8,)
    assert np.issubdtype(a.dtype, np.integer)
    assert s.dtype == np.float32 and r.dtype == np.float32 and d.dtype == np.float32


def test_buffer_sample_keeps_transitions_together():
    buf = _filled_buffer(n=20)
    s, a, r, s2, d = buf.sample(10)
    # every transition was built so that s = i, a = i % 2, r = i, s2 = i + 1
    assert np.allclose(s[:, 0], r)
    assert np.allclose(s2[:, 0], r + 1)
    assert np.array_equal(a, r.astype(int) % 2)
    assert np.array_equal(d, (r == 19).astype(np.float32))


def test_buffer_sample_without_replacement():
    buf = _filled_buffer(n=10)
    _, _, r, _, _ = buf.sample(10)
    assert sorted(r.tolist()) == [float(i) for i in range(10)]


def test_buffer_seed_is_reproducible():
    r1 = _filled_buffer(n=50, seed=3).sample(10)[2]
    r2 = _filled_buffer(n=50, seed=3).sample(10)[2]
    assert np.array_equal(r1, r2)


# ---------- 1. QNetwork ----------

def test_q_network_output_shape():
    QNetwork = load_task(__file__, "1-q_network.py", "QNetwork")
    torch.manual_seed(0)
    net = QNetwork(4, 2)
    assert isinstance(net, nn.Module)
    out = net(torch.randn(7, 4))
    assert out.shape == (7, 2)


def test_q_network_hidden_size_and_trainable():
    QNetwork = load_task(__file__, "1-q_network.py", "QNetwork")
    small = QNetwork(4, 2, hidden=16)
    big = QNetwork(4, 2, hidden=64)
    n_small = sum(p.numel() for p in small.parameters())
    n_big = sum(p.numel() for p in big.parameters())
    assert n_big > n_small
    out = small(torch.randn(3, 4)).sum()
    out.backward()
    assert all(p.grad is not None for p in small.parameters())


# ---------- 2. Epsilon schedule ----------

def test_linear_epsilon_endpoints():
    linear_epsilon = load_task(__file__, "2-epsilon.py", "linear_epsilon")
    assert linear_epsilon(0) == pytest.approx(1.0)
    assert linear_epsilon(10_000) == pytest.approx(0.05)
    assert linear_epsilon(50_000) == pytest.approx(0.05)


def test_linear_epsilon_midpoint_and_custom():
    linear_epsilon = load_task(__file__, "2-epsilon.py", "linear_epsilon")
    assert linear_epsilon(5_000) == pytest.approx(0.525)
    assert linear_epsilon(25, start=0.5, end=0.0, decay_steps=100) == pytest.approx(0.375)


def test_linear_epsilon_monotonic():
    linear_epsilon = load_task(__file__, "2-epsilon.py", "linear_epsilon")
    values = [linear_epsilon(s, decay_steps=1000) for s in range(0, 2000, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))


# ---------- 3. select_action ----------

def test_select_action_greedy_when_epsilon_zero():
    select_action = load_task(__file__, "3-select_action.py", "select_action")
    net = FixedQ([[0.1, 0.9, 0.3], [2.0, 1.0, 0.0]])
    assert select_action(net, np.array([0.0, 0.0]), 0.0, 3) == 1
    assert select_action(net, np.array([1.0, 0.0]), 0.0, 3) == 0
    assert isinstance(select_action(net, np.array([0.0, 0.0]), 0.0, 3), int)


def test_select_action_random_when_epsilon_one():
    select_action = load_task(__file__, "3-select_action.py", "select_action")
    net = FixedQ([[0.1, 0.9, 0.3]])
    np.random.seed(0)
    torch.manual_seed(0)
    seen = {select_action(net, np.array([0.0, 0.0]), 1.0, 3) for _ in range(200)}
    assert seen == {0, 1, 2}


# ---------- 4. TD targets ----------

def test_td_targets_by_hand():
    td_targets = load_task(__file__, "4-td_target.py", "td_targets")
    target = FixedQ([[1.0, 3.0], [-2.0, -1.0], [5.0, 4.0]])
    next_states = torch.tensor([[0.0], [1.0], [2.0]])
    rewards = torch.tensor([1.0, 0.5, 2.0])
    dones = torch.tensor([0.0, 0.0, 1.0])
    y = td_targets(rewards, next_states, dones, target, 0.9)
    # 1 + 0.9*3 = 3.7 ; 0.5 + 0.9*(-1) = -0.4 ; done -> just 2.0
    assert torch.allclose(y, torch.tensor([3.7, -0.4, 2.0]))


def test_td_targets_have_no_gradient():
    td_targets = load_task(__file__, "4-td_target.py", "td_targets")
    torch.manual_seed(0)
    target = nn.Linear(4, 2)
    y = td_targets(torch.ones(5), torch.randn(5, 4), torch.zeros(5), target, 0.99)
    assert y.shape == (5,)
    assert not y.requires_grad


# ---------- 5. DQN loss ----------

def test_dqn_loss_matches_huber_on_chosen_actions():
    dqn_loss = load_task(__file__, "5-dqn_loss.py", "dqn_loss")
    q = FixedQ([[1.0, 2.0], [0.0, 4.0]])
    target = FixedQ([[0.0, 1.0], [3.0, 2.0]])
    states = np.array([[0.0], [1.0]], dtype=np.float32)
    actions = np.array([1, 0])
    rewards = np.array([1.0, 0.0], dtype=np.float32)
    next_states = np.array([[1.0], [0.0]], dtype=np.float32)
    dones = np.array([0.0, 1.0], dtype=np.float32)
    loss = dqn_loss(q, target, (states, actions, rewards, next_states, dones), 0.5)
    # Q(s,a) = [2, 0]; targets = [1 + 0.5*3, 0] = [2.5, 0]
    expected = torch.nn.functional.smooth_l1_loss(torch.tensor([2.0, 0.0]),
                                                  torch.tensor([2.5, 0.0]))
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(expected.item())


def test_dqn_loss_backprops_only_into_online_net():
    dqn_loss = load_task(__file__, "5-dqn_loss.py", "dqn_loss")
    torch.manual_seed(0)
    q_net = nn.Linear(4, 2)
    target_net = copy.deepcopy(q_net)
    batch = (np.random.randn(6, 4).astype(np.float32), np.array([0, 1, 0, 1, 1, 0]),
             np.ones(6, dtype=np.float32), np.random.randn(6, 4).astype(np.float32),
             np.zeros(6, dtype=np.float32))
    dqn_loss(q_net, target_net, batch, 0.99).backward()
    assert q_net.weight.grad is not None
    assert target_net.weight.grad is None


# ---------- 6. Target network sync ----------

def test_hard_update_copies_weights():
    hard_update = load_task(__file__, "6-sync.py", "hard_update")
    torch.manual_seed(0)
    online, target = nn.Linear(4, 2), nn.Linear(4, 2)
    hard_update(target, online)
    for p, t in zip(online.parameters(), target.parameters()):
        assert torch.equal(p, t)
    # a copy, not shared storage
    with torch.no_grad():
        online.weight.add_(1.0)
    assert not torch.equal(online.weight, target.weight)


def test_soft_update_blends_weights():
    soft_update = load_task(__file__, "6-sync.py", "soft_update")
    online, target = nn.Linear(2, 1), nn.Linear(2, 1)
    with torch.no_grad():
        online.weight.fill_(1.0)
        online.bias.fill_(1.0)
        target.weight.fill_(0.0)
        target.bias.fill_(0.0)
    soft_update(target, online, 0.1)
    assert torch.allclose(target.weight, torch.full((1, 2), 0.1))
    soft_update(target, online, 0.1)
    assert torch.allclose(target.bias, torch.tensor([0.19]))
    assert torch.allclose(online.weight, torch.ones(1, 2))  # online untouched


# ---------- 7/8. Training on CartPole (slow) ----------

@pytest.mark.slow
def test_train_runs_short():
    pytest.importorskip("gymnasium")
    train_dqn = load_task(__file__, "7-train.py", "train_dqn")
    q_net, rewards = train_dqn(total_steps=3_000, seed=0)
    assert isinstance(q_net, nn.Module)
    assert len(rewards) > 0


@pytest.mark.slow
def test_dqn_solves_cartpole():
    pytest.importorskip("gymnasium")
    train_dqn = load_task(__file__, "7-train.py", "train_dqn")
    evaluate = load_task(__file__, "8-evaluate.py", "evaluate")
    q_net, _ = train_dqn(seed=0)
    assert evaluate(q_net, episodes=100) >= 475


# ---------- 9. Atari (bonus) ----------

def test_atari_q_network_shape():
    AtariQNetwork = load_task(__file__, "9-atari.py", "AtariQNetwork")
    net = AtariQNetwork(4)
    frames = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    assert net(frames).shape == (2, 4)


@pytest.mark.slow
def test_make_atari_env_observation():
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")  # AtariPreprocessing needs opencv-python
    make_atari_env = load_task(__file__, "9-atari.py", "make_atari_env")
    env = make_atari_env("ALE/Breakout-v5", seed=0)
    obs, _ = env.reset(seed=0)
    assert np.asarray(obs).shape == (4, 84, 84)
    assert env.action_space.n == 4
    env.close()
