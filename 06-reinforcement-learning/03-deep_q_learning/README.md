# Deep Q-learning

**Status:** Modernized. The original asked you to plug Breakout into `keras-rl2`'s ready-made `DQNAgent`. That library was archived in 2021 and doesn't run on current Keras or gymnasium, so here you build DQN yourself in PyTorch, piece by piece, solve CartPole with it, and then (bonus) point it at Breakout.

## Why this matters
A Q-table needs one row per state, which is hopeless when the state is a screen of pixels or a vector of real numbers. DQN replaces the table with a neural network that *predicts* Q-values, and adds two tricks (experience replay and a target network) that make that stable. This is the algorithm that learned to play Atari games from pixels in 2013-2015 and kicked off deep RL. Writing each piece yourself is the best way to understand why each one is there.

## Concepts to understand first
- **What is function approximation?** Instead of storing $Q(s,a)$ in a table, train a network $Q_\theta(s)$ that outputs one Q-value per action. Similar states get similar values automatically, so the agent generalizes to states it has never seen.
- **What is the DQN loss?** For a transition $(s, a, r, s', \text{done})$, the target is $y = r + \gamma (1 - \text{done}) \max_{a'} Q_{\text{target}}(s', a')$. Train $Q_\theta(s, a)$ to match $y$, usually with the Huber (smooth L1) loss, which is like squared error near zero but less sensitive to big outliers.
- **Why "done" matters.** If the episode *terminated* (the pole fell), there is no future reward, so the target is just $r$. If it was only *truncated* (time limit hit), the future still exists, so store `terminated`, not `terminated or truncated`, as "done".
- **What is experience replay?** Store transitions in a big buffer and train on random mini-batches from it. Consecutive steps are highly correlated; random sampling breaks that correlation and lets each experience be reused many times.
- **What is a target network?** A frozen copy of the Q-network used only to compute targets. If the same network produced both prediction and target, every update would move the target too ("chasing your own tail"). Copy the weights over every N steps (hard update) or blend a little every step (soft/Polyak update).
- **Why an epsilon schedule?** Early on the network knows nothing, so explore a lot; later, mostly exploit. A linear decay from 1.0 to a small value over the first chunk of training is simple and works well.
- **What is CartPole?** Balance a pole on a cart by pushing left or right. Observation: 4 numbers (cart position and velocity, pole angle and angular velocity). +1 reward per step, max 500 steps. "Solved" here means an average score of at least 475 over 100 evaluation episodes.

## Resources
- [Mnih et al., *Human-level control through deep reinforcement learning* (Nature, 2015)](https://www.nature.com/articles/nature14236)
- [Mnih et al., *Playing Atari with Deep Reinforcement Learning* (2013, arXiv)](https://arxiv.org/abs/1312.5602)
- [Hugging Face Deep RL Course, Unit 3: Deep Q-learning with Atari](https://huggingface.co/learn/deep-rl-course/unit3/introduction)
- [PyTorch tutorial: Reinforcement Learning (DQN)](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) (read it *after* you've tried, it contains a full solution)
- [CleanRL DQN docs](https://docs.cleanrl.dev/rl-algorithms/dqn/) (clear explanations of each hyperparameter)
- [Gymnasium: CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) · [Gymnasium Atari wrappers](https://gymnasium.farama.org/api/wrappers/misc_wrappers/)
- [Sutton & Barto (free), chapter 9-11 on function approximation](http://incompleteideas.net/book/the-book-2nd.html)

## Rules
- Allowed imports: `numpy`, `torch` (and `torch.nn`, `torch.nn.functional`), `gymnasium`, `copy`, `random`, and `ale_py` for the bonus.
- No RL libraries (no Stable-Baselines3, no keras-rl) in your solution files.
- Device-agnostic: accept a `device` argument and move tensors/models there. A small MLP on CartPole is usually fastest on CPU; MPS/CUDA help for the Atari bonus.
- Later files reuse earlier ones with `__import__('0-replay_buffer').ReplayBuffer` and so on.
- Use `np.random` for exploration randomness (the seeded tests rely on it).

## Setup
`uv pip install torch "gymnasium[classic-control]"`. For the bonus: `uv pip install ale-py opencv-python` (gymnasium's Atari preprocessing wrapper needs OpenCV). ale-py ships the Atari ROMs, so no separate download is needed.

## Tasks

### 0. Replay buffer
**File:** `0-replay_buffer.py` · **Class:** `ReplayBuffer(capacity, seed=None)`

A fixed-size memory of transitions.
- `push(state, action, reward, next_state, done)`: store one transition. Once `capacity` is reached, the newest transition overwrites the oldest.
- `sample(batch_size)`: return a tuple of 5 NumPy arrays `(states, actions, rewards, next_states, dones)`, drawn uniformly at random **without replacement**. Shapes: `(B, obs_dim)`, `(B,)`, `(B,)`, `(B, obs_dim)`, `(B,)`. Dtypes: float32, integer, float32, float32, float32 (done as 0.0/1.0). Keep each transition's five parts together.
- `__len__()`: how many transitions are stored.
- `seed`: seeds the buffer's own random generator, so sampling is reproducible.

```
buf = ReplayBuffer(5, seed=0)
(push 12 transitions with rewards 0, 1, ..., 11)
len(buf)                     ->  5
buf.sample(5)[2] (sorted)    ->  [7., 8., 9., 10., 11.]
```

### 1. Q-network
**File:** `1-q_network.py` · **Class:** `QNetwork(obs_dim, n_actions, hidden=128)` (an `nn.Module`)

An MLP mapping a batch of observations `(B, obs_dim)` to Q-values `(B, n_actions)`. Two hidden layers of size `hidden` with ReLU are plenty for CartPole. No activation on the output (Q-values can be any real number).
```
QNetwork(4, 2)(torch.randn(7, 4)).shape    ->  torch.Size([7, 2])
```

### 2. Epsilon schedule
**File:** `2-epsilon.py` · **Signature:** `linear_epsilon(step, start=1.0, end=0.05, decay_steps=10000)`

Return epsilon for a given step: linear from `start` at step 0 to `end` at `decay_steps`, then constant at `end`.
```
linear_epsilon(0)          ->  1.0
linear_epsilon(5000)       ->  0.525
linear_epsilon(20000)      ->  0.05
```

### 3. Choosing an action
**File:** `3-select_action.py` · **Signature:** `select_action(q_net, state, epsilon, n_actions, device="cpu")`

Epsilon-greedy with a network: with probability `epsilon` return a random action in `[0, n_actions)`; otherwise return the action with the largest predicted Q-value for the single `state` (a NumPy array of shape `(obs_dim,)`). Return a plain Python `int`. No gradients needed here.
```
select_action(q_net, obs, 0.0, 2)    ->  0 or 1, whichever Q(obs) is larger
```

### 4. TD targets
**File:** `4-td_target.py` · **Signature:** `td_targets(rewards, next_states, dones, target_net, gamma)`

All inputs are tensors (`rewards`, `dones`: shape `(B,)`; `next_states`: `(B, obs_dim)`). Return the tensor of targets $y = r + \gamma (1 - \text{done}) \max_{a'} Q_{\text{target}}(s', a')$, shape `(B,)`. It must not carry gradients (targets are treated as constants).
```
target Q-values for 3 next states: [1, 3], [-2, -1], [5, 4]
rewards [1, 0.5, 2], dones [0, 0, 1], gamma 0.9
td_targets(...)    ->  tensor([ 3.7, -0.4,  2.0])
```

### 5. DQN loss
**File:** `5-dqn_loss.py` · **Signature:** `dqn_loss(q_net, target_net, batch, gamma, device="cpu")`

`batch` is the tuple returned by `ReplayBuffer.sample`. Return the mean Huber (smooth L1) loss between $Q_\theta(s, a)$ (the online network's value for the action actually taken) and the TD targets from task 4. Gradients should flow into `q_net` only.
```
Q(s,a) = [2, 0], targets = [2.5, 0]
dqn_loss(...)      ->  tensor(0.0625)
```

### 6. Target network sync
**File:** `6-sync.py` · **Signatures:** `hard_update(target_net, q_net)` and `soft_update(target_net, q_net, tau)`

- `hard_update`: make every parameter of `target_net` equal to `q_net`'s (a copy, not shared memory).
- `soft_update`: move each target parameter a fraction `tau` toward the online one: $\theta_{\text{target}} \leftarrow \tau\,\theta + (1-\tau)\,\theta_{\text{target}}$. Leave `q_net` unchanged.
```
online weights all 1, target weights all 0
soft_update(target, online, 0.1)  ->  target weights all 0.1
soft_update(target, online, 0.1)  ->  target weights all 0.19
```

### 7. Training loop
**File:** `7-train.py` · **Signature:** `train_dqn(env_id="CartPole-v1", total_steps=100_000, seed=0, ..., device=None, save_path=None)`

Put it all together and return `(q_net, episode_rewards)`, where `episode_rewards` is the list of total rewards of each finished training episode.
- You choose the other hyperparameters (buffer size, batch size, learning rate, how often to train and to sync the target, epsilon schedule, warm-up steps before training starts) and add them as keyword arguments with defaults.
- Seed everything (`torch`, `np.random`, the env's reset and action space) from `seed`.
- With **your default arguments**, the network you return must solve CartPole (see task 8). Training scores are noisy; returning the best network seen during training (by a moving average of recent episode scores) rather than the last one is allowed and smart.
- If `save_path` is given, save the returned network's `state_dict` there with `torch.save`.

On a recent MacBook CPU a good implementation solves it in roughly 1-3 minutes and 50k-100k steps.
```
q_net, rewards = train_dqn(seed=0)
len(rewards)                 ->  several hundred episodes
average of the last 100      ->  well above 100, climbing toward 500
```

### 8. Evaluate
**File:** `8-evaluate.py` · **Signature:** `evaluate(q_net, env_id="CartPole-v1", episodes=100, seed=1000, device="cpu", render=False)`

Play `episodes` episodes with the greedy policy (no exploration), resetting episode `i` with seed `seed + i`, and return the mean total reward as a float. With `render=True`, create the env with `render_mode="human"` so you can watch it.
```
evaluate(q_net)              ->  500.0  (anything >= 475 counts as solved)
```

### 9. Breakout (bonus)
**File:** `9-atari.py` · **Signatures:** `make_atari_env(env_id="ALE/Breakout-v5", seed=0, render_mode=None)` and class `AtariQNetwork(n_actions)`

- `make_atari_env`: return a Breakout env whose observations are a stack of the last 4 grayscale 84x84 frames, shape `(4, 84, 84)`, dtype uint8. Use gymnasium's own wrappers: `AtariPreprocessing` (frame skip 4, resize, grayscale) and `FrameStackObservation`. Two gotchas: ale-py's environments must be registered with gymnasium first (`gym.register_envs(ale_py)` after importing `ale_py`), and `AtariPreprocessing` wants the base env created with `frameskip=1` because it does the skipping itself.
- `AtariQNetwork`: the CNN from the 2015 Nature paper: three conv layers (32 filters 8x8 stride 4, 64 filters 4x4 stride 2, 64 filters 3x3 stride 1, each followed by ReLU), then a 512-unit dense layer and the output layer. It takes the raw uint8 frames `(B, 4, 84, 84)`; scale them to [0, 1] inside `forward`.
- Then train it with your task 7 loop (adapted: bigger buffer, `AtariQNetwork`, uint8 storage to save memory, Adam with lr ~1e-4, target sync every ~1000 steps, epsilon decaying over ~1M steps). Real progress takes millions of steps, i.e. hours on a laptop (much faster on MPS/CUDA). Save the weights and use `evaluate`-style greedy play with `render_mode="human"` to watch your agent, which is what the original `play.py` did.
```
env = make_atari_env()
env.reset(seed=0)[0].shape                               ->  (4, 84, 84)
AtariQNetwork(4)(<uint8 tensor (2, 4, 84, 84)>).shape    ->  torch.Size([2, 4])
```

### Bonus: compare with a library
Once yours works, install [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (`uv pip install stable-baselines3`) and train its `DQN` on CartPole-v1. Compare the learning curve, wall-clock time and final score with yours, and read through its hyperparameter defaults. Which ones did you choose differently, and does it matter? (Write this up; there are no tests.)

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 06-reinforcement-learning/03-deep_q_learning -m "not slow"   # the building blocks, seconds
pytest 06-reinforcement-learning/03-deep_q_learning                 # also trains and checks CartPole is solved (minutes)
```
