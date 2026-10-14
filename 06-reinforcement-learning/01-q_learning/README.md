# Q-learning

**Status:** Modernized (gymnasium). Same tasks as the original; it now targets the gymnasium ≥ 1.0 API, where `reset()` returns `(obs, info)` and `step()` returns five values.

## Why this matters
Everything you've built so far learned from labeled examples. Reinforcement learning (RL) is different: an agent learns by *trying things* and getting rewards. Q-learning is the simplest RL algorithm that really works, and every modern method (DQN, the RL behind game-playing agents, RLHF for LLMs) builds on the ideas you'll write here: states, actions, rewards, value estimates, and the explore-vs-exploit tradeoff.

## Concepts to understand first
- **What is an agent, and what is an environment?** The agent is the learner that picks actions. The environment is everything else: it takes an action, moves to a new state, and hands back a reward.
- **What is a Markov Decision Process (MDP)?** A formal description of an RL problem: a set of states, a set of actions, transition probabilities (where you might land after an action) and rewards. "Markov" means the next state depends only on the current state and action, not on the history.
- **What is a policy?** A rule for choosing an action in each state. It can be deterministic ("in state 3, go right") or random ("go right 70% of the time").
- **What is a return, and the discount factor $\gamma$?** The return is the sum of future rewards, with a reward $k$ steps away multiplied by $\gamma^k$. With $\gamma$ close to 1 the agent is patient; close to 0 it's greedy for immediate reward.
- **What is a value function? An action-value (Q) function?** $V(s)$ is the expected return from state $s$. $Q(s, a)$ is the expected return if you take action $a$ in state $s$ and act well afterwards. If you know $Q$, the best action is simply the one with the highest $Q(s, a)$.
- **What is the Bellman equation?** A recursive definition of value: the value of an action now is the immediate reward plus the discounted value of the best action from the next state, $Q(s,a) = r + \gamma \max_{a'} Q(s', a')$.
- **What is Q-learning?** Start with a table of zeros and, after every step, nudge $Q(s,a)$ a little (learning rate $\alpha$) toward $r + \gamma \max_{a'} Q(s',a')$. Repeat over many episodes and the table converges to good values.
- **What is epsilon-greedy?** With probability $\epsilon$ pick a random action (explore), otherwise pick the best known action (exploit). Start with $\epsilon = 1$ (all exploring) and shrink it as the agent learns.
- **What is FrozenLake?** A tiny grid world: start at `S`, reach the goal `G`, walk on frozen tiles `F`, avoid holes `H`. On slippery ice your move only goes where you intended one time in three.

## Resources
- [Sutton & Barto, *Reinforcement Learning: An Introduction* (free), chapters 1, 3 and 6.5 (Q-learning)](http://incompleteideas.net/book/the-book-2nd.html)
- [Hugging Face Deep RL Course, Unit 2: Q-learning](https://huggingface.co/learn/deep-rl-course/unit2/introduction)
- [David Silver's RL lectures (UCL)](https://www.davidsilver.uk/teaching/)
- [Gymnasium: basic usage](https://gymnasium.farama.org/introduction/basic_usage/)
- [Gymnasium: FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- [OpenAI Spinning Up: key concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

## Rules
- Allowed imports: `import numpy as np` and `import gymnasium as gym`.
- Use the gymnasium ≥ 1.0 API: `reset()` returns `(state, info)`, `step()` returns `(state, reward, terminated, truncated, info)`.
- Use `np.random` (not Python's `random`) for every random choice, so seeded tests are reproducible.

## Setup
`uv pip install "gymnasium[classic-control]"` (already in the root `requirements.txt`).

## Tasks

### 0. Load the environment
**File:** `0-load_env.py` · **Signature:** `load_frozen_lake(desc=None, map_name=None, is_slippery=False)`

Return gymnasium's `FrozenLake-v1` environment.
- `desc`: `None` or a list of lists of single-letter strings describing a custom map.
- `map_name`: `None` or the name of a built-in map (`"4x4"` or `"8x8"`).
- If both are `None`, gymnasium generates a random 8x8 map.
- `is_slippery`: whether the ice is slippery.
- Create it with `render_mode="ansi"` so that `env.render()` returns the board as a string (you'll need that in task 4).

```
env = load_frozen_lake()
env.unwrapped.desc.shape            ->  (8, 8)
len(env.unwrapped.P[0][0])          ->  1      (one outcome per action: not slippery)
env = load_frozen_lake(is_slippery=True)
len(env.unwrapped.P[0][0])          ->  3
env = load_frozen_lake(desc=[['S','F','F'], ['F','H','H'], ['F','F','G']])
env.unwrapped.desc                  ->  [[b'S' b'F' b'F']
                                         [b'F' b'H' b'H']
                                         [b'F' b'F' b'G']]
```

### 1. Initialize the Q-table
**File:** `1-q_init.py` · **Signature:** `q_init(env)`

Return a NumPy array of zeros with one row per state and one column per action.
```
q_init(load_frozen_lake()).shape                    ->  (64, 4)
q_init(load_frozen_lake(map_name='4x4')).shape      ->  (16, 4)
```

### 2. Epsilon-greedy
**File:** `2-epsilon_greedy.py` · **Signature:** `epsilon_greedy(Q, state, epsilon)`

Return the index of the next action.
- Draw one number with `np.random.uniform` between 0 and 1. If it is below `epsilon`, explore: pick a random action with `np.random.randint` over all actions. Otherwise exploit: pick the action with the highest Q-value for `state`.

```
Q = <9x4 zeros>, with Q[7] set to [0.5, 0.7, 1, -1]
np.random.seed(0); epsilon_greedy(Q, 7, 0.5)    ->  2
np.random.seed(1); epsilon_greedy(Q, 7, 0.5)    ->  0
```

### 3. Q-learning
**File:** `3-q_learning.py` · **Signature:** `train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)`

Train the Q-table with Q-learning and return `(Q, total_rewards)`.
- `episodes`: how many episodes to play. `max_steps`: cap on steps per episode.
- `alpha`: learning rate. `gamma`: discount factor.
- `epsilon`: starting exploration rate. After each episode, decay it toward `min_epsilon`; `epsilon_decay` controls how fast (exponential decay is the usual choice).
- FrozenLake gives 0 reward for falling in a hole. Change that to **-1** so the agent learns holes are bad.
- Reuse your task 2 function: `epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy`.
- `total_rewards` is a list with the total reward of each episode.

```
np.random.seed(0)
env = load_frozen_lake(desc=[['S','F','F'], ['F','H','H'], ['F','F','G']])
Q, total_rewards = train(env, q_init(env))
len(total_rewards)                  ->  5000
mean of the last 1000 rewards       ->  around 0.9 (exact numbers depend on your implementation)
```

### 4. Play
**File:** `4-play.py` · **Signature:** `play(env, Q, max_steps=100)`

Let the trained agent play one episode, always exploiting the Q-table (no randomness).
- Return `(total_reward, rendered_outputs)`, where `rendered_outputs` is a list of the `env.render()` strings: the starting board, then the board after every step, including the final one.

```
total, outputs = play(env, Q)      (Q from task 3 on the 3x3 map)
total                               ->  1.0
len(outputs)                        ->  5    (start + 4 moves)
print(outputs[-1])                  ->    (Right)
                                         SFF
                                         FHH
                                         FFG     (in a terminal, the agent's square, here G, has a red background)
```

### Write-up (optional)
Write a short blog post or note explaining Q-learning to a friend: what the table means, why epsilon decays, and what changes when you turn `is_slippery=True` on.

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 06-reinforcement-learning/01-q_learning
```
