# Temporal Difference Learning

**Status:** Modernized (gymnasium). Same tasks as the original, written for the gymnasium ≥ 1.0 API (`reset()` → `(obs, info)`, `step()` → 5 values).

## Why this matters
Q-learning updated its table after every single step. But how far ahead should an update look? Monte Carlo waits until the episode is over and uses the real return; temporal difference (TD) learning updates immediately using its own guess about the future. TD(λ) and SARSA(λ) blend the two with *eligibility traces*, one of the most elegant ideas in RL, and the reason RL agents can learn from long, delayed rewards.

## Concepts to understand first
- **What is prediction vs. control?** Prediction: given a fixed policy, estimate how good each state is ($V$). Control: find a good policy (usually by learning $Q$). Tasks 0 and 1 are prediction, task 2 is control.
- **What is Monte Carlo (MC) estimation?** Play a full episode, then compute the actual discounted return $G_t$ from each visited state, and nudge $V(s_t)$ toward it: $V(s) \leftarrow V(s) + \alpha (G_t - V(s))$. Unbiased but noisy, and it must wait for the episode to end.
- **What is "first-visit" MC?** If a state shows up several times in one episode, only its first appearance is used for the update in that episode.
- **What is a TD error?** $\delta = r + \gamma V(s') - V(s)$: the difference between what you predicted and a slightly better prediction made one step later. TD(0) moves $V(s)$ by $\alpha \delta$ right away. If $s'$ is terminal there is no future, so $V(s')$ counts as 0.
- **What is bootstrapping?** Updating a guess using another guess ($V(s')$). It's what makes TD fast and online, and what MC avoids.
- **What is an eligibility trace?** A short memory, one number per state (or state-action pair), of how recently and how often it was visited. Every step, all traces decay by $\gamma\lambda$ and the current state's trace goes up by 1. A TD error then updates *every* state in proportion to its trace, so credit flows back to the states that led here.
- **What does $\lambda$ do?** $\lambda = 0$ gives plain one-step TD. $\lambda = 1$ behaves like Monte Carlo. Values in between trade off bias and variance.
- **What is SARSA?** On-policy TD control. It learns $Q(s,a)$ from the tuple (State, Action, Reward, next State, next Action): the target is $r + \gamma Q(s', a')$, where $a'$ is the action you will actually take next (chosen epsilon-greedily), not the best one. SARSA(λ) adds eligibility traces over state-action pairs.

## Resources
- [Sutton & Barto (free): chapter 5 (Monte Carlo), chapter 6 (TD, SARSA), chapter 12 (eligibility traces)](http://incompleteideas.net/book/the-book-2nd.html)
- [David Silver, Lecture 4: Model-free prediction and Lecture 5: Model-free control](https://www.davidsilver.uk/teaching/)
- [Hugging Face Deep RL Course, Unit 2 (MC vs TD)](https://huggingface.co/learn/deep-rl-course/unit2/mc-vs-td)
- [Gymnasium: FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- [Gymnasium: basic usage](https://gymnasium.farama.org/introduction/basic_usage/)

## Rules
- Allowed imports: `import numpy as np` (gymnasium is only used by the caller, you receive `env`).
- Use the gymnasium ≥ 1.0 API.
- Use `np.random` for randomness so seeded tests are reproducible.
- Update the arrays you're given (`V`, `Q`) and return them.

## Setup used by the examples
The examples (and tests) use `gym.make('FrozenLake8x8-v1')`, which is the slippery 8x8 lake, seeded with `env.reset(seed=0)` and `np.random.seed(0)`. The policy for tasks 0 and 1 is a random "walker": with probability 1/2 it prefers moving right, otherwise down, and it avoids stepping onto a neighbouring hole when it can (falling back to up/left). `V` starts at -1 for holes and 1 for every other tile. You'll write this policy yourself when you try the examples; the tests have their own copy.

## Tasks

### 0. Monte Carlo
**File:** `0-monte_carlo.py` · **Signature:** `monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`

Estimate the value function of `policy` with first-visit Monte Carlo and return `V`.
- `V`: array of shape `(s,)` with the current value estimates.
- `policy`: a function that takes a state and returns an action.
- Play each episode to the end (or `max_steps`), then walk backward through it computing returns.

```
V = monte_carlo(env, V, policy)
V.shape            ->  (64,)
V for holes        ->  still -1 (an episode ends when you fall in, so holes are never updated)
V[0]               ->  a value between -1 and 1, no longer the initial 1
```
On the 3x3 lake `SFF / FHH / FFG` (not slippery) with the fixed policy down, down, right, right, and `V` starting at zeros, `V[0]` converges to $0.99^3 \approx 0.9703$.

### 1. TD(λ)
**File:** `1-td_lambtha.py` · **Signature:** `td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`

Estimate the value function of `policy` with backward-view TD(λ) using eligibility traces, and return `V`.
- `lambtha`: the trace decay factor λ (spelled that way because `lambda` is a Python keyword).
- Reset the traces at the start of every episode.

```
V = td_lambtha(env, V, policy, 0.9)
V.shape            ->  (64,)
```
On the 3x3 example above it converges to the same values as Monte Carlo, for any λ.

### 2. SARSA(λ)
**File:** `2-sarsa_lambtha.py` · **Signature:** `sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)`

Learn a Q-table with SARSA(λ) and return `Q`.
- `Q`: array of shape `(s, a)`.
- Choose actions epsilon-greedily (you can reuse your idea from the Q-learning project, but put the function in this file).
- Decay epsilon toward `min_epsilon` after each episode, like in Q-learning.

```
np.random.seed(0); Q = np.random.uniform(size=(64, 4))
Q = sarsa_lambtha(env, Q, 0.9)
Q.shape            ->  (64, 4)
```
On the 3x3 lake starting from zeros, the greedy policy from the returned `Q` walks straight to the goal.

### Write-up (optional)
Explain in your own words, with a small drawing, how an eligibility trace spreads one reward back over the path that led to it. Compare λ = 0, 0.5 and 0.9 on the 8x8 lake.

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 06-reinforcement-learning/02-temporal_difference
```
