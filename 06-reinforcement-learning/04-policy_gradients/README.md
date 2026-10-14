# Policy Gradients

**Status:** Modernized (gymnasium). Same NumPy-only tasks as the original, on gymnasium's `CartPole-v1` with the ≥ 1.0 API.

## Why this matters
Everything so far learned *values* and derived a policy from them ("pick the action with the highest Q"). Policy gradient methods skip the middleman: they learn the policy directly, as a probability distribution over actions, and push up the probability of actions that led to high returns. REINFORCE, the algorithm you'll write here, is the ancestor of PPO, the workhorse behind robotics controllers and the RLHF step used to train chat LLMs. Doing it in plain NumPy forces you to derive the gradient yourself.

## Concepts to understand first
- **What is a stochastic policy?** A function $\pi(a \mid s)$ giving the probability of each action in state $s$. Here it's linear-softmax: multiply the state (1x4) by a weight matrix (4x2) to get one score per action, then apply softmax to turn scores into probabilities.
- **What is softmax, and why subtract the max?** $\text{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$. Subtracting $\max(z)$ from every score first doesn't change the result but prevents $e^{z}$ from overflowing.
- **What is the policy gradient theorem, intuitively?** To increase expected return, nudge the weights in the direction that makes the actions you took more likely, scaled by how good the outcome was: $\nabla J \approx \sum_t G_t \nabla \log \pi(a_t \mid s_t)$.
- **What is $\nabla \log \pi$ for a linear-softmax policy?** For weights $W$, state $s$ and chosen action $a$: $\nabla_W \log \pi(a \mid s) = s^\top (\mathbb{1}_a - \pi(\cdot \mid s))$, an outer product of the state with "one-hot of the chosen action minus the probabilities". Deriving this on paper is the heart of task 1.
- **What is REINFORCE?** Monte Carlo policy gradient: play a whole episode by sampling actions from $\pi$, compute the discounted return $G_t$ from every step, then update $W \leftarrow W + \alpha \, G_t \, \nabla \log \pi(a_t \mid s_t)$ for each step. It's gradient *ascent*, since we maximize return.
- **Why is it noisy?** Returns vary a lot between episodes, so the gradient estimate has high variance. That's why the learning rate is tiny and training takes thousands of episodes. (Baselines and actor-critic methods, the next step after this project, reduce that variance.)

## Resources
- [Sutton & Barto (free), chapter 13: Policy gradient methods](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Spinning Up: Intro to policy optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Andrej Karpathy: Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/)
- [Hugging Face Deep RL Course, Unit 4: Policy gradient with PyTorch](https://huggingface.co/learn/deep-rl-course/unit4/introduction)
- [David Silver, Lecture 7: Policy gradient methods](https://www.davidsilver.uk/teaching/)
- [Gymnasium: CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## Rules
- `import numpy as np` only in `policy_gradient.py`. `train.py` may also import nothing else but your own `policy_gradient` module (the environment is passed in).
- Use `np.random` for every random choice (seeded tests).
- Gymnasium ≥ 1.0 API: `reset()` → `(state, info)`, `step()` → `(state, reward, terminated, truncated, info)`; an episode ends when terminated **or** truncated.

## Setup used by the examples
`env = gym.make('CartPole-v1')`, seeded with `env.reset(seed=0)`, `np.random.seed(0)` and `random.seed(0)`. CartPole's state is 4 numbers and it has 2 actions (push left, push right).

## Tasks

### 0. Simple policy function
**File:** `policy_gradient.py` · **Signature:** `policy(matrix, weight)`

Return the action probabilities: the softmax of `matrix @ weight`. With a state of shape `(1, 4)` and weights `(4, 2)`, the result has shape `(1, 2)` and sums to 1. Make it numerically stable.
```
weight = [[4.17022005e-01, 7.20324493e-01],
          [1.14374817e-04, 3.02332573e-01],
          [1.46755891e-01, 9.23385948e-02],
          [1.86260211e-01, 3.45560727e-01]]
state  = [[-0.04428214,  0.01636746,  0.01196594, -0.03095031]]
policy(state, weight)    ->  [[0.50351642 0.49648358]]
```

### 1. Monte Carlo policy gradient
**File:** `policy_gradient.py` · **Signature:** `policy_gradient(state, weight)`

Using `policy`, sample an action from the policy's probabilities (with `np.random`) and compute the gradient of $\log \pi(\text{action} \mid \text{state})$ with respect to `weight`. Return `(action, gradient)`; the gradient has the same shape as `weight`.
- `state` comes straight from the environment, shape `(4,)`.
```
np.random.seed(0); weight = np.random.rand(4, 2); state, _ = env.reset(seed=0)
action, grad = policy_gradient(state, weight)
action        ->  0 or 1
grad.shape    ->  (4, 2)
```

### 2. Training
**File:** `train.py` · **Signature:** `train(env, nb_episodes, alpha=0.000045, gamma=0.98)`

Implement REINFORCE and return the list of scores (score = sum of rewards in an episode).
- Start from random weights of shape `(observations, actions)` drawn with `np.random.rand`.
- For each episode: play until it ends using `policy_gradient`, remembering each step's gradient and reward; then, for every step, compute the discounted return from that step to the end and update the weights by `alpha * gradient * return`.
- After each episode print `Episode: {episode} Score: {score}`.
- Import your helper with `policy_gradient = __import__('policy_gradient').policy_gradient`.
```
scores = train(env, 10000)
(prints)      Episode: 0 Score: 22.0
              Episode: 1 Score: 62.0
              ...                      (your exact numbers will differ)
len(scores)   ->  10000
average score per block of 1000 episodes: roughly 50 at the start, climbing to 300-400+ by the end
```

### 3. Animate iteration
**File:** `train.py` · **Signature:** `train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False)`

Add the optional `show_result` parameter. When it's `True`, render the environment during every 1000th episode (episode 0, 1000, 2000, ...) so you can watch the agent improve. To see a window, the caller creates the env with `gym.make('CartPole-v1', render_mode="human")`.
```
train(env, 10000, 0.000045, 0.98, True)    ->  a CartPole window appears for episodes 0, 1000, 2000, ...
```

### Write-up (optional)
Derive $\nabla_W \log \pi(a \mid s)$ for the linear-softmax policy on paper and explain each term. Then compare: how does REINFORCE's learning curve differ from your DQN's on the same environment?

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 06-reinforcement-learning/04-policy_gradients
```
The full 10,000-episode training test is marked slow; skip it with `-m "not slow"`.
