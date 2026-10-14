# Hidden Markov Models

**Status:** Original

## Why this matters

Lots of data is a sequence where the thing you care about is hidden: the weather behind someone's activity log, the part of speech behind a word, the gene region behind a DNA letter. Hidden Markov Models are the classic tool for this, and the algorithms you'll write (forward, backward, Viterbi, Baum-Welch) are dynamic programming at its best. They're also a gentle preview of ideas behind RNNs and reinforcement learning.

## Concepts to understand first

- **What is a Markov chain?** A system that hops between states where the next state depends only on the current one. A transition matrix `P` holds the hop probabilities: `P[i, j]` is the chance of going from state `i` to state `j`, so each row sums to 1.
- **How do you get the distribution after t steps?** If `s` is a row vector of current state probabilities, one step later it's `s @ P`. After t steps it's `s` times `P` multiplied by itself t times.
- **What is a regular chain and a steady state?** A chain is regular if some power of `P` has every entry positive (from anywhere you can reach anywhere, in the same number of steps). Then the chain forgets where it started and settles into a unique steady state $\pi$ with $\pi P = \pi$: a left eigenvector of `P` with eigenvalue 1.
- **What is an absorbing chain?** An absorbing state is one you can never leave (`P[i, i] = 1`). A chain is absorbing if it has at least one absorbing state and *every* state can eventually reach one.
- **What makes a Markov model "hidden"?** You never see the states, only observations. Each hidden state has an emission distribution: `Emission[i, k]` is the chance of seeing observation `k` while in state `i`.
- **Why not just try every hidden path?** With N states and T steps there are $N^T$ paths. Forward, backward and Viterbi reuse partial results so the cost is only $N^2 T$.
- **Forward vs. Viterbi?** Both sweep left to right. Forward *adds* over all ways to reach a state (total probability). Viterbi takes the *max* (the single best way) and remembers which choice won, so it can trace back the best path.
- **What does Baum-Welch do?** It's EM for HMMs. E-step: use forward and backward to compute how likely each state and each transition was at every time step. M-step: re-estimate the transition and emission matrices from those expected counts.

## Resources

- [Setosa: Markov chains, explained visually](https://setosa.io/ev/markov-chains/)
- [Jurafsky & Martin, Speech and Language Processing (free draft): appendix on Hidden Markov Models](https://web.stanford.edu/~jurafsky/slp3/)
- [Rabiner (1989): A Tutorial on Hidden Markov Models](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf)
- [Wikipedia: Absorbing Markov chain](https://en.wikipedia.org/wiki/Absorbing_Markov_chain)
- [Wikipedia: Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)
- [Wikipedia: Baum-Welch algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)

## Rules

- NumPy only: `import numpy as np`.
- "Returns None on failure": on inputs of the wrong type or shape (or rows that don't sum to 1), return `None` (or `None, None`).
- Start each file with `#!/usr/bin/env python3`; document every module and function.

**Notation for tasks 3-6** (same in every function):
- `Observation`: `(T,)` array of observation indices.
- `Emission`: `(N, M)`, `Emission[i, k]` = chance of observing `k` in hidden state `i`.
- `Transition`: `(N, N)`, `Transition[i, j]` = chance of moving from hidden state `i` to `j`.
- `Initial`: `(N, 1)`, chance of starting in each hidden state.

Examples for tasks 3-5 use this small HMM: `Emission = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]`, `Transition = [[0.8, 0.2], [0.3, 0.7]]`, `Initial = [[0.6], [0.4]]` and `Observation = [0, 2, 1, 2, 2]`.

## Tasks

### 0. Markov Chain

**File:** `0-markov_chain.py` · **Function:** `markov_chain(P, s, t=1)`

- `P`: `(n, n)` transition matrix. `s`: `(1, n)` starting probabilities. `t`: number of steps.
- **Returns** `(1, n)`: the probability of being in each state after `t` steps. `None` on failure.

```
>>> P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3],
...               [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
>>> markov_chain(P, np.array([[1, 0, 0, 0]]), 300)
array([[0.24949, 0.26335, 0.23394, 0.25321]])
```

### 1. Regular Chains

**File:** `1-regular.py` · **Function:** `regular(P)`

- **Returns** `(1, n)`: the steady state probabilities, or `None` if `P` is invalid **or the chain is not regular**.
- Remember a regular `P` can contain zeros, as long as some power of it is all positive. (For n states, checking powers up to $(n-1)^2 + 1$ is enough.)

```
>>> regular(np.array([[0.6, 0.4], [0.3, 0.7]]))
array([[0.42857, 0.57143]])
>>> regular(np.array([[0., 1.], [0.5, 0.5]]))
array([[0.33333, 0.66667]])
>>> print(regular(np.eye(2)))
None
```

### 2. Absorbing Chains

**File:** `2-absorbing.py` · **Function:** `absorbing(P)`

- **Returns** `True` if the chain is absorbing, `False` if not, `None` on invalid input.
- Absorbing states can be anywhere in the matrix, not only at the top.

```
>>> absorbing(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.5, 0.5], [0, 0.5, 0.5, 0]]))
True
>>> absorbing(np.array([[0.6, 0.4], [0.3, 0.7]]))
False
>>> absorbing(np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))   # states 1 and 2 never get absorbed
False
```

### 3. The Forward Algorithm

**File:** `3-forward.py` · **Function:** `forward(Observation, Emission, Transition, Initial)`

- **Returns** `(P, F)`: `P` is the likelihood of the whole observation sequence; `F` is `(N, T)` where `F[i, t]` is the probability of seeing the first `t + 1` observations *and* being in state `i` at time `t`. `(None, None)` on failure.

```
>>> P, F = forward(Observation, Emission, Transition, Initial)
>>> P
0.003831
>>> F
array([[0.42   , 0.0348 , 0.0096 , 0.00125, 0.00034],
       [0.04   , 0.0672 , 0.0162 , 0.00796, 0.00349]])
```

### 4. The Viterbi Algorithm

**File:** `4-viterbi.py` · **Function:** `viterbi(Observation, Emission, Transition, Initial)`

- **Returns** `(path, P)`: `path` is a list of length `T`, the most likely sequence of hidden states, and `P` is the probability of that path together with the observations. `(None, None)` on failure.

```
>>> viterbi(Observation, Emission, Transition, Initial)
([0, 1, 1, 1, 1], 0.0018670176)
```

### 5. The Backward Algorithm

**File:** `5-backward.py` · **Function:** `backward(Observation, Emission, Transition, Initial)`

- **Returns** `(P, B)`: `P` is the likelihood of the observations (it must equal the forward result); `B` is `(N, T)` where `B[i, t]` is the probability of the *future* observations after time `t`, given state `i` at time `t`. The last column is all ones. `(None, None)` on failure.

```
>>> P, B = backward(Observation, Emission, Transition, Initial)
>>> P
0.003831
>>> B
array([[0.00725, 0.0229 , 0.07   , 0.2    , 1.     ],
       [0.01965, 0.04515, 0.195  , 0.45   , 1.     ]])
```

### 6. The Baum-Welch Algorithm

**File:** `6-baum_welch.py` · **Function:** `baum_welch(Observations, Transition, Emission, Initial, iterations=1000)`

- Note the argument order: `Transition` comes before `Emission` here.
- `Transition` and `Emission` are initial guesses; run up to `iterations` rounds of EM and stop early if both stop changing.
- **Returns** `(Transition, Emission)`, the learned matrices (rows still sum to 1). `(None, None)` on failure.
- You may copy your forward and backward functions into this file.

```
>>> T, E = baum_welch(obs, T_guess, E_guess, Initial, iterations=50)
>>> T.shape, E.shape, np.allclose(T.sum(axis=1), 1)
((2, 2), (2, 3), True)
```
The test generates 120 observations from a known 2-state HMM, starts from a flat transition matrix and a random emission matrix, and checks that the likelihood under your learned matrices is much higher than under the guesses. Don't expect to recover the true matrices exactly: with little data, and with hidden states that can swap labels, the answer is only "a good fit".

### Write-up (optional)

Explain the Markov property, then walk through the forward and Viterbi algorithms on a tiny example of your own (the classic one is guessing the weather from whether someone carried an umbrella).

## Data

No downloads. To make your own test sequences, sample a hidden path with `np.random.choice` (first state from `Initial`, each next state from the current state's row of `Transition`), then sample one observation per hidden state from its row of `Emission`.

## Check your work

From the `ml-curriculum` folder:

```bash
pytest 05-unsupervised/03-hmm
```
