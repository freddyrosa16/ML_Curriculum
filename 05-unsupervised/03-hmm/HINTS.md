# Hints: Hidden Markov Models

Open one hint at a time. Try for a while before opening the next.

## 0. Markov Chain

<details><summary>Hint 1</summary>

One step of the chain is one matrix product of the state row vector with `P`.
</details>

<details><summary>Hint 2</summary>

Do that `t` times, or raise `P` to the power `t` first; NumPy's linear-algebra module has a function for integer matrix powers.
</details>

## 1. Regular Chains

<details><summary>Hint 1</summary>

Two separate questions: (a) is the chain regular? (b) if so, what's the steady state?
</details>

<details><summary>Hint 2</summary>

For (a), keep multiplying `P` by itself and check whether every entry is positive. Stop after the bound given in the README; if you never got there, it's not regular.
</details>

<details><summary>Hint 3</summary>

For (b), the steady state is a *left* eigenvector of `P` with eigenvalue 1, which is an ordinary (right) eigenvector of the transpose. Pick the eigenvalue closest to 1, take the real part of its eigenvector, and rescale it so its entries sum to 1.
</details>

## 2. Absorbing Chains

<details><summary>Hint 1</summary>

Find the absorbing states first: the ones whose diagonal entry is 1. If there are none, you already know the answer.
</details>

<details><summary>Hint 2</summary>

Think of the chain as a graph: there's an edge from `i` to `j` when `P[i, j] > 0`. The question becomes "can every node reach an absorbing node?"
</details>

<details><summary>Hint 3</summary>

Grow a set of "states known to reach absorption": start with the absorbing states, then repeatedly add any state with an edge into the set. When the set stops growing, check whether it contains everything. (Another route: the fundamental-matrix approach from the Wikipedia page, but only after reordering states so absorbing ones come first.)
</details>

## 3. The Forward Algorithm

<details><summary>Hint 1</summary>

Fill `F` one column (one time step) at a time, left to right. The first column only needs `Initial` and the emission of the first observation.
</details>

<details><summary>Hint 2</summary>

To be in state `j` at time `t`, you were in some state `i` at `t - 1`, moved from `i` to `j`, then emitted the observation. Add over every `i`.
</details>

<details><summary>Hint 3</summary>

"Add over every previous state" for all `j` at once is a single vector-matrix product of the previous column with `Transition`, then an element-wise multiply by the emission column for the current observation. The likelihood is the sum of the last column.
</details>

## 4. The Viterbi Algorithm

<details><summary>Hint 1</summary>

Same table and same sweep as forward, but replace "add over previous states" with "take the best previous state".
</details>

<details><summary>Hint 2</summary>

Alongside the probability table, keep a second `(N, T)` table that remembers *which* previous state gave the max (a backpointer).
</details>

<details><summary>Hint 3</summary>

At the end, start from the best state in the last column and follow the backpointers from right to left, building the path in reverse. Return plain Python ints in the list.
</details>

## 5. The Backward Algorithm

<details><summary>Hint 1</summary>

Mirror image of forward: fill `B` from right to left, starting with a column of ones.
</details>

<details><summary>Hint 2</summary>

From state `i` at time `t`, you move to some `j`, emit observation `t + 1` from `j`, then generate the rest of the future from `j`. Add over every `j`.
</details>

<details><summary>Hint 3</summary>

The final likelihood combines the first column of `B` with the starting probabilities and the first emission. Check it equals the forward `P`.
</details>

## 6. The Baum-Welch Algorithm

<details><summary>Hint 1</summary>

Each iteration: run forward and backward with the current matrices, compute two kinds of expected counts, then turn the counts into new matrices.
</details>

<details><summary>Hint 2</summary>

The two quantities (see Rabiner's tutorial, or Jurafsky & Martin): gamma, the probability of being in state `i` at time `t` given all observations (forward times backward, normalized); and xi, the probability of going from `i` at `t` to `j` at `t + 1` (forward at `t`, transition, emission at `t + 1`, backward at `t + 1`, normalized). xi has shape `(N, N, T - 1)`.
</details>

<details><summary>Hint 3</summary>

New transition: expected `i → j` transitions divided by expected time spent in `i` (only over the first `T - 1` steps). New emission: expected time in `i` while observing `k`, divided by expected time in `i`. A boolean mask "observation equals k" selects the right time steps.
</details>
