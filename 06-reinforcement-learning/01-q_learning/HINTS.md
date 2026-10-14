# Hints: Q-learning

Open one hint at a time. Try for a while before opening the next.

## 0. Load the environment
<details><summary>Hint 1</summary>
This is a one-liner wrapped in a function. Gymnasium creates every environment through the same factory function, which takes the environment's id plus keyword arguments that get passed on to the environment.
</details>
<details><summary>Hint 2</summary>
The FrozenLake page in the gymnasium docs lists the keyword arguments it accepts. Three of them have exactly the same names as your function's parameters, and the render mode is a fourth.
</details>
<details><summary>Hint 3</summary>
`env` is wrapped in helper layers (time limit, checks). To look at the map or the transition table, reach through to the underlying environment with `.unwrapped`.
</details>

## 1. Initialize the Q-table
<details><summary>Hint 1</summary>
One row per state, one column per action. You only need two numbers from the environment.
</details>
<details><summary>Hint 2</summary>
Every gymnasium environment describes its states and actions with "space" objects: one for observations, one for actions. For FrozenLake both are discrete spaces, and a discrete space knows how many values it has.
</details>

## 2. Epsilon-greedy
<details><summary>Hint 1</summary>
It's one coin flip followed by one of two choices. The order of random calls matters for the seeded test: draw the uniform number first, and only call the integer sampler when you explore.
</details>
<details><summary>Hint 2</summary>
"Best known action" means the column with the largest value in the row of the Q-table that belongs to the current state. NumPy has a function that returns the *index* of the largest element.
</details>
<details><summary>Hint 3</summary>
The number of actions is the Q-table's second dimension, so you don't need the environment here at all.
</details>

## 3. Q-learning
<details><summary>Hint 1</summary>
Two loops: an outer loop over episodes, an inner loop over steps. Each episode starts with a reset; each step picks an action, takes it, updates one cell of the Q-table, then moves to the new state. Stop the inner loop when the episode ends (terminated *or* truncated).
</details>
<details><summary>Hint 2</summary>
The update moves $Q(s,a)$ a fraction $\alpha$ of the way toward the target $r + \gamma \max_{a'} Q(s', a')$. If the new state is terminal there's no future, so the target is just $r$. A hole is a terminal state where the reward came back as 0: that's where you swap in -1.
</details>
<details><summary>Hint 3</summary>
For the epsilon schedule, a common choice is exponential decay: epsilon starts at its initial value and approaches `min_epsilon` as $e^{-\text{decay} \cdot \text{episode}}$ shrinks. Update it once per episode, after the inner loop. Remember the starting epsilon before you begin overwriting it.
</details>

## 4. Play
<details><summary>Hint 1</summary>
Same step loop as training, minus learning and minus randomness: always take the best action from the Q-table.
</details>
<details><summary>Hint 2</summary>
Render once right after the reset (the starting board) and once after every step, collecting the strings in a list. That's why the list is one longer than the number of moves.
</details>
