# Hints: Temporal Difference Learning

Open one hint at a time. Try for a while before opening the next.

## 0. Monte Carlo
<details><summary>Hint 1</summary>
Split each episode into two phases: first *collect* (play until the episode ends, remembering each state you were in and the reward you got after it), then *learn* (go over what you collected).
</details>
<details><summary>Hint 2</summary>
Returns are easiest to compute backward. Starting from the last step with a running return of 0, each earlier return is "this step's reward plus gamma times the return that came after it".
</details>
<details><summary>Hint 3</summary>
For first-visit, while walking backward at position t, only update the state if it does not appear anywhere earlier in the episode (positions 0 to t-1). Then move V of that state a fraction alpha toward the return.
</details>

## 1. TD(λ)
<details><summary>Hint 1</summary>
Unlike Monte Carlo, you learn inside the step loop, right after each step. You need one extra array, the same shape as V, holding the eligibility traces. It starts at zero every episode.
</details>
<details><summary>Hint 2</summary>
Each step: compute the TD error (reward plus discounted value of the next state, minus value of the current state; the next state's value counts as zero if the episode terminated), decay all traces by gamma times lambda, bump the current state's trace by one.
</details>
<details><summary>Hint 3</summary>
Then update all states at once: every value moves by alpha times the TD error times its own trace. That's a single whole-array NumPy operation, no loop over states.
</details>

## 2. SARSA(λ)
<details><summary>Hint 1</summary>
It's TD(λ) where states become state-action pairs: the traces now have the shape of Q, and you need to know the *next action* before you can compute the TD error.
</details>
<details><summary>Hint 2</summary>
Pick the first action right after the reset. Inside the loop: take the action, observe the next state, choose the next action epsilon-greedily, compute the error using Q of the next state and next action, update, then make "next" become "current".
</details>
<details><summary>Hint 3</summary>
The trace and update steps mirror task 1 exactly, just indexed by (state, action). Epsilon decays once per episode with the same exponential schedule you used in Q-learning.
</details>
