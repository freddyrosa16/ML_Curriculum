# Hints: Policy Gradients

Open one hint at a time. Try for a while before opening the next.

## 0. Simple policy function
<details><summary>Hint 1</summary>
Two steps: a matrix product to get one score per action, then softmax over those scores.
</details>
<details><summary>Hint 2</summary>
For stability, subtract the largest score from all scores before exponentiating. The probabilities don't change because the same factor cancels from top and bottom.
</details>

## 1. Monte Carlo policy gradient
<details><summary>Hint 1</summary>
First get the probabilities from `policy`. Then sample: NumPy can choose an index from a range with given probabilities. Make sure the probabilities you pass are a flat 1-D array that sums to 1.
</details>
<details><summary>Hint 2</summary>
Derive the gradient of the log of a softmax output on paper. For the score of action $j$, the derivative of $\log \pi(a)$ is 1 minus $\pi(a)$ when $j = a$, and minus $\pi(j)$ otherwise: a one-hot vector minus the probability vector.
</details>
<details><summary>Hint 3</summary>
Each score is the state dotted with one column of the weights, so the chain rule multiplies by the state. Combining a 4-vector (the state) with a 2-vector (one-hot minus probabilities) into a 4x2 matrix is an outer product.
</details>

## 2. Training
<details><summary>Hint 1</summary>
Each episode has two phases: play (collect gradients and rewards, step by step, until terminated or truncated), then learn (one weight update per step you took).
</details>
<details><summary>Hint 2</summary>
The return from step $t$ is the reward at $t$, plus gamma times the reward at $t+1$, plus gamma squared times the reward at $t+2$, and so on to the end. The power of gamma depends on how many steps ahead the reward is, not on the reward's value.
</details>
<details><summary>Hint 3</summary>
Computing returns backward (like Monte Carlo in the TD project) gives you all of them in one pass. The update is gradient *ascent*: add alpha times the step's gradient times its return to the weights.
</details>

## 3. Animate iteration
<details><summary>Hint 1</summary>
Inside the step loop, call the environment's render method, but only when the flag is on and the episode number is a multiple of 1000.
</details>
