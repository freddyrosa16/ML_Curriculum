# Hints: Deep Q-learning

Open one hint at a time. Try for a while before opening the next.

## 0. Replay buffer
<details><summary>Hint 1</summary>
Think "circular buffer": a plain list that grows until it's full, plus a position index that wraps around to 0 when it reaches the capacity, so new items overwrite the oldest slot.
</details>
<details><summary>Hint 2</summary>
Store each transition as one tuple, so its five parts can never get out of sync. For sampling, pick random *indices* first (NumPy's random generator objects can choose indices without replacement), then gather the tuples at those indices.
</details>
<details><summary>Hint 3</summary>
To turn a list of tuples into five arrays, "unzip" them (Python's built-in zip with the star operator does this), then stack each group into one array and cast to the dtype the task asks for. Seed a dedicated generator in the constructor with the `seed` argument.
</details>

## 1. Q-network
<details><summary>Hint 1</summary>
This is the same kind of model you built in the PyTorch basics project: a subclass of `nn.Module` with layers created in the constructor and used in `forward`.
</details>
<details><summary>Hint 2</summary>
A container module that runs layers in order keeps it very short: linear, ReLU, linear, ReLU, linear. The last layer has `n_actions` outputs and no activation.
</details>

## 2. Epsilon schedule
<details><summary>Hint 1</summary>
Draw it: a straight line from (0, start) to (decay_steps, end), then flat.
</details>
<details><summary>Hint 2</summary>
"Fraction of the way through the decay" is step divided by decay_steps, capped at 1. Epsilon is start plus that fraction times the (negative) distance from start to end.
</details>

## 3. Choosing an action
<details><summary>Hint 1</summary>
Same coin flip as the Q-learning project. The only new part is getting a Q-value out of a network for a single state.
</details>
<details><summary>Hint 2</summary>
Networks expect a batch. Convert the state to a float32 tensor on the right device and add a batch dimension of size 1 in front. The output then has shape (1, n_actions).
</details>
<details><summary>Hint 3</summary>
Wrap the forward pass in PyTorch's no-gradient context, take the index of the max along the action dimension, and convert the resulting one-element tensor to a Python int.
</details>

## 4. TD targets
<details><summary>Hint 1</summary>
Write the formula from the README, then translate it term by term. Every operation is element-wise over the batch except the max over actions.
</details>
<details><summary>Hint 2</summary>
Tensor max along a dimension returns both values and indices; you want the values, along the action dimension. Multiplying by (1 - done) zeroes out the future for terminal transitions.
</details>
<details><summary>Hint 3</summary>
Compute everything inside the no-gradient context, so the result is a constant as far as autograd is concerned.
</details>

## 5. DQN loss
<details><summary>Hint 1</summary>
Two halves: the prediction (online network's Q-value for the action that was actually taken) and the target (task 4). Then one loss function call.
</details>
<details><summary>Hint 2</summary>
Convert the five NumPy arrays to tensors first (actions as integers, the rest as floats). The online network gives (B, n_actions); to pick one column per row using the actions, look at PyTorch's `gather` (the index needs an extra dimension, and you squeeze it away afterwards).
</details>
<details><summary>Hint 3</summary>
PyTorch's functional module has the smooth L1 loss built in; its default reduction is the mean you need.
</details>

## 6. Target network sync
<details><summary>Hint 1</summary>
Every module can export all its weights as a dictionary and load weights from such a dictionary. That's the whole hard update.
</details>
<details><summary>Hint 2</summary>
For the soft update, walk over the two networks' parameters side by side (they're in the same order) and update the target's values in place. Do it without tracking gradients.
</details>

## 7. Training loop
<details><summary>Hint 1</summary>
One loop over environment steps (not episodes). Each step: pick epsilon from the schedule, choose an action, step the env, push the transition, and if the episode ended, record its score and reset. Separately, once you have enough transitions (a warm-up), sample a batch and take one optimizer step; and every N steps, sync the target network.
</details>
<details><summary>Hint 2</summary>
Common culprits when it doesn't learn: storing `terminated or truncated` as done (store only `terminated`), forgetting `zero_grad`, syncing the target too often or never, a learning rate that's too large, or no warm-up. Gradient-norm clipping is a cheap stabilizer.
</details>
<details><summary>Hint 3</summary>
Reasonable starting points for CartPole: buffer of tens of thousands, batch around 64-128, Adam with a learning rate around 1e-3 to 5e-4, gamma 0.99, target sync every few hundred steps, epsilon decaying over the first ~20% of training to about 0.02. DQN on CartPole often learns, then collapses, then recovers: keep a copy of the best weights by a moving average of episode scores.
</details>

## 8. Evaluate
<details><summary>Hint 1</summary>
It's the play function from the Q-learning project with the network instead of a table, and an outer loop over episodes.
</details>
<details><summary>Hint 2</summary>
Put the network in evaluation mode and use the no-gradient context. Keep going until the episode is terminated or truncated (CartPole truncates at 500).
</details>

## 9. Breakout (bonus)
<details><summary>Hint 1</summary>
Wrappers nest like layers of an onion: make the base env, wrap it in the preprocessing wrapper, then wrap that in the frame stacker. Check the wrappers' docs for their parameter names.
</details>
<details><summary>Hint 2</summary>
For the CNN, work out the spatial size after each conv layer with (input - kernel) / stride + 1: 84 → 20 → 9 → 7. So the flattened size feeding the dense layer is 64 × 7 × 7.
</details>
