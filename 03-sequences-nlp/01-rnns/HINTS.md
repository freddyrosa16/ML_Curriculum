# Hints: RNNs

Open one hint at a time. Try for a while before opening the next.

## 0. RNN Cell
<details><summary>Hint 1</summary>
Draw the shapes. You have a batch of hidden states and a batch of inputs, side by side. If you glue them together along the feature axis, how wide is the result? That width is why `Wh` has `i + h` rows.
</details>
<details><summary>Hint 2</summary>
There are two steps: (1) joined inputs → weights → bias → tanh gives the new hidden state; (2) new hidden state → output weights → bias → softmax gives y. Each is one matrix product plus a bias.
</details>
<details><summary>Hint 3</summary>
Look up NumPy's concatenate function and its `axis` argument. For softmax, the sum must run over each row separately (one probability distribution per example), so read about `keepdims`. Subtracting each row's max before exponentiating keeps the numbers from overflowing and doesn't change the result.
</details>

## 1. RNN
<details><summary>Hint 1</summary>
The cell already knows how to do one step. This function only has to call it once per time step and remember the results.
</details>
<details><summary>Hint 2</summary>
Pre-allocate the two result arrays with the shapes from the task. Slot 0 of H holds the starting state. The hidden state produced at step k goes into slot k + 1, because slot 0 is already taken.
</details>
<details><summary>Hint 3</summary>
The output size isn't passed in, but the cell's output weight matrix knows it (check its shape).
</details>

## 2. GRU Cell
<details><summary>Hint 1</summary>
A GRU is three small "RNN-like" computations: two gates (sigmoid) and one candidate (tanh). Then you blend.
</details>
<details><summary>Hint 2</summary>
Both gates read the same joined hidden-state-and-input. The candidate is different: before joining, the previous hidden state is scaled element-wise by the reset gate, so the reset gate controls how much of the past the candidate sees.
</details>
<details><summary>Hint 3</summary>
The final blend is a weighted average between the candidate and the old state, with z as the weight on the candidate and (1 − z) on the old state. Read the GRU section in d2l if you need the equation. Sigmoid is 1 / (1 + e^(−x)).
</details>

## 3. LSTM Cell
<details><summary>Hint 1</summary>
Count what you need: three gates (forget, update, output) and one candidate. All four read the same joined input.
</details>
<details><summary>Hint 2</summary>
The cell state update is "keep some old + add some new": the forget gate multiplies the old cell state, the update gate multiplies the candidate, and you add the two.
</details>
<details><summary>Hint 3</summary>
Colah's LSTM post walks through these exact equations, one diagram per gate. The output y comes from the new hidden state, not the cell state.
</details>

## 4. Deep RNN
<details><summary>Hint 1</summary>
You need two loops: one over time and one over layers. Which one goes outside? Think about what layer 2 needs at step k before it can run.
</details>
<details><summary>Hint 2</summary>
At each time step, go up the stack. Layer 0's "input" is the data. Layer j's "input" is the hidden state that layer j − 1 just produced at this same step. Each layer's "previous hidden state" comes from the previous time step at its own level.
</details>
<details><summary>Hint 3</summary>
Only the top layer's output goes in Y. Every layer still returns an output; you just ignore the others.
</details>

## 5–7. Bidirectional Cell
<details><summary>Hint 1</summary>
`forward` is the same calculation as the RNN cell's hidden-state step, just with the forward-direction weights. No output yet.
</details>
<details><summary>Hint 2</summary>
`backward` is the mirror image: same formula, backward weights, and the "previous" state is really the state from the *next* time step, because this direction reads the sequence end-to-start.
</details>
<details><summary>Hint 3</summary>
For `output`, the matrix product works on a 3-D array directly: multiplying `(t, m, 2h)` by `(2h, o)` gives `(t, m, o)`. Apply softmax over the last axis.
</details>

## 8. Bidirectional RNN
<details><summary>Hint 1</summary>
Run two separate passes: one walking forward in time from `h_0`, one walking backward from `h_t`. Store each direction's states in its own array.
</details>
<details><summary>Hint 2</summary>
Give each direction's array one extra slot for its starting state: at the start for forward, at the end for backward. Then the states for the real time steps line up and you can drop the extra slot from each.
</details>
<details><summary>Hint 3</summary>
Join the two directions along the last axis (forward first), then hand that to the cell's `output` method. Python's `range` can count down if you give it a negative step.
</details>
