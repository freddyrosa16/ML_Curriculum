# RNNs

**Status:** Original

## Why this matters
Text, audio, prices and sensor readings all arrive as *sequences*, where order carries meaning. Recurrent neural networks handle this by keeping a running memory (the hidden state) that they update one step at a time. Building the simple RNN, GRU and LSTM cells by hand shows you exactly what that memory is and why gates fixed the original RNN's forgetfulness. Every later sequence model, including attention, is easier to follow once you have done this.

## Concepts to understand first
- **What makes data "sequential"?** Each item depends on the ones before it. Shuffling the words of a sentence or the days of a price chart destroys information, so the model must read things in order.
- **What is a hidden state?** A vector the network carries from one time step to the next. At each step it mixes the old hidden state with the new input to produce a new one, like a running summary of everything seen so far.
- **Why are the same weights used at every time step?** The cell is one small network applied over and over. Sharing weights means the model can handle sequences of any length and learns one rule for "how to update memory", not a different rule per position.
- **What is the vanishing gradient problem?** When you backpropagate through many time steps, the gradient gets multiplied by similar factors again and again. If those factors are below 1 it shrinks toward zero, so a plain RNN struggles to learn long-range dependencies.
- **What do gates do in a GRU or LSTM?** A gate is a sigmoid layer that outputs numbers between 0 and 1, used to multiply something element-wise. It acts like a dial: "keep this much of the old memory, add this much of the new candidate". That lets information (and gradient) flow across many steps.
- **How does an LSTM differ from a GRU?** The LSTM keeps two vectors, a cell state $c$ (long-term memory) and a hidden state $h$ (what it shows the outside), and uses forget, update (input) and output gates. The GRU merges them into one hidden state and uses just an update gate and a reset gate.
- **What do "deep" and "bidirectional" mean for RNNs?** Deep: stack several RNN layers, where layer 2 reads layer 1's hidden states as its input. Bidirectional: run one RNN left-to-right and another right-to-left, then join their hidden states so each position sees both past and future context.
- **What is softmax?** It turns a vector of scores into probabilities: exponentiate each score, then divide by the sum. Every cell here produces its output $y$ with a softmax.

## Resources
- [Andrej Karpathy: The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Christopher Olah: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (the clearest diagrams of LSTM gates)
- [Dive into Deep Learning: Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
- [Dive into Deep Learning: Modern RNNs (GRU, LSTM, deep, bidirectional)](https://d2l.ai/chapter_recurrent-modern/index.html)
- [StatQuest: Recurrent Neural Networks, clearly explained](https://www.youtube.com/watch?v=AsNTP8Kwu80)
- [StatQuest: Long Short-Term Memory (LSTM), clearly explained](https://www.youtube.com/watch?v=YCzL96nL7j0)

## Rules
- NumPy only: `import numpy as np`. No other imports.
- Forward pass only: no backpropagation in this project.
- **Concatenation order matters.** Whenever a cell combines the hidden state with the input, put the hidden state first: `[h_prev, x_t]` along the feature axis (axis 1). The weight shapes below assume this.
- Weights go on the right side of the matrix multiplication: inputs times weights, not weights times inputs.
- Initialize weight matrices with `np.random.normal` (standard normal), **in the order the task lists them**, and biases with zeros. The tests check the order by seeding NumPy.
- Hidden states use `tanh`, gates use sigmoid, outputs use softmax.

## Tasks

Notation used below: `i` = input size, `h` = hidden size, `o` = output size, `m` = batch size, `t` = number of time steps.

### 0. RNN Cell
**File:** `0-rnn_cell.py`
**Class:** `RNNCell` with `__init__(self, i, h, o)` and `forward(self, h_prev, x_t)`

- Constructor creates public attributes, in this order: `Wh` of shape `(i + h, h)`, `Wy` of shape `(h, o)`, `bh` of shape `(1, h)`, `by` of shape `(1, o)`. `Wh`/`bh` turn the joined `[h_prev, x_t]` into the next hidden state; `Wy`/`by` turn the hidden state into an output.
- `forward` takes `h_prev` with shape `(m, h)` and `x_t` with shape `(m, i)`, and returns `h_next, y`:
  - `h_next`: the new hidden state, shape `(m, h)`, using tanh
  - `y`: the output, shape `(m, o)`, using softmax

```python
>>> cell = RNNCell(10, 15, 5)
>>> cell.Wh.shape, cell.Wy.shape, cell.bh.shape, cell.by.shape
((25, 15), (15, 5), (1, 15), (1, 5))
>>> h_next, y = cell.forward(np.zeros((8, 15)), np.random.randn(8, 10))
>>> h_next.shape, y.shape, np.allclose(y.sum(axis=1), 1)
((8, 15), (8, 5), True)
```

### 1. RNN
**File:** `1-rnn.py`
**Function:** `def rnn(rnn_cell, X, h_0):`

Run a full forward pass over a sequence.
- `rnn_cell`: an `RNNCell`
- `X`: input data, shape `(t, m, i)`
- `h_0`: initial hidden state, shape `(m, h)`
- Returns `H, Y`:
  - `H`: every hidden state **including** `h_0` at index 0, shape `(t + 1, m, h)`
  - `Y`: every output, shape `(t, m, o)`

```python
>>> H, Y = rnn(cell, np.random.randn(6, 8, 10), np.zeros((8, 15)))
>>> H.shape, Y.shape
((7, 8, 15), (6, 8, 5))
```

### 2. GRU Cell
**File:** `2-gru_cell.py`
**Class:** `GRUCell` with `__init__(self, i, h, o)` and `forward(self, h_prev, x_t)`

- Constructor attributes, in this order: `Wz`, `Wr`, `Wh` (each `(i + h, h)`), `Wy` (`(h, o)`), then biases `bz`, `br`, `bh` (each `(1, h)`) and `by` (`(1, o)`). `z` is the update gate, `r` the reset gate, `h` the candidate hidden state.
- `forward` returns `h_next, y` with the same shapes as the RNN cell.
- How the GRU combines things: the update gate decides how much of the candidate replaces the old state. A `z` of 1 means "take the candidate fully". The reset gate scales `h_prev` *before* it is joined with `x_t` to compute the candidate.

```python
>>> gru = GRUCell(10, 15, 5)
>>> h_next, y = gru.forward(np.random.randn(8, 15), np.random.randn(8, 10))
>>> h_next.shape, y.shape
((8, 15), (8, 5))
```

### 3. LSTM Cell
**File:** `3-lstm_cell.py`
**Class:** `LSTMCell` with `__init__(self, i, h, o)` and `forward(self, h_prev, c_prev, x_t)`

- Constructor attributes, in this order: `Wf`, `Wu`, `Wc`, `Wo` (each `(i + h, h)`), `Wy` (`(h, o)`), then `bf`, `bu`, `bc`, `bo` (each `(1, h)`) and `by` (`(1, o)`). `f` = forget gate, `u` = update (input) gate, `c` = candidate cell state, `o` = output gate.
- `forward` takes `h_prev` and `c_prev` (both `(m, h)`) and `x_t` (`(m, i)`), and returns `h_next, c_next, y`.
- All four of `f`, `u`, the candidate and `o` are computed from the same joined `[h_prev, x_t]`. The new cell state keeps part of the old one (forget gate) and adds part of the candidate (update gate). The new hidden state is the output gate times tanh of the new cell state.

```python
>>> lstm = LSTMCell(10, 15, 5)
>>> h, c, y = lstm.forward(np.zeros((8, 15)), np.zeros((8, 15)), np.random.randn(8, 10))
>>> h.shape, c.shape, y.shape
((8, 15), (8, 15), (8, 5))
```

### 4. Deep RNN
**File:** `4-deep_rnn.py`
**Function:** `def deep_rnn(rnn_cells, X, h_0):`

- `rnn_cells`: a list of `l` `RNNCell`s, layer 0 first
- `X`: shape `(t, m, i)`
- `h_0`: initial hidden states for every layer, shape `(l, m, h)`
- Returns `H, Y`:
  - `H`: shape `(t + 1, l, m, h)`, with `H[0]` equal to `h_0`
  - `Y`: the outputs of the **last** layer, shape `(t, m, o)`
- Layer 0 reads `X[step]`. Every higher layer reads the new hidden state of the layer below at the same step.

```python
>>> cells = [RNNCell(10, 15, 1), RNNCell(15, 15, 1), RNNCell(15, 15, 5)]
>>> H, Y = deep_rnn(cells, np.random.randn(6, 8, 10), np.zeros((3, 8, 15)))
>>> H.shape, Y.shape
((7, 3, 8, 15), (6, 8, 5))
```

### 5. Bidirectional Cell Forward
**File:** `5-bi_forward.py`
**Class:** `BidirectionalCell` with `__init__(self, i, h, o)` and `forward(self, h_prev, x_t)`

- Constructor attributes, in this order: `Whf`, `Whb` (each `(i + h, h)`), `Wy` (`(2 * h, o)`), then `bhf`, `bhb` (each `(1, h)`) and `by` (`(1, o)`). `f` = forward direction, `b` = backward direction. `Wy` is twice as tall because the output reads both directions joined together.
- `forward` computes the next hidden state in the **forward** direction and returns `h_next` (shape `(m, h)`).

### 6. Bidirectional Cell Backward
**File:** `6-bi_backward.py`
Copy your class from task 5 and add `backward(self, h_next, x_t)`: compute the hidden state for the **backward** direction (which walks from the end of the sequence to the start), using the backward weights. Join as `[h_next, x_t]`. Returns `h_prev`, shape `(m, h)`.

### 7. Bidirectional Output
**File:** `7-bi_output.py`
Copy your class from task 6 and add `output(self, H)`:
- `H`: the joined hidden states from both directions, shape `(t, m, 2 * h)`, not including the initial states
- Returns `Y`, shape `(t, m, o)`, the softmax output at every time step

```python
>>> bi = BidirectionalCell(10, 15, 5)
>>> bi.output(np.random.randn(6, 8, 30)).shape
(6, 8, 5)
```

### 8. Bidirectional RNN
**File:** `8-bi_rnn.py`
**Function:** `def bi_rnn(bi_cell, X, h_0, h_t):`

- `bi_cell`: a `BidirectionalCell` from task 7
- `X`: shape `(t, m, i)`
- `h_0`: starting hidden state for the forward direction, `(m, h)`
- `h_t`: starting hidden state for the backward direction (it starts at the end), `(m, h)`
- Returns `H, Y`:
  - `H`: at each time step, the forward hidden state joined with the backward hidden state (forward first), shape `(t, m, 2 * h)`. Don't include `h_0` or `h_t`.
  - `Y`: shape `(t, m, o)`

```python
>>> H, Y = bi_rnn(bi, np.random.randn(6, 8, 10), np.zeros((8, 15)), np.zeros((8, 15)))
>>> H.shape, Y.shape
((6, 8, 30), (6, 8, 5))
```

### 9. Concept check (optional)
No code. Answer in your own words, in a file `9-answers.md` if you like:
1. What does RNN stand for, and what kind of data are RNNs built for?
2. Sketch (on paper) a basic RNN, a deep RNN and a bidirectional RNN. What is different about how information flows in each?

## Check your work
From `ml-curriculum/`:
```bash
pytest 03-sequences-nlp/01-rnns
```
