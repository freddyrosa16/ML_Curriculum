# Regularization

**Status:** Original. The NumPy tasks are unchanged; the TensorFlow layer tasks (2, 3, 6) now use PyTorch. Task 3 became "L2 via weight decay", because that's how PyTorch does it.

## Why this matters
A big network can memorize its training set and still fail on new data (high variance). Regularization is the toolbox for that: penalize large weights (L2), randomly switch off neurons while training (dropout), and stop before the model starts memorizing (early stopping). You'll use at least one of these in nearly every model you train.

## Concepts to understand first
- **What is overfitting?** The model learns the noise in the training set along with the signal. Training error keeps falling while validation error starts rising.
- **What does L2 regularization do?** It adds a penalty for big weights to the cost: $J_{reg} = J + \frac{\lambda}{2m}\sum_l \lVert W^{[l]} \rVert_F^2$, where $\lVert W \rVert_F^2$ is the sum of the squares of every entry. Smaller weights make a smoother function that can't twist itself around every training point. $\lambda$ controls how hard you push.
- **Why is it also called "weight decay"?** The penalty's gradient is $\frac{\lambda}{m} W$, so each gradient step also shrinks every weight a little toward zero. Optimizers like `torch.optim.SGD` implement exactly that with a `weight_decay` argument.
- **What is dropout?** During training, each hidden neuron is kept with probability `keep_prob` and zeroed otherwise, with a fresh random choice every step. The network can't rely on any single neuron, so it learns redundant, robust features. At test time nothing is dropped.
- **What is "inverted" dropout?** After zeroing, divide the surviving activations by `keep_prob`. That keeps the expected value of each activation the same in training and testing, so you don't need to rescale anything at test time.
- **What is early stopping?** Watch the validation cost while training. If it hasn't improved by at least some `threshold` for `patience` checks in a row, stop and keep the best model.

## Resources
- [Andrew Ng, Improving Deep Neural Networks (Coursera, free to audit), week 1: regularization and dropout](https://www.coursera.org/learn/deep-neural-network)
- [Dive into Deep Learning: Weight decay](https://d2l.ai/chapter_linear-regression/weight-decay.html) and [Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html)
- [Srivastava et al. (2014): Dropout, the original paper (JMLR)](https://jmlr.org/papers/v15/srivastava14a.html)
- [StatQuest: Regularization Part 1 - Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30)
- [CS231n: Regularization notes](https://cs231n.github.io/neural-networks-2/#reg)
- [PyTorch docs: `nn.Dropout`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html) and [`torch.optim.SGD` (weight_decay, parameter groups)](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)

## Rules
- **NumPy tasks** (0, 1, 4, 5, 7): only `import numpy as np` (task 7 needs no imports). You may loop over layers, not over examples.
- **PyTorch tasks** (2, 3, 6): `import torch` / `from torch import nn` only.
- NumPy networks here use the classification project's layout: `X` is `(nx, m)`, one-hot `Y` is `(classes, m)`, `weights` is a dict `W1, b1, ..., WL, bL`, `cache` is a dict `A0 ... AL`. Hidden layers use **tanh**, the output layer uses **softmax**.

## Tasks

### 0. L2 regularization cost
**File:** `0-l2_reg_cost.py` · **Function:** `l2_reg_cost(cost, lambtha, weights, L, m)`

`cost` is the unregularized cost, `lambtha` is $\lambda$, `weights` the weights dict, `L` the number of layers, `m` the number of examples. Return the cost with the L2 penalty added (penalize the `W`s only, not the biases).

```
>>> weights = {'W1': np.array([[1., 2.], [3., 4.]]), 'b1': np.zeros((2, 1)),
...            'W2': np.array([[1., -1.]]), 'b2': np.zeros((1, 1))}
>>> l2_reg_cost(0.5, 0.1, weights, 2, 10)
0.66
```

### 1. Gradient descent with L2 regularization
**File:** `1-l2_reg_gradient_descent.py` · **Function:** `l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L)`

One step of gradient descent on the L2-regularized cost (tanh hidden layers, softmax output), updating `weights` **in place**. Returns nothing.

### 2. L2 regularization cost (PyTorch)
**File:** `2-l2_reg_cost.py` · **Function:** `l2_reg_cost(cost, model, lambtha)`

`cost` is a scalar tensor, `model` an `nn.Module`. Return `cost + lambtha * (sum of the squares of the weights of every nn.Linear layer in the model)`, as a tensor that gradients can flow through. Don't penalize biases.

(This is the same idea as task 0 with the constant folded into `lambtha`.)

```
>>> model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
>>> # with weights [[1, 2], [3, 4]] and [[1, -1]]:
>>> l2_reg_cost(torch.tensor(0.5), model, 0.01)
tensor(0.8200, grad_fn=<AddBackward0>)
```

### 3. L2 via weight decay (PyTorch)
**File:** `3-l2_reg_optimizer.py` · **Function:** `l2_reg_optimizer(model, alpha, lambtha)`

In practice you rarely add the L2 term to the loss yourself; you ask the optimizer to decay the weights. Return a `torch.optim.SGD` optimizer for all of `model`'s parameters with learning rate `alpha`, where:
- weight matrices (parameters with more than one dimension) get `weight_decay=lambtha`,
- biases (one-dimensional parameters) get no weight decay.

Use parameter groups to give the two kinds different settings.

```
>>> opt = l2_reg_optimizer(nn.Linear(3, 2), 0.1, 0.01)
>>> [(len(g['params']), g['weight_decay']) for g in opt.param_groups]
[(1, 0.01), (1, 0.0)]
```

### 4. Forward propagation with dropout
**File:** `4-dropout_forward_prop.py` · **Function:** `dropout_forward_prop(X, weights, L, keep_prob)`

Forward pass with inverted dropout on every hidden layer (not the output). Return a dict holding the outputs `A0 ... AL` and the dropout masks `D1 ... D(L-1)`: arrays of 0s and 1s with the same shape as the layer's output. Draw masks with `np.random`.

```
>>> np.random.seed(0)
>>> X = np.random.randn(3, 5)
>>> weights = {'W1': np.random.randn(4, 3), 'b1': np.zeros((4, 1)),
...            'W2': np.random.randn(2, 4), 'b2': np.zeros((2, 1))}
>>> cache = dropout_forward_prop(X, weights, 2, 0.5)
>>> sorted(cache)
['A0', 'A1', 'A2', 'D1']
>>> cache['A2'].sum(axis=0)
array([1., 1., 1., 1., 1.])
```

### 5. Gradient descent with dropout
**File:** `5-dropout_gradient_descent.py` · **Function:** `dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L)`

One step of gradient descent for a network trained with dropout, using the masks saved in `cache` (the same neurons that were dropped going forward get no gradient going back). Update `weights` in place. Returns nothing.

### 6. Create a layer with dropout (PyTorch)
**File:** `6-dropout_create_layer.py` · **Function:** `dropout_create_layer(n_in, n, activation, keep_prob)`

Return an `nn.Module` that applies `nn.Linear(n_in, n)`, then `activation` (a module like `nn.Tanh()`), then dropout that **keeps** each unit with probability `keep_prob`. Careful: PyTorch's `nn.Dropout(p)` takes the probability of *dropping*.

In `.train()` mode roughly `1 - keep_prob` of the outputs should be zero; in `.eval()` mode none should be dropped.

### 7. Early stopping
**File:** `7-early_stopping.py` · **Function:** `early_stopping(cost, opt_cost, threshold, patience, count)`

`cost` is the current validation cost, `opt_cost` the best one so far, `threshold` how much better a cost must be to count as an improvement, `patience` how many non-improving checks you tolerate, `count` how many you've had so far. Return `(should_stop, new_count)`:
- If `opt_cost - cost > threshold`, the count resets to 0. Otherwise it goes up by 1.
- Stop when the count reaches `patience`.

```
>>> early_stopping(1.0, 1.5, 0.1, 3, 2)
(False, 0)
>>> early_stopping(1.45, 1.5, 0.1, 3, 1)
(False, 2)
>>> early_stopping(1.45, 1.5, 0.1, 3, 2)
(True, 3)
```

### 8. Write-up (bonus)
Train the same small network on MNIST (from `01-classification/data/MNIST.npz`) with no regularization, with L2 and with dropout, and plot training vs validation cost for each. Write a few paragraphs on what you see.

## Check your work
```bash
pytest 01-supervised-foundations/04-regularization
```
