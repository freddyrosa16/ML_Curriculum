# Optimization

**Status:** Original. The NumPy tasks are unchanged; the tasks that used TensorFlow (6, 8, 10, 12, 14) now use PyTorch's `torch.optim`, LR schedulers and `nn.BatchNorm1d`.

## Why this matters
Plain gradient descent works, but it's slow and fussy. The tricks in this project (normalizing inputs, mini-batches, momentum, RMSProp, Adam, learning-rate decay, batch norm) are why modern networks train in minutes instead of days. You'll write each one in NumPy so you know exactly what it does, then meet the one-line PyTorch version you'll use in practice.

## Concepts to understand first
- **Why normalize features?** If one feature ranges 0-1 and another 0-10000, the cost surface is a long, thin valley and gradient descent zig-zags. Subtracting each feature's mean and dividing by its standard deviation makes the valley round, so steps go straight to the bottom.
- **What are mini-batches?** Instead of computing the gradient on all `m` examples (slow) or one example (noisy), you use small chunks of, say, 32-256. You shuffle first so each batch is a fair sample. One pass over all batches is an *epoch*.
- **What is an exponentially weighted moving average?** A running average that favors recent values: $v_t = \beta v_{t-1} + (1-\beta)\,\theta_t$. With $\beta = 0.9$ it roughly averages the last 10 values. Because $v_0 = 0$, early values are too small; dividing by $1 - \beta^t$ ("bias correction") fixes that.
- **What does momentum do?** It applies that moving average to the gradients: $v = \beta_1 v + (1-\beta_1)\,dW$, then $W = W - \alpha v$. Like a ball rolling downhill, it builds speed in consistent directions and damps the zig-zags.
- **What does RMSProp do?** It keeps a moving average of *squared* gradients, $s = \beta_2 s + (1-\beta_2)\,dW^2$, and divides the step by $\sqrt{s} + \epsilon$. Directions with large, jumpy gradients get smaller steps; flat directions get bigger ones.
- **What is Adam?** Momentum plus RMSProp, both bias-corrected: $W = W - \alpha \frac{\hat v}{\sqrt{\hat s} + \epsilon}$ where $\hat v = v / (1-\beta_1^t)$ and $\hat s = s / (1-\beta_2^t)$. It's the default optimizer for most deep learning today.
- **Why decay the learning rate?** Big steps help early on; near the minimum they make you bounce around it. Inverse time decay shrinks the rate as $\alpha_t = \frac{\alpha_0}{1 + \text{decay\_rate}\cdot\lfloor t / \text{decay\_step}\rfloor}$, so it drops in stair steps.
- **What is batch normalization?** Normalizing the *inputs* helps, so why not normalize every layer's pre-activation too? For each mini-batch, subtract the batch mean and divide by the batch std, then let the network rescale with learned $\gamma$ and shift with learned $\beta$: $\tilde Z = \gamma \frac{Z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$.

## Resources
- [Andrew Ng, Improving Deep Neural Networks (Coursera, free to audit), week 2: optimization algorithms](https://www.coursera.org/learn/deep-neural-network)
- [Dive into Deep Learning: Optimization algorithms (momentum, RMSProp, Adam, schedulers)](https://d2l.ai/chapter_optimization/index.html)
- [Sebastian Ruder: An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/)
- [Distill: Why Momentum Really Works](https://distill.pub/2017/momentum/)
- [Kingma & Ba (2014): Adam paper](https://arxiv.org/abs/1412.6980) · [Ioffe & Szegedy (2015): Batch Normalization paper](https://arxiv.org/abs/1502.03167)
- [PyTorch docs: `torch.optim`](https://docs.pytorch.org/docs/stable/optim.html) and [`nn.BatchNorm1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)

## Rules
- **NumPy tasks** (0, 1, 2, 3, 4, 5, 7, 9, 11, 13): only `import numpy as np`. Task 4 needs no imports at all. No loops except where the task says so (task 3 may loop over batches, task 4 over data points).
- **PyTorch tasks** (6, 8, 10, 12, 14): `import torch` / `from torch import nn` only. Use the built-in optimizers, schedulers and layers. Don't reimplement them.
- In this project data is laid out **one example per row**: `X` has shape `(m, nx)`. (This is the PyTorch convention too, and the opposite of the classification project.)

## Tasks

### 0. Normalization constants
**File:** `0-norm_constants.py` · **Function:** `normalization_constants(X)`

`X` is `(m, nx)`. Return `(mean, std)` of each feature, each of shape `(nx,)`.

```
>>> X = np.array([[1., 10.], [2., 20.], [3., 60.]])
>>> normalization_constants(X)
(array([ 2., 30.]), array([ 0.8165, 21.6025]))
```

### 1. Normalize
**File:** `1-normalize.py` · **Function:** `normalize(X, m, s)`

Standardize `X` using the means `m` and standard deviations `s` from task 0. Return the normalized array.

```
>>> normalize(X, *normalization_constants(X))
array([[-1.2247, -0.9258],
       [ 0.    , -0.4629],
       [ 1.2247,  1.3887]])
```

### 2. Shuffle data
**File:** `2-shuffle_data.py` · **Function:** `shuffle_data(X, Y)`

Shuffle the rows of `X` `(m, nx)` and `Y` `(m, ny)` **the same way** (row `i` of X must stay paired with row `i` of Y). Use `np.random.permutation`. Return `(X_shuffled, Y_shuffled)`.

```
>>> np.random.seed(0)
>>> shuffle_data(X, np.array([[0], [1], [2]]))
(array([[ 3., 60.], [ 2., 20.], [ 1., 10.]]), array([[2], [1], [0]]))
```

### 3. Mini-batch
**File:** `3-mini_batch.py` · **Function:** `create_mini_batches(X, Y, batch_size)`

Shuffle `X` and `Y` with your `shuffle_data` from task 2 (import it with `__import__('2-shuffle_data')`), then cut them into consecutive mini-batches of `batch_size` rows. The last batch holds whatever is left and may be smaller. Return a list of `(X_batch, Y_batch)` tuples.

```
>>> np.random.seed(0)
>>> X = np.arange(10).reshape(5, 2); Y = np.arange(5).reshape(5, 1)
>>> [(xb.tolist(), yb.ravel().tolist()) for xb, yb in create_mini_batches(X, Y, 2)]
[([[4, 5], [0, 1]], [2, 0]), ([[2, 3], [6, 7]], [1, 3]), ([[8, 9]], [4])]
```

### 4. Moving average
**File:** `4-moving_average.py` · **Function:** `moving_average(data, beta)`

`data` is a list of numbers. Return a list of the exponentially weighted moving averages **with bias correction**, one per data point.

```
>>> moving_average([72, 78, 71, 68, 66], 0.9)
[72.0, 75.1579, 73.6236, 71.9884, 70.5260]   # (rounded here)
```

### 5. Momentum
**File:** `5-momentum.py` · **Function:** `update_variables_momentum(alpha, beta1, var, grad, v)`

One step of gradient descent with momentum. `v` is the previous first moment. Return `(updated_var, new_v)`.

```
>>> var, grad = np.array([1., 2.]), np.array([.5, -1.])
>>> update_variables_momentum(0.1, 0.9, var, grad, np.zeros(2))
(array([0.995, 2.01 ]), array([ 0.05, -0.1 ]))
```

### 6. Momentum upgraded (PyTorch)
**File:** `6-momentum.py` · **Function:** `create_momentum_op(params, alpha, beta1)`

Return a `torch.optim.SGD` optimizer over `params` with learning rate `alpha` and momentum `beta1`.

Note: PyTorch's momentum is written slightly differently from task 5 ($v = \beta v + g$, without the $(1-\beta)$ factor). It's the same idea; the learning rate just absorbs the scale.

```
>>> model = torch.nn.Linear(3, 1)
>>> opt = create_momentum_op(model.parameters(), 0.01, 0.9)
>>> type(opt).__name__, opt.param_groups[0]['lr'], opt.param_groups[0]['momentum']
('SGD', 0.01, 0.9)
```

### 7. RMSProp
**File:** `7-RMSProp.py` · **Function:** `update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s)`

One RMSProp step. `s` is the previous second moment. Return `(updated_var, new_s)`.

```
>>> update_variables_RMSProp(0.1, 0.9, 1e-8, var, grad, np.zeros(2))
(array([0.6838, 2.3162]), array([0.025, 0.1  ]))
```

### 8. RMSProp upgraded (PyTorch)
**File:** `8-RMSProp.py` · **Function:** `create_RMSProp_op(params, alpha, beta2, epsilon)`

Return a `torch.optim.RMSprop` optimizer matching task 7. Careful: PyTorch's argument named `alpha` is *not* the learning rate. Read the docs to map each of our hyperparameters to the right argument. One step of your optimizer should give exactly the same result as task 7.

### 9. Adam
**File:** `9-Adam.py` · **Function:** `update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t)`

One Adam step with bias correction. `t` is the step number, starting at 1. Return `(updated_var, new_v, new_s)` (the **uncorrected** moments).

```
>>> update_variables_Adam(0.1, 0.9, 0.99, 1e-8, var, grad, np.zeros(2), np.zeros(2), 1)
(array([0.9, 2.1]), array([ 0.05, -0.1 ]), array([0.0025, 0.01  ]))
```

### 10. Adam upgraded (PyTorch)
**File:** `10-Adam.py` · **Function:** `create_Adam_op(params, alpha, beta1, beta2, epsilon)`

Return a `torch.optim.Adam` optimizer matching task 9. A few steps of it should match your NumPy Adam.

### 11. Learning rate decay
**File:** `11-learning_rate_decay.py` · **Function:** `learning_rate_decay(alpha, decay_rate, global_step, decay_step)`

Return the learning rate after `global_step` steps using stepwise inverse time decay (formula in the concepts above).

```
>>> [learning_rate_decay(0.1, 1, step, 10) for step in (0, 9, 10, 25)]
[0.1, 0.1, 0.05, 0.0333...]
```

### 12. Learning rate decay upgraded (PyTorch)
**File:** `12-learning_rate_decay.py` · **Function:** `learning_rate_decay(optimizer, decay_rate, decay_step)`

Return a scheduler from `torch.optim.lr_scheduler` that applies the same stepwise inverse time decay to `optimizer`, where one call to `scheduler.step()` = one gradient descent step. The optimizer's starting learning rate is $\alpha_0$.

```
>>> opt = torch.optim.SGD(model.parameters(), lr=0.1)
>>> sched = learning_rate_decay(opt, 1, 10)
>>> # after 0, 9, 10 and 20 calls to opt.step(); sched.step():
>>> # opt.param_groups[0]['lr'] -> 0.1, 0.1, 0.05, 0.0333...
```

### 13. Batch normalization
**File:** `13-batch_norm.py` · **Function:** `batch_norm(Z, gamma, beta, epsilon)`

`Z` is `(m, n)`: the un-activated outputs of a layer for one mini-batch. `gamma` and `beta` are `(1, n)`. Return the batch-normalized `Z`.

```
>>> Z = np.array([[1., 2.], [3., 6.], [5., 10.]])
>>> batch_norm(Z, np.ones((1, 2)), np.zeros((1, 2)), 1e-7)
array([[-1.2247, -1.2247],
       [ 0.    ,  0.    ],
       [ 1.2247,  1.2247]])
```

### 14. Batch normalization upgraded (PyTorch)
**File:** `14-batch_norm.py` · **Function:** `create_batch_norm_layer(n_in, n, activation)`

Return an `nn.Module` (an `nn.Sequential` is perfect) that does, in order:
1. `nn.Linear(n_in, n)` **without a bias** (batch norm subtracts the mean anyway, so a bias would be wasted), with weights initialized by Xavier/Glorot normal initialization,
2. `nn.BatchNorm1d(n)` with `eps=1e-7` (its $\gamma$ starts at 1, $\beta$ at 0),
3. `activation`, an activation module such as `nn.Tanh()`.

```
>>> create_batch_norm_layer(4, 3, torch.nn.Tanh())
Sequential(
  (0): Linear(in_features=4, out_features=3, bias=False)
  (1): BatchNorm1d(3, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)
  (2): Tanh()
)
```
In training mode its batch-norm step should give the same numbers as your task 13.

### 15. Write-up (bonus)
Write a short post explaining momentum, RMSProp and Adam, with a plot comparing all three (plus plain gradient descent) minimizing a simple 2D function like $f(x, y) = x^2 + 10y^2$. Use your own NumPy functions from this project.

## Check your work
```bash
pytest 01-supervised-foundations/02-optimization
```
