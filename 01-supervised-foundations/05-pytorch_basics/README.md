# PyTorch Basics

**Status:** Modernized. Replaces the original course's TensorFlow 1 `tensorflow` project (placeholders, sessions, `tf.train.Saver`). Every task maps to the same step in PyTorch: placeholders → tensors, `create_layer` → `nn.Linear`, `forward_prop` → an `nn.Module`, train op → optimizer step, Saver → checkpoint dict.

## Why this matters
You've built networks by hand in NumPy, including every gradient. PyTorch does the boring parts for you: it computes gradients automatically (autograd), provides layers and optimizers, and runs the same code on a CPU, an Apple GPU (MPS) or an NVIDIA GPU. This project rebuilds your MNIST classifier in PyTorch so you can see that the concepts carry over one-to-one.

## Concepts to understand first
- **What is a tensor?** PyTorch's version of a NumPy array: an n-dimensional grid of numbers with a `dtype` and a `device` (where it lives: `cpu`, `mps`, `cuda`). Most NumPy operations have a tensor twin.
- **What is autograd?** If a tensor has `requires_grad=True`, PyTorch records every operation done with it. Calling `.backward()` on a scalar result walks that record backwards (the chain rule, like your backprop) and stores the gradient in each input's `.grad`.
- **What is an `nn.Module`?** A class that holds parameters (weights) and defines `forward(x)`. You build models by putting layers in `__init__` and calling them in `forward`. Calling `model(x)` runs `forward`.
- **What is `nn.Linear`?** A fully connected layer computing `x @ W.T + b`. Note the layout flip from the classification project: in PyTorch each **row** of `x` is an example, so `x` is `(m, nx)`.
- **Why do PyTorch classifiers output "logits"?** Raw, un-softmaxed scores. `F.cross_entropy` applies log-softmax internally (more numerically stable than doing it yourself), and takes class *indices* as labels, not one-hot vectors.
- **What is the training loop?** Every step: clear old gradients (`optimizer.zero_grad()`), forward pass, compute the loss, `loss.backward()`, `optimizer.step()`. That's it. Everything else is bookkeeping.
- **What are `model.train()`, `model.eval()` and `torch.no_grad()`?** `train`/`eval` switch layers like dropout and batch norm between their two behaviors. `no_grad` turns off gradient recording when you're only measuring, which saves memory and time.
- **What is a checkpoint?** A file holding what you need to rebuild the model later: the architecture settings plus the `state_dict` (a dictionary of all parameter tensors). `torch.save` writes it, `torch.load` reads it.

## Resources
- [PyTorch: Learn the Basics (official tutorial series)](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) - tensors, autograd, building a model, the optimization loop, save/load
- [PyTorch: A Gentle Introduction to `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [PyTorch: What is `torch.nn` really?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html) - builds from raw tensors up to `nn.Module`, very close to this project
- [Dive into Deep Learning: Builders' guide (modules, parameters, saving)](https://d2l.ai/chapter_builders-guide/index.html)
- [Andrej Karpathy: Building makemore (bigram model in PyTorch)](https://www.youtube.com/watch?v=PaCmpygFfXo) - see the training loop written live
- [PyTorch docs: `torch.nn.functional.cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) and [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

## Rules
- Use `torch` (with `torch.nn` and `torch.nn.functional`) and `numpy`. No TensorFlow or Keras.
- Data is `(m, nx)`: one example per row. Labels are class indices, shape `(m,)`.
- Later tasks should import earlier ones, e.g. `Network = __import__('3-forward_prop').Network`.
- Code should be device-agnostic: never hard-code `"cuda"`. Task 7 picks the best available device.

## Data
MNIST from the classification project: `01-classification/data/MNIST.npz` (keys `X_train`, `Y_train`, `X_valid`, `Y_valid`, `X_test`, `Y_test`). Flatten each image to 784 numbers so `X` is `(m, 784)`. The tests use small synthetic data; one optional slow test trains on MNIST.

## Tasks

### 0. Tensors (was: placeholders)
**File:** `0-to_tensors.py` · **Function:** `to_tensors(X, Y, device="cpu")`

`X` is a NumPy array `(m, nx)`, `Y` a NumPy array of integer labels `(m,)`. Return `(X_t, Y_t)`: `X` as a `float32` tensor and `Y` as an `int64` tensor, both on `device`.

```
>>> to_tensors(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
(tensor([[1., 2.],
        [3., 4.]]), tensor([0, 1]))
```

### 1. Autograd
**File:** `1-autograd.py` · **Function:** `gradient_at(f, x)`

`f` is a Python function built from tensor operations (like `lambda t: t ** 3 + 2 * t`), `x` a number. Return `(f(x), f'(x))` as two Python floats, computing the derivative with autograd (no finite differences, no doing the calculus yourself). Use `float64` for precision.

```
>>> gradient_at(lambda t: t ** 3 + 2 * t, 2.0)
(12.0, 14.0)
>>> gradient_at(torch.sin, 0.0)
(0.0, 1.0)
```

### 2. Create a layer
**File:** `2-create_layer.py` · **Function:** `create_layer(n_in, n, activation=None)`

Return a module that applies `nn.Linear(n_in, n)` followed by the activation named by `activation`: `'tanh'`, `'relu'`, `'sigmoid'`, or `None` for no activation. Initialize the weights with Xavier/Glorot **uniform** initialization and the bias with zeros. An unknown activation name raises `ValueError`.

```
>>> create_layer(784, 256, 'tanh')
Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): Tanh()
)
>>> create_layer(256, 10)
Linear(in_features=256, out_features=10, bias=True)
```

### 3. Forward propagation
**File:** `3-forward_prop.py` · **Class:** `Network(nn.Module)`

`Network(nx, layer_sizes, activations)` stacks one `create_layer` per entry: layer `i` has `layer_sizes[i]` nodes and activation `activations[i]`. Store the three constructor arguments as attributes `nx`, `layer_sizes`, `activations`. If the two lists have different lengths, raise `ValueError`. `forward(x)` passes `x` through every layer and returns the output.

```
>>> net = Network(784, [256, 256, 10], ['tanh', 'tanh', None])
>>> net(torch.zeros(5, 784)).shape
torch.Size([5, 10])
```

### 4. Accuracy
**File:** `4-calculate_accuracy.py` · **Function:** `calculate_accuracy(y, y_pred)`

`y` is `(m,)` class indices, `y_pred` is `(m, classes)` scores. Return the fraction of examples whose highest score is the right class, as a 0-dimensional float tensor.

```
>>> y = torch.tensor([0, 1, 2, 1])
>>> y_pred = torch.tensor([[2., 1, 0], [0, 3, 1], [1, 0, .5], [0, 0, 1]])
>>> calculate_accuracy(y, y_pred)
tensor(0.5000)
```

### 5. Loss
**File:** `5-calculate_loss.py` · **Function:** `calculate_loss(y, y_pred)`

Return the mean softmax cross-entropy loss between logits `y_pred` and labels `y`, as a scalar tensor.

```
>>> calculate_loss(y, y_pred)
tensor(0.8273)
```

### 6. Train op
**File:** `6-create_train_op.py` · **Functions:** `create_train_op(model, alpha)` and `train_step(model, optimizer, x, y)`

- `create_train_op` returns a plain gradient descent optimizer (`torch.optim.SGD`, no momentum) over the model's parameters with learning rate `alpha`.
- `train_step` performs one full training step on the batch `(x, y)`: clear gradients, forward, loss, backward, update. Return the loss (computed before the update) as a Python float.

```
>>> opt = create_train_op(net, 0.1)
>>> train_step(net, opt, x, y)
2.4552...     # some float; it should shrink if you call it repeatedly
```

### 7. Train
**File:** `7-train.py` · **Function:** `train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="model.pt", device=None)`

Build a `Network`, train it with full-batch gradient descent for `iterations` steps, and save a checkpoint. Inputs are NumPy arrays.
- Also write a helper `get_device()` that returns `cuda` if available, else `mps` if available, else `cpu`. Use it when `device` is `None`; otherwise use the given device.
- At iteration 0, every 100 iterations, and after the last iteration, print (cost and accuracy measured in eval mode with no gradients, *before* that iteration's update):
  ```
  After {i} iterations:
  	Training Cost: {cost}
  	Training Accuracy: {accuracy}
  	Validation Cost: {cost}
  	Validation Accuracy: {accuracy}
  ```
  (each metric line starts with a tab)
- Save with `torch.save` a dict with keys `"nx"`, `"layer_sizes"`, `"activations"` and `"state_dict"` (tensors moved to the CPU, so the file loads anywhere).
- Return `save_path`.

On MNIST, `[256, 256, 10]` with `['tanh', 'tanh', None]`, `alpha=0.1` and 300 iterations on 10,000 training images gets about 90% validation accuracy in roughly 10 seconds on a laptop CPU.

### 8. Evaluate
**File:** `8-evaluate.py` · **Function:** `evaluate(X, Y, save_path)`

Load the checkpoint from task 7 (on the CPU), rebuild the `Network`, load its weights and evaluate it on `(X, Y)`. Return `(y_pred, accuracy, loss)`: the `(m, classes)` output tensor, and accuracy and loss as Python floats. Use `weights_only=True` when loading, which is the safe default for files you didn't create.

```
>>> y_pred, acc, loss = evaluate(X_valid, Y_valid, 'model.pt')
>>> y_pred.shape, round(acc, 2)
(torch.Size([2000, 10]), 0.91)
```

### 9. Write-up (bonus)
Put your NumPy `DeepNeuralNetwork` from the classification project next to your PyTorch version. Write a short comparison: which lines of NumPy turned into which PyTorch calls, and what does autograd replace?

## Check your work
```bash
pytest 01-supervised-foundations/05-pytorch_basics
```
Add `-m "not slow"` to skip the MNIST training test.
