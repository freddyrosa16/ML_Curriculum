# PyTorch Training

**Status:** Modernized. Replaces the original course's `keras` project. The same 14 tasks (build a model two ways, optimizer, one-hot, train, validation, early stopping, LR decay, save best, save/load model, weights, config, test, predict) done the PyTorch way: `nn.Sequential` vs subclassing, `DataLoader`, a hand-written training loop, `lr_scheduler`, `state_dict`.

## Why this matters
Keras hides the training loop behind `model.fit()`. In PyTorch you write the loop yourself, which feels like more work at first but means you understand (and can change) every step. By the end of this project you'll have a reusable `train_model` function with mini-batches, validation, early stopping, learning-rate decay and checkpointing: the skeleton of almost every PyTorch project you'll write later.

## Concepts to understand first
- **`nn.Sequential` vs subclassing `nn.Module`?** `nn.Sequential` is a list of layers applied in order: quick and readable for simple stacks. Subclassing lets you write any `forward` you like (skip connections, multiple inputs, conditionals). Both produce the same kind of object.
- **What are `Dataset` and `DataLoader`?** A `Dataset` knows how to return example `i`. A `DataLoader` wraps it and hands you shuffled mini-batches, one per loop iteration. `TensorDataset` turns tensors you already have into a `Dataset`.
- **What is an epoch?** One full pass over the training data. Metrics are usually reported once per epoch.
- **Why a validation set?** Training loss only tells you how well the model fits data it has seen. Validation loss, measured on held-out data with dropout off (`model.eval()`) and no gradients, tells you how well it generalizes, and it's what early stopping and checkpointing watch.
- **What does an LR scheduler do?** It changes the optimizer's learning rate as training goes on. You call `scheduler.step()` at a fixed point (here: once per epoch). Common ones: `StepLR` (drop by a factor every N epochs), `ReduceLROnPlateau` (drop when validation loss stalls), `LambdaLR` (any formula you want).
- **`state_dict` vs saving the whole model?** A `state_dict` is just a dictionary of parameter tensors: small, portable, safe to load (`weights_only=True`), but you need the code to rebuild the architecture first. Saving the whole model pickles the Python object: convenient, but loading runs pickle, which can execute arbitrary code, so only do it with files you trust.
- **Why save the config separately?** A small JSON file describing the architecture (sizes, activations, dropout) lets you rebuild an empty model without the original script, then load weights into it.

## Resources
- [PyTorch: Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch: Optimizing model parameters (the training loop)](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [PyTorch: Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch docs: `torch.optim` - how to adjust learning rate](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Dive into Deep Learning: Builders' guide](https://d2l.ai/chapter_builders-guide/index.html)
- [Andrej Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)

## Rules
- `torch` (with `torch.nn`, `torch.nn.functional`, `torch.utils.data`), plus `json` and `math` from the standard library. No Keras, no Lightning.
- Data passed to tasks 4-13 are tensors: `data` is float32 `(m, nx)`, `labels` are int64 class indices `(m,)`.
- Work on whatever device the model is on: move each batch to the model's device (you can read it from any of its parameters).
- Activations are given by name: `'relu'`, `'tanh'`, `'sigmoid'`, or `None` (no activation, used for the output layer, which produces logits).

## Data
Any classification data works. MNIST from `01-classification/data/MNIST.npz` is a good choice (flatten to `(m, 784)`, convert with `torch.tensor`). The tests use small synthetic data.

## Tasks

### 0. Sequential
**File:** `0-sequential.py` · **Function:** `build_model(nx, layers, activations, keep_prob)`

Return an `nn.Sequential` with one `nn.Linear` per entry of `layers` (layer `i` has `layers[i]` outputs), each followed by its activation (if not `None`), and an `nn.Dropout` that keeps units with probability `keep_prob` after every layer **except the last**. Raise `ValueError` if `layers` and `activations` differ in length.

Also attach the settings to the model as a dict attribute `config` with keys `"nx"`, `"layers"`, `"activations"`, `"keep_prob"` (task 11 uses it).

```
>>> model = build_model(784, [256, 256, 10], ['relu', 'relu', None], 0.8)
>>> model
Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.19999999999999996, inplace=False)
  (3): Linear(in_features=256, out_features=256, bias=True)
  (4): ReLU()
  (5): Dropout(p=0.19999999999999996, inplace=False)
  (6): Linear(in_features=256, out_features=10, bias=True)
)
>>> model.config['layers']
[256, 256, 10]
```

### 1. Subclassing
**File:** `1-input.py` · **Function:** `build_model(nx, layers, activations, keep_prob)`

Build the same architecture, but as your own subclass of `nn.Module` (not `nn.Sequential`) with a hand-written `forward`. It must have the same parameters in the same order, so that copying task 0's weights into it gives identical outputs in eval mode. Raise `ValueError` on mismatched lengths.

### 2. Optimize
**File:** `2-optimize.py` · **Function:** `optimize_model(network, alpha, beta1, beta2, lambtha=0.0)`

Return `(optimizer, loss_fn)`: an Adam optimizer over the network's parameters with learning rate `alpha`, betas `beta1` and `beta2`, and L2 weight decay `lambtha`; and a cross-entropy loss module.

```
>>> opt, loss_fn = optimize_model(model, 0.001, 0.9, 0.999)
>>> type(opt).__name__, type(loss_fn).__name__
('Adam', 'CrossEntropyLoss')
```

### 3. One-hot
**File:** `3-one_hot.py` · **Function:** `one_hot(labels, classes=None)`

Return a float32 one-hot tensor of shape `(m, classes)` (note: examples on rows, unlike the classification project). `labels` may be a list, NumPy array or tensor. If `classes` is `None`, use the largest label + 1.

You won't need one-hot labels for `CrossEntropyLoss` (it takes class indices), but you will for many other losses and metrics.

```
>>> one_hot([2, 0, 1], 4)
tensor([[0., 0., 1., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.]])
```

### 4. Train
**File:** `4-train.py` · **Function:** `train_model(network, optimizer, loss_fn, data, labels, batch_size, epochs, verbose=True, shuffle=False)`

Train with mini-batches from a `DataLoader` (over a `TensorDataset`) for `epochs` epochs. Return a history dict:
- `history["loss"]`: one value per epoch, the average training loss over that epoch's batches (weight each batch by its size).
- `history["accuracy"]`: the training accuracy over the epoch, measured on the batches as you train.

If `verbose`, print one line per epoch: `Epoch {e}/{epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}` (epochs numbered from 1).

```
>>> history = train_model(model, opt, loss_fn, X_train, Y_train, 64, 3)
Epoch 1/3 - loss: 0.5013 - accuracy: 0.8471
Epoch 2/3 - loss: 0.2386 - accuracy: 0.9301
Epoch 3/3 - loss: 0.1846 - accuracy: 0.9452
>>> len(history["loss"])
3
```
(Your numbers will differ.)

### 5. Validate
**File:** `5-train.py` · Add `validation_data=None` (after `epochs`).

If `validation_data` is a tuple `(X_valid, Y_valid)`, evaluate it after each epoch in eval mode without gradients, and record `history["val_loss"]` and `history["val_accuracy"]`. Append ` - val_loss: {:.4f} - val_accuracy: {:.4f}` to the printed line.

### 6. Early stopping
**File:** `6-train.py` · Add `early_stopping=False, patience=0` (after `validation_data`).

Only when `early_stopping` is on **and** there is validation data: keep track of the best (lowest) validation loss so far. An epoch whose validation loss is not strictly lower than the best counts as a non-improving epoch; an improving epoch resets the count. Stop training once the count reaches `patience` (treat a `patience` of 0 like 1).

### 7. Learning rate decay
**File:** `7-train.py` · Add `learning_rate_decay=False, alpha=0.1, decay_rate=1` (after `patience`).

- From now on, also record `history["lr"]`: the learning rate in use during each epoch.
- Only when `learning_rate_decay` is on **and** there is validation data: use a scheduler from `torch.optim.lr_scheduler` so that epoch `e` (counting from 0) trains with learning rate `alpha / (1 + decay_rate * e)`. Step the scheduler once per epoch.

```
>>> h = train_model(..., validation_data=(Xv, Yv), learning_rate_decay=True, alpha=0.01, decay_rate=1, epochs=4)
>>> h["lr"]
[0.01, 0.005, 0.0033333333333333335, 0.0025]
```

### 8. Save the best
**File:** `8-train.py` · Add `save_best=False, filepath=None` (after `decay_rate`).

When `save_best` is on and there is validation data, save the network's `state_dict` to `filepath` every time the validation loss reaches a new best. After training, that file holds the weights from the best epoch, not the last one.

### 9. Save and load a model
**File:** `9-model.py` · **Functions:** `save_model(network, filename)` and `load_model(filename)`

Save and load the **entire** model object (architecture and weights). `load_model` returns the model, loaded onto the CPU. Since this unpickles Python objects, you'll need to opt out of PyTorch's safe `weights_only` default. Add a comment saying why that's only OK for files you trust.

### 10. Save and load weights
**File:** `10-weights.py` · **Functions:** `save_weights(network, filename)` and `load_weights(network, filename)`

Save only the `state_dict`. `load_weights` loads it into the given (already built) network in place, safely (`weights_only=True`, onto the CPU), and returns nothing.

### 11. Save and load a config
**File:** `11-config.py` · **Functions:** `save_config(network, filename)` and `load_config(filename)`

`save_config` writes `network.config` (from task 0) to a JSON file. `load_config` reads it and returns a **new, untrained** model built with task 0's `build_model`.

```
>>> save_config(model, 'config.json')
>>> fresh = load_config('config.json')
>>> fresh.config == model.config
True
```

### 12. Test
**File:** `12-test.py` · **Function:** `test_model(network, data, labels)`

Return `(loss, accuracy)` of the network on the whole set as Python floats, using cross-entropy, in eval mode and without gradients.

### 13. Predict
**File:** `13-predict.py` · **Function:** `predict(network, data)`

Return the predicted class **probabilities** (softmax of the outputs), shape `(m, classes)`, as a CPU tensor, computed in eval mode without gradients.

```
>>> probs = predict(model, X_test[:5])
>>> probs.shape, probs.sum(dim=1)
(torch.Size([5, 10]), tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000]))
```

### 14. Put it together (bonus)
Write a short script (outside the tests) that builds a model for MNIST, trains it with validation, early stopping, LR decay and save-best, then reloads the best weights and reports test accuracy. Aim for 97%+. Try swapping the `LambdaLR` decay for `StepLR` or `ReduceLROnPlateau` and compare.

## Check your work
```bash
pytest 01-supervised-foundations/06-pytorch_training
```
