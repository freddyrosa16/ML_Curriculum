# Convolutional Neural Networks

**Status:** Original + Modernized. Tasks 0-3 (forward and backward passes) are the original NumPy tasks. Tasks 4-5 (LeNet-5) moved from TensorFlow 1 / Keras to PyTorch.

## Why this matters
Convolutional layers are the reason computers can recognize images. In `00-math/06-convolutions_and_pooling` you slid a filter over an image; here you turn that into a real network layer, including the backward pass that lets it learn. Once you've written backprop for a conv layer by hand, PyTorch's `nn.Conv2d` stops being magic. Then you build LeNet-5, one of the first CNNs that worked, in PyTorch.

## Concepts to understand first
- **What does a convolutional layer compute?** Each filter is a small block of weights (height × width × input channels). It slides across the input, and at every position it multiplies element-wise with the patch under it and sums everything to one number, then adds a bias. One filter gives one output channel; `c_new` filters give `c_new` channels.
- **What are "same" and "valid" padding?** "Valid" means no padding, so the output shrinks. "Same" pads with zeros so that (for stride 1) the output keeps the input's height and width. In this project, "same" uses $p_h = \lfloor((h_{prev}-1)\cdot s_h + k_h - h_{prev})/2\rfloor$ zeros on top *and* bottom (same idea for width).
- **How big is the output?** $h_{new} = \lfloor (h_{prev} + 2p_h - k_h)/s_h \rfloor + 1$, and the same for width.
- **What does pooling do?** It shrinks the feature map by summarizing each window with its max (max pooling) or mean (average pooling). It has no weights to learn and works on each channel separately.
- **How does the gradient flow back through a conv layer?** Each output value was built from one input patch and one filter. So in the backward pass, each output's gradient `dZ` flows back to that patch (scaled by the filter weights), and to the filter (scaled by the patch). You add up all those contributions. The bias gradient is just the sum of `dZ`.
- **How does the gradient flow back through pooling?** For max pooling, only the input that was the max gets the gradient; the others get zero. For average pooling, the gradient is shared equally across the window.
- **What is "He" (Kaiming) initialization?** Starting weights drawn from a normal distribution with standard deviation $\sqrt{2/\text{fan\_in}}$, where fan_in is the number of inputs to a unit. It keeps signals from shrinking or exploding through ReLU layers.

## Resources
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) — the clearest written guide, with an animated conv demo
- [CS231n: Backpropagation, intuitions](https://cs231n.github.io/optimization-2/)
- [d2l.ai Chapter 7: Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html) — includes LeNet in PyTorch
- [A guide to convolution arithmetic (Dumoulin & Visin)](https://arxiv.org/abs/1603.07285) — padding/stride/output-size pictures
- [3Blue1Brown: But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)
- [PyTorch docs: `nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) and [Training a classifier tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [LeCun et al. 1998, Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (the LeNet paper, for the curious)

## Rules
- Tasks 0-3: **NumPy only** (`import numpy as np`). Arrays use the "channels last" layout `(m, h, w, c)`.
- Tasks 0-3: you may loop over output height, output width and channels/filters, but **not** over the examples `m` (vectorize across the batch). Task 3 is the one exception if you need it: looping over examples is accepted there, but try without.
- Tasks 4-5: **PyTorch** (`import torch`, `import torch.nn as nn`). PyTorch uses "channels first" `(N, C, H, W)`.
- Write each task in the file named in the task, inside this folder.

## Tasks

### 0. Convolutional forward prop
**File:** `0-conv_forward.py` · **Signature:** `def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):`

Forward pass of a conv layer.
- `A_prev`: `(m, h_prev, w_prev, c_prev)` input.
- `W`: `(kh, kw, c_prev, c_new)` filters. `b`: `(1, 1, 1, c_new)` biases.
- `activation`: a function applied to the result (e.g. ReLU).
- `padding`: `"same"` or `"valid"` (see Concepts for the "same" formula). `stride`: `(sh, sw)`.
- **Returns:** the activated output, shape `(m, h_new, w_new, c_new)`.

```python
>>> A = conv_forward(images, W, b, relu, padding="valid")   # images: (50000, 28, 28, 1), W: (3, 3, 1, 2)
>>> A.shape
(50000, 26, 26, 2)
```

### 1. Pooling forward prop
**File:** `1-pool_forward.py` · **Signature:** `def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):`

- `A_prev`: `(m, h_prev, w_prev, c_prev)`. `kernel_shape`: `(kh, kw)`. `stride`: `(sh, sw)`. No padding.
- `mode`: `'max'` or `'avg'`.
- **Returns:** the pooled output `(m, h_new, w_new, c_prev)`.

```python
>>> pool_forward(images, (2, 2), stride=(2, 2)).shape   # images: (50000, 28, 28, 2)
(50000, 14, 14, 2)
```

### 2. Convolutional back prop
**File:** `2-conv_backward.py` · **Signature:** `def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):`

- `dZ`: `(m, h_new, w_new, c_new)`, gradient of the loss with respect to the conv output **before** the activation.
- `A_prev`, `W`, `b`, `padding`, `stride`: exactly as in task 0 (use the same padding rule as your forward pass).
- **Returns:** `(dA_prev, dW, db)` with the same shapes as `A_prev`, `W` and `b`.

```python
>>> dA_prev, dW, db = conv_backward(dZ, A_prev, W, b, padding="valid")  # A_prev: (10, 28, 28, 1), W: (3, 3, 1, 2)
>>> dA_prev.shape, dW.shape, db.shape
((10, 28, 28, 1), (3, 3, 1, 2), (1, 1, 1, 2))
```
The tests compare your gradients with PyTorch's autograd, so they must be exact (up to float rounding).

### 3. Pooling back prop
**File:** `3-pool_backward.py` · **Signature:** `def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):`

- `dA`: `(m, h_new, w_new, c)`, gradient with respect to the pooling output.
- `A_prev`: `(m, h_prev, w_prev, c)`, the pooling input. `kernel_shape`, `stride`, `mode`: as in task 1.
- **Returns:** `dA_prev`, same shape as `A_prev`.

```python
>>> pool_backward(dA, A_prev, (2, 2), stride=(2, 2), mode='avg').shape   # A_prev: (10, 28, 28, 2)
(10, 28, 28, 2)
```

### 4. LeNet-5 in PyTorch
**File:** `4-lenet5.py` · **Signature:** `def lenet5():`

Return an `nn.Module` (an `nn.Sequential` is fine) that maps `(N, 1, 28, 28)` images to `(N, 10)` **logits** (no softmax: `nn.CrossEntropyLoss` applies it for you). Layers, in order:
1. Conv: 6 filters, 5×5, "same" padding, ReLU
2. Max pool 2×2, stride 2
3. Conv: 16 filters, 5×5, no padding, ReLU
4. Max pool 2×2, stride 2
5. Flatten, then fully connected 120 units, ReLU
6. Fully connected 84 units, ReLU
7. Fully connected 10 units (output)

Initialize every conv/linear weight with He normal (Kaiming normal for ReLU) and every bias with zeros.

```python
>>> model = lenet5()
>>> model(torch.zeros(4, 1, 28, 28)).shape
torch.Size([4, 10])
>>> sum(p.numel() for p in model.parameters())
61706
```

### 5. Train LeNet-5
**File:** `5-train_lenet5.py` · **Signature:** `def train_lenet5(model, X, Y, epochs=5, batch_size=32, lr=1e-3, device=None):`

Write the training loop yourself.
- `X`: images `(N, 1, 28, 28)` (NumPy array or tensor, floats). `Y`: integer labels `(N,)`.
- Use Adam with learning rate `lr` and cross-entropy loss. Shuffle every epoch and go through the data in mini-batches of `batch_size`.
- `device`: if `None`, pick `"cuda"`, then `"mps"`, then `"cpu"`, whichever is available.
- **Returns:** a dict `{"loss": [...], "accuracy": [...]}` with one average training loss and one training accuracy (fraction between 0 and 1) per epoch.

```python
>>> history = train_lenet5(lenet5(), X_train, Y_train, epochs=5)
>>> [round(a, 3) for a in history["accuracy"]]   # on MNIST; your numbers will differ a little
[0.943, 0.981, 0.986, 0.989, 0.991]
```
Then evaluate on the MNIST test set yourself: you should get about 98-99%.

### 6. Write-up (optional)
Write a short blog post (or notebook) explaining convolution, padding and pooling to someone who knows basic neural networks. Pictures help.

## Data
- Tasks 0-3 tests use random arrays; no download needed.
- For tasks 4-5 (and to look at your conv outputs with matplotlib), use MNIST from torchvision: `torchvision.datasets.MNIST(root="data", train=True, download=True)`. It downloads about 50 MB. Its `.data` attribute holds the images as a `(60000, 28, 28)` uint8 tensor and `.targets` the labels. Divide by 255 and add a channel axis to get `(N, 1, 28, 28)` floats for PyTorch, or `(N, 28, 28, 1)` for your NumPy functions.

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 02-vision/01-cnn
```
Tasks 0-3 tests use PyTorch only as a reference to compare against, so you need `torch` installed.
