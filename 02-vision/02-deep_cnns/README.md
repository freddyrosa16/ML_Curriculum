# Deep CNNs

**Status:** Modernized. Same architectures as the original (Inception, ResNet-50, DenseNet-121), built in PyTorch instead of Keras. Blocks are now `nn.Module` classes, and the tests check your parameter counts against `torchvision`'s own models.

## Why this matters
Stacking more conv layers stopped helping around 2014: deeper networks got *worse* because the signal and the gradients faded on the way through. Three ideas fixed that and are still everywhere in modern vision models: run several filter sizes side by side (Inception), add shortcut connections so a block only has to learn a correction (ResNet), and let every layer see all earlier feature maps (DenseNet). Building them yourself teaches you to read any architecture diagram and turn it into code.

## Concepts to understand first
- **What is a 1×1 convolution for?** It mixes channels at each pixel without looking at neighbours. It's a cheap way to shrink (or grow) the channel count, e.g. squeeze 256 channels to 64 before an expensive 3×3 conv.
- **What is an Inception block?** Four branches run on the same input in parallel: a 1×1 conv; a 1×1 then 3×3; a 1×1 then 5×5; and a 3×3 max pool followed by a 1×1. Their outputs (same height and width) are stacked along the channel axis.
- **What is a residual (skip) connection?** Instead of learning a mapping $H(x)$ directly, a block learns $F(x)$ and outputs $F(x) + x$. If the best thing to do is "nothing", the block just pushes $F$ to zero. Gradients also flow straight through the `+ x` path, which makes very deep networks trainable.
- **Identity vs projection block?** If the main path keeps the shape, you can add `x` as is (identity block). If it changes the channel count or downsamples with a stride, the shortcut needs its own 1×1 conv (with the same stride) to match shapes (projection block).
- **What is batch normalization?** It normalizes each channel over the batch to mean 0 and variance 1, then applies a learned scale and shift. It speeds up and stabilizes training. Because BN already has a learned shift, the conv right before it doesn't need a bias.
- **What is a dense block?** Each layer takes the concatenation of *all* previous outputs in the block and adds `growth_rate` new channels. So channels grow linearly: `in + layers × growth_rate`.
- **What is a transition layer?** It sits between dense blocks: a 1×1 conv shrinks channels by a `compression` factor (0.5 halves them) and a 2×2 average pool halves the height and width.
- **Why do my parameter counts matter?** Counting parameters is the quickest way to check you built the architecture correctly. A conv layer has $k_h \cdot k_w \cdot c_{in} \cdot c_{out}$ weights (+ $c_{out}$ biases if it has them); a BN layer has $2c$.

## Resources
- [d2l.ai Chapter 8: Modern CNNs](https://d2l.ai/chapter_convolutional-modern/index.html) — GoogLeNet, ResNet and DenseNet in PyTorch, with diagrams
- [Going Deeper with Convolutions (Szegedy et al., 2014)](https://arxiv.org/abs/1409.4842) — Inception; Table 1 lists every filter count
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385) — ResNet; Table 1 has ResNet-50
- [Densely Connected Convolutional Networks (Huang et al., 2016)](https://arxiv.org/abs/1608.06993) — DenseNet; Table 1 has DenseNet-121
- [CS231n: CNN architectures](https://cs231n.github.io/convolutional-networks/#case)
- [PyTorch docs: `nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), [`nn.BatchNorm2d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [torchvision models](https://pytorch.org/vision/stable/models.html) — compare your network with `torchvision.models.resnet50(weights=None)` by printing both

## Rules
- PyTorch only: `import torch`, `import torch.nn as nn`. Don't import `torchvision.models` in your solutions (the tests use it only as a reference).
- Layout is `(N, C, H, W)`. Final networks return **logits** (no softmax).
- Every conv uses He (Kaiming) normal initialization if you set one yourself; PyTorch's default init is also accepted.
- Inside ResNet and DenseNet, convs that are followed by batch norm have **no bias** (`bias=False`). Inception has no batch norm, so its convs keep their bias.
- Later tasks reuse earlier ones. Import them with `__import__('0-inception_block').InceptionBlock` (file names start with digits, so a normal `import` won't work).

## Tasks

### 0. Inception block
**File:** `0-inception_block.py` · **Class:** `InceptionBlock(nn.Module)` · **Init:** `__init__(self, in_channels, filters)`

- `filters` is `(F1, F3R, F3, F5R, F5, FPP)`: filters in the 1×1 branch; the 1×1 "reduce" before the 3×3, and the 3×3; the 1×1 reduce before the 5×5, and the 5×5; and the 1×1 after the max pool.
- All convs are followed by ReLU. Use padding so every branch keeps the input's height and width. The max pool is 3×3 with stride 1.
- Store the total output channel count in `self.out_channels`.
- `forward(x)` returns the four branches concatenated along the channel axis.

```python
>>> block = InceptionBlock(192, (64, 96, 128, 16, 32, 32))
>>> block(torch.zeros(1, 192, 28, 28)).shape
torch.Size([1, 256, 28, 28])
>>> block.out_channels
256
```

### 1. Inception network
**File:** `1-inception_network.py` · **Signature:** `def inception_network(num_classes=1000):`

Build GoogLeNet from Table 1 of the Inception paper (skip the two auxiliary classifiers). Input `(N, 3, 224, 224)`, output `(N, num_classes)`. In order:
1. Conv 7×7, 64 filters, stride 2 + ReLU; max pool 3×3 stride 2
2. Conv 1×1, 64 + ReLU; conv 3×3, 192 + ReLU; max pool 3×3 stride 2
3. Inception 3a `(64, 96, 128, 16, 32, 32)`, 3b `(128, 128, 192, 32, 96, 64)`; max pool 3×3 stride 2
4. Inception 4a `(192, 96, 208, 16, 48, 64)`, 4b `(160, 112, 224, 24, 64, 64)`, 4c `(128, 128, 256, 24, 64, 64)`, 4d `(112, 144, 288, 32, 64, 64)`, 4e `(256, 160, 320, 32, 128, 128)`; max pool 3×3 stride 2
5. Inception 5a `(256, 160, 320, 32, 128, 128)`, 5b `(384, 192, 384, 48, 128, 128)`
6. Average pool 7×7 (stride 1), flatten, dropout 0.4, linear to `num_classes`

Use padding so the feature map goes 224 → 112 → 56 → 28 → 14 → 7.

```python
>>> model = inception_network()
>>> model(torch.zeros(2, 3, 224, 224)).shape
torch.Size([2, 1000])
>>> sum(p.numel() for p in model.parameters())
6998552
```

### 2. Identity block
**File:** `2-identity_block.py` · **Class:** `IdentityBlock(nn.Module)` · **Init:** `__init__(self, in_channels, filters)`

- `filters` is `(F11, F3, F12)`. Main path: 1×1 conv (F11) → BN → ReLU → 3×3 conv (F3, same size) → BN → ReLU → 1×1 conv (F12) → BN.
- Add the block's input to the main path, then ReLU.
- The shapes must match, so `F12` has to equal `in_channels`: raise a `ValueError` if it doesn't.

```python
>>> IdentityBlock(256, (64, 64, 256))(torch.zeros(1, 256, 56, 56)).shape
torch.Size([1, 256, 56, 56])
```

### 3. Projection block
**File:** `3-projection_block.py` · **Class:** `ProjectionBlock(nn.Module)` · **Init:** `__init__(self, in_channels, filters, s=2)`

- Same main path as the identity block, except the first 1×1 conv uses stride `s`.
- Shortcut: a 1×1 conv with F12 filters and stride `s`, then BN.
- Output: ReLU(main + shortcut).

```python
>>> ProjectionBlock(64, (64, 64, 256), s=1)(torch.zeros(1, 64, 56, 56)).shape
torch.Size([1, 256, 56, 56])
>>> ProjectionBlock(256, (128, 128, 512))(torch.zeros(1, 256, 56, 56)).shape
torch.Size([1, 512, 28, 28])
```
(The original paper puts the stride on the first 1×1 conv, as here. torchvision moves it to the 3×3 conv, "ResNet v1.5". The parameter count is the same either way.)

### 4. ResNet-50
**File:** `4-resnet50.py` · **Signature:** `def resnet50(num_classes=1000):`

Input `(N, 3, 224, 224)`, output logits `(N, num_classes)`:
1. Conv 7×7, 64 filters, stride 2 (no bias) → BN → ReLU → max pool 3×3 stride 2 (padded so 112 → 56)
2. Stage 1: projection block `(64, 64, 256)` with **s=1**, then 2 identity blocks
3. Stage 2: projection `(128, 128, 512)` (s=2), then 3 identity blocks
4. Stage 3: projection `(256, 256, 1024)`, then 5 identity blocks
5. Stage 4: projection `(512, 512, 2048)`, then 2 identity blocks
6. Global average pool, flatten, linear to `num_classes`

```python
>>> model = resnet50()
>>> sum(p.numel() for p in model.parameters())   # identical to torchvision's resnet50
25557032
```

### 5. Dense block
**File:** `5-dense_block.py` · **Class:** `DenseBlock(nn.Module)` · **Init:** `__init__(self, in_channels, growth_rate, layers)`

- Build `layers` bottleneck layers. Each one: BN → ReLU → 1×1 conv with `4 * growth_rate` filters → BN → ReLU → 3×3 conv with `growth_rate` filters (same size). No conv biases.
- Each layer's input is everything so far concatenated along channels; its output is concatenated on.
- Set `self.out_channels` to the final channel count.

```python
>>> block = DenseBlock(64, growth_rate=32, layers=6)
>>> block(torch.zeros(1, 64, 56, 56)).shape
torch.Size([1, 256, 56, 56])
>>> block.out_channels
256
```

### 6. Transition layer
**File:** `6-transition_layer.py` · **Class:** `TransitionLayer(nn.Module)` · **Init:** `__init__(self, in_channels, compression)`

- BN → ReLU → 1×1 conv with `int(in_channels * compression)` filters (no bias) → 2×2 average pool, stride 2.
- Set `self.out_channels`.

```python
>>> t = TransitionLayer(256, 0.5)
>>> t(torch.zeros(1, 256, 56, 56)).shape
torch.Size([1, 128, 28, 28])
```

### 7. DenseNet-121
**File:** `7-densenet121.py` · **Signature:** `def densenet121(growth_rate=32, compression=0.5, num_classes=1000):`

Input `(N, 3, 224, 224)`, output logits:
1. Conv 7×7 with `2 * growth_rate` filters, stride 2 (no bias) → BN → ReLU → max pool 3×3 stride 2
2. Dense block (6 layers) → transition → dense block (12) → transition → dense block (24) → transition → dense block (16)
3. BN → ReLU → global average pool → flatten → linear to `num_classes`

```python
>>> model = densenet121()
>>> sum(p.numel() for p in model.parameters())   # identical to torchvision's densenet121
7978856
```
(The original Holberton version had no compression by default and an extra batch norm on the raw input. This version follows the paper's DenseNet-BC and torchvision.)

### 8. Write-up (optional)
In your own words: why did plain deep networks get worse with depth, and how does each of these three architectures address it? Half a page is enough.

## Data
None. These tasks only build architectures, and every test uses tensors of zeros or random numbers. If you want to train one, CIFAR-10 from `torchvision.datasets.CIFAR10` is the usual small dataset (expect slow training on a laptop CPU; Apple MPS helps).

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 02-vision/02-deep_cnns
```
