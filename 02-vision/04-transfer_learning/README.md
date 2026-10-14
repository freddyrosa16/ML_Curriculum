# Transfer Learning

**Status:** Modernized. Same goal as the original (reach at least 87% on CIFAR-10 by reusing an ImageNet model), done with a torchvision ResNet-50 in PyTorch instead of a Keras application. The single big script is now split into small, testable steps, including the "run the frozen backbone once and cache the features" trick.

## Why this matters
Training a big vision model from scratch takes millions of images and days of GPU time. But a model trained on ImageNet has already learned general features (edges, textures, shapes, object parts) that are useful for almost any image task. Transfer learning reuses those features and only trains a small new part for your problem. It's how most real-world vision models are built, and it lets you get strong results on a laptop.

## Concepts to understand first
- **What is a pretrained model?** A network whose weights were already trained on a large dataset (here ImageNet: 1.28 million photos in 1000 classes). torchvision downloads those weights for you.
- **Backbone vs head?** The backbone is everything up to the last layer: it turns an image into a feature vector (2048 numbers for ResNet-50). The head is the final layer that turns features into class scores. ImageNet's head predicts 1000 classes; you replace it with one for CIFAR-10's 10.
- **What does "freezing" mean?** Setting `requires_grad = False` on the backbone's parameters, so the optimizer never changes them. Only the head learns. Fewer trainable parameters means faster training and less overfitting.
- **Why cache features?** If the backbone is frozen, it gives the same output for the same image every epoch. So you can run it once over the whole dataset, save the feature vectors, and then train the head on those vectors in seconds. This is often called a "linear probe".
- **Why resize and normalize?** The pretrained model expects what it saw in training: roughly 224×224 RGB images, normalized per channel with ImageNet's mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)` (on the 0-1 scale). CIFAR-10 images are only 32×32, so upscaling helps the features a lot.
- **What are `train()` and `eval()` modes?** Layers like dropout and batch norm behave differently while training. For feature extraction (and testing), call `model.eval()` and wrap the work in `torch.no_grad()` so no gradients are tracked.
- **What is fine-tuning?** After training the head, you can optionally unfreeze the last block(s) of the backbone and keep training with a small learning rate. It usually adds a few points of accuracy but costs much more compute.

## Resources
- [PyTorch tutorial: Transfer Learning for Computer Vision](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [torchvision: Models and pre-trained weights](https://pytorch.org/vision/stable/models.html) — how `weights=` works and each model's preprocessing
- [CS231n: Transfer Learning notes](https://cs231n.github.io/transfer-learning/)
- [d2l.ai 14.2: Fine-Tuning](https://d2l.ai/chapter_computer-vision/fine-tuning.html)
- [How transferable are features in deep neural networks? (Yosinski et al., 2014)](https://arxiv.org/abs/1411.1792)
- [CIFAR-10 dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)

## Rules
- PyTorch + torchvision: `import torch`, `import torch.nn as nn`, `from torchvision import models, datasets`. NumPy is fine for data handling.
- Write device-agnostic code: use `"cuda"` if available, else `"mps"` (Apple Silicon), else `"cpu"`.
- Each function must also work with `weights=None` (random weights) so the tests can run without downloads.
- Import earlier tasks with `__import__('0-transfer').preprocess_data` and so on.

## Tasks

### 0. Preprocess
**File:** `0-transfer.py` · **Signature:** `def preprocess_data(X, Y, size=224):`

- `X`: NumPy `uint8` array `(m, 32, 32, 3)` (CIFAR-10's raw format: channels last). `Y`: labels, shape `(m,)` (array or list).
- Return `(X_p, Y_p)`:
  - `X_p`: `float32` tensor `(m, 3, size, size)`: channels first, scaled to [0, 1], resized to `size × size` (bilinear), then normalized with the ImageNet mean and std above.
  - `Y_p`: `int64` tensor `(m,)`.

```python
>>> X_p, Y_p = preprocess_data(train.data[:8], train.targets[:8])
>>> X_p.shape, X_p.dtype, Y_p.dtype
(torch.Size([8, 3, 224, 224]), torch.float32, torch.int64)
```

### 1. Build the model
**File:** `1-build_model.py` · **Signature:** `def build_model(num_classes=10, weights="DEFAULT"):`

- Load torchvision's ResNet-50 with the given `weights` (`"DEFAULT"` downloads the best ImageNet weights, about 100 MB; `None` gives random weights).
- Freeze every parameter, then replace the final fully connected layer (`model.fc`) with a new `nn.Linear` from 2048 features to `num_classes`. The new layer must be the **only** trainable part.

```python
>>> model = build_model(10, weights=None)
>>> sum(p.numel() for p in model.parameters() if p.requires_grad)
20490
>>> model(torch.zeros(2, 3, 224, 224)).shape
torch.Size([2, 10])
```

### 2. Extract (cache) features
**File:** `2-extract_features.py` · **Signature:** `def extract_features(backbone, X, batch_size=128, device=None, transform=None):`

- `backbone`: any `nn.Module` (for ResNet-50, everything except `fc`).
- `X`: a dataset you can slice (`X[i:i + batch_size]`), e.g. a NumPy array or tensor.
- `transform`: optional function applied to each raw batch before the backbone (e.g. your `preprocess_data`, so you never hold 50,000 upscaled images in memory at once).
- Run the backbone in eval mode, without gradients, batch by batch on `device` (pick one automatically if `None`).
- **Returns:** a CPU float tensor `(N, D)` (flatten each output to a vector).

```python
>>> feats = extract_features(backbone, train.data[:256], transform=to_tensor)
>>> feats.shape
torch.Size([256, 2048])
```

### 3. Train the head
**File:** `3-train_head.py` · **Signature:** `def train_head(features, labels, num_classes=10, epochs=20, lr=1e-3, batch_size=256, device=None, seed=0):`

- Create an `nn.Linear(D, num_classes)` and train it on the cached features with Adam and cross-entropy, shuffled mini-batches. Seed with `torch.manual_seed(seed)`.
- **Returns:** the trained `nn.Linear`, on the CPU.

```python
>>> head = train_head(f_train, y_train, epochs=30)
>>> head
Linear(in_features=2048, out_features=10, bias=True)
```

### 4. Train on CIFAR-10 (goal: at least 87%)
**File:** `4-train_cifar10.py` · **Signature:** `def main(data_root="data", out_path="cifar10.pt", size=224):`

Put it all together and run it with `if __name__ == "__main__": main()`:
1. Load CIFAR-10 train and test sets with `torchvision.datasets.CIFAR10`.
2. Build the model with pretrained weights. Extract features for both sets (the backbone is the model without its `fc` layer).
3. Train the head on the training features, then measure accuracy on the test features. Print it.
4. Put the trained head back into the model as `model.fc` and save `model.state_dict()` to `out_path`. Return the test accuracy.

Target: **test accuracy ≥ 87%**. A frozen ResNet-50 at 224×224 typically lands around 88-90%.

Expect feature extraction to be the slow part: 60,000 images through ResNet-50 takes roughly 10-30 minutes on an Apple Silicon GPU (MPS) and over an hour on CPU. Tips: save the features with `torch.save` so you only extract them once, and if it's too slow, try `size=160` first.

```
$ python 4-train_cifar10.py
test accuracy: 0.8934
```

### 5. Fine-tuning (bonus)
Unfreeze the last ResNet stage (`model.layer4`) and train it together with the head for a few epochs with a small learning rate (e.g. 1e-4), using random flips and crops from the data augmentation project. How much does it help, and how much longer does it take?

### 6. Write-up (optional)
A short post: what transfer learning is, what you tried, what worked and what didn't, and your final accuracy.

## Data
- CIFAR-10: `torchvision.datasets.CIFAR10(root="data", train=True, download=True)` (and `train=False` for the test set). About 170 MB. `.data` is a `(50000, 32, 32, 3)` uint8 array and `.targets` a list of labels.
- Pretrained ResNet-50 weights download automatically (about 100 MB) the first time you call `models.resnet50(weights="DEFAULT")`.
- The non-slow tests download nothing.

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 02-vision/04-transfer_learning -m "not slow"
pytest 02-vision/04-transfer_learning -m slow   # after task 4 has saved cifar10.pt here
```
The slow test loads your `cifar10.pt`, downloads the CIFAR-10 test set and checks you reached 87%.
