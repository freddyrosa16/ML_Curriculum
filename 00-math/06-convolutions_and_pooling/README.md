# Convolutions and Pooling

**Status:** Original

## Why this matters
Convolutional neural networks (CNNs) power image classification, object detection and much of computer vision. Their core operation is simple: slide a small grid of numbers (a *kernel*) across an image and take a weighted sum at every position. Writing it by hand with NumPy, batch and all, makes Stage 2's CNN projects much easier, because you'll know exactly what `Conv2d` and `MaxPool2d` do and how padding and stride change the output shape.

## Concepts to understand first
- **What is a convolution in deep learning?** Place the kernel on a patch of the image, multiply them element by element, and sum to get one output number. Slide and repeat. (Strictly speaking that's *cross-correlation*; the mathematical convolution flips the kernel first. Deep learning keeps the name anyway.)
- **What is a kernel (filter)?** A small matrix, e.g. 3x3. Different kernels detect different patterns: the kernel with columns of 1, 0, -1 responds to vertical edges. In a CNN the kernel values are learned.
- **What is padding, and what are "valid" and "same"?** Padding adds a border of zeros around the image. *Valid* means no padding, so the output shrinks. *Same* adds just enough padding that, with stride 1, the output has the same size as the input. For an odd kernel of height $k_h$ that's $\lfloor k_h/2 \rfloor$ rows of zeros on the top and on the bottom.
- **What is stride?** How far the kernel jumps between positions. Stride 2 skips every other position and roughly halves the output size.
- **How big is the output?** For input height $h$, kernel height $k_h$, padding $p_h$ on each side and stride $s_h$: $\left\lfloor \frac{h + 2p_h - k_h}{s_h} \right\rfloor + 1$. Same formula for the width.
- **What are channels?** A color image has 3 values per pixel (red, green, blue), so its shape is (height, width, 3). A kernel for it has shape (kh, kw, 3) and sums over all channels, producing a single 2D output. Using `nc` different kernels produces `nc` output channels.
- **What is pooling?** Sliding a window and keeping one summary per position: the maximum (*max pooling*) or the average (*average pooling*). There are no weights. It shrinks the image and makes features a little more robust to small shifts.
- **Why process a batch of images at once?** Images come in stacks with shape (m, h, w) or (m, h, w, c). If you loop over output *positions* and let NumPy handle all m images (and channels) at each position, you get one fast vectorized operation per position instead of millions of Python steps.

## Resources
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [Dive into Deep Learning: Convolutions for images, padding and stride, pooling](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
- [Dumoulin & Visin: A guide to convolution arithmetic for deep learning (paper)](https://arxiv.org/abs/1603.07285)
- [conv_arithmetic animations (GitHub)](https://github.com/vdumoulin/conv_arithmetic)
- [3Blue1Brown: But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)
- [NumPy: `np.pad`](https://numpy.org/doc/stable/reference/generated/numpy.pad.html)

## Rules
- `import numpy as np` only.
- Tasks 0-4 and 6: **at most two `for` loops** in total, and no other loops (no `while`, no comprehensions). Task 5: at most three. The tests count them.
- Inputs: grayscale images have shape `(m, h, w)`, color images `(m, h, w, c)`. Return a NumPy array.
- Pad with zeros.
- For `padding='same'`, pad by `kh // 2` rows (top and bottom) and `kw // 2` columns (left and right). The tests only use odd-sized kernels with `'same'`.

## Data
The tests use random arrays, so no download is needed. To *see* what your edge-detection kernel does, try it on real images: MNIST digits (e.g. via `torchvision.datasets.MNIST` once you install PyTorch in Stage 1) or any photo loaded as an array with `matplotlib.image.imread`, then show the input and output side by side with `plt.imshow`.

## Tasks

### 0. Valid Convolution
**File:** `0-convolve_grayscale_valid.py` · **Signature:** `convolve_grayscale_valid(images, kernel)`

`images`: shape `(m, h, w)`. `kernel`: shape `(kh, kw)`. Return the valid convolution of every image, shape `(m, h - kh + 1, w - kw + 1)`.
```
images.shape = (50000, 28, 28), kernel = the 3x3 vertical-edge kernel
convolve_grayscale_valid(images, kernel).shape  ->  (50000, 26, 26)
one 4x4 image holding 0..15 with a 2x2 kernel of ones  ->  [[10 14 18] [26 30 34] [42 46 50]]
```

### 1. Same Convolution
**File:** `1-convolve_grayscale_same.py` · **Signature:** `convolve_grayscale_same(images, kernel)`

Same inputs; the output has the same height and width as the input.
```
convolve_grayscale_same(<(50000, 28, 28)>, <3x3>).shape  ->  (50000, 28, 28)
```

### 2. Convolution with Padding
**File:** `2-convolve_grayscale_padding.py` · **Signature:** `convolve_grayscale_padding(images, kernel, padding)`

`padding` is a tuple `(ph, pw)`: pad `ph` rows on the top **and** bottom, `pw` columns on the left **and** right.
```
convolve_grayscale_padding(<(50000, 28, 28)>, <3x3>, (2, 4)).shape  ->  (50000, 30, 34)
```

### 3. Strided Convolution
**File:** `3-convolve_grayscale.py` · **Signature:** `convolve_grayscale(images, kernel, padding='same', stride=(1, 1))`

`padding` is `'same'`, `'valid'` or a tuple `(ph, pw)`. `stride` is `(sh, sw)`.
```
convolve_grayscale(<(50000, 28, 28)>, <3x3>, padding='valid', stride=(2, 2)).shape  ->  (50000, 13, 13)
```

### 4. Convolution with Channels
**File:** `4-convolve_channels.py` · **Signature:** `convolve_channels(images, kernel, padding='same', stride=(1, 1))`

`images`: shape `(m, h, w, c)`. `kernel`: shape `(kh, kw, c)`. Sum over the channels, so the output has shape `(m, out_h, out_w)`.
```
convolve_channels(<(10000, 32, 32, 3)>, <(3, 3, 3)>, padding='valid').shape  ->  (10000, 30, 30)
```

### 5. Multiple Kernels
**File:** `5-convolve.py` · **Signature:** `convolve(images, kernels, padding='same', stride=(1, 1))`

`kernels`: shape `(kh, kw, c, nc)`, i.e. `nc` separate kernels. Output shape `(m, out_h, out_w, nc)`, where `[..., i]` is the result of kernel `i`.
```
convolve(<(10000, 32, 32, 3)>, <(3, 3, 3, 3)>, padding='valid').shape  ->  (10000, 30, 30, 3)
```

### 6. Pooling
**File:** `6-pool.py` · **Signature:** `pool(images, kernel_shape, stride, mode='max')`

`images`: shape `(m, h, w, c)`. `kernel_shape`: `(kh, kw)`. `stride`: `(sh, sw)`. `mode`: `'max'` or `'avg'`. No padding. Pool each channel separately, so the output has shape `(m, out_h, out_w, c)`.
```
pool(<(10000, 32, 32, 3)>, (2, 2), (2, 2), mode='avg').shape  ->  (10000, 16, 16, 3)
one 4x4 single-channel image holding 0..15, (2, 2) window, stride (2, 2):
  max -> [[5 7] [13 15]]     avg -> [[2.5 4.5] [10.5 12.5]]
```

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 00-math/06-convolutions_and_pooling
```
