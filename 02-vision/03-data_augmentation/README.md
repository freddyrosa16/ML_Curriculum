# Data Augmentation

**Status:** Modernized. Same six augmentations as the original, done with `torchvision.transforms.v2` on PyTorch tensors instead of `tf.image`. The "automation" blog task is now an optional write-up, and there's a new task that combines everything into a training pipeline.

## Why this matters
A model only learns what its training data shows it. If every cat photo is upright and well lit, it may fail on a tilted, dim one. Data augmentation creates new, slightly changed copies of your images (flipped, cropped, recoloured) each time they're used, so the model learns that those changes don't matter. It's one of the cheapest ways to reduce overfitting, and it's in practically every vision training script.

## Concepts to understand first
- **How is an image stored in PyTorch?** As a tensor of shape `(C, H, W)`: channels (3 for RGB), height, width. Pixels are either `uint8` in 0..255 or floats in 0..1. torchvision's functions accept both and keep the dtype.
- **What is data augmentation?** Applying random, label-preserving changes to training images. A horizontally flipped dog is still a dog. Augmentation is applied on the fly, so the model sees a slightly different version every epoch.
- **Which changes are "label-preserving"?** It depends on the task. Flipping is fine for animals but not for digits ("b" vs "d") or text. Always ask: would a human still give it the same label?
- **What do brightness, contrast and hue mean?** Brightness scales all pixel values (darker/lighter). Contrast pushes pixels toward or away from the image's average gray level. Hue rotates colours around the colour wheel (red → yellow → green ...) while keeping brightness and saturation.
- **Why only augment training data?** Validation and test images should look like real inputs, so you measure honest performance. You usually only resize/normalize them.
- **What is `transforms.v2`?** torchvision's current transform API. It works on tensors, PIL images, and also on bounding boxes and masks together with the image (handy for detection). There's a functional form (`torchvision.transforms.v2.functional`, deterministic: you pass the exact angle/factor) and a class form (`v2.RandomCrop`, `v2.ColorJitter`, random: it draws parameters for you).

## Resources
- [torchvision: Transforming and augmenting images](https://pytorch.org/vision/stable/transforms.html) — the official guide to `transforms.v2`
- [torchvision: Getting started with transforms v2 (example gallery)](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html)
- [torchvision: Illustration of transforms](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html) — pictures of every transform
- [d2l.ai 14.1: Image Augmentation](https://d2l.ai/chapter_computer-vision/image-augmentation.html)
- [A survey on Image Data Augmentation for Deep Learning (Shorten & Khoshgoftaar, 2019)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)
- [Wikipedia: HSL and HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) — what "hue" actually is

## Rules
- Use PyTorch and torchvision: `import torch`, `from torchvision.transforms import v2`, `from torchvision.transforms.v2 import functional as F`.
- Every function takes an image tensor of shape `(C, H, W)` (dtype `uint8` or float in [0, 1]) and returns a tensor with the same dtype.
- Random functions must use PyTorch's random number generator (so `torch.manual_seed` makes them reproducible). Don't use Python's `random` or NumPy's.

## Tasks

### 0. Flip
**File:** `0-flip.py` · **Signature:** `def flip_image(image):`

Return the image mirrored left-to-right.
```python
>>> img = torch.arange(6, dtype=torch.uint8).reshape(1, 2, 3)
>>> flip_image(img)
tensor([[[2, 1, 0],
         [5, 4, 3]]], dtype=torch.uint8)
```

### 1. Crop
**File:** `1-crop.py` · **Signature:** `def crop_image(image, size):`

Return a random crop of `size = (h, w)`, taken from a random position inside the image (no padding).
```python
>>> crop_image(torch.zeros(3, 224, 224, dtype=torch.uint8), (200, 200)).shape
torch.Size([3, 200, 200])
```

### 2. Rotate
**File:** `2-rotate.py` · **Signature:** `def rotate_image(image):`

Rotate the image 90° **counter-clockwise**. For a non-square image, height and width swap, and no pixels may be lost or blurred.
```python
>>> img = torch.arange(6, dtype=torch.uint8).reshape(1, 2, 3)
>>> rotate_image(img)
tensor([[[2, 5],
         [1, 4],
         [0, 3]]], dtype=torch.uint8)
```

### 3. Contrast
**File:** `3-contrast.py` · **Signature:** `def change_contrast(image, lower, upper):`

Change the contrast by a random factor drawn uniformly from `[lower, upper]`. A factor of 1 leaves the image unchanged, 0 gives a flat gray image, above 1 increases contrast.
```python
>>> torch.manual_seed(0)
>>> out = change_contrast(img, 0.5, 3.0)   # img: (3, 64, 64) uint8
>>> out.shape, out.dtype
(torch.Size([3, 64, 64]), torch.uint8)
```

### 4. Brightness
**File:** `4-brightness.py` · **Signature:** `def change_brightness(image, max_delta):`

Multiply the brightness by a random factor drawn uniformly from `[1 - max_delta, 1 + max_delta]` (this is how torchvision defines brightness; the original TensorFlow task *added* a random delta instead). Values stay within the valid range.
```python
>>> torch.manual_seed(0)
>>> change_brightness(img, 0.3).shape   # img: (3, 64, 64)
torch.Size([3, 64, 64])
```

### 5. Hue
**File:** `5-hue.py` · **Signature:** `def change_hue(image, delta):`

Shift the hue of an RGB image by exactly `delta`, a number in `[-0.5, 0.5]` (a fraction of a full turn of the colour wheel). Raise a `ValueError` if `delta` is outside that range.
```python
>>> change_hue(img, 0.0).shape    # img: (3, 64, 64); 0 changes nothing
torch.Size([3, 64, 64])
```

### 6. A training pipeline
**File:** `6-pipeline.py` · **Signature:** `def augmentation_pipeline(size):`

Return one callable transform (e.g. a `v2.Compose`) that turns a raw training image (a `uint8` tensor `(3, H, W)` or a PIL image) into a ready-to-train float tensor:
- random crop that is then resized to `(size, size)` (look up `RandomResizedCrop`),
- random horizontal flip,
- random brightness/contrast/saturation/hue jitter (keep it mild),
- output `float32` with values in [0, 1].

```python
>>> pipe = augmentation_pipeline(96)
>>> out = pipe(img)          # img: (3, 120, 160) uint8
>>> out.shape, out.dtype
(torch.Size([3, 96, 96]), torch.float32)
```

### 7. Write-up (optional)
Write a short post: what is data augmentation, when is it useful, which augmentations would be wrong for a given task (give examples), and how could augmentation be automated (look up AutoAugment, RandAugment and TrivialAugment, which `transforms.v2` also provides).

## Data
Tests use random tensors, so no download is needed. To see your augmentations on real photos, load any JPEG with `torchvision.io.decode_image(path)` (it returns a `(C, H, W)` uint8 tensor) and plot it with matplotlib (`plt.imshow` needs `(H, W, C)`, so permute the axes first). Or use the photos in `02-vision/07-neural_style_transfer/data/`. For a dataset, `torchvision.datasets.Flowers102(root="data", download=True)` is about 350 MB.

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 02-vision/03-data_augmentation
```
