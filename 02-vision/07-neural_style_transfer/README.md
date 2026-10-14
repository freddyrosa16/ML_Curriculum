# Neural Style Transfer

**Status:** Modernized. Same algorithm and the same `NST` class as the original (Gatys et al., VGG19, Gram matrices, content/style/variation costs, Adam on the image), built with PyTorch and torchvision's VGG19 instead of TensorFlow/Keras. Everything lives in one file, and the variational cost is part of the total cost from the start instead of being bolted on at the end.

## Why this matters
Style transfer repaints a photo in the style of a painting. It's a fun result, but the ideas behind it are deep: a trained CNN's layers hold a description of an image's *content* (deep layers: what's where) and its *style* (correlations between feature maps: textures, colours, brush strokes). And instead of training the network, you keep it frozen and use gradient descent to change the **image itself**. That "optimize the input" trick shows up again in adversarial examples, feature visualization and model inversion.

## Concepts to understand first
- **Why use a pretrained VGG19?** VGG19, trained on ImageNet, is a plain stack of 3×3 convs and pools. Its features are known to work well for style. We only use its convolutional part (`features`), never its classifier, and we freeze it.
- **What represents content?** The activations of one deep layer (here the ReLU after `conv5_2`). Two images with similar activations there show similar things in similar places, even if colours and textures differ. Content cost = mean squared difference of those activations.
- **What represents style?** For a layer with $C$ channels, the **Gram matrix** $G_{ij}$ is the dot product of feature map $i$ with feature map $j$ over all positions, divided by the number of positions. It measures which features fire together, while ignoring *where* they fire. That's texture and style. Style cost = mean squared difference of Gram matrices, averaged over 5 layers.
- **What is total variation?** The sum of absolute differences between neighbouring pixels, horizontally and vertically. Penalizing it keeps the image smooth and removes high-frequency noise.
- **How do the costs combine?** $J = \alpha J_{content} + \beta J_{style} + \text{var} \cdot J_{var}$. A bigger $\alpha$ keeps more of the photo; a bigger $\beta$ paints more style.
- **How can you optimize an image?** Make the image a tensor with `requires_grad=True`, compute $J$, call backward, and let Adam update the pixels. The network's weights never change. After each step, clamp pixels back into [0, 1].
- **Why avg-pool instead of max-pool?** The original paper found that replacing VGG's max pooling with average pooling gives smoother, nicer-looking gradients for image generation.

## Resources
- [A Neural Algorithm of Artistic Style (Gatys et al., 2015)](https://arxiv.org/abs/1508.06576) — the original paper, short and readable
- [PyTorch tutorial: Neural Transfer Using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [d2l.ai 14.12: Neural Style Transfer](https://d2l.ai/chapter_computer-vision/neural-style.html)
- [Very Deep Convolutional Networks (VGG paper, Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- [torchvision: `vgg19`](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html) — print `vgg19().features` to see the layer indices
- [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/) — the same "optimize the input" idea, beautifully illustrated

## Rules
- PyTorch + torchvision (`import torch`, `import torch.nn as nn`, `from torchvision import models`), NumPy for the input images.
- Everything goes in **one file, `neural_style.py`**, containing a class `NST`. Each task adds methods. A task's tests are skipped until its methods exist. Most tests build a full `NST`, so they need tasks 0-3 done first.
- Images given to `NST` are NumPy arrays `(h, w, 3)`, `uint8`, RGB. Inside the class, images are tensors `(1, 3, h, w)`, float in [0, 1].
- Validate inputs and raise errors with **exactly** the messages given below (the tests check them).
- Tests use `weights=None` (random VGG weights, no download) and tiny images. Your code must work for any image size and on any device.

## Tasks

### 0. Initialize
**Class attributes:**
- `style_layers = [1, 6, 11, 20, 29]`: positions in `vgg19().features` of the ReLUs after `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`.
- `content_layer = 31`: the ReLU after `conv5_2`.

**Constructor:** `__init__(self, style_image, content_image, alpha=1e4, beta=1, var=10, weights="DEFAULT", max_dim=512, device=None)`
- If `style_image` is not a NumPy array of shape `(h, w, 3)`, raise `TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")`. Same for `content_image` (with `content_image` in the message).
- If `alpha` is not a non-negative number, raise `TypeError("alpha must be a non-negative number")`; same for `beta` and `var`.
- `device`: if `None`, pick cuda, then mps, then cpu.
- Store `self.style_image` and `self.content_image` (scaled with `scale_image`, on the device), `self.alpha`, `self.beta`, `self.var`, `self.device`, and anything else you'll need later (like `weights`).
- When you get to tasks 1 and 3, also call `self.load_model()` and then `self.generate_features()` at the end of `__init__`.

**Static method:** `scale_image(image, max_dim=512)`
- If `image` is not a NumPy array of shape `(h, w, 3)`, raise `TypeError("image must be a numpy.ndarray with shape (h, w, 3)")`.
- Resize (bicubic) so the longest side is `max_dim` and the aspect ratio is kept (round the other side to the nearest integer), scale pixels to [0, 1] (clamp after resizing), and return a `float32` tensor `(1, 3, h_new, w_new)`.

```python
>>> NST.scale_image(np.zeros((100, 200, 3), dtype=np.uint8)).shape
torch.Size([1, 3, 256, 512])
```

### 1. Load the model
**Method:** `load_model(self)`

Set `self.model` to an `nn.Sequential` with the first 32 layers of `models.vgg19(weights=self.weights).features` (indices 0 to `content_layer`), where every max-pool is replaced by a 2×2, stride-2 average pool. Freeze all parameters, put it in eval mode and on the device.
```python
>>> len(nst.model), sum(isinstance(m, nn.AvgPool2d) for m in nst.model)
(32, 4)
```
Tip: torchvision's VGG uses `ReLU(inplace=True)`. Swapping in `ReLU(inplace=False)` avoids surprises when you keep activations around.

### 2. Gram matrix
**Static method:** `gram_matrix(input_layer)`

- If `input_layer` is not a tensor of rank 4, raise `TypeError("input_layer must be a tensor of rank 4")`.
- For input `(1, C, H, W)`, return the `(1, C, C)` Gram matrix divided by `H * W`.
```python
>>> NST.gram_matrix(torch.tensor([[[[1., 2.]], [[3., 4.]]]]))
tensor([[[ 2.5000,  5.5000],
         [ 5.5000, 12.5000]]])
```

### 3. Extract features
**Methods:** `get_features(self, image)` and `generate_features(self)`

- `get_features(image)`: normalize the `(1, 3, h, w)` image with the ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`, run it through `self.model` layer by layer, and return `(style_outputs, content_output)`: a list of the 5 activations at `style_layers`, and the activation at `content_layer`.
- `generate_features()`: without tracking gradients, compute and store `self.gram_style_features` (list of the 5 Gram matrices of the style image) and `self.content_feature` (content activation of the content image).
```python
>>> style_outputs, content_output = nst.get_features(nst.content_image)   # golden_gate.jpg, max_dim=512
>>> [s.shape[1] for s in style_outputs], content_output.shape
([64, 128, 256, 512, 512], torch.Size([1, 512, 21, 32]))
```

### 4. Layer style cost
**Method:** `layer_style_cost(self, style_output, gram_target)`

- `style_output` must be a rank-4 tensor, else `TypeError("style_output must be a tensor of rank 4")`.
- `gram_target` must be a tensor of shape `(1, C, C)` (C = channels of `style_output`), else `TypeError("gram_target must be a tensor of shape [1, C, C]")` with the actual number, e.g. `[1, 64, 64]`.
- Return the mean of the squared differences between the Gram matrix of `style_output` and `gram_target` (a 0-dimensional tensor).

### 5. Style cost
**Method:** `style_cost(self, style_outputs)`

- `style_outputs` must be a list with one entry per style layer, else `TypeError("style_outputs must be a list with a length of 5")` (use the real length of `style_layers`).
- Return the layer style costs against `self.gram_style_features`, weighted equally (each `1/5`) and summed.

### 6. Content cost
**Method:** `content_cost(self, content_output)`

- `content_output` must be a tensor with the same shape as `self.content_feature`, else `TypeError("content_output must be a tensor of shape (1, 512, 21, 32)")` with the actual shape written as a Python tuple.
- Return the mean squared difference with `self.content_feature`.

### 7. Variational cost
**Static method:** `variational_cost(generated_image)`

- Must be a rank-4 tensor, else `TypeError("generated_image must be a tensor of rank 4")`.
- Return the total variation: the sum of absolute differences between vertically adjacent pixels plus the same for horizontally adjacent pixels, over all channels.
```python
>>> img = torch.zeros(1, 3, 2, 2); img[0, 0] = torch.tensor([[0., 1.], [2., 4.]])
>>> NST.variational_cost(img)
tensor(8.)
```

### 8. Total cost
**Method:** `total_cost(self, generated_image)`

- `generated_image` must be a tensor with the same shape as `self.content_image`, else `TypeError("generated_image must be a tensor of shape (1, 3, h, w)")` with the real shape as a tuple.
- Return `(J, J_content, J_style, J_var)` with $J = \alpha J_{content} + \beta J_{style} + \text{var} \cdot J_{var}$.
```python
>>> J, J_content, J_style, J_var = nst.total_cost(nst.content_image)
>>> J_content
tensor(0.)
```

### 9. Compute gradients
**Method:** `compute_grads(self, generated_image)`

Same shape check and message as task 8. Return `(gradient, J, J_content, J_style, J_var)`, where `gradient` is $\partial J / \partial \text{image}$ with the image's shape. Don't modify the tensor you were given.

### 10. Generate the image
**Method:** `generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99)`

Validate, in this order, raising exactly:
- `iterations` not an int: `TypeError("iterations must be an integer")`; not positive: `ValueError("iterations must be positive")`
- `step` (if not `None`) not an int: `TypeError("step must be an integer")`; not in `(0, iterations)`: `ValueError("step must be positive and less than iterations")`
- `lr` not a number: `TypeError("lr must be a number")`; not positive: `ValueError("lr must be positive")`
- `beta1` not a float: `TypeError("beta1 must be a float")`; outside [0, 1]: `ValueError("beta1 must be in the range [0, 1]")`. Same for `beta2`.

Then start from a copy of the content image and run Adam (`lr`, `betas=(beta1, beta2)`) on the image for `iterations` steps, clamping to [0, 1] after each step. Evaluate the cost at iterations 0 to `iterations` (inclusive). If `step` is set, print `Cost at iteration {i}: {J}, content {J_content}, style {J_style}, var {J_var}` every `step` iterations and at the last one. Keep the image with the lowest cost seen.

**Returns** `(image, cost)`: the best image as a NumPy array `(h, w, 3)` in [0, 1], and its cost as a Python float.
```python
>>> nst = NST(style, content)
>>> image, cost = nst.generate_image(iterations=2000, step=100, lr=0.01)
Cost at iteration 0: ..., content 0.0, style ..., var ...
...                      (the numbers depend on your images and settings)
>>> plt.imshow(image)
```

### 11. Experiments (optional)
Try different `alpha`/`beta`/`var` values, different style layers, starting from random noise instead of the content image, and your own photos. Write up what changes and why.

## Data
- `data/starry_night.jpg` (style) and `data/golden_gate.jpg` (content) are included. Load them with Pillow: open the file, convert to RGB, and turn it into a NumPy array.
- The pretrained VGG19 weights download automatically the first time you use `weights="DEFAULT"`: about 550 MB, cached in `~/.cache/torch`. Non-slow tests use `weights=None` and download nothing.
- Speed: at `max_dim=512`, each iteration runs VGG19 forward and backward on a large image. Expect around 0.1-0.5 s per iteration on an Apple Silicon GPU (MPS) and a few seconds on CPU. Start with `max_dim=256`.

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 02-vision/07-neural_style_transfer -m "not slow"
pytest 02-vision/07-neural_style_transfer -m slow   # downloads VGG19 weights, runs 100 real iterations
```
