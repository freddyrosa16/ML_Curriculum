# Hints: Data Augmentation

Open one hint at a time.

## 0. Flip
<details><summary>Hint 1</summary>
Which axis of a `(C, H, W)` tensor runs left to right? Mirroring means reversing the order along that one axis.
</details>
<details><summary>Hint 2</summary>
The functional module of `transforms.v2` has a ready-made function for this, named after the direction of the flip.
</details>
<details><summary>Hint 3</summary>
Plain PyTorch can do it too: there's a tensor function that reverses given dimensions. Either way, it's a single call.
</details>

## 1. Crop
<details><summary>Hint 1</summary>
A random crop means: pick a random top-left corner such that the whole crop fits, then take that rectangle.
</details>
<details><summary>Hint 2</summary>
In `transforms.v2`, random transforms are classes. You create one with its settings (here, the size), then call it on the image like a function.
</details>
<details><summary>Hint 3</summary>
Look at `v2.RandomCrop`. Its default is no padding, which is what you want.
</details>

## 2. Rotate
<details><summary>Hint 1</summary>
Draw a tiny 2×3 grid of numbers on paper and turn it a quarter turn to the left. Which pixel ends up top-left? Compare with the README example.
</details>
<details><summary>Hint 2</summary>
torchvision's rotate function uses degrees, and positive angles mean counter-clockwise. By default it keeps the original canvas size, which would cut off corners of a non-square image. Find the option that grows the canvas to fit.
</details>
<details><summary>Hint 3</summary>
Alternatively, PyTorch has a function made exactly for quarter turns that never interpolates. Pay attention to which two dimensions you rotate in and which direction "1 turn" goes.
</details>

## 3. Contrast
<details><summary>Hint 1</summary>
Split the task in two: draw a random factor, then apply a contrast change with that exact factor.
</details>
<details><summary>Hint 2</summary>
To draw a uniform random number between two values with PyTorch's generator, look at `torch.rand` (which gives values in [0, 1)) and think how to stretch and shift it to [lower, upper]. Tensors also have an in-place uniform fill method.
</details>
<details><summary>Hint 3</summary>
The functional module has an "adjust contrast" function that takes the image and a plain Python float factor.
</details>

## 4. Brightness
<details><summary>Hint 1</summary>
Same structure as contrast: random factor first, then a deterministic adjustment.
</details>
<details><summary>Hint 2</summary>
The factor's range is centred on 1: from 1 minus max_delta to 1 plus max_delta.
</details>
<details><summary>Hint 3</summary>
Use the functional "adjust brightness" function. It clamps values to the valid range for you.
</details>

## 5. Hue
<details><summary>Hint 1</summary>
This one isn't random: the shift is given to you.
</details>
<details><summary>Hint 2</summary>
Hue lives in a different colour space (HSV), so changing it by hand means converting RGB to HSV, shifting H around the circle, and converting back. You don't need to do that yourself here.
</details>
<details><summary>Hint 3</summary>
The functional module has an "adjust hue" function. Check what it does when the delta is out of range: you may get the required error for free.
</details>

## 6. A training pipeline
<details><summary>Hint 1</summary>
`v2.Compose` takes a list of transforms and applies them in order. Think about what order makes sense: convert, crop/resize, flip, colour, convert dtype.
</details>
<details><summary>Hint 2</summary>
To accept PIL images as well as tensors, start with the transform that turns any input into a torchvision image. To end with floats in [0, 1], use the dtype transform with scaling turned on.
</details>
<details><summary>Hint 3</summary>
`v2.ColorJitter` covers brightness, contrast, saturation and hue in one transform. Small values (a few tenths, and a few hundredths for hue) keep images realistic.
</details>
