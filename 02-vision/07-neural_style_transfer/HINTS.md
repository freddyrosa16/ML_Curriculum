# Hints: Neural Style Transfer

Open one hint at a time.

## 0. Initialize
<details><summary>Hint 1</summary>
Do all the checks first, before storing anything, in the order listed. Three conditions make an image valid: it's a NumPy array, it has 3 dimensions, and its last dimension is 3.
</details>
<details><summary>Hint 2</summary>
For the numbers, "is it an int or a float" is one `isinstance` check with a tuple of types. Note that Python treats `True` as an int, so you may want to exclude bools explicitly.
</details>
<details><summary>Hint 3</summary>
In `scale_image`: compute the scale factor from the longest side, convert the array to a float tensor, move channels first and add a batch axis, divide by 255, then resize with `torch.nn.functional`'s interpolation function in bicubic mode. Bicubic can overshoot slightly, which is why you clamp at the end.
</details>

## 1. Load the model
<details><summary>Hint 1</summary>
Print `models.vgg19(weights=None).features` and find indices 0 to 31. You'll see where the max-pools sit.
</details>
<details><summary>Hint 2</summary>
An `nn.Sequential` can be sliced like a list and iterated over. Build a new list of layers, swapping each max-pool for an average pool as you go, and wrap it in a new `nn.Sequential`.
</details>
<details><summary>Hint 3</summary>
Freezing means turning off `requires_grad` for every parameter. Eval mode and moving to the device are one method call each.
</details>

## 2. Gram matrix
<details><summary>Hint 1</summary>
Think of each of the C channels as a long vector of H×W numbers. The Gram matrix is all the dot products between those C vectors.
</details>
<details><summary>Hint 2</summary>
Reshape the input from `(1, C, H, W)` to `(1, C, H*W)`. Then a batched matrix product of that tensor with its own transpose (swapping the last two axes) gives `(1, C, C)`.
</details>
<details><summary>Hint 3</summary>
Don't forget to divide by the number of positions. Check your function on the README example by hand: channel 0 is [1, 2], channel 1 is [3, 4], so the top-left entry is (1·1 + 2·2) / 2.
</details>

## 3. Extract features
<details><summary>Hint 1</summary>
Store the ImageNet mean and std as tensors shaped to broadcast over the channel axis (3 values, size 1 elsewhere). Make sure they're on the same device as the image.
</details>
<details><summary>Hint 2</summary>
Instead of calling the whole model at once, loop over its layers with their index (`enumerate`), passing the output of each into the next. When the index is in `style_layers`, keep the output; when it equals `content_layer`, keep that one too.
</details>
<details><summary>Hint 3</summary>
In `generate_features`, wrap the work in PyTorch's no-gradient context: these targets are constants, not something to optimize.
</details>

## 4. Layer style cost
<details><summary>Hint 1</summary>
Do the two checks first. The number of channels C is dimension 1 of `style_output`.
</details>
<details><summary>Hint 2</summary>
Compare shapes as tuples: `tuple(tensor.shape)` makes comparison with `(1, C, C)` easy.
</details>
<details><summary>Hint 3</summary>
The cost is a two-step calculation: the Gram matrix of the output, then the mean of the squared difference with the target.
</details>

## 5. Style cost
<details><summary>Hint 1</summary>
Pair each style output with its matching target Gram matrix (`zip` is perfect for this).
</details>
<details><summary>Hint 2</summary>
Each layer's weight is one divided by the number of style layers.
</details>
<details><summary>Hint 3</summary>
Python's built-in `sum` works on a generator of tensors and returns a tensor.
</details>

## 6. Content cost
<details><summary>Hint 1</summary>
Build the error message with the stored content feature's shape converted to a tuple, so it prints with parentheses.
</details>
<details><summary>Hint 2</summary>
The rest is a mean squared error between two tensors of the same shape.
</details>
<details><summary>Hint 3</summary>
Either the explicit "subtract, square, mean" or `torch.nn.functional`'s MSE function gives the same number.
</details>

## 7. Variational cost
<details><summary>Hint 1</summary>
Vertical neighbours: compare every row with the row below it. That's two slices of the height axis, one dropping the first row and one dropping the last, subtracted.
</details>
<details><summary>Hint 2</summary>
Do the same along the width axis for horizontal neighbours.
</details>
<details><summary>Hint 3</summary>
Take absolute values and sum everything from both. Check with the README example: the column differences are 2 and 3, the row differences 1 and 2.
</details>

## 8. Total cost
<details><summary>Hint 1</summary>
You already have every piece: features of the generated image, then three costs, then the weighted sum.
</details>
<details><summary>Hint 2</summary>
The variational cost uses the raw image in [0, 1], not the normalized one.
</details>
<details><summary>Hint 3</summary>
Return all four values; the individual costs are useful for printing progress.
</details>

## 9. Compute gradients
<details><summary>Hint 1</summary>
You need a version of the image that autograd tracks, without changing the caller's tensor: detach it, clone it, and turn on `requires_grad` for the copy.
</details>
<details><summary>Hint 2</summary>
Compute the total cost from that copy. Then either call backward on J and read the copy's `.grad`, or use `torch.autograd.grad`, which returns the gradient directly.
</details>
<details><summary>Hint 3</summary>
Detach the costs before returning them so they don't keep the whole graph alive in memory.
</details>

## 10. Generate the image
<details><summary>Hint 1</summary>
Get all the validation passing first (the tests check each message), then write the loop.
</details>
<details><summary>Hint 2</summary>
The optimizer needs a list of the tensors it updates: here that's just the image (a leaf tensor with `requires_grad`). You can give it the gradient from `compute_grads` by setting the image's `.grad` before calling the optimizer's step.
</details>
<details><summary>Hint 3</summary>
After each step, clamp the image in place inside a no-gradient block. To keep the best result, store a detached copy whenever the cost is lower than the best so far. At the end, drop the batch axis, move channels last, and convert to NumPy.
</details>
