# Hints: Convolutional Neural Networks

Open one hint at a time. Try for a while before opening the next.

## 0. Convolutional forward prop
<details><summary>Hint 1</summary>
Start by working out three things on paper before writing anything: how much padding you need, the output height and width, and the shape of the empty output array. If those are right, the rest is filling in numbers.
</details>
<details><summary>Hint 2</summary>
Pad the input once, up front, with zeros on the height and width axes only (not on the examples or channels axes). Then think of the output as a grid: for each output row, output column and filter, there is exactly one input patch that produces it. The patch's top-left corner moves by the stride.
</details>
<details><summary>Hint 3</summary>
You can handle all examples at once: slicing the padded input with a colon on the first axis gives you the same patch for every example. Multiply that patch by one filter (broadcasting does the work) and sum over the three axes that belong to the patch (height, width, channels), leaving one number per example. Add the bias, then apply the activation.
</details>

## 1. Pooling forward prop
<details><summary>Hint 1</summary>
This is the same sliding-window walk as task 0, but simpler: no padding, no weights, and channels stay separate.
</details>
<details><summary>Hint 2</summary>
Because pooling works per channel, you don't need to loop over channels at all. Take the window for all examples and all channels at once.
</details>
<details><summary>Hint 3</summary>
NumPy's max and mean functions accept a tuple of axes. Reduce over the window's height and width axes only, and you'll get one value per example and channel.
</details>

## 2. Convolutional back prop
<details><summary>Hint 1</summary>
The bias gradient is the easy one: each bias was added to every position of its output channel, for every example. So its gradient is the sum of dZ over everything except the channel axis (keep the dimensions so the shape matches b).
</details>
<details><summary>Hint 2</summary>
Recreate the padded input exactly as in the forward pass, and make a zero array the same shape to collect the gradient for the padded input. Then walk over the same output positions as in task 0. At each position, one patch and one filter produced the output, and the gradient dZ at that spot "flows back" to both of them.
</details>
<details><summary>Hint 3</summary>
Use the chain rule on "output = sum of patch × filter". The derivative with respect to the filter is the patch; with respect to the patch it's the filter. So: the filter's gradient grows by the patch times that output gradient (summed over examples), and the patch's region of the padded-gradient array grows by the filter times that output gradient (one per example; broadcasting needs a few extra axes on dZ). Contributions from overlapping windows add up. At the end, cut the padding off to return to the original input size.
</details>

## 3. Pooling back prop
<details><summary>Hint 1</summary>
Pooling has no weights, so you only need the gradient for its input. Start from zeros the shape of A_prev and walk over the same windows as the forward pass, adding each output's gradient back into its window.
</details>
<details><summary>Hint 2</summary>
For average pooling, every input in the window contributed equally (one part in kh×kw), so each one gets that same fraction of the output's gradient.
</details>
<details><summary>Hint 3</summary>
For max pooling, only the winner of each window affected the output. Build a boolean mask that is true where the window equals its own max, and use it to send the gradient only to that position. Remember to add (not overwrite), since windows can overlap when the stride is smaller than the kernel.
</details>

## 4. LeNet-5 in PyTorch
<details><summary>Hint 1</summary>
The layer list in the README maps one-to-one onto PyTorch building blocks: a 2D convolution, a ReLU, a 2D max-pool, a flatten and a linear layer. An `nn.Sequential` lets you stack them in order.
</details>
<details><summary>Hint 2</summary>
For "same" padding with a 5×5 kernel and stride 1, ask yourself how many zeros on each side keep a 28-wide image 28 wide. To size the first linear layer, trace the shape through by hand: 28 → conv → pool → conv → pool. What is the height, width and channel count at the end?
</details>
<details><summary>Hint 3</summary>
For initialization, look in `torch.nn.init` for the Kaiming normal function (set its nonlinearity to ReLU) and the function that fills a tensor with zeros. Loop over the model's modules and apply them to every conv and linear layer.
</details>

## 5. Train LeNet-5
<details><summary>Hint 1</summary>
Every PyTorch training loop has the same five steps per batch: forward pass, compute loss, clear old gradients, backward pass, optimizer step. Write those first; everything else is bookkeeping.
</details>
<details><summary>Hint 2</summary>
Convert X and Y to tensors with the right dtypes (labels must be integers for cross-entropy). Move the model and each batch to the device. For shuffling, a random permutation of the indices each epoch is all you need.
</details>
<details><summary>Hint 3</summary>
To report the epoch averages, keep running totals: add each batch's loss multiplied by the batch size, and count correct predictions (the predicted class is the index of the largest logit). Divide both by N at the end of the epoch. Use `.item()` to turn one-number tensors into plain floats.
</details>
