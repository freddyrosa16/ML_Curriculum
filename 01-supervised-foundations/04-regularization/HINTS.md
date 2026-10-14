# Hints: Regularization

## 0. L2 regularization cost
<details><summary>Hint 1</summary>
Loop over the layer numbers 1 to L and add up each weight matrix's squared Frobenius norm: the sum of the squares of all its entries.
</details>
<details><summary>Hint 2</summary>
Scale that total by $\lambda / (2m)$ and add it to the original cost. Make sure you square: the plain Frobenius norm is a square root of what you need.
</details>

## 1. Gradient descent with L2 regularization
<details><summary>Hint 1</summary>
This is the deep network backward pass from the classification project, with two changes: hidden layers are tanh (so a different activation derivative), and each weight gradient gets one extra term.
</details>
<details><summary>Hint 2</summary>
The extra term is the derivative of the penalty with respect to that layer's weights. Differentiate $\frac{\lambda}{2m}\sum W^2$ with respect to $W$. Biases get no extra term.
</details>
<details><summary>Hint 3</summary>
As before, compute the error for the previous layer using the old weights before you update the current layer.
</details>

## 2. L2 regularization cost (PyTorch)
<details><summary>Hint 1</summary>
`model.modules()` walks every submodule, nested ones included. Keep only the ones that are `nn.Linear`.
</details>
<details><summary>Hint 2</summary>
Each linear layer has a `.weight` tensor. Squaring and summing with tensor operations (not `.item()` or NumPy) keeps the result connected to autograd.
</details>

## 3. L2 via weight decay
<details><summary>Hint 1</summary>
Optimizers accept a list of dicts instead of a plain parameter list. Each dict holds its own `params` and can override settings like `weight_decay`.
</details>
<details><summary>Hint 2</summary>
Sort `model.parameters()` into two lists by their number of dimensions (tensors have an `ndim` attribute).
</details>

## 4. Forward propagation with dropout
<details><summary>Hint 1</summary>
Each hidden layer: compute the tanh output as usual, then build a mask of the same shape where each entry is 1 with probability `keep_prob`. A uniform random number compared to `keep_prob` gives you exactly that.
</details>
<details><summary>Hint 2</summary>
Apply the mask by element-wise multiplication, then divide by `keep_prob` (inverted dropout). Save the mask in the cache under `D` plus the layer number; the backward pass will need it.
</details>
<details><summary>Hint 3</summary>
The last layer is softmax with no dropout and no mask.
</details>

## 5. Gradient descent with dropout
<details><summary>Hint 1</summary>
The forward pass multiplied by the mask and divided by `keep_prob`. When the error flows back into a hidden layer's output, it has to go through those same two operations.
</details>
<details><summary>Hint 2</summary>
So the order at each hidden layer, going backwards: send the error back through the next layer's weights, apply the same mask and scaling, then multiply by the tanh derivative.
</details>
<details><summary>Hint 3</summary>
Watch out: the tanh derivative $1 - \tanh^2(Z)$ needs the *plain* tanh output, but the cache stores the masked-and-scaled version. For the neurons that were kept, how do you get from one to the other? (For dropped neurons the error is already zero, so it doesn't matter.)
</details>

## 6. Create a layer with dropout
<details><summary>Hint 1</summary>
`nn.Sequential` of three modules. Translate `keep_prob` into the probability PyTorch wants.
</details>
<details><summary>Hint 2</summary>
You don't handle train vs eval yourself: `nn.Dropout` checks the module's mode for you.
</details>

## 7. Early stopping
<details><summary>Hint 1</summary>
Two decisions in order: first update the count (reset or increment), then compare the new count to the patience.
</details>
