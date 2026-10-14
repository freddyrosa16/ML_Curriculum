# Hints: Generative Adversarial Networks

Open one hint at a time. Try for a while before opening the next.

## 0. Simple GAN

<details><summary>Hint 1</summary>

Picking random rows without repeats: shuffle all the row indices and keep the first `size`. PyTorch can make a random permutation of `0 … n-1` directly.
</details>

<details><summary>Hint 2</summary>

Both losses are mean squared errors against a target tensor full of +1 or −1 with the same shape as the scores. PyTorch has helpers that make a tensor of ones shaped like another tensor.
</details>

<details><summary>Hint 3</summary>

In `train_step`, the discriminator update must not touch the generator: detach the fake batch first. For the generator update you *do* want gradients to flow through the discriminator into the generator, but only the generator's optimizer should step. Clear each optimizer's gradients before its backward pass so leftovers don't pile up. Return `.item()` of the losses.
</details>

## 1. Wasserstein GAN with weight clipping

<details><summary>Hint 1</summary>

Copy your task 0 class and change three things: the two losses and one extra step in the discriminator loop.
</details>

<details><summary>Hint 2</summary>

The losses are just means of scores, with signs chosen so that *minimizing* them does what each network wants.
</details>

<details><summary>Hint 3</summary>

Clip right after each discriminator optimizer step. Changing parameters in place must happen outside autograd's tracking (look for a context manager that turns gradients off), and tensors have an in-place clamp method.
</details>

## 2. Wasserstein GAN with gradient penalty

<details><summary>Hint 1</summary>

For the interpolation, the random `u` needs shape `(batch, 1, 1, …)`, with as many trailing 1s as the samples have extra dimensions, so it broadcasts over everything except the batch axis.
</details>

<details><summary>Hint 2</summary>

For the penalty: make the interpolated batch a leaf tensor that requires gradients, run the critic on it, and ask `torch.autograd.grad` for the gradient of the (summed) scores with respect to that input. Summing works because each score depends only on its own example.
</details>

<details><summary>Hint 3</summary>

Pass the option that keeps the graph of this gradient so the penalty can be backpropagated. Then flatten each example's gradient into one vector, take its length, subtract 1, square, and average over the batch.
</details>

## 3. Generating faces

<details><summary>Hint 1</summary>

Write down the shape after every layer before you code. The generator doubles the image size three times: 2 → 4 → 8 → 16. The discriminator halves it four times: 16 → 8 → 4 → 2 → 1.
</details>

<details><summary>Hint 2</summary>

The reshape from a flat 2048-vector to `(512, 2, 2)` can be a layer inside `nn.Sequential`; look at `nn.Unflatten`. For "same" padding with a 3×3 kernel, use padding 1.
</details>

<details><summary>Hint 3</summary>

If your parameter counts don't match the README, compare layer by layer: a conv layer has (in × out × 3 × 3) weights plus `out` biases, and batch norm has 2 learnable numbers per channel.
</details>

## 4. Our own WGAN-GP

<details><summary>Hint 1</summary>

You can inherit from your task 2 class (load it with `__import__('2-wgan_gp')`) and add one method.
</details>

<details><summary>Hint 2</summary>

`torch.load` reads the saved dictionary back (use its safe "weights only" mode), and every `nn.Module` has a method that loads such a dictionary into itself.
</details>
