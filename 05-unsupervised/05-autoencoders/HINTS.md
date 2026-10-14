# Hints: Autoencoders

Open one hint at a time. Try for a while before opening the next.

## 0. "Vanilla" Autoencoder

<details><summary>Hint 1</summary>

Think of the layer sizes as a list: input, each hidden size, then latent. Every neighbouring pair in that list is one linear layer. The decoder's list is the same idea run backwards.
</details>

<details><summary>Hint 2</summary>

Build the layers in a loop into a Python list, then wrap them in `nn.Sequential`. The only thing that changes for the last layer is its activation.
</details>

<details><summary>Hint 3</summary>

`auto` needs both a `forward` and a `loss` method, so it has to be your own `nn.Module` subclass that holds the encoder and decoder as attributes (assigning them in `__init__` registers their parameters). PyTorch's functional module has a binary cross-entropy function that averages by default.
</details>

## 1. Sparse Autoencoder

<details><summary>Hint 1</summary>

The network is identical to task 0. Only the loss changes.
</details>

<details><summary>Hint 2</summary>

In `loss`, run the encoder and decoder as separate steps so you have the code in hand, not just the reconstruction.
</details>

<details><summary>Hint 3</summary>

Absolute values of the code, summed across each row, averaged across the batch, times `lambtha`, added to the BCE. Remember to store `lambtha` on the model.
</details>

## 2. Convolutional Autoencoder

<details><summary>Hint 1</summary>

Write down the shape after every layer for the example `(1, 28, 28)` with filters `[16, 8, 8]` before coding. Every layer you add should match one line of that list.
</details>

<details><summary>Hint 2</summary>

For a 3×3 kernel, "same" padding means a padding of 1 and "valid" means 0. Pooling of 7 down to 4 needs the max-pool option that rounds up instead of down. For upsampling, look at `nn.Upsample`.
</details>

<details><summary>Hint 3</summary>

The decoder's first convolution takes `latent_dims[0]` input channels. In the decoder loop, check whether you're on the last iteration to pick the padding. The final convolution outputs `input_dims[0]` channels.
</details>

## 3. Variational Autoencoder

<details><summary>Hint 1</summary>

The encoder now has a shared body and two heads, and returns three things, so it needs to be its own `nn.Module` subclass rather than a plain `nn.Sequential`.
</details>

<details><summary>Hint 2</summary>

For the sample: the standard deviation is the exponential of half the log-variance. PyTorch can make standard-normal noise with the same shape, dtype and device as an existing tensor. Check `self.training` to decide whether to sample.
</details>

<details><summary>Hint 3</summary>

For the loss, ask BCE for a sum instead of a mean, then divide by the batch size yourself. The KL formula is in the README; sum it over everything, then divide by the batch size too.
</details>

## 4. Training loop

<details><summary>Hint 1</summary>

The training step is always the same four beats: clear old gradients, compute the loss, backpropagate, let the optimizer step.
</details>

<details><summary>Hint 2</summary>

For shuffled mini-batches, make a random permutation of the row indices each epoch and slice it in chunks of `batch_size`. Make the permutation on the same device as the data.
</details>

<details><summary>Hint 3</summary>

To get an honest epoch average when the last batch is smaller, weight each batch's loss by its size. Use `.item()` to turn a scalar tensor into a Python float. For the device check, look at `torch.cuda.is_available()` and the MPS equivalent under `torch.backends`.
</details>
