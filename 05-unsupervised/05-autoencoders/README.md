# Autoencoders

**Status:** Modernized. Was Keras (`compile` with Adam + binary cross-entropy); now PyTorch, where you write the loss and the training loop yourself. Images are channels-first `(C, H, W)` as PyTorch expects.

## Why this matters

An autoencoder learns to squeeze data through a narrow bottleneck and rebuild it. Whatever survives the squeeze is a compact summary of the data, learned with no labels at all. That idea is everywhere: denoising, anomaly detection, compression, pretraining, and the latent space behind modern image generators (Stable Diffusion runs inside a VAE's latent space). Variational autoencoders add probability on top, which turns the decoder into a generator of brand-new samples.

## Concepts to understand first

- **What are the encoder, bottleneck and decoder?** The encoder maps an input to a small code (the latent vector), the decoder maps the code back to input space. Because the code is small, the network can't just copy; it has to learn the structure of the data.
- **Why binary cross-entropy for images?** With pixel values in [0, 1] and a sigmoid output, BCE treats each pixel as "how bright, as a probability" and gives strong gradients when the reconstruction is confidently wrong.
- **What makes an autoencoder "sparse"?** Adding a penalty $\lambda \sum |z_i|$ on the code pushes most code values to zero, so each input is described by a few active features. It's the same L1 idea as Lasso regression.
- **Why convolutions?** Images have local structure. Convolutions with pooling shrink the image while keeping spatial patterns; the decoder mirrors that with upsampling and convolutions.
- **What does a VAE encode?** Not a single point but a distribution: a mean $\mu$ and a log-variance $\log\sigma^2$ for each latent dimension.
- **What is the reparameterization trick?** To sample $z \sim \mathcal{N}(\mu, \sigma^2)$ and still backpropagate, write $z = \mu + \sigma \cdot \epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$. The randomness lives in $\epsilon$, so gradients flow through $\mu$ and $\sigma$.
- **What does the KL term do?** $\mathrm{KL} = -\tfrac12 \sum (1 + \log\sigma^2 - \mu^2 - \sigma^2)$ pulls every encoded distribution toward a standard normal. That keeps the latent space smooth and filled in, so decoding a random $z \sim \mathcal{N}(0, I)$ gives a sensible sample.
- **What do `model.train()` and `model.eval()` change?** They flip a flag (`self.training`) that layers like dropout, batch norm, and here your VAE encoder can check. You'll sample in train mode and use the mean in eval mode.

## Resources

- [Deep Learning book, chapter 14: Autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html)
- [Stanford UFLDL: Autoencoders](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)
- [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
- [Kingma & Welling (2013): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Doersch (2016): Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- [PyTorch: torch.nn reference](https://pytorch.org/docs/stable/nn.html)

## Rules

- PyTorch only (`torch`, `torch.nn`, `torch.nn.functional`). torchvision only for loading data, outside your solution files.
- `encoder`, `decoder` and `auto` must all be `nn.Module`s. `auto(x)` returns the reconstruction.
- Every `auto` also has a method `loss(x)` that returns the scalar training loss for a batch `x`. This replaces Keras' `compile(...)`: the loss lives with the model, so one training loop (task 4) works for all of them.
- Activations: ReLU everywhere except the decoder's last layer (sigmoid), and, in the VAE, the mean and log-variance layers (no activation).
- Code must run on any device: don't hard-code `"cuda"`; create new tensors on the same device as your inputs.

## Tasks

### 0. "Vanilla" Autoencoder

**File:** `0-vanilla.py` · **Function:** `autoencoder(input_dims, hidden_layers, latent_dims)`

- `input_dims`: int, size of each input vector. `hidden_layers`: list of ints, the encoder's hidden layer sizes (the decoder uses them reversed). `latent_dims`: int, size of the code.
- Fully connected (`nn.Linear`) layers only. The latent layer also uses ReLU.
- `auto.loss(x)`: mean binary cross-entropy between `auto(x)` and `x`.
- **Returns** `(encoder, decoder, auto)`.

```
>>> encoder, decoder, auto = autoencoder(784, [128, 64], 32)
>>> x = torch.rand(5, 784)
>>> encoder(x).shape, decoder(encoder(x)).shape
(torch.Size([5, 32]), torch.Size([5, 784]))
>>> auto.loss(x)
tensor(0.69..., grad_fn=<...>)
```

### 1. Sparse Autoencoder

**File:** `1-sparse.py` · **Function:** `autoencoder(input_dims, hidden_layers, latent_dims, lambtha)`

- Same architecture as task 0.
- `auto.loss(x)`: mean binary cross-entropy **plus** `lambtha` times the L1 norm of the code, where the L1 norm is summed over latent units and averaged over the batch.

```
>>> encoder, decoder, auto = autoencoder(784, [128, 64], 32, 10e-6)
>>> auto.loss(torch.rand(5, 784)).shape
torch.Size([])
```

### 2. Convolutional Autoencoder

**File:** `2-convolutional.py` · **Function:** `autoencoder(input_dims, filters, latent_dims)`

- `input_dims`: `(C, H, W)`. `filters`: list of conv widths for the encoder (reversed for the decoder). `latent_dims`: `(C', H', W')`, the shape the encoder produces.
- **Encoder:** for each entry of `filters`, a 3×3 convolution with "same" padding and ReLU, followed by 2×2 max pooling. When a size is odd, round up (7 → 4); PyTorch's max-pool has an option for that.
- **Decoder:** for each entry of the reversed `filters`, a 3×3 convolution with ReLU followed by 2× upsampling. All of these use "same" padding **except the last one in this loop, which uses no padding** (it trims 2 pixels, e.g. 16 → 14). Then one final 3×3 "same" convolution back to `C` channels with sigmoid, and no upsampling.
- `auto.loss(x)`: mean binary cross-entropy.

```
>>> encoder, decoder, auto = autoencoder((1, 28, 28), [16, 8, 8], (8, 4, 4))
>>> x = torch.rand(3, 1, 28, 28)
>>> encoder(x).shape
torch.Size([3, 8, 4, 4])
>>> auto(x).shape
torch.Size([3, 1, 28, 28])
```
Trace the decoder sizes by hand: 4 → 8 → 16 → 14 (no padding) → 28 → 28.

### 3. Variational Autoencoder

**File:** `3-variational.py` · **Function:** `autoencoder(input_dims, hidden_layers, latent_dims)`

- **Encoder:** hidden layers as before, then two separate linear heads (no activation): the mean and the log-variance. `encoder(x)` returns `(z, mean, log_var)`, each `(batch, latent_dims)`.
  - In train mode (`self.training` is `True`), `z` is sampled with the reparameterization trick.
  - In eval mode, `z` is simply the mean.
- **Decoder:** reversed hidden layers, sigmoid output.
- `auto(x)`: encode, then decode `z`.
- `auto.loss(x)`: binary cross-entropy **summed** over the features, plus the KL divergence summed over latent dimensions; both averaged over the batch.

```
>>> encoder, decoder, auto = autoencoder(784, [512], 2)
>>> z, mean, log_var = encoder(torch.rand(4, 784))
>>> z.shape, mean.shape, log_var.shape
(torch.Size([4, 2]), torch.Size([4, 2]), torch.Size([4, 2]))
```

### 4. Training loop

**File:** `4-train.py` · **Functions:** `get_device()` and `train(auto, X, epochs=10, batch_size=32, lr=1e-3, device=None)`

Keras' `fit` did this for you; now you write it.
- `get_device()`: return a `torch.device`, preferring CUDA, then Apple's MPS, then CPU.
- `train(...)`: move the model and data to `device` (`None` means use `get_device()`), put the model in train mode, and run `epochs` passes over `X` in shuffled mini-batches of `batch_size`, using Adam with learning rate `lr` and the model's own `loss`.
- **Returns** a list of `epochs` Python floats: the average loss over each epoch.

```
>>> encoder, decoder, auto = autoencoder(784, [128, 64], 32)   # from task 0
>>> history = train(auto, X_train, epochs=5, batch_size=256)
>>> len(history), history[-1] < history[0]
(5, True)
```

### Try it on MNIST (optional, not tested)

Train each model on MNIST for a few epochs and plot a row of originals above their reconstructions. For the VAE with `latent_dims=2`, decode a grid of `z` values (for example from −2 to 2 in each direction) to see the whole latent space at once, and decode random normal `z` vectors to generate new digits.

### Write-up (optional)

Explain the difference between an autoencoder and a VAE, and why sampling random codes works for one but not the other.

## Data

Tests use small synthetic data and need no downloads.

For the optional MNIST part: `torchvision.datasets.MNIST` with `download=True` and the `ToTensor` transform gives 28×28 images with values in [0, 1] (about 50 MB the first time). Flatten each image to 784 numbers for tasks 0, 1 and 3; keep the `(1, 28, 28)` shape for task 2. A few epochs train in a minute or two on a laptop CPU or Apple MPS.

## Check your work

From the `ml-curriculum` folder:

```bash
pytest 05-unsupervised/05-autoencoders
```
