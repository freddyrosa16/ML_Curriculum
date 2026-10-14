# Generative Adversarial Networks

**Status:** Modernized. Was Keras (`keras.Model` subclasses with a custom `train_step`); now PyTorch `nn.Module`s with explicit optimizers. The original 16×16 face dataset isn't public, so task 3 uses the Olivetti faces (via scikit-learn) shrunk to 16×16.

## Why this matters

A GAN learns to generate new data by setting two networks against each other: a generator that forges samples and a discriminator that tries to catch the forgeries. GANs powered the first photorealistic generated faces and are still used for super-resolution, image-to-image translation and as a training signal inside other generators. Building them teaches you to control *two* optimizers, to think about what a loss is really measuring, and why training can be unstable, plus the clever fixes (Wasserstein distance, gradient penalty) that made it reliable.

## Concepts to understand first

- **What are the generator and discriminator?** The generator turns random noise $z$ into a sample. The discriminator (called the *critic* in WGANs) scores a sample: high for "looks real", low for "looks fake". Each is trained to beat the other.
- **Why `detach()` the fakes when training the discriminator?** The discriminator step should update only the discriminator. Detaching the fake batch stops gradients from flowing back into the generator.
- **What losses does the simple GAN use?** A least-squares version: the discriminator pushes its scores on real data toward +1 and on fakes toward −1 (mean squared error), and the generator pushes the scores of its fakes toward +1. Squared error gives smoother gradients than the original log-loss.
- **What is the Wasserstein distance, intuitively?** The "earth mover's" cost of reshaping one distribution into another. Unlike the original GAN loss, it still gives a useful gradient when real and fake distributions don't overlap at all.
- **What is the WGAN critic loss?** The critic maximizes (mean score on real) − (mean score on fake), so it minimizes the negative of that. The generator minimizes −(mean score on fake).
- **Why must the critic be 1-Lipschitz, and how?** The Wasserstein trick only works if the critic's output can't change faster than its input (gradient norm at most 1). The first WGAN forced this crudely by *clipping* every weight into [−1, 1].
- **What is the gradient penalty?** A softer, better way: sample points between real and fake examples, compute the gradient of the critic's score with respect to those input points, and penalize $(\|\nabla\|_2 - 1)^2$. Added to the critic loss with weight $\lambda = 10$.
- **Why `create_graph=True`?** The penalty is built *from* a gradient. To train the critic on it you need the gradient of that gradient, so autograd must keep the graph of the first gradient computation.

## Resources

- [Goodfellow et al. (2014): Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Google: GAN course for developers](https://developers.google.com/machine-learning/gan)
- [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)
- [Mao et al. (2016): Least Squares GANs](https://arxiv.org/abs/1611.04076)
- [Arjovsky et al. (2017): Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Gulrajani et al. (2017): Improved Training of Wasserstein GANs (WGAN-GP)](https://arxiv.org/abs/1704.00028)
- [PyTorch tutorial: DCGAN faces](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [PyTorch: torch.autograd.grad](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html)

## Rules

- PyTorch only in your solution files (`torch`, `torch.nn`, `torch.nn.functional`).
- Each GAN class is an `nn.Module` subclass. It does not have a `forward`; its job is `train_step`.
- `latent_generator` is a function you're given: `latent_generator(k)` returns a `(k, latent_dim)` tensor of noise.
- Don't hard-code a device: sample indices and random numbers on the same device as the data they go with.

## Tasks

### 0. Simple GAN

**File:** `0-simple_gan.py` · **Class:** `Simple_GAN`

`__init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005)`
- Store every argument under the same name (`self.generator`, `self.real_examples`, ...), plus `self.beta_1 = .5` and `self.beta_2 = .9`.
- Create two Adam optimizers with that learning rate and those betas: `self.gen_optimizer` over the generator's parameters and `self.disc_optimizer` over the discriminator's.

Methods:
- `get_fake_sample(self, size=None)`: `size` generated examples (`None` means `batch_size`).
- `get_real_sample(self, size=None)`: `size` rows of `real_examples` chosen at random, with no row picked twice.
- `discriminator_loss(self, real_output, fake_output)`: MSE of real scores against +1 plus MSE of fake scores against −1.
- `generator_loss(self, fake_output)`: MSE of fake scores against +1.
- `train_step(self)`: `disc_iter` discriminator updates (fresh real and fake batches each time), then one generator update. **Returns** `{"discr_loss": float, "gen_loss": float}` (the last values, as Python floats).

```
>>> gan = Simple_GAN(G, D, latent_generator, real, batch_size=32)
>>> gan.get_fake_sample().shape, gan.get_real_sample(7).shape
(torch.Size([32, 2]), torch.Size([7, 2]))
>>> gan.discriminator_loss(torch.tensor([[1.], [0.]]), torch.tensor([[-1.], [1.]]))
tensor(2.5000)
>>> gan.train_step()
{'discr_loss': ..., 'gen_loss': ...}
```

### 1. Wasserstein GAN with weight clipping

**File:** `1-wgan_clip.py` · **Class:** `WGAN_clip`

Same constructor, attributes, sampling methods and return format as task 0, but:
- `discriminator_loss(real_output, fake_output)`: mean fake score minus mean real score.
- `generator_loss(fake_output)`: minus the mean fake score.
- After **every** discriminator update, clamp every discriminator parameter into [−1, 1] (in place, without tracking gradients).

```
>>> gan.discriminator_loss(torch.tensor([[2.], [4.]]), torch.tensor([[1.], [-1.]]))
tensor(-3.)
```

### 2. Wasserstein GAN with gradient penalty

**File:** `2-wgan_gp.py` · **Class:** `WGAN_GP`

Constructor adds `lambda_gp=10` at the end of the argument list; store it as `self.lambda_gp`. Here `self.beta_1 = .3` (and `self.beta_2 = .9`). Same WGAN losses as task 1, **no clipping**, plus:
- `get_interpolated_sample(self, real_sample, fake_sample)`: for each example pick one random `u` uniformly in [0, 1] and return `u * real + (1 - u) * fake`. Works for any input shape: `u` has one value per example, broadcast over all other dimensions.
- `gradient_penalty(self, interpolated_sample)`: the mean over the batch of $(\|g\|_2 - 1)^2$, where $g$ is the gradient of the critic's output with respect to that example's input, and the norm runs over **all** non-batch dimensions. It must stay differentiable so the critic can learn from it.
- `train_step(self)`: each critic update minimizes the WGAN loss plus `lambda_gp` times the penalty. **Returns** `{"discr_loss": float, "gen_loss": float, "gp": float}` (`discr_loss` without the penalty).

```
>>> # a linear critic D(x) = w . x with w = (3, 4) has gradient norm 5 everywhere
>>> gan = WGAN_GP(G, linear_critic, latent_generator, real)
>>> gan.gradient_penalty(torch.randn(10, 2))
tensor(16., grad_fn=<...>)
```

### 3. Generating faces

**File:** `3-generate_faces.py` · **Function:** `convolutional_GenDiscr()`

**Returns** `(generator, discriminator)` for 16×16 single-channel images (values in [−1, 1]). All convolutions are 3×3 with "same" padding.
- **Generator** (input `(batch, 16)`): linear to 2048 → reshape to `(512, 2, 2)` → then three blocks of [2× upsample → conv → batch norm → activation] with 64, 16 and 1 output channels. The first two blocks use ReLU, the last uses tanh. Output `(batch, 1, 16, 16)`.
- **Discriminator** (input `(batch, 1, 16, 16)`): four blocks of [conv → 2×2 max pool → ReLU] with 32, 64, 128 and 256 channels → flatten → linear to 1. Output `(batch, 1)`.

```
>>> G, D = convolutional_GenDiscr()
>>> G(torch.randn(8, 16)).shape
torch.Size([8, 1, 16, 16])
>>> D(G(torch.randn(8, 16))).shape
torch.Size([8, 1])
>>> sum(p.numel() for p in G.parameters()), sum(p.numel() for p in D.parameters())
(339331, 388097)
```
Then (not tested, but the fun part): train your `WGAN_GP` with these networks on the faces dataset (see Data) and show a grid of generated faces every few hundred steps. Expect blurry but recognizable faces after some minutes of training.

### 4. Our own "Wasserstein GAN with gradient penalty"

**File:** `4-wgan_gp.py` · **Class:** `WGAN_GP`

Everything from task 2 (import it or copy it), plus:
- `replace_weights(self, gen_path, disc_path)`: load generator and discriminator weights from files saved with `torch.save(model.state_dict(), path)`.

Use this to save your best face generator and reload it later without retraining.

### Write-up (optional)

Describe your face-GAN experiments: which settings you tried, what failed (mode collapse? exploding critic?), and how weight clipping compared with the gradient penalty. Include your best grid of generated faces.

## Data

Tests use tiny synthetic data, plus one slow test that downloads the faces.

**Faces for task 3:** scikit-learn's `sklearn.datasets.fetch_olivetti_faces()` downloads 400 grey 64×64 face photos (about 4 MB, cached after the first time). Its `.images` field is a `(400, 64, 64)` array with values in [0, 1]. To match the original project: shrink each image to 16×16 by averaging each 4×4 block of pixels, add a channel axis to get `(400, 1, 16, 16)`, and rescale from [0, 1] to [−1, 1] (times 2, minus 1) to match the generator's tanh output. Alternatively, MNIST from `torchvision.datasets.MNIST` resized to 16×16 works the same way if you'd rather generate digits.

## Check your work

From the `ml-curriculum` folder:

```bash
pytest 05-unsupervised/06-gan               # includes the slow faces download test
pytest 05-unsupervised/06-gan -m "not slow"
```
