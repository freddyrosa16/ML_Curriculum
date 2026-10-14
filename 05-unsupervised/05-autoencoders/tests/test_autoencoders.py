"""Tests for 05-unsupervised/05-autoencoders (PyTorch)."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from helpers import load_task  # noqa: E402


def low_rank_data(n=256, d=32, k=4, seed=0):
    """Vectors in (0, 1) that secretly depend on only k numbers each."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, k, generator=g)
    W = torch.randn(k, d, generator=g)
    return torch.sigmoid(z @ W)


def blob_images(n=64, size=28, seed=0):
    """Grey images, each with one bright Gaussian blob at a random spot. Shape (n, 1, size, size)."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.rand(n, 2, generator=g) * (size - 8) + 4
    ys, xs = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    d2 = (ys[None] - centers[:, 0, None, None]) ** 2 + (xs[None] - centers[:, 1, None, None]) ** 2
    return torch.exp(-d2 / 18.0).unsqueeze(1).clamp(1e-4, 1 - 1e-4)


def quick_train(auto, X, steps=150, lr=1e-2):
    """A few full-batch Adam steps; returns (first_loss, last_loss)."""
    opt = torch.optim.Adam(auto.parameters(), lr=lr)
    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = auto.loss(X)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    with torch.no_grad():
        last = auto.loss(X).item()
    return first, last


def linear_widths(module):
    return [m.out_features for m in module.modules() if isinstance(m, nn.Linear)]


# ---------- Task 0: vanilla ----------

def test_vanilla_shapes_and_layers():
    autoencoder = load_task(__file__, "0-vanilla.py", "autoencoder")
    torch.manual_seed(0)
    encoder, decoder, auto = autoencoder(784, [128, 64], 32)
    for m in (encoder, decoder, auto):
        assert isinstance(m, nn.Module)
    x = torch.rand(5, 784)
    code = encoder(x)
    assert code.shape == (5, 32)
    assert torch.all(code >= 0)  # ReLU on the latent layer too
    out = decoder(code)
    assert out.shape == (5, 784)
    assert torch.all((out > 0) & (out < 1))  # sigmoid output
    assert torch.allclose(auto(x), decoder(encoder(x)))
    assert linear_widths(encoder) == [128, 64, 32]
    assert linear_widths(decoder) == [64, 128, 784]


def test_vanilla_loss_is_bce():
    autoencoder = load_task(__file__, "0-vanilla.py", "autoencoder")
    torch.manual_seed(0)
    _, _, auto = autoencoder(32, [16], 4)
    x = low_rank_data()
    loss = auto.loss(x)
    assert loss.ndim == 0
    assert torch.isclose(loss, F.binary_cross_entropy(auto(x), x))


def test_vanilla_learns():
    autoencoder = load_task(__file__, "0-vanilla.py", "autoencoder")
    torch.manual_seed(0)
    _, _, auto = autoencoder(32, [64], 8)
    first, last = quick_train(auto, low_rank_data())
    assert last < first - 0.05


# ---------- Task 1: sparse ----------

def test_sparse_shapes_and_penalty():
    autoencoder = load_task(__file__, "1-sparse.py", "autoencoder")
    torch.manual_seed(0)
    encoder, decoder, auto = autoencoder(32, [16, 8], 4, 0.01)
    x = low_rank_data()
    code = encoder(x)
    assert code.shape == (256, 4) and decoder(code).shape == (256, 32)
    expected = F.binary_cross_entropy(decoder(code), x) + 0.01 * code.abs().sum(dim=1).mean()
    assert torch.isclose(auto.loss(x), expected, atol=1e-6)


def test_sparse_penalty_scales_with_lambtha():
    autoencoder = load_task(__file__, "1-sparse.py", "autoencoder")
    x = low_rank_data()
    torch.manual_seed(1)
    _, _, a0 = autoencoder(32, [16], 4, 0.0)
    torch.manual_seed(1)
    _, _, a1 = autoencoder(32, [16], 4, 1.0)
    assert torch.isclose(a0.loss(x), F.binary_cross_entropy(a0(x), x))
    assert a1.loss(x) >= a0.loss(x)


def test_sparse_learns():
    autoencoder = load_task(__file__, "1-sparse.py", "autoencoder")
    torch.manual_seed(0)
    _, _, auto = autoencoder(32, [64], 8, 1e-4)
    first, last = quick_train(auto, low_rank_data())
    assert last < first - 0.05


# ---------- Task 2: convolutional ----------

def test_conv_shapes_and_layers():
    autoencoder = load_task(__file__, "2-convolutional.py", "autoencoder")
    torch.manual_seed(0)
    encoder, decoder, auto = autoencoder((1, 28, 28), [16, 8, 8], (8, 4, 4))
    x = torch.rand(3, 1, 28, 28)
    code = encoder(x)
    assert code.shape == (3, 8, 4, 4)
    out = decoder(code)
    assert out.shape == (3, 1, 28, 28)
    assert torch.all((out > 0) & (out < 1))
    assert auto(x).shape == (3, 1, 28, 28)
    enc_convs = [m for m in encoder.modules() if isinstance(m, nn.Conv2d)]
    dec_convs = [m for m in decoder.modules() if isinstance(m, nn.Conv2d)]
    assert [c.out_channels for c in enc_convs] == [16, 8, 8]
    assert [c.out_channels for c in dec_convs] == [8, 8, 16, 1]
    assert all(c.kernel_size == (3, 3) for c in enc_convs + dec_convs)
    assert dec_convs[-2].padding in ((0, 0), "valid")


def test_conv_three_channels():
    autoencoder = load_task(__file__, "2-convolutional.py", "autoencoder")
    encoder, decoder, auto = autoencoder((3, 32, 32), [8, 4], (4, 8, 8))
    x = torch.rand(2, 3, 32, 32)
    assert encoder(x).shape == (2, 4, 8, 8)
    # 8 -> 16 (same) -> valid conv 14 -> 28 -> final conv keeps 28
    assert decoder(encoder(x)).shape == (2, 3, 28, 28)


def test_conv_learns():
    autoencoder = load_task(__file__, "2-convolutional.py", "autoencoder")
    torch.manual_seed(0)
    _, _, auto = autoencoder((1, 28, 28), [8, 8, 4], (4, 4, 4))
    X = blob_images()
    first, last = quick_train(auto, X, steps=40, lr=5e-3)
    assert last < first * 0.9


# ---------- Task 3: variational ----------

def test_vae_shapes():
    autoencoder = load_task(__file__, "3-variational.py", "autoencoder")
    torch.manual_seed(0)
    encoder, decoder, auto = autoencoder(784, [512], 2)
    x = torch.rand(4, 784)
    z, mean, log_var = encoder(x)
    assert z.shape == mean.shape == log_var.shape == (4, 2)
    out = decoder(z)
    assert out.shape == (4, 784)
    assert torch.all((out > 0) & (out < 1))
    assert auto(x).shape == (4, 784)


def test_vae_samples_in_train_mode_only():
    autoencoder = load_task(__file__, "3-variational.py", "autoencoder")
    torch.manual_seed(0)
    encoder, _, auto = autoencoder(32, [16], 3)
    x = low_rank_data(n=8)
    auto.train()
    z1, m1, _ = encoder(x)
    z2, m2, _ = encoder(x)
    assert torch.allclose(m1, m2)
    assert not torch.allclose(z1, z2)  # z is a random sample around the mean
    auto.eval()
    z3, m3, _ = encoder(x)
    assert torch.allclose(z3, m3)  # no sampling in eval mode


def test_vae_loss_includes_kl():
    autoencoder = load_task(__file__, "3-variational.py", "autoencoder")
    torch.manual_seed(0)
    encoder, decoder, auto = autoencoder(32, [16], 3)
    x = low_rank_data(n=64)
    auto.eval()  # deterministic: z = mean
    with torch.no_grad():
        _, mean, log_var = encoder(x)
        recon = decoder(mean)
        bce = F.binary_cross_entropy(recon, x, reduction="sum") / 64
        kl = -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp()) / 64
        loss = auto.loss(x)
    assert kl > 0
    assert torch.isclose(loss, bce + kl, rtol=1e-5)


def test_vae_learns():
    autoencoder = load_task(__file__, "3-variational.py", "autoencoder")
    torch.manual_seed(0)
    _, _, auto = autoencoder(32, [64], 4)
    auto.train()
    first, last = quick_train(auto, low_rank_data(), steps=200)
    assert last < first * 0.95


# ---------- Task 4: training loop ----------

def test_train_loop_cpu():
    train = load_task(__file__, "4-train.py", "train")
    autoencoder = load_task(__file__, "0-vanilla.py", "autoencoder")
    torch.manual_seed(0)
    _, _, auto = autoencoder(32, [64], 8)
    history = train(auto, low_rank_data(), epochs=8, batch_size=32, lr=1e-2, device="cpu")
    assert isinstance(history, list) and len(history) == 8
    assert all(isinstance(h, float) for h in history)
    assert history[-1] < history[0]


def test_train_loop_works_for_vae():
    train = load_task(__file__, "4-train.py", "train")
    autoencoder = load_task(__file__, "3-variational.py", "autoencoder")
    torch.manual_seed(0)
    _, _, auto = autoencoder(32, [64], 4)
    history = train(auto, low_rank_data(), epochs=6, batch_size=64, lr=1e-2, device="cpu")
    assert len(history) == 6 and history[-1] < history[0]


def test_get_device():
    get_device = load_task(__file__, "4-train.py", "get_device")
    assert get_device().type in ("cpu", "cuda", "mps")
