"""Tests for 05-unsupervised/06-gan (PyTorch)."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from helpers import load_task  # noqa: E402

LATENT = 4


def mlp(n_in, n_out, hidden=16):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


def latent_generator(k):
    return torch.randn(k, LATENT)


def toy_setup(seed=0, n_real=500):
    """2-D 'real' data around (3, 3), a small generator and a small critic."""
    torch.manual_seed(seed)
    real = torch.randn(n_real, 2) * 0.5 + 3.0
    return mlp(LATENT, 2), mlp(2, 1), real


def params_snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def changed(before, module):
    return any(not torch.equal(b, p) for b, p in zip(before, module.parameters()))


class LinearCritic(nn.Module):
    """D(x) = w . flatten(x): its input-gradient is w for every sample."""

    def __init__(self, w):
        super().__init__()
        self.w = nn.Parameter(w.flatten().clone())

    def forward(self, x):
        return (x.flatten(start_dim=1) @ self.w).unsqueeze(1)


class SquareCritic(nn.Module):
    """D(x) = s * sum of x^2 with s = 1: its input-gradient is 2x."""

    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.ones(()))  # so the optimizer has something to hold

    def forward(self, x):
        return self.s * (x.flatten(start_dim=1) ** 2).sum(dim=1, keepdim=True)


CLASSES = [("0-simple_gan.py", "Simple_GAN"), ("1-wgan_clip.py", "WGAN_clip"), ("2-wgan_gp.py", "WGAN_GP")]


# ---------- Shared behaviour of tasks 0, 1 and 2 ----------

@pytest.mark.parametrize("filename, cls", CLASSES)
def test_init_attributes(filename, cls):
    GAN = load_task(__file__, filename, cls)
    G, D, real = toy_setup()
    gan = GAN(G, D, latent_generator, real)
    assert isinstance(gan, nn.Module)
    assert gan.generator is G and gan.discriminator is D
    assert gan.latent_generator is latent_generator and gan.real_examples is real
    assert gan.batch_size == 200 and gan.disc_iter == 2 and gan.learning_rate == .005
    assert isinstance(gan.gen_optimizer, torch.optim.Adam)
    assert isinstance(gan.disc_optimizer, torch.optim.Adam)
    gen_params = {id(p) for g in gan.gen_optimizer.param_groups for p in g["params"]}
    disc_params = {id(p) for g in gan.disc_optimizer.param_groups for p in g["params"]}
    assert gen_params == {id(p) for p in G.parameters()}
    assert disc_params == {id(p) for p in D.parameters()}


@pytest.mark.parametrize("filename, cls", CLASSES)
def test_samples(filename, cls):
    GAN = load_task(__file__, filename, cls)
    G, D, real = toy_setup()
    gan = GAN(G, D, latent_generator, real, batch_size=32)
    assert gan.get_fake_sample().shape == (32, 2)
    assert gan.get_fake_sample(7).shape == (7, 2)
    r = gan.get_real_sample()
    assert r.shape == (32, 2)
    # rows really come from real_examples, and no row is picked twice
    rows = {tuple(x.tolist()) for x in real}
    picked = [tuple(x.tolist()) for x in gan.get_real_sample(100)]
    assert all(p in rows for p in picked) and len(set(picked)) == 100


@pytest.mark.parametrize("filename, cls", CLASSES)
def test_train_step_updates_both_networks(filename, cls):
    GAN = load_task(__file__, filename, cls)
    G, D, real = toy_setup()
    gan = GAN(G, D, latent_generator, real, batch_size=64)
    g0, d0 = params_snapshot(G), params_snapshot(D)
    out = gan.train_step()
    assert {"discr_loss", "gen_loss"} <= set(out)
    assert all(isinstance(out[k], float) for k in ("discr_loss", "gen_loss"))
    assert changed(g0, G) and changed(d0, D)


# ---------- Task 0: Simple GAN losses ----------

def test_simple_gan_losses():
    GAN = load_task(__file__, "0-simple_gan.py", "Simple_GAN")
    G, D, real = toy_setup()
    gan = GAN(G, D, latent_generator, real)
    real_out = torch.tensor([[1.0], [0.0]])
    fake_out = torch.tensor([[-1.0], [1.0]])
    # real targets +1: errors 0 and 1 -> 0.5; fake targets -1: errors 0 and 2 -> 2.0
    assert torch.isclose(gan.discriminator_loss(real_out, fake_out), torch.tensor(2.5))
    # generator wants +1: errors 2 and 0 -> 2.0
    assert torch.isclose(gan.generator_loss(fake_out), torch.tensor(2.0))


# ---------- Task 1: WGAN with clipping ----------

def test_wgan_losses():
    GAN = load_task(__file__, "1-wgan_clip.py", "WGAN_clip")
    G, D, real = toy_setup()
    gan = GAN(G, D, latent_generator, real)
    real_out = torch.tensor([[2.0], [4.0]])
    fake_out = torch.tensor([[1.0], [-1.0]])
    assert torch.isclose(gan.discriminator_loss(real_out, fake_out), torch.tensor(-3.0))
    assert torch.isclose(gan.generator_loss(fake_out), torch.tensor(0.0))


def test_wgan_clip_keeps_weights_in_range():
    GAN = load_task(__file__, "1-wgan_clip.py", "WGAN_clip")
    G, D, real = toy_setup()
    with torch.no_grad():  # start with weights far outside [-1, 1]
        for p in D.parameters():
            p.mul_(50.0)
    gan = GAN(G, D, latent_generator, real, batch_size=64)
    for _ in range(3):
        gan.train_step()
        for p in D.parameters():
            assert p.min() >= -1.0 and p.max() <= 1.0
    # the clipping really happened: some weights sit exactly on the boundary
    assert any(torch.any(p.abs() == 1.0) for p in D.parameters())


# ---------- Task 2: WGAN with gradient penalty ----------

def test_interpolated_sample_lies_between():
    GAN = load_task(__file__, "2-wgan_gp.py", "WGAN_GP")
    G, D, real = toy_setup()
    gan = GAN(G, D, latent_generator, real)
    torch.manual_seed(1)
    a = torch.randn(50, 3, 4, 4)
    b = torch.randn(50, 3, 4, 4)
    x = gan.get_interpolated_sample(a, b)
    assert x.shape == a.shape
    # x = u * a + (1 - u) * b with ONE u per sample, 0 <= u <= 1
    u = ((x - b) / (a - b)).flatten(start_dim=1)
    assert torch.allclose(u, u[:, :1].expand_as(u), atol=1e-4)
    assert torch.all(u[:, 0] >= -1e-6) and torch.all(u[:, 0] <= 1 + 1e-6)
    assert u[:, 0].std() > 0.1  # different samples get different u


def test_gradient_penalty_linear_critic():
    GAN = load_task(__file__, "2-wgan_gp.py", "WGAN_GP")
    G, _, real = toy_setup()
    w = torch.tensor([3.0, 4.0])  # ||w|| = 5, so the penalty is (5 - 1)^2 = 16
    gan = GAN(G, LinearCritic(w), latent_generator, real)
    gp = gan.gradient_penalty(torch.randn(10, 2))
    assert gp.ndim == 0
    assert torch.isclose(gp, torch.tensor(16.0))


def test_gradient_penalty_norm_over_all_non_batch_dims():
    GAN = load_task(__file__, "2-wgan_gp.py", "WGAN_GP")
    G, _, real = toy_setup()
    gan = GAN(G, SquareCritic(), latent_generator, real)
    torch.manual_seed(2)
    x = torch.randn(6, 1, 4, 4)
    norms = 2 * x.flatten(start_dim=1).norm(dim=1)
    assert torch.isclose(gan.gradient_penalty(x), ((norms - 1) ** 2).mean(), rtol=1e-5)


def test_gradient_penalty_is_differentiable():
    """The penalty must be part of the graph so the critic is trained to reduce it."""
    GAN = load_task(__file__, "2-wgan_gp.py", "WGAN_GP")
    G, _, real = toy_setup()
    critic = LinearCritic(torch.tensor([3.0, 4.0]))
    gan = GAN(G, critic, latent_generator, real)
    gp = gan.gradient_penalty(torch.randn(10, 2))
    gp.backward()
    # d/dw (||w|| - 1)^2 = 2 (||w|| - 1) w / ||w|| = 2 * 4 * w / 5
    assert torch.allclose(critic.w.grad, torch.tensor([4.8, 6.4]), atol=1e-4)


def test_wgan_gp_defaults_and_train_step():
    GAN = load_task(__file__, "2-wgan_gp.py", "WGAN_GP")
    G, D, real = toy_setup()
    gan = GAN(G, D, latent_generator, real, batch_size=64)
    assert gan.lambda_gp == 10
    out = gan.train_step()
    assert "gp" in out and isinstance(out["gp"], float) and out["gp"] >= 0


@pytest.mark.slow
def test_wgan_gp_learns_toy_distribution():
    GAN = load_task(__file__, "2-wgan_gp.py", "WGAN_GP")
    _, _, real = toy_setup(n_real=2000)

    def leaky_mlp(n_in, n_out):
        return nn.Sequential(nn.Linear(n_in, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32),
                             nn.LeakyReLU(0.2), nn.Linear(32, n_out))

    G, D = leaky_mlp(LATENT, 2), leaky_mlp(2, 1)
    gan = GAN(G, D, latent_generator, real, batch_size=128, disc_iter=5, learning_rate=1e-3)
    with torch.no_grad():
        start = torch.linalg.norm(gan.get_fake_sample(2000).mean(0) - real.mean(0))
    for _ in range(600):
        gan.train_step()
    with torch.no_grad():
        end = torch.linalg.norm(gan.get_fake_sample(2000).mean(0) - real.mean(0))
    # GANs wobble, so only ask for a clear move toward the real data
    assert end < 0.5 * start


# ---------- Task 3: convolutional generator and discriminator ----------

def test_conv_gen_discr_shapes():
    convolutional_GenDiscr = load_task(__file__, "3-generate_faces.py", "convolutional_GenDiscr")
    torch.manual_seed(0)
    G, D = convolutional_GenDiscr()
    assert isinstance(G, nn.Module) and isinstance(D, nn.Module)
    z = torch.randn(8, 16)
    imgs = G(z)
    assert imgs.shape == (8, 1, 16, 16)
    assert torch.all(imgs >= -1) and torch.all(imgs <= 1)  # tanh output
    assert D(imgs).shape == (8, 1)


def test_conv_gen_discr_param_counts():
    convolutional_GenDiscr = load_task(__file__, "3-generate_faces.py", "convolutional_GenDiscr")
    G, D = convolutional_GenDiscr()
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)  # noqa: E731
    assert count(G) == 34816 + 294976 + 9232 + 145 + 2 * (64 + 16 + 1)
    assert count(D) == 320 + 18496 + 73856 + 295168 + 257


def test_conv_models_train_in_wgan_gp():
    convolutional_GenDiscr = load_task(__file__, "3-generate_faces.py", "convolutional_GenDiscr")
    GAN = load_task(__file__, "2-wgan_gp.py", "WGAN_GP")
    torch.manual_seed(0)
    G, D = convolutional_GenDiscr()
    real = torch.rand(64, 1, 16, 16) * 2 - 1
    gan = GAN(G, D, lambda k: torch.randn(k, 16), real, batch_size=16, disc_iter=1)
    out = gan.train_step()
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values())


@pytest.mark.slow
def test_faces_dataset_end_to_end():
    """Downloads the Olivetti faces (about 4 MB) through scikit-learn."""
    datasets = pytest.importorskip("sklearn.datasets")
    convolutional_GenDiscr = load_task(__file__, "3-generate_faces.py", "convolutional_GenDiscr")
    GAN = load_task(__file__, "2-wgan_gp.py", "WGAN_GP")
    faces = torch.tensor(datasets.fetch_olivetti_faces().images, dtype=torch.float32)  # (400, 64, 64) in [0, 1]
    faces = nn.functional.avg_pool2d(faces.unsqueeze(1), 4) * 2 - 1  # (400, 1, 16, 16) in [-1, 1]
    torch.manual_seed(0)
    G, D = convolutional_GenDiscr()
    gan = GAN(G, D, lambda k: torch.randn(k, 16), faces, batch_size=32, disc_iter=2, learning_rate=1e-3)
    for _ in range(20):
        out = gan.train_step()
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values())


# ---------- Task 4: replace_weights ----------

def test_replace_weights(tmp_path):
    GAN = load_task(__file__, "4-wgan_gp.py", "WGAN_GP")
    G, D, real = toy_setup()
    gan = GAN(G, D, latent_generator, real)
    torch.manual_seed(5)
    G2, D2 = mlp(LATENT, 2), mlp(2, 1)
    torch.save(G2.state_dict(), tmp_path / "gen.pt")
    torch.save(D2.state_dict(), tmp_path / "disc.pt")
    gan.replace_weights(str(tmp_path / "gen.pt"), str(tmp_path / "disc.pt"))
    for a, b in zip(gan.generator.parameters(), G2.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(gan.discriminator.parameters(), D2.parameters()):
        assert torch.equal(a, b)
    assert gan.lambda_gp == 10 and hasattr(gan, "gradient_penalty")
