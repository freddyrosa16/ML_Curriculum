"""Tests for 02-vision/06-face_verification.

Non-slow tests use synthetic "faces" (random patterns) and synthetic embeddings.
"""
import re

import numpy as np
import pytest

from helpers import load_task, project_dir

torch = pytest.importorskip("torch")
nn = torch.nn


def toy_faces(n_people=6, per_person=12, size=16, noise=0.3, seed=0):
    """Each 'person' is a fixed random pattern; their photos add noise."""
    rng = np.random.default_rng(seed)
    patterns = rng.uniform(0, 1, (n_people, size, size))
    labels = np.repeat(np.arange(n_people), per_person)
    images = patterns[labels] + rng.normal(0, noise, (len(labels), size, size))
    return np.clip(images, 0, 1).astype(np.float32), labels


# ---------- Task 0 (slow: downloads LFW) ----------
@pytest.mark.slow
def test_load_faces_lfw():
    load_faces = load_task(__file__, "0-load_faces.py", "load_faces")
    images, labels, names = load_faces(min_faces_per_person=20, resize=0.5)
    assert images.ndim == 3 and images.dtype == np.float32
    assert images.min() >= 0 and images.max() <= 1
    assert len(images) == len(labels)
    assert len(np.unique(labels)) == len(names) == 62
    assert "George W Bush" in list(names)


# ---------- Task 1 ----------
def test_split_by_identity_is_disjoint_and_complete():
    split = load_task(__file__, "1-split_by_identity.py", "split_by_identity")
    labels = np.repeat(np.arange(10), [3, 5, 2, 8, 4, 4, 6, 2, 3, 7])
    train_idx, test_idx = split(labels, test_fraction=0.2, seed=0)
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(np.concatenate([train_idx, test_idx]).tolist()) == list(range(len(labels)))
    train_ids, test_ids = set(labels[train_idx]), set(labels[test_idx])
    assert train_ids.isdisjoint(test_ids)
    assert len(test_ids) == 2 and len(train_ids) == 8


def test_split_by_identity_seeded():
    split = load_task(__file__, "1-split_by_identity.py", "split_by_identity")
    labels = np.repeat(np.arange(20), 3)
    a = split(labels, 0.25, seed=1)
    b = split(labels, 0.25, seed=1)
    assert np.array_equal(a[1], b[1])
    assert len(set(labels[a[1]])) == 5
    tests = {tuple(sorted(set(labels[split(labels, 0.25, seed=s)[1]]))) for s in range(5)}
    assert len(tests) > 1, "different seeds should pick different people"


# ---------- Task 2 ----------
def test_generate_triplets_valid():
    gen = load_task(__file__, "2-generate_triplets.py", "generate_triplets")
    labels = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])   # person 2 has a single photo
    trips = gen(labels, 500, seed=0)
    assert trips.shape == (500, 3)
    assert np.issubdtype(trips.dtype, np.integer)
    a, p, n = trips.T
    assert np.all(a != p)
    assert np.all(labels[a] == labels[p])
    assert np.all(labels[a] != labels[n])
    assert not np.any(labels[a] == 2), "a person with one photo can't be an anchor"
    assert set(labels[n]) >= {2}, "...but can be a negative"
    assert len(set(labels[a])) == 3


def test_generate_triplets_seeded_and_errors():
    gen = load_task(__file__, "2-generate_triplets.py", "generate_triplets")
    labels = np.repeat(np.arange(4), 5)
    assert np.array_equal(gen(labels, 20, seed=3), gen(labels, 20, seed=3))
    with pytest.raises(ValueError):
        gen(np.array([0, 1, 2]), 5)          # nobody has two photos
    with pytest.raises(ValueError):
        gen(np.array([0, 0, 0]), 5)          # no negatives possible


# ---------- Task 3 ----------
def test_triplet_loss_hand_example():
    triplet_loss = load_task(__file__, "3-triplet_loss.py", "triplet_loss")
    a = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    p = torch.tensor([[1.0, 0.0], [0.1, 0.0], [1.0, 1.0]])
    n = torch.tensor([[0.0, 0.5], [0.0, 1.0], [1.0, 1.0]])
    # squared distances: pos = [1, 0.01, 0], neg = [0.25, 1, 0]
    per_sample = triplet_loss(a, p, n, alpha=0.2, reduction="none")
    assert torch.allclose(per_sample, torch.tensor([0.95, 0.0, 0.2]))
    assert torch.isclose(triplet_loss(a, p, n, alpha=0.2), torch.tensor(1.15 / 3))
    assert torch.isclose(triplet_loss(a, p, n), torch.tensor(1.15 / 3)), "default alpha is 0.2"


def test_triplet_loss_is_differentiable_and_validates():
    triplet_loss = load_task(__file__, "3-triplet_loss.py", "triplet_loss")
    torch.manual_seed(0)
    a = torch.randn(8, 4, requires_grad=True)
    loss = triplet_loss(a, torch.randn(8, 4), torch.randn(8, 4), alpha=1.0)
    assert loss.ndim == 0
    loss.backward()
    assert a.grad is not None and a.grad.shape == a.shape
    with pytest.raises(ValueError):
        triplet_loss(a, a, a, reduction="sum")


# ---------- Task 4 ----------
def test_embedding_net_shapes_and_unit_norm():
    EmbeddingNet = load_task(__file__, "4-embedding_net.py", "EmbeddingNet")
    torch.manual_seed(0)
    net = EmbeddingNet().eval()
    with torch.no_grad():
        out = net(torch.rand(5, 1, 62, 47))     # LFW size at resize=0.5
    assert tuple(out.shape) == (5, 128)
    assert torch.allclose(out.norm(dim=1), torch.ones(5), atol=1e-5)
    net3 = EmbeddingNet(in_channels=3, embedding_dim=32).eval()
    with torch.no_grad():
        assert tuple(net3(torch.rand(2, 3, 32, 32)).shape) == (2, 32)


def test_embedding_net_is_small():
    EmbeddingNet = load_task(__file__, "4-embedding_net.py", "EmbeddingNet")
    n = sum(p.numel() for p in EmbeddingNet().parameters())
    assert 10_000 < n < 2_000_000


# ---------- Task 5 ----------
def test_train_triplet_learns_toy_faces():
    EmbeddingNet = load_task(__file__, "4-embedding_net.py", "EmbeddingNet")
    train = load_task(__file__, "5-train_triplet.py", "train_triplet")
    torch.manual_seed(0)
    images, labels = toy_faces()
    model = EmbeddingNet(embedding_dim=16)
    history = train(model, images, labels, epochs=6, triplets_per_epoch=256,
                    batch_size=32, lr=1e-3, alpha=0.2, device="cpu", seed=0)
    assert len(history) == 6 and all(isinstance(h, float) for h in history)
    assert history[-1] < history[0]
    model.cpu().eval()
    with torch.no_grad():
        emb = model(torch.tensor(images).unsqueeze(1)).numpy()
    d = ((emb[:, None] - emb[None]) ** 2).sum(-1)
    same = labels[:, None] == labels[None]
    off_diag = ~np.eye(len(labels), dtype=bool)
    assert d[same & off_diag].mean() < 0.5 * d[~same].mean()


# ---------- Task 6 ----------
class FlattenNormalize(nn.Module):
    """A fake 'embedding model': flatten and scale to unit length."""

    def forward(self, x):
        x = x.flatten(1)
        return x / x.norm(dim=1, keepdim=True)


def test_face_verification_embed_and_distance():
    FaceVerification = load_task(__file__, "6-verification.py", "FaceVerification")
    fv = FaceVerification(FlattenNormalize(), threshold=0.5)
    images = np.zeros((3, 2, 2), dtype=np.float32)
    images[0, 0, 0] = 1
    images[1, 0, 0] = 2
    images[2, 1, 1] = 1
    emb = fv.embed(images)
    assert isinstance(emb, np.ndarray) and emb.shape == (3, 4)
    assert np.isclose(fv.distance(emb[0], emb[1]), 0.0)
    assert np.isclose(fv.distance(emb[0], emb[2]), 2.0)   # squared distance of orthogonal unit vectors
    assert np.allclose(fv.distance(emb, emb[0]), [0.0, 0.0, 2.0])


def test_face_verification_verify():
    FaceVerification = load_task(__file__, "6-verification.py", "FaceVerification")
    fv = FaceVerification(FlattenNormalize(), threshold=0.5)
    a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 0.2], [0.0, 0.0]], dtype=np.float32)
    c = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    same, d = fv.verify(a, b)
    assert same is True or same == np.True_
    assert isinstance(d, float) and d < 0.5
    same, d = fv.verify(a, c)
    assert not same and np.isclose(d, 2.0)


def test_face_verification_uses_eval_mode():
    FaceVerification = load_task(__file__, "6-verification.py", "FaceVerification")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 8), nn.Dropout(0.9))
    fv = FaceVerification(model, threshold=1.0)
    x = np.random.default_rng(0).random((2, 2, 2)).astype(np.float32)
    assert np.allclose(fv.embed(x), fv.embed(x))


# ---------- Task 7 ----------
def test_roc_points():
    roc_points = load_task(__file__, "7-threshold.py", "roc_points")
    distances = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2])
    is_same = np.array([1, 1, 0, 0, 0, 1], dtype=bool)
    fpr, tpr = roc_points(distances, is_same, [0.0, 0.3, 0.5, 1.0])
    assert np.allclose(tpr, [0, 2 / 3, 1, 1])
    assert np.allclose(fpr, [0, 0, 1 / 3, 1])


def test_best_threshold():
    best_threshold = load_task(__file__, "7-threshold.py", "best_threshold")
    distances = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2])
    is_same = np.array([1, 1, 0, 0, 0, 1], dtype=bool)
    t, acc = best_threshold(distances, is_same)
    # t = 0.2 -> 5/6 correct; t = 0.4 -> 5/6 too; pick the smallest
    assert np.isclose(t, 0.2) and np.isclose(acc, 5 / 6)
    t, acc = best_threshold([0.1, 0.2, 0.9, 1.0], [True, True, False, False])
    assert np.isclose(t, 0.2) and np.isclose(acc, 1.0)


# ---------- Task 8: ethics write-up ----------
def test_ethics_write_up():
    path = project_dir(__file__) / "ethics.md"
    if not path.exists():
        pytest.skip("ethics.md not written yet")
    text = path.read_text(encoding="utf-8")
    words = re.findall(r"[A-Za-z']+", text)
    assert len(words) >= 250, f"ethics.md has {len(words)} words; aim for at least 250"
    lower = text.lower()
    for topic in ("bias", "consent", "surveillance"):
        assert topic in lower, f"ethics.md should discuss {topic}"
