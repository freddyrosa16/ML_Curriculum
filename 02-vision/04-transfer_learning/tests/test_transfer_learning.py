"""Tests for 02-vision/04-transfer_learning.

Non-slow tests never download anything (models are built with weights=None).
"""
import numpy as np
import pytest

from helpers import load_task, project_dir

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
nn = torch.nn


# ---------- Task 0 ----------
def test_preprocess_shapes_and_types():
    preprocess_data = load_task(__file__, "0-transfer.py", "preprocess_data")
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
    Y = np.array([0, 1, 2, 3, 9])
    X_p, Y_p = preprocess_data(X, Y)
    assert isinstance(X_p, torch.Tensor) and isinstance(Y_p, torch.Tensor)
    assert tuple(X_p.shape) == (5, 3, 224, 224)
    assert X_p.dtype == torch.float32
    assert Y_p.dtype == torch.int64 and Y_p.tolist() == [0, 1, 2, 3, 9]


def test_preprocess_normalizes_with_imagenet_stats():
    preprocess_data = load_task(__file__, "0-transfer.py", "preprocess_data")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    X = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    X[0] = np.round(mean * 255).astype(np.uint8)        # the "average" colour -> ~0
    X[1] = 255                                           # white -> (1 - mean) / std
    X_p, _ = preprocess_data(X, np.zeros(2), size=64)
    assert tuple(X_p.shape) == (2, 3, 64, 64)
    assert torch.allclose(X_p[0], torch.zeros_like(X_p[0]), atol=0.02)
    expected_white = torch.tensor((1 - mean) / std, dtype=torch.float32).view(3, 1, 1)
    assert torch.allclose(X_p[1], expected_white.expand(3, 64, 64), atol=1e-3)


def test_preprocess_channel_order():
    preprocess_data = load_task(__file__, "0-transfer.py", "preprocess_data")
    X = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    X[..., 0] = 255   # pure red
    X_p, _ = preprocess_data(X, np.zeros(1), size=32)
    assert X_p[0, 0].mean() > 2 and X_p[0, 1].mean() < -1.5


# ---------- Task 1 ----------
def test_build_model_head_and_frozen_backbone():
    build_model = load_task(__file__, "1-build_model.py", "build_model")
    torch.manual_seed(0)
    model = build_model(num_classes=10, weights=None)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 2048 and model.fc.out_features == 10
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(name.startswith("fc.") for name in trainable)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 2048 * 10 + 10
    model.eval()
    with torch.no_grad():
        assert tuple(model(torch.zeros(2, 3, 64, 64)).shape) == (2, 10)


def test_build_model_num_classes():
    build_model = load_task(__file__, "1-build_model.py", "build_model")
    assert build_model(num_classes=3, weights=None).fc.out_features == 3


# ---------- Task 2 ----------
def test_extract_features_eval_mode_batches_and_no_grad():
    extract_features = load_task(__file__, "2-extract_features.py", "extract_features")
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(12, 5), nn.Dropout(0.5))
    X = torch.randn(10, 3, 2, 2)
    feats = extract_features(backbone, X, batch_size=3, device="cpu")
    assert tuple(feats.shape) == (10, 5)
    assert not feats.requires_grad
    assert feats.device.type == "cpu"
    backbone.eval()
    with torch.no_grad():
        assert torch.allclose(feats, backbone(X), atol=1e-6)   # dropout was off


def test_extract_features_applies_transform_per_batch():
    extract_features = load_task(__file__, "2-extract_features.py", "extract_features")
    backbone = nn.Flatten()
    X = np.arange(7 * 4, dtype=np.uint8).reshape(7, 2, 2)
    seen = []

    def transform(xb):
        seen.append(len(xb))
        return torch.as_tensor(xb, dtype=torch.float32) * 2

    feats = extract_features(backbone, X, batch_size=4, device="cpu", transform=transform)
    assert seen == [4, 3]
    assert torch.allclose(feats, torch.tensor(X, dtype=torch.float32).reshape(7, 4) * 2)


# ---------- Task 3 ----------
def test_train_head_learns_separable_features():
    train_head = load_task(__file__, "3-train_head.py", "train_head")
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 3, (4, 16))
    labels = rng.integers(0, 4, 800)
    features = centers[labels] + rng.normal(0, 1, (800, 16))
    head = train_head(torch.tensor(features, dtype=torch.float32), torch.tensor(labels),
                      num_classes=4, epochs=20, lr=1e-2, batch_size=64, device="cpu")
    assert isinstance(head, nn.Linear)
    assert head.in_features == 16 and head.out_features == 4
    with torch.no_grad():
        acc = (head(torch.tensor(features, dtype=torch.float32)).argmax(1).numpy() == labels).mean()
    assert acc > 0.95


# ---------- Task 4 (slow) ----------
@pytest.mark.slow
def test_saved_cifar10_model_reaches_target():
    """Needs cifar10.pt from task 4. Downloads the CIFAR-10 test set (~170 MB)."""
    build_model = load_task(__file__, "1-build_model.py", "build_model")
    preprocess_data = load_task(__file__, "0-transfer.py", "preprocess_data")
    folder = project_dir(__file__)
    weights_path = folder / "cifar10.pt"
    if not weights_path.exists():
        pytest.skip("cifar10.pt not found: run your task 4 script first")
    from torchvision import datasets
    test = datasets.CIFAR10(str(folder / "data"), train=False, download=True)
    model = build_model(10, weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(test.data), 200):
            X_p, Y_p = preprocess_data(test.data[i:i + 200], test.targets[i:i + 200])
            correct += (model(X_p.to(device)).argmax(1).cpu() == Y_p).sum().item()
    acc = correct / len(test.data)
    assert acc >= 0.87, f"test accuracy {acc:.3f} is below the 87% target"
