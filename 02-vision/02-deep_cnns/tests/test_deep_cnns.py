"""Tests for 02-vision/02-deep_cnns. torchvision is used only as a reference."""
import pytest

from helpers import load_task

torch = pytest.importorskip("torch")
nn = torch.nn


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def conv_params(cin, cout, k, bias=True):
    return k * k * cin * cout + (cout if bias else 0)


def bn_params(c):
    return 2 * c


# ---------- Task 0 ----------
def test_inception_block_shape_and_channels():
    InceptionBlock = load_task(__file__, "0-inception_block.py", "InceptionBlock")
    block = InceptionBlock(192, (64, 96, 128, 16, 32, 32))
    assert isinstance(block, nn.Module)
    out = block(torch.zeros(2, 192, 28, 28))
    assert tuple(out.shape) == (2, 256, 28, 28)
    assert block.out_channels == 256


def test_inception_block_params():
    InceptionBlock = load_task(__file__, "0-inception_block.py", "InceptionBlock")
    cin, (F1, F3R, F3, F5R, F5, FPP) = 192, (64, 96, 128, 16, 32, 32)
    expected = (conv_params(cin, F1, 1) + conv_params(cin, F3R, 1) + conv_params(F3R, F3, 3)
                + conv_params(cin, F5R, 1) + conv_params(F5R, F5, 5) + conv_params(cin, FPP, 1))
    assert n_params(InceptionBlock(cin, (F1, F3R, F3, F5R, F5, FPP))) == expected


def test_inception_block_branches_order_and_relu():
    InceptionBlock = load_task(__file__, "0-inception_block.py", "InceptionBlock")
    torch.manual_seed(0)
    block = InceptionBlock(8, (1, 2, 3, 2, 4, 5)).eval()
    out = block(torch.randn(1, 8, 9, 9))
    assert tuple(out.shape) == (1, 13, 9, 9)
    assert torch.all(out >= 0)  # every branch ends with a ReLU


# ---------- Task 1 ----------
def test_inception_network():
    inception_network = load_task(__file__, "1-inception_network.py", "inception_network")
    torch.manual_seed(0)
    model = inception_network().eval()
    assert n_params(model) == 6998552
    with torch.no_grad():
        assert tuple(model(torch.zeros(2, 3, 224, 224)).shape) == (2, 1000)
    assert any(isinstance(m, nn.Dropout) and m.p == pytest.approx(0.4) for m in model.modules())


def test_inception_network_num_classes():
    inception_network = load_task(__file__, "1-inception_network.py", "inception_network")
    model = inception_network(num_classes=10).eval()
    with torch.no_grad():
        assert tuple(model(torch.zeros(1, 3, 224, 224)).shape) == (1, 10)


# ---------- Task 2 ----------
def test_identity_block():
    IdentityBlock = load_task(__file__, "2-identity_block.py", "IdentityBlock")
    block = IdentityBlock(256, (64, 64, 256))
    out = block(torch.randn(2, 256, 14, 14))
    assert tuple(out.shape) == (2, 256, 14, 14)
    assert torch.all(out >= 0)
    expected = (conv_params(256, 64, 1, False) + bn_params(64) + conv_params(64, 64, 3, False)
                + bn_params(64) + conv_params(64, 256, 1, False) + bn_params(256))
    assert n_params(block) == expected


def test_identity_block_adds_input():
    IdentityBlock = load_task(__file__, "2-identity_block.py", "IdentityBlock")
    block = IdentityBlock(4, (2, 2, 4))
    for p in block.parameters():   # zero every weight -> main path is all zeros
        nn.init.zeros_(p)
    block.eval()
    x = torch.randn(1, 4, 5, 5)
    with torch.no_grad():
        assert torch.allclose(block(x), torch.relu(x))


def test_identity_block_rejects_bad_filters():
    IdentityBlock = load_task(__file__, "2-identity_block.py", "IdentityBlock")
    with pytest.raises(ValueError):
        IdentityBlock(64, (64, 64, 256))


# ---------- Task 3 ----------
def test_projection_block_shapes():
    ProjectionBlock = load_task(__file__, "3-projection_block.py", "ProjectionBlock")
    out = ProjectionBlock(256, (128, 128, 512))(torch.randn(2, 256, 56, 56))
    assert tuple(out.shape) == (2, 512, 28, 28)
    assert torch.all(out >= 0)
    out = ProjectionBlock(64, (64, 64, 256), s=1)(torch.randn(2, 64, 56, 56))
    assert tuple(out.shape) == (2, 256, 56, 56)


def test_projection_block_params():
    ProjectionBlock = load_task(__file__, "3-projection_block.py", "ProjectionBlock")
    expected = (conv_params(256, 128, 1, False) + bn_params(128) + conv_params(128, 128, 3, False)
                + bn_params(128) + conv_params(128, 512, 1, False) + bn_params(512)
                + conv_params(256, 512, 1, False) + bn_params(512))
    assert n_params(ProjectionBlock(256, (128, 128, 512))) == expected


# ---------- Task 4 ----------
def test_resnet50_matches_torchvision_param_count():
    resnet50 = load_task(__file__, "4-resnet50.py", "resnet50")
    tv = pytest.importorskip("torchvision")
    model = resnet50()
    assert n_params(model) == n_params(tv.models.resnet50(weights=None)) == 25557032


def test_resnet50_forward():
    resnet50 = load_task(__file__, "4-resnet50.py", "resnet50")
    model = resnet50(num_classes=10).eval()
    with torch.no_grad():
        assert tuple(model(torch.zeros(2, 3, 224, 224)).shape) == (2, 10)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bns) == 53  # 1 stem + 16 blocks x 3 + 4 shortcuts


# ---------- Task 5 ----------
def test_dense_block():
    DenseBlock = load_task(__file__, "5-dense_block.py", "DenseBlock")
    block = DenseBlock(64, 32, 6)
    assert block.out_channels == 64 + 6 * 32
    x = torch.randn(2, 64, 14, 14)
    out = block(x)
    assert tuple(out.shape) == (2, 256, 14, 14)
    # the input is kept unchanged as the first channels
    assert torch.allclose(out[:, :64], x)


def test_dense_block_params():
    DenseBlock = load_task(__file__, "5-dense_block.py", "DenseBlock")
    k, cin, L = 12, 24, 3
    expected = 0
    for i in range(L):
        c = cin + i * k
        expected += bn_params(c) + conv_params(c, 4 * k, 1, False) + bn_params(4 * k) + conv_params(4 * k, k, 3, False)
    assert n_params(DenseBlock(cin, k, L)) == expected


# ---------- Task 6 ----------
def test_transition_layer():
    TransitionLayer = load_task(__file__, "6-transition_layer.py", "TransitionLayer")
    t = TransitionLayer(256, 0.5)
    assert t.out_channels == 128
    assert tuple(t(torch.randn(2, 256, 56, 56)).shape) == (2, 128, 28, 28)
    assert n_params(t) == bn_params(256) + conv_params(256, 128, 1, False)
    assert TransitionLayer(100, 0.33).out_channels == 33


# ---------- Task 7 ----------
def test_densenet121_matches_torchvision_param_count():
    densenet121 = load_task(__file__, "7-densenet121.py", "densenet121")
    tv = pytest.importorskip("torchvision")
    model = densenet121()
    assert n_params(model) == n_params(tv.models.densenet121(weights=None)) == 7978856


def test_densenet121_forward_and_options():
    densenet121 = load_task(__file__, "7-densenet121.py", "densenet121")
    model = densenet121(growth_rate=32, compression=0.5, num_classes=10).eval()
    with torch.no_grad():
        assert tuple(model(torch.zeros(1, 3, 224, 224)).shape) == (1, 10)
    assert n_params(densenet121(compression=1.0)) == 13908584
