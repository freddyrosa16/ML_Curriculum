"""Tests for 02-vision/03-data_augmentation."""
import pytest

from helpers import load_task

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torchvision.transforms.v2 import functional as TF  # noqa: E402


def rand_uint8(shape=(3, 32, 48), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)


def rand_float(shape=(3, 32, 48), seed=0, high=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g) * high


# ---------- Task 0 ----------
def test_flip_small_example():
    flip_image = load_task(__file__, "0-flip.py", "flip_image")
    img = torch.arange(6, dtype=torch.uint8).reshape(1, 2, 3)
    assert torch.equal(flip_image(img), torch.tensor([[[2, 1, 0], [5, 4, 3]]], dtype=torch.uint8))


def test_flip_keeps_shape_dtype_and_is_involution():
    flip_image = load_task(__file__, "0-flip.py", "flip_image")
    img = rand_uint8()
    out = flip_image(img)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert torch.equal(out[:, :, 0], img[:, :, -1])
    assert torch.equal(flip_image(out), img)


# ---------- Task 1 ----------
def test_crop_shape_and_is_a_real_window():
    crop_image = load_task(__file__, "1-crop.py", "crop_image")
    # every pixel value is unique, so we can find where the crop came from
    img = torch.arange(40 * 50, dtype=torch.float32).reshape(1, 40, 50)
    torch.manual_seed(0)
    for _ in range(10):
        out = crop_image(img, (20, 30))
        assert out.shape == (1, 20, 30)
        top, left = divmod(int(out[0, 0, 0].item()), 50)
        assert torch.equal(out, img[:, top:top + 20, left:left + 30])


def test_crop_is_random_and_seeded():
    crop_image = load_task(__file__, "1-crop.py", "crop_image")
    img = rand_uint8((3, 64, 64))
    torch.manual_seed(1)
    outs = [crop_image(img, (16, 16)) for _ in range(8)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])
    torch.manual_seed(1)
    assert torch.equal(crop_image(img, (16, 16)), outs[0])


def test_crop_full_size_returns_image():
    crop_image = load_task(__file__, "1-crop.py", "crop_image")
    img = rand_uint8((3, 10, 12))
    assert torch.equal(crop_image(img, (10, 12)), img)


# ---------- Task 2 ----------
def test_rotate_small_example():
    rotate_image = load_task(__file__, "2-rotate.py", "rotate_image")
    img = torch.arange(6, dtype=torch.uint8).reshape(1, 2, 3)
    assert torch.equal(rotate_image(img), torch.tensor([[[2, 5], [1, 4], [0, 3]]], dtype=torch.uint8))


def test_rotate_non_square_exact():
    rotate_image = load_task(__file__, "2-rotate.py", "rotate_image")
    img = rand_uint8((3, 20, 30))
    out = rotate_image(img)
    assert out.shape == (3, 30, 20) and out.dtype == torch.uint8
    four = rotate_image(rotate_image(rotate_image(out)))
    assert torch.equal(four, img)


# ---------- Task 3 ----------
def test_contrast_factor_one_is_identity():
    change_contrast = load_task(__file__, "3-contrast.py", "change_contrast")
    img = rand_float()
    assert torch.allclose(change_contrast(img, 1.0, 1.0), img, atol=1e-6)


def test_contrast_fixed_factor():
    change_contrast = load_task(__file__, "3-contrast.py", "change_contrast")
    img = rand_uint8()
    out = change_contrast(img, 2.0, 2.0)
    assert out.dtype == torch.uint8 and out.shape == img.shape
    assert torch.equal(out, TF.adjust_contrast(img, 2.0))


def test_contrast_random_factor_in_range():
    change_contrast = load_task(__file__, "3-contrast.py", "change_contrast")
    img = rand_float()
    lo, hi = TF.adjust_contrast(img, 0.5), TF.adjust_contrast(img, 1.5)
    low, high = torch.minimum(lo, hi), torch.maximum(lo, hi)
    torch.manual_seed(0)
    outs = [change_contrast(img, 0.5, 1.5) for _ in range(5)]
    for out in outs:
        assert torch.all(out >= low - 1e-5) and torch.all(out <= high + 1e-5)
    assert any(not torch.allclose(outs[0], o) for o in outs[1:])


# ---------- Task 4 ----------
def test_brightness_zero_delta_is_identity():
    change_brightness = load_task(__file__, "4-brightness.py", "change_brightness")
    img = rand_float()
    assert torch.allclose(change_brightness(img, 0.0), img)


def test_brightness_scales_all_pixels_by_one_factor():
    change_brightness = load_task(__file__, "4-brightness.py", "change_brightness")
    img = rand_float(high=0.5) + 0.01   # no clipping possible with max_delta 0.5
    torch.manual_seed(0)
    factors = []
    for _ in range(5):
        out = change_brightness(img, 0.5)
        ratio = out / img
        assert torch.allclose(ratio, ratio.flatten()[0].expand_as(ratio), atol=1e-4)
        factors.append(ratio.flatten()[0].item())
    assert all(0.5 - 1e-4 <= f <= 1.5 + 1e-4 for f in factors)
    assert len({round(f, 4) for f in factors}) > 1


def test_brightness_uint8_stays_valid():
    change_brightness = load_task(__file__, "4-brightness.py", "change_brightness")
    img = rand_uint8()
    out = change_brightness(img, 0.9)
    assert out.dtype == torch.uint8 and out.shape == img.shape


# ---------- Task 5 ----------
def test_hue_matches_definition():
    change_hue = load_task(__file__, "5-hue.py", "change_hue")
    img = rand_float()
    out = change_hue(img, 0.2)
    assert out.shape == img.shape
    assert torch.allclose(out, TF.adjust_hue(img, 0.2), atol=1e-5)
    assert torch.allclose(change_hue(img, 0.0), img, atol=1e-5)


def test_hue_pure_red_to_green():
    change_hue = load_task(__file__, "5-hue.py", "change_hue")
    red = torch.zeros(3, 4, 4)
    red[0] = 1.0
    out = change_hue(red, 1 / 3)   # a third of a turn: red -> green
    assert torch.allclose(out[1], torch.ones(4, 4), atol=1e-4)
    assert torch.allclose(out[0], torch.zeros(4, 4), atol=1e-4)


def test_hue_out_of_range_raises():
    change_hue = load_task(__file__, "5-hue.py", "change_hue")
    with pytest.raises(ValueError):
        change_hue(rand_float(), 0.7)


# ---------- Task 6 ----------
def test_pipeline_output_on_tensor_and_pil():
    pipe_fn = load_task(__file__, "6-pipeline.py", "augmentation_pipeline")
    pipe = pipe_fn(48)
    img = rand_uint8((3, 60, 80))
    torch.manual_seed(0)
    out = pipe(img)
    assert out.shape == (3, 48, 48)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    pil = TF.to_pil_image(img)
    out2 = pipe(pil)
    assert out2.shape == (3, 48, 48) and out2.dtype == torch.float32


def test_pipeline_is_random():
    pipe_fn = load_task(__file__, "6-pipeline.py", "augmentation_pipeline")
    pipe = pipe_fn(32)
    img = rand_uint8((3, 60, 80))
    torch.manual_seed(0)
    outs = [pipe(img) for _ in range(4)]
    assert any(not torch.allclose(outs[0], o) for o in outs[1:])
