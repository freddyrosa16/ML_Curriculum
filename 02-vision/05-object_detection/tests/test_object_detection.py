"""Tests for 02-vision/05-object_detection.

No real YOLO weights are needed: the tests feed hand-made YOLO-shaped outputs
and a fake "model". torchvision's NMS is used only as a reference.
"""
import pathlib

import numpy as np
import pytest

from helpers import load_task

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
CLASSES = str(DATA / "coco_classes.txt")
ANCHORS = np.array([[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]])
GRIDS = (13, 26, 52)


def get_yolo(*methods):
    Yolo = load_task(__file__, "yolo.py", "Yolo")
    for m in methods:
        if not hasattr(Yolo, m):
            pytest.skip(f"Yolo.{m} not written yet")
    return Yolo


def make_yolo(*methods, class_t=0.6, nms_t=0.5, model=None):
    Yolo = get_yolo(*methods)
    return Yolo(model, CLASSES, class_t, nms_t, ANCHORS)


def blank_outputs(batch=None, value=-10.0):
    """Raw outputs where every confidence/class logit is very negative."""
    outs = []
    for g in GRIDS:
        shape = (g, g, 3, 85) if batch is None else (batch, g, g, 3, 85)
        out = np.zeros(shape)
        out[..., 4:] = value
        outs.append(out)
    return outs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------- Task 0: init ----------
def test_init_stores_everything():
    yolo = make_yolo()
    assert yolo.model is None
    assert len(yolo.class_names) == 80
    assert yolo.class_names[0] == "person" and yolo.class_names[16] == "dog"
    assert yolo.class_t == 0.6 and yolo.nms_t == 0.5
    assert np.array_equal(yolo.anchors, ANCHORS)
    assert tuple(yolo.input_size) == (416, 416)


def test_init_custom_input_size():
    Yolo = get_yolo()
    yolo = Yolo(None, CLASSES, 0.5, 0.4, ANCHORS, input_size=(320, 480))
    assert tuple(yolo.input_size) == (320, 480)


# ---------- Task 1: process_outputs ----------
def test_process_outputs_shapes_and_ranges():
    yolo = make_yolo("process_outputs")
    rng = np.random.default_rng(0)
    outs = [rng.standard_normal((g, g, 3, 85)) for g in GRIDS]
    originals = [o.copy() for o in outs]
    boxes, confs, probs = yolo.process_outputs(outs, np.array([500, 700]))
    assert len(boxes) == len(confs) == len(probs) == 3
    for g, b, c, p in zip(GRIDS, boxes, confs, probs):
        assert b.shape == (g, g, 3, 4)
        assert c.shape == (g, g, 3, 1)
        assert p.shape == (g, g, 3, 80)
        assert np.all((c > 0) & (c < 1)) and np.all((p > 0) & (p < 1))
        assert np.all(b[..., 2] > b[..., 0]) and np.all(b[..., 3] > b[..., 1])
    for o, orig in zip(outs, originals):
        assert np.array_equal(o, orig), "process_outputs must not modify its input"


def test_process_outputs_known_cell():
    yolo = make_yolo("process_outputs")
    outs = [np.zeros((g, g, 3, 85)) for g in GRIDS]
    outs[1][4, 7, 2, :4] = [1.0, -0.5, 0.3, 0.2]   # row 4, column 7, anchor 2, 26x26 grid
    outs[1][4, 7, 2, 4] = 2.0
    outs[1][4, 7, 2, 5 + 16] = -1.0
    H, W = 500, 700
    boxes, confs, probs = yolo.process_outputs(outs, np.array([H, W]))
    bx = (sigmoid(1.0) + 7) / 26
    by = (sigmoid(-0.5) + 4) / 26
    bw = 59 * np.exp(0.3) / 416
    bh = 119 * np.exp(0.2) / 416
    expected = [(bx - bw / 2) * W, (by - bh / 2) * H, (bx + bw / 2) * W, (by + bh / 2) * H]
    assert np.allclose(boxes[1][4, 7, 2], expected)
    assert np.isclose(confs[1][4, 7, 2, 0], sigmoid(2.0))
    assert np.isclose(probs[1][4, 7, 2, 16], sigmoid(-1.0))
    # all-zero cell of the 13x13 grid, anchor 0: centred in its cell, anchor-sized
    bx, by = 0.5 / 13, 0.5 / 13
    bw, bh = 116 / 416, 90 / 416
    expected = [(bx - bw / 2) * W, (by - bh / 2) * H, (bx + bw / 2) * W, (by + bh / 2) * H]
    assert np.allclose(boxes[0][0, 0, 0], expected)


def test_process_outputs_non_square_grid():
    yolo = make_yolo("process_outputs")
    out = np.zeros((2, 3, 3, 85))      # 2 rows, 3 columns
    boxes, _, _ = yolo.process_outputs([out], np.array([416, 416]))
    centers_x = (boxes[0][..., 0] + boxes[0][..., 2]) / 2
    centers_y = (boxes[0][..., 1] + boxes[0][..., 3]) / 2
    assert np.allclose(centers_x[1, 2, 0], (2.5 / 3) * 416)
    assert np.allclose(centers_y[1, 2, 0], (1.5 / 2) * 416)


# ---------- Task 2: filter_boxes ----------
def test_filter_boxes_keeps_only_confident_boxes():
    yolo = make_yolo("process_outputs", "filter_boxes", class_t=0.6)
    outs = blank_outputs()
    outs[0][5, 5, 1, 4] = 5.0          # confident box ...
    outs[0][5, 5, 1, 5 + 16] = 5.0     # ... that is a dog: score ~0.987
    outs[2][10, 20, 0, 4] = 5.0
    outs[2][10, 20, 0, 5 + 2] = 0.0    # car with prob 0.5 -> score ~0.497, dropped
    outs[1][3, 3, 0, 4] = 3.0
    outs[1][3, 3, 0, 5 + 0] = 3.0      # person: score ~0.907
    boxes, confs, probs = yolo.process_outputs(outs, np.array([416, 416]))
    fb, fc, fs = yolo.filter_boxes(boxes, confs, probs)
    assert fb.shape == (2, 4) and fc.shape == (2,) and fs.shape == (2,)
    order = np.argsort(fc)
    assert fc[order].tolist() == [0, 16]
    assert np.allclose(fs[order], [sigmoid(3.0) ** 2, sigmoid(5.0) ** 2])
    assert np.allclose(fb[order][1], boxes[0][5, 5, 1])


def test_filter_boxes_threshold_is_inclusive_and_empty_ok():
    yolo = make_yolo("filter_boxes", class_t=0.25)
    boxes = [np.ones((1, 1, 1, 4))]
    confs = [np.full((1, 1, 1, 1), 0.5)]
    probs = [np.array([[[[0.5, 0.1]]]])]
    fb, fc, fs = yolo.filter_boxes(boxes, confs, probs)
    assert fc.tolist() == [0] and np.allclose(fs, [0.25])
    yolo.class_t = 0.9
    fb, fc, fs = yolo.filter_boxes(boxes, confs, probs)
    assert fb.shape == (0, 4) and fc.shape == (0,) and fs.shape == (0,)


# ---------- Task 3: non_max_suppression ----------
def test_nms_hand_example():
    yolo = make_yolo("non_max_suppression", nms_t=0.5)
    boxes = np.array([
        [0, 0, 10, 10],     # class 1, 0.9  -> kept
        [1, 1, 11, 11],     # class 1, 0.8  -> IoU 81/119 = 0.68 with box 0: removed
        [20, 20, 30, 30],   # class 1, 0.7  -> no overlap: kept
        [0, 0, 10, 10],     # class 0, 0.6  -> other class: kept
        [5, 0, 15, 10],     # class 1, 0.95 -> IoU 50/150 = 0.33 with box 0: both kept
    ], dtype=float)
    classes = np.array([1, 1, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.95])
    pb, pc, ps = yolo.non_max_suppression(boxes, classes, scores)
    assert pc.tolist() == [0, 1, 1, 1]
    assert np.allclose(ps, [0.6, 0.95, 0.9, 0.7])
    assert np.allclose(pb, boxes[[3, 4, 0, 2]])


def test_nms_matches_torchvision():
    torch = pytest.importorskip("torch")
    ops = pytest.importorskip("torchvision.ops")
    yolo = make_yolo("non_max_suppression", nms_t=0.4)
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 300, (200, 2))
    wh = rng.uniform(20, 120, (200, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1)
    classes = rng.integers(0, 3, 200)
    scores = rng.uniform(0.5, 1, 200)
    pb, pc, ps = yolo.non_max_suppression(boxes, classes, scores)
    keep = ops.batched_nms(torch.tensor(boxes), torch.tensor(scores), torch.tensor(classes), 0.4).numpy()
    assert len(pb) == len(keep)
    assert sorted(np.round(ps, 10).tolist()) == sorted(np.round(scores[keep], 10).tolist())
    # sorted by class, then by descending score
    assert np.all(np.diff(pc) >= 0)
    for c in np.unique(pc):
        assert np.all(np.diff(ps[pc == c]) <= 0)


# ---------- Task 4: load_images ----------
def test_load_images(tmp_path):
    Yolo = get_yolo("load_images")
    images, paths = Yolo.load_images(str(DATA / "yolo"))
    names = [pathlib.Path(p).name for p in paths]
    assert names == sorted(["dog.jpg", "eagle.jpg", "giraffe.jpg", "horses.jpg", "person.jpg", "takagaki.jpg"])
    assert len(images) == 6
    for img in images:
        assert isinstance(img, np.ndarray) and img.dtype == np.uint8
        assert img.ndim == 3 and img.shape[2] == 3
    # RGB, not BGR: make a pure red PNG and load it
    from PIL import Image
    Image.new("RGB", (8, 4), (255, 0, 0)).save(tmp_path / "red.png")
    (tmp_path / "notes.txt").write_text("not an image")
    imgs, ps = Yolo.load_images(str(tmp_path))
    assert len(imgs) == 1 and imgs[0].shape == (4, 8, 3)
    assert imgs[0][0, 0].tolist() == [255, 0, 0]


# ---------- Task 5: preprocess_images ----------
def test_preprocess_images():
    yolo = make_yolo("preprocess_images")
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (300, 500, 3), dtype=np.uint8),
              np.full((100, 80, 3), 255, dtype=np.uint8)]
    pimages, shapes = yolo.preprocess_images(images)
    assert pimages.shape == (2, 416, 416, 3)
    assert pimages.min() >= 0 and pimages.max() <= 1
    assert np.allclose(pimages[1], 1.0, atol=1e-3)
    assert np.array_equal(shapes, [[300, 500], [100, 80]])


# ---------- Task 6: show_boxes ----------
def test_show_boxes_saves_file(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    yolo = make_yolo("show_boxes")
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    yolo.show_boxes(image, np.array([[10, 20, 60, 80]]), np.array([16]), np.array([0.93]),
                    "test.png", save_dir=str(tmp_path / "detections"), show=False)
    assert (tmp_path / "detections" / "test.png").exists()


# ---------- Task 7: predict ----------
class FakeModel:
    """Pretends to detect one dog in every image, in the 13x13 grid cell (6, 6)."""

    def __init__(self):
        self.calls = []

    def __call__(self, batch):
        self.calls.append(batch.shape)
        outs = blank_outputs(batch=batch.shape[0])
        outs[0][:, 6, 6, 0, 4] = 6.0
        outs[0][:, 6, 6, 0, 5 + 16] = 6.0
        outs[0][:, 6, 6, 1, 4] = 5.0      # a second, overlapping dog box -> removed by NMS
        outs[0][:, 6, 6, 1, 5 + 16] = 5.0
        return outs


def test_predict_with_fake_model(tmp_path):
    model = FakeModel()
    yolo = make_yolo("predict", "process_outputs", "filter_boxes", "non_max_suppression",
                     "load_images", "preprocess_images", model=model, nms_t=0.3)
    predictions, paths = yolo.predict(str(DATA / "yolo"))
    assert model.calls == [(6, 416, 416, 3)], "call the model once on the whole batch"
    assert len(predictions) == len(paths) == 6
    for (boxes, classes, scores), path in zip(predictions, paths):
        assert classes.tolist() == [16]
        assert np.isclose(scores[0], sigmoid(6.0) ** 2)
        from PIL import Image
        w, h = Image.open(path).size
        cx = (boxes[0][0] + boxes[0][2]) / 2
        cy = (boxes[0][1] + boxes[0][3]) / 2
        assert np.isclose(cx, w / 2) and np.isclose(cy, h / 2)


# ---------- Bonus: ultralytics ----------
@pytest.mark.slow
def test_ultralytics_finds_the_dog():
    pytest.importorskip("ultralytics")
    detect = load_task(__file__, "yolo_ultralytics.py", "ultralytics_detect")
    boxes, names, scores = detect(str(DATA / "yolo" / "dog.jpg"))
    assert len(boxes) == len(names) == len(scores)
    assert "dog" in names
