# Object Detection (YOLO v3)

**Status:** Original + Modernized. You still write all of YOLO v3's post-processing (decoding boxes, filtering, non-max suppression) in NumPy. What changed: the original loaded a Keras `yolo.h5` model that you can't easily get or run today, so the `Yolo` class now accepts any model function (or none), and the tests feed it YOLO-shaped outputs. Images are loaded with Pillow and drawn with matplotlib instead of OpenCV. A bonus task compares your pipeline with a modern detector from the `ultralytics` package.

## Why this matters
Classification answers "what is in this image?". Detection answers "what is where?", with a box around every object. YOLO ("You Only Look Once") was the model that made detection fast enough for real-time video, and the post-processing steps you'll write (decode → filter → non-max suppression) are used by nearly every detector today, including the newest YOLO versions.

## Concepts to understand first
- **How does YOLO see an image?** It splits the image into a grid (YOLO v3 uses three grids: 13×13, 26×26 and 52×52 for a 416×416 input, for large, medium and small objects). Each grid cell predicts a few boxes, one per **anchor box**.
- **What is an anchor box?** A typical box shape (width, height in input pixels), picked by clustering the training boxes. The network predicts how to stretch and shift its anchor rather than a box from scratch. Each grid scale has 3 anchors.
- **What does each prediction contain?** 85 numbers for COCO: $t_x, t_y, t_w, t_h$ (box offsets), one objectness score ("is there an object here?") and 80 class scores. All are raw (before any sigmoid).
- **How do you turn the offsets into a box?** For the cell in column $c_x$ and row $c_y$ of a grid with $g_w$ columns and $g_h$ rows, and an anchor of size $(p_w, p_h)$:
  $b_x = (\sigma(t_x) + c_x)/g_w$, $b_y = (\sigma(t_y) + c_y)/g_h$, $b_w = p_w e^{t_w} / \text{input}_w$, $b_h = p_h e^{t_h} / \text{input}_h$.
  These are the box centre and size as fractions of the image. Multiply by the original image's width/height to get pixels, and convert centre/size to corners $(x_1, y_1, x_2, y_2)$.
- **What is the box score?** objectness × class probability (both after a sigmoid). Each box keeps only its best class.
- **What is IoU?** Intersection over Union: the overlap area of two boxes divided by the area they cover together. 1 means identical, 0 means no overlap.
- **What is non-max suppression (NMS)?** The model usually fires several overlapping boxes for one object. NMS keeps the highest-scoring box, deletes every other box of the same class that overlaps it by more than a threshold (IoU), and repeats with what's left.

## Resources
- [YOLOv3: An Incremental Improvement (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) — short and funny; section 2.1 has the box equations
- [You Only Look Once (Redmon et al., 2015)](https://arxiv.org/abs/1506.02640) — the original idea
- [d2l.ai 14.4: Anchor Boxes](https://d2l.ai/chapter_computer-vision/anchor.html) — IoU and NMS explained with code you can compare against your understanding
- [d2l.ai 14.3: Object Detection and Bounding Boxes](https://d2l.ai/chapter_computer-vision/bounding-box.html)
- [torchvision.ops.nms docs](https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html) — the library version of task 3
- [Ultralytics YOLO docs](https://docs.ultralytics.com/) — for the bonus task
- [COCO dataset](https://cocodataset.org/#explore) — the 80 classes the model knows

## Rules
- NumPy for all post-processing (`import numpy as np`). Pillow (`from PIL import Image`) for loading/resizing, matplotlib for drawing, `os` for files. No PyTorch, OpenCV or TensorFlow in `yolo.py`.
- Everything goes in **one file, `yolo.py`**, containing a class `Yolo`. Each task adds a method. The tests for a task are skipped until its method exists.
- Methods must not change the arrays you pass in.
- Images are NumPy arrays `(h, w, 3)`, RGB, `uint8`.

## Tasks

### 0. Initialize Yolo
**Method:** `__init__(self, model, classes_path, class_t, nms_t, anchors, input_size=(416, 416))`

- `model`: a function that takes a batch of preprocessed images `(n, input_h, input_w, 3)` and returns a list of raw outputs (one per grid scale), each of shape `(n, grid_h, grid_w, anchor_boxes, 4 + 1 + classes)`. May be `None`.
- `classes_path`: text file with one class name per line, in index order (use `data/coco_classes.txt`).
- `class_t`: box score threshold for filtering. `nms_t`: IoU threshold for NMS.
- `anchors`: array `(outputs, anchor_boxes, 2)` of anchor `[width, height]` in input pixels.
- `input_size`: `(input_h, input_w)` the model expects.
- Store them as `self.model`, `self.class_names` (list of strings), `self.class_t`, `self.nms_t`, `self.anchors`, `self.input_size`.

```python
>>> anchors = np.array([[[116, 90], [156, 198], [373, 326]],
...                     [[30, 61], [62, 45], [59, 119]],
...                     [[10, 13], [16, 30], [33, 23]]])
>>> yolo = Yolo(None, "data/coco_classes.txt", 0.6, 0.5, anchors)
>>> len(yolo.class_names), yolo.class_names[16]
(80, 'dog')
```

### 1. Process outputs
**Method:** `process_outputs(self, outputs, image_size)`

- `outputs`: list of raw arrays for **one** image, each `(grid_h, grid_w, anchor_boxes, 4 + 1 + classes)`.
- `image_size`: `[image_h, image_w]` of the original image.
- **Returns** `(boxes, box_confidences, box_class_probs)`, three lists with one array per output:
  - `boxes`: `(grid_h, grid_w, anchor_boxes, 4)` with `(x1, y1, x2, y2)` in original-image pixels (formula in Concepts),
  - `box_confidences`: `(grid_h, grid_w, anchor_boxes, 1)`, sigmoid of the objectness,
  - `box_class_probs`: `(grid_h, grid_w, anchor_boxes, classes)`, sigmoid of the class scores.

```python
>>> outs = [np.random.randn(g, g, 3, 85) for g in (13, 26, 52)]
>>> boxes, confs, probs = yolo.process_outputs(outs, np.array([500, 700]))
>>> boxes[0].shape, confs[1].shape, probs[2].shape
((13, 13, 3, 4), (26, 26, 3, 1), (52, 52, 3, 80))
```

### 2. Filter boxes
**Method:** `filter_boxes(self, boxes, box_confidences, box_class_probs)`

For every box, compute the score of each class (confidence × class probability) and keep the best class. Keep only boxes whose best score is **≥ `class_t`**.

**Returns** `(filtered_boxes, box_classes, box_scores)`: arrays of shape `(?, 4)`, `(?,)` and `(?,)`, gathered from all outputs (any order).

```python
>>> fb, fc, fs = yolo.filter_boxes(boxes, confs, probs)
>>> fb.shape[1], fb.shape[0] == fc.shape[0] == fs.shape[0]
(4, True)
```

### 3. Non-max suppression
**Method:** `non_max_suppression(self, filtered_boxes, box_classes, box_scores)`

Do NMS separately for each class: repeatedly keep the highest-scoring remaining box and drop the other boxes of that class whose IoU with it is **greater than** `nms_t`. Compute IoU with continuous coordinates: a box's area is $(x_2 - x_1)(y_2 - y_1)$, with no "+1".

**Returns** `(box_predictions, predicted_box_classes, predicted_box_scores)`, ordered by class index (ascending), and within a class by score (descending).

```python
>>> boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=float)
>>> yolo.non_max_suppression(boxes, np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
(array([[ 0.,  0., 10., 10.],
       [20., 20., 30., 30.]]), array([1, 1]), array([0.9, 0.7]))
```

### 4. Load images
**Method:** `load_images(folder_path)`, a `@staticmethod`

Load every image file (`.jpg`, `.jpeg`, `.png`, `.bmp`; ignore other files) in the folder, **sorted by file name**, as RGB `uint8` arrays.

**Returns** `(images, image_paths)`: a list of arrays and a list of path strings.

```python
>>> images, paths = Yolo.load_images("data/yolo")
>>> len(images), paths[0].endswith("dog.jpg"), images[0].shape
(6, True, (576, 768, 3))
```

### 5. Preprocess images
**Method:** `preprocess_images(self, images)`

Resize each image to `input_size` with bicubic interpolation and scale pixels to [0, 1].

**Returns** `(pimages, image_shapes)`: `(n, input_h, input_w, 3)` floats, and `(n, 2)` with each original `[h, w]`.

```python
>>> pimages, shapes = yolo.preprocess_images(images)
>>> pimages.shape, shapes[0]
((6, 416, 416, 3), array([576, 768]))
```

### 6. Show boxes
**Method:** `show_boxes(self, image, boxes, box_classes, box_scores, file_name, save_dir=None, show=True)`

Draw the image with matplotlib, a rectangle for every box, and a label above it with the class name and the score rounded to 2 decimals (e.g. `dog 0.98`). Use `file_name` as the title. If `save_dir` is given, create it if needed and save the figure there as `file_name`. If `show` is true, display it. Close the figure at the end.

```python
>>> yolo.show_boxes(images[0], pb, pc, ps, "dog.jpg", save_dir="detections", show=False)
>>> os.path.exists("detections/dog.jpg")
True
```

### 7. Predict
**Method:** `predict(self, folder_path, show=False, save_dir=None)`

Run the whole pipeline on every image in a folder: load, preprocess, call `self.model` **once** on the whole batch, then for each image process its outputs, filter, and apply NMS. If `show` or `save_dir` is set, call `show_boxes` for each image (using its file name).

**Returns** `(predictions, image_paths)`, where `predictions[i]` is the `(boxes, classes, scores)` tuple from NMS for image `i`.

```python
>>> predictions, paths = yolo.predict("data/yolo")    # with a real or fake model
>>> len(predictions), len(predictions[0])
(6, 3)
```

### 8. Compare with a modern YOLO (bonus)
**File:** `yolo_ultralytics.py` · **Signature:** `def ultralytics_detect(image_path, model_name="yolov8n.pt", conf=0.25):`

Install the package with `pip install ultralytics` (it is large, and it downloads `yolov8n.pt`, about 6 MB, on first use). Run its YOLO model on one image and return `(boxes, class_names, scores)`: an `(n, 4)` array of `(x1, y1, x2, y2)` pixels, a list of class-name strings, and an `(n,)` array. Run it on the images in `data/yolo/` and compare with what you'd expect. Then read how the ultralytics results object exposes boxes, and notice it already applied the same filter + NMS steps you wrote.

```python
>>> boxes, names, scores = ultralytics_detect("data/yolo/dog.jpg")
>>> sorted(set(names))
['bicycle', 'car', 'dog', 'truck']
```
(Exact detections can vary by model version.)

### 9. Write-up (optional)
Explain anchor boxes, IoU and NMS in a short post, with a picture of your detections.

## Data
- `data/coco_classes.txt`: the 80 COCO class names in YOLO's index order.
- `data/yolo/`: six test photos (dog, eagle, giraffe, horses, person, takagaki).
- No model weights are needed for the tasks: tests use hand-made outputs and a fake model. The original Darknet YOLO v3 weights (`yolov3.weights` from the YOLO author's site) need a separate converter to run in Python, so we skip them; task 8 gives you a real detector instead.

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 02-vision/05-object_detection -m "not slow"
pytest 02-vision/05-object_detection -m slow   # bonus, needs `pip install ultralytics`
```
