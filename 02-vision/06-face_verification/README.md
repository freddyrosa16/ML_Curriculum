# Face Verification

**Status:** Restored. This project was in the 2019-2021 version of the original course and later dropped. It's rebuilt with modern tools: no dlib (hard to install), a pre-cropped face dataset from scikit-learn instead of manual face alignment, and PyTorch instead of Keras. The core ideas are the same: embeddings, triplet loss, one-shot verification with a distance threshold, and a required ethics write-up.

## Why this matters
A normal classifier can only recognize the classes it was trained on. Face verification has to work for people it has *never seen*: "are these two photos the same person?" The trick is to learn an **embedding**, a short vector per face, where photos of the same person land close together and different people land far apart. The same idea powers phone unlock, image search, recommendation systems and the text embeddings behind RAG in Stage 4. And because faces are personal, this is also the project where you think hard about bias, consent and surveillance.

## Concepts to understand first
- **Verification vs identification?** Verification is one-to-one: "is this person who they claim to be?" Identification is one-to-many: "who is this, out of everyone in my database?" Both use the same embeddings.
- **What is an embedding?** A function (here a CNN) that maps an image to a vector, e.g. 128 numbers. We normalize it to length 1, so all embeddings lie on a sphere and distances are comparable.
- **What is one-shot learning?** Recognizing someone from a single example photo. With a good embedding you don't retrain anything for a new person: you store one embedding and compare new photos against it.
- **What is triplet loss?** Take an anchor photo $a$, a positive $p$ (same person) and a negative $n$ (different person). We want $\lVert f(a)-f(p) \rVert^2 + \alpha < \lVert f(a)-f(n) \rVert^2$: the positive closer than the negative by at least a margin $\alpha$. The loss is $\max(\lVert f(a)-f(p) \rVert^2 - \lVert f(a)-f(n) \rVert^2 + \alpha,\ 0)$, which is zero once a triplet is "easy enough".
- **Why split by identity?** If the same person appears in both the training and test sets, you're measuring memorization. A fair test uses people the model has never seen.
- **How do you pick the threshold?** Compute distances for many pairs whose answer you know (same / different), then choose the threshold that makes the most correct calls. Lower thresholds mean fewer false matches but more false rejections.
- **What is an ROC curve?** For each possible threshold, plot the false positive rate (different people wrongly accepted) against the true positive rate (same person correctly accepted). It shows the whole trade-off at once.

## Resources
- [FaceNet: A Unified Embedding for Face Recognition and Clustering (Schroff et al., 2015)](https://arxiv.org/abs/1503.03832) — where triplet loss for faces comes from
- [Olivier Moindrot: Triplet Loss and Online Triplet Mining](https://omoindrot.github.io/triplet-loss) — clear explanation of triplets and how to choose them
- [scikit-learn: `fetch_lfw_people`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html)
- [StatQuest: ROC and AUC, clearly explained](https://www.youtube.com/watch?v=4jRBRDbJemM)
- [Gender Shades (Buolamwini & Gebru, 2018)](https://proceedings.mlr.press/v81/buolamwini18a.html) — error rates of face systems across skin type and gender
- [NIST: Face Recognition Vendor Test, Part 3: Demographic Effects (2019)](https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.8280.pdf)
- [Holberton School panel: The importance of ethics & diversity in Machine Learning (43 min)](https://www.youtube.com/watch?v=PWFpmaPjdjc) — for the ethics write-up
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — pretrained face detector (MTCNN) and embedding model, for the bonus

## Rules
- PyTorch for models and losses (`import torch`, `import torch.nn as nn`), NumPy for data handling, scikit-learn only to download LFW (task 0).
- Images are grayscale, shape `(n, h, w)`, floats in [0, 1]. Models take `(N, 1, H, W)` tensors.
- Distances are **squared** Euclidean distances between embeddings.
- Import earlier tasks with `__import__('2-generate_triplets').generate_triplets`, etc.

## Tasks

### 0. Load faces
**File:** `0-load_faces.py` · **Signature:** `def load_faces(min_faces_per_person=20, resize=0.5, data_home=None):`

Download (first time only) and load LFW's pre-cropped faces with scikit-learn's `fetch_lfw_people`, keeping people with at least `min_faces_per_person` photos.
**Returns** `(images, labels, names)`: `float32` images `(n, h, w)` scaled to [0, 1] (check what range scikit-learn gives you!), integer labels `(n,)`, and the array of person names (index = label).
```python
>>> images, labels, names = load_faces()
>>> images.shape, len(names)
((3023, 62, 47), 62)
```

### 1. Split by identity
**File:** `1-split_by_identity.py` · **Signature:** `def split_by_identity(labels, test_fraction=0.2, seed=0):`

Choose `round(test_fraction × number_of_people)` people at random (at least 1) using `np.random.default_rng(seed)`, and put **all** of their photos in the test set.
**Returns** `(train_idx, test_idx)`: index arrays that together cover every photo exactly once, with no person in both.
```python
>>> train_idx, test_idx = split_by_identity(labels, 0.2)
>>> len(set(labels[train_idx]) & set(labels[test_idx]))
0
```

### 2. Generate triplets
**File:** `2-generate_triplets.py` · **Signature:** `def generate_triplets(labels, n_triplets, seed=0):`

Build `n_triplets` random triplets of indices `(anchor, positive, negative)`: anchor and positive are two **different** photos of the same person, and the negative is a photo of someone else. Only people with 2+ photos can be anchors (anyone can be a negative). Use `np.random.default_rng(seed)`. Raise a `ValueError` if no person has 2+ photos or there's only one person.
**Returns** an integer array `(n_triplets, 3)`.
```python
>>> generate_triplets(labels, 4)
array([[1520,  974,   12],
       [ 233, 2911,  808],
       [2045, 1688, 2990],
       [ 671,  402, 1311]])      # your numbers will differ
```

### 3. Triplet loss
**File:** `3-triplet_loss.py` · **Signature:** `def triplet_loss(anchor, positive, negative, alpha=0.2, reduction="mean"):`

Inputs are tensors `(n, d)` of embeddings. Compute the triplet loss from Concepts for each row. With `reduction="mean"` return the mean (a 0-dimensional tensor); with `"none"` return the `(n,)` per-triplet losses; anything else raises `ValueError`. It must work with autograd.
```python
>>> a = torch.tensor([[0., 0.]]); p = torch.tensor([[1., 0.]]); n = torch.tensor([[0., .5]])
>>> triplet_loss(a, p, n)
tensor(0.9500)
```

### 4. Embedding network
**File:** `4-embedding_net.py` · **Class:** `EmbeddingNet(nn.Module)` · **Init:** `__init__(self, in_channels=1, embedding_dim=128)`

A small CNN: a few conv → batch norm → ReLU → max-pool blocks, then a global average pool, then a linear layer to `embedding_dim`, then L2-normalize each output to length 1. It must accept any image size of at least 16×16. Keep it small (between 10 thousand and 2 million parameters).
```python
>>> net = EmbeddingNet()
>>> e = net(torch.rand(5, 1, 62, 47))
>>> e.shape, e.norm(dim=1)
(torch.Size([5, 128]), tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], grad_fn=...))
```

### 5. Train with triplets
**File:** `5-train_triplet.py` · **Signature:** `def train_triplet(model, images, labels, epochs=10, triplets_per_epoch=1024, batch_size=64, lr=1e-3, alpha=0.2, device=None, seed=0):`

- Accept `images` as `(n, h, w)` (add the channel axis yourself) or `(n, 1, h, w)`.
- Seed with `torch.manual_seed(seed)`. Each epoch, generate fresh triplets (task 2, with seed `seed + epoch`), go through them in batches, embed anchors, positives and negatives with the model, and minimize the triplet loss (task 3) with Adam.
- `device`: if `None`, pick cuda, then mps, then cpu.
- **Returns** a list with the mean loss of each epoch (plain floats).
```python
>>> history = train_triplet(EmbeddingNet(), images[train_idx], labels[train_idx], epochs=10)
>>> [round(h, 3) for h in history]
[0.152, 0.121, 0.108, ...]      # should go down
```

### 6. Verification
**File:** `6-verification.py` · **Class:** `FaceVerification`

- `__init__(self, model, threshold, device=None)`: store the model (in eval mode, on `device`, default `"cpu"`) and the threshold.
- `embed(self, images)`: images `(n, h, w)` or `(n, 1, h, w)` → NumPy embeddings `(n, d)`. No gradients.
- `distance(e1, e2)`, a `@staticmethod`: squared Euclidean distance along the last axis (works for two vectors, or a matrix against a vector).
- `verify(self, image1, image2)`: two single images `(h, w)` → `(is_same, distance)`, where `is_same` is `distance <= threshold` (a bool) and `distance` a Python float.
```python
>>> fv = FaceVerification(model, threshold=0.9)
>>> fv.verify(images[0], images[1])
(True, 0.4127)
```

### 7. ROC and threshold selection
**File:** `7-threshold.py` · **Signatures:** `def roc_points(distances, is_same, thresholds):` and `def best_threshold(distances, is_same):`

Both take distances for many pairs and booleans saying which pairs really are the same person. A pair is predicted "same" when `distance <= t`.
- `roc_points` returns `(fpr, tpr)` arrays, one value per threshold.
- `best_threshold` tries every distinct distance value as the threshold and returns `(threshold, accuracy)` for the most accurate one (the smallest threshold if there's a tie).
```python
>>> best_threshold([0.1, 0.2, 0.9, 1.0], [True, True, False, False])
(0.2, 1.0)
```
Use it on pairs built from your **test** people to choose `FaceVerification`'s threshold, and plot the ROC curve with matplotlib.

### 8. Ethics write-up (required)
**File:** `ethics.md` (in this folder)

Write at least 250 words, in your own words, covering:
- **Bias:** who is in LFW (look at the names and photos), and what happens when error rates differ between groups of people.
- **Consent:** where face datasets come from and whether the people in them agreed.
- **Surveillance:** how the same technology can be used for phone unlock and for tracking people in public; where you'd draw the line.

Use at least one real source (the Gender Shades paper, the NIST report, the Holberton panel, or news reports). The test checks only the length and that the three topics are mentioned; the thinking is up to you.

### 9. End-to-end on LFW (bonus)
Put it together: load LFW, split by identity, train on the training people, build same/different pairs from the test people, pick a threshold, and report test accuracy and the ROC curve. Then compare with a pretrained model: `pip install facenet-pytorch` gives you MTCNN (face detection/alignment) and an InceptionResnetV1 embedding model trained on millions of faces. (It pins specific torch versions, so install it in a separate virtual environment.) How far is your small model from it, and why?

## Data
- **LFW (Labeled Faces in the Wild):** downloaded by `sklearn.datasets.fetch_lfw_people` on first use (about 200 MB, cached in `~/scikit_learn_data`). With `resize=0.5` the faces are 62×47 grayscale, already cropped around the face. Only task 0's test (marked slow) downloads it.
- All other tests use small synthetic "faces" (random patterns plus noise), so they run in seconds.

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 02-vision/06-face_verification -m "not slow"
pytest 02-vision/06-face_verification -m slow   # downloads LFW
```
