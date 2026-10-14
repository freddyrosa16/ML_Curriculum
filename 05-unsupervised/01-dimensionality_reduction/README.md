# Dimensionality Reduction

**Status:** Original (PCA), plus the t-SNE tasks from the full original project restored as bonus tasks.

## Why this matters

Real data often has hundreds of columns, but most of the useful information lives in just a few directions. PCA finds those directions so you can compress data, remove noise, and speed up other models. t-SNE goes further: it squeezes high-dimensional data into 2-D so you can *see* clusters with your own eyes. You'll use both constantly for exploring datasets and inspecting embeddings.

## Concepts to understand first

- **What does "variance along a direction" mean?** Project every point onto a line and measure how spread out the projections are. PCA looks for the line (then the next, perpendicular line, and so on) where that spread is largest.
- **What is an eigenvector of a covariance matrix?** A direction that the matrix only stretches, never rotates. For the covariance matrix $X^TX$, the eigenvectors are the principal directions and the eigenvalues are how much variance lies along each one.
- **What does SVD give you?** Any matrix can be written $X = U \Sigma V^T$. The columns of $V$ are the principal directions, and the squared singular values $\sigma_i^2$ are proportional to the variance along each direction. That's why "fraction of variance kept" uses $\sigma^2$, not $\sigma$.
- **Why must the data be centered first?** PCA measures spread around the mean. If the data isn't centered, the first "direction" just points at the mean instead of describing the spread.
- **What is perplexity in t-SNE?** Roughly "how many neighbours each point should care about". t-SNE tunes a separate Gaussian width for every point so that its neighbourhood has the perplexity you asked for.
- **What is entropy (in bits)?** $H = -\sum p \log_2 p$ measures how spread out a probability distribution is. Perplexity is $2^H$, so hitting a target perplexity means hitting a target entropy.
- **What is KL divergence?** $\mathrm{KL}(P\|Q)=\sum P \log(P/Q)$ measures how badly distribution $Q$ approximates $P$. t-SNE moves the 2-D points to make $Q$ (similarities in 2-D) match $P$ (similarities in the original space).
- **Why a Student-t distribution in 2-D?** Its heavy tails let moderately distant points sit far apart in the map, which fixes the "everything gets crushed into the middle" problem.

## Resources

- [StatQuest: PCA, step by step](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [3Blue1Brown: Eigenvectors and eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues)
- [Mathematics for Machine Learning, chapter 10 (PCA)](https://mml-book.github.io/)
- [StatQuest: t-SNE, clearly explained](https://www.youtube.com/watch?v=NEaUSP4YerM)
- [van der Maaten & Hinton (2008): Visualizing Data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html)
- [Distill: How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- [Laurens van der Maaten's t-SNE page](https://lvdmaaten.github.io/tsne/)

## Rules

- NumPy only: `import numpy as np`. No scikit-learn in your solutions.
- Later files may import earlier ones, e.g. the t-SNE file can reuse your PCA, affinities, gradients and cost.
- Start each file with `#!/usr/bin/env python3` and give every module and function a docstring.

## Tasks

### 0. PCA

**File:** `0-pca.py` · **Function:** `pca(X, var=0.95)`

- `X`: array of shape `(n, d)`, already centered (every column has mean 0).
- `var`: the fraction of the total variance the projection must keep.
- **Returns** `W`, shape `(d, nd)`: the first `nd` principal directions as columns, where `nd` is the **smallest** number of components whose variance adds up to at least `var` of the total.
- Variance along component *i* is proportional to the *square* of its singular value. (The original course summed the singular values themselves, which isn't the variance. Use the squares.)

Example, with `X` a centered `(50, 6)` dataset whose columns are `a, b, c, 2a, -5b, 10c` for three independent standard-normal columns `a, b, c`:

```
>>> W = pca(X)
>>> W.shape
(6, 2)
>>> W = pca(X, 0.99)
>>> W.shape
(6, 3)
```
With 3 components, projecting to `X @ W` and back with `@ W.T` rebuilds `X` almost perfectly (mean squared error around `1e-29`), because the data really only has 3 independent directions.

### 1. PCA v2

**File:** `1-pca.py` · **Function:** `pca(X, ndim)`

- `X`: array of shape `(n, d)`, **not** necessarily centered.
- `ndim`: the number of dimensions to keep.
- **Returns** `T`, shape `(n, ndim)`: `X` projected onto its top `ndim` principal directions.

```
>>> T = pca(X, 50)      # X is (2500, 784)
>>> T.shape
(2500, 50)
```
Signs of individual columns may differ from someone else's answer. That's fine: a principal direction and its negative describe the same line.

### 2. Initialize t-SNE (bonus)

**File:** `2-P_init.py` · **Function:** `P_init(X, perplexity)`

- `X`: `(n, d)` dataset. `perplexity`: the target perplexity.
- **Returns** `(D, P, betas, H)`:
  - `D`, `(n, n)`: squared Euclidean distance between every pair of points, zeros on the diagonal.
  - `P`, `(n, n)`: all zeros (filled in by task 4).
  - `betas`, `(n, 1)`: all ones. `beta` is the precision $1/(2\sigma^2)$ of each point's Gaussian.
  - `H`: the entropy, in bits, that corresponds to `perplexity`.

```
>>> D, P, betas, H = P_init(np.array([[0., 0], [3, 4], [6, 8]]), 2.0)
>>> D
array([[  0.,  25., 100.],
       [ 25.,   0.,  25.],
       [100.,  25.,   0.]])
>>> betas.shape, H
((3, 1), 1.0)
```

### 3. Entropy (bonus)

**File:** `3-entropy.py` · **Function:** `HP(Di, beta)`

- `Di`: `(n - 1,)` squared distances from point *i* to every *other* point.
- `beta`: the precision of point *i*'s Gaussian.
- **Returns** `(Hi, Pi)`: `Pi` is the `(n - 1,)` array of affinities $p_{j|i} \propto e^{-\beta D_{ij}}$ (normalized to sum to 1), and `Hi` is its Shannon entropy in bits.

```
>>> HP(np.array([1., 2., 4., 8.]), 0.5)
(1.4720..., array([0.5377, 0.3261, 0.12  , 0.0162]))
```

### 4. P affinities (bonus)

**File:** `4-P_affinities.py` · **Function:** `P_affinities(X, tol=1e-5, perplexity=30.0)`

- For every point, search for the `beta` whose entropy is within `tol` of the target `H` (binary search), and store that point's affinities in row *i* of `P` (diagonal stays 0).
- **Returns** the symmetric joint affinities `P`, shape `(n, n)`: average `P` with its transpose and divide by `n`, so the whole matrix sums to 1.
- Use your `P_init` and `HP`.

```
>>> X = np.array([[0., 0], [1, 0], [0, 2], [3, 3], [4, 1], [-2, 1]])
>>> P = P_affinities(X, perplexity=2.0)
>>> P.sum(), P[0, 1]
(1.0, 0.12947...)
```

### 5. Q affinities (bonus)

**File:** `5-Q_affinities.py` · **Function:** `Q_affinities(Y)`

- `Y`: `(n, ndim)` low-dimensional points.
- **Returns** `(Q, num)`: `num[i, j]` is $1/(1 + \|y_i - y_j\|^2)$ with zeros on the diagonal, and `Q` is `num` divided by its total.

```
>>> Q, num = Q_affinities(np.array([[0., 0], [3, 4], [0, 1]]))
>>> num
array([[0.    , 0.0385, 0.5   ],
       [0.0385, 0.    , 0.0526],
       [0.5   , 0.0526, 0.    ]])
>>> Q.sum()
1.0
```

### 6. Gradients (bonus)

**File:** `6-grads.py` · **Function:** `grads(Y, P)`

- **Returns** `(dY, Q)`: `dY` is the `(n, ndim)` gradient of the t-SNE cost with respect to `Y`, and `Q` comes from your `Q_affinities(Y)`.
- Leave out the constant factor 4 that appears in the paper's gradient formula (your learning rate absorbs it).
- The test checks your `dY` against a numerical gradient of your cost (task 7), so finish task 7 too.

```
>>> P = np.array([[0, .3, .2], [.3, 0, 0], [.2, 0, 0]])
>>> dY, Q = grads(np.array([[0., 0], [3, 4], [0, 1]]), P)
>>> dY
array([[-0.0309,  0.0703],
       [ 0.0238,  0.0341],
       [ 0.007 , -0.1044]])
```

### 7. Cost (bonus)

**File:** `7-cost.py` · **Function:** `cost(P, Q)`

- **Returns** the KL divergence $\sum P \log(P/Q)$ (natural log). Avoid `log(0)`: floor both `P` and `Q` at `1e-12` first.

```
>>> P = np.array([[0, .3, .2], [.3, 0, 0], [.2, 0, 0]])
>>> Q = np.array([[0, .1, .4], [.1, 0, 0], [.4, 0, 0]])
>>> cost(P, Q)
0.38190...
```

### 8. t-SNE (bonus)

**File:** `8-tsne.py` · **Function:** `tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500)`

- First reduce `X` to `idims` dimensions with your `1-pca.py`.
- Compute `P` with your `P_affinities`.
- Start `Y` from `np.random.randn(n, ndims)`.
- Run `iterations` steps of gradient descent with momentum: the new `Y` is the old `Y` minus `lr` times the gradient, plus `a` times the previous step. Use `a = 0.5` for the first 20 iterations and `a = 0.8` after that.
- **Early exaggeration:** multiply `P` by 4 for the first 100 iterations.
- Re-center `Y` (subtract its column means) after every update.
- Every 100 iterations, print `Cost at iteration {i}: {cost}` (iterations counted from 1).
- **Returns** `Y`, shape `(n, ndims)`.

Example, with `X` holding 60 points in 10-D drawn around 3 well-separated centres:

```
>>> Y = tsne(X, idims=5, perplexity=10.0, iterations=300, lr=100)
Cost at iteration 100: ...
Cost at iteration 200: ...
Cost at iteration 300: ...
>>> Y.shape
(60, 2)
```

### Write-up (optional)

Write a short post comparing PCA and t-SNE on the same dataset: what each one preserves, what each one throws away, and why distances between clusters in a t-SNE plot shouldn't be taken literally.

## Data

The tests use small synthetic data, so you don't need to download anything.

To try your code on real images (optional):
- The original course used `mnist2500_X.txt` and `mnist2500_labels.txt` (2,500 MNIST digits flattened to 784 numbers each). They come in the Python implementation zip on [Laurens van der Maaten's t-SNE page](https://lvdmaaten.github.io/tsne/). Load the text file with NumPy.
- Or, with no download: scikit-learn's built-in digits dataset (`sklearn.datasets.load_digits`, 1,797 8×8 digits). Using scikit-learn just to *load* data is fine; your solutions stay NumPy-only. Plot the 2-D t-SNE output with matplotlib, colouring each point by its digit label.

## Check your work

From the `ml-curriculum` folder:

```bash
pytest 05-unsupervised/01-dimensionality_reduction
```
