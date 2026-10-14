# Multivariate Probability

**Status:** Original

## Why this matters
Real data has many features at once, and they move together: taller people tend to weigh more. The covariance matrix captures those relationships, and the multivariate normal distribution is the workhorse model built on it. You'll meet both again in PCA, Gaussian mixture models, Gaussian processes and variational autoencoders.

## Concepts to understand first
- **What is a joint distribution?** The probability of several random variables taking values *together*, like (height, weight) pairs, rather than each one separately.
- **What is the mean of a dataset with many features?** Just the per-feature average: a vector with one entry per dimension.
- **What is covariance?** How two variables vary together. Positive: when one is above its mean, the other tends to be too. Negative: they move in opposite directions. Near zero: no linear relationship. $\text{cov}(X, Y) = \frac{1}{n-1}\sum_i (x_i - \bar{x})(y_i - \bar{y})$.
- **What is a covariance matrix?** For $d$ features, a $d \times d$ symmetric matrix: entry $(i, j)$ is the covariance of features $i$ and $j$, and the diagonal holds each feature's variance. With centered data $X_c$ (rows = data points), it's $\frac{1}{n-1} X_c^\top X_c$.
- **Why divide by $n - 1$?** The sample mean is fitted to the same data, which makes the raw spread look slightly smaller than the true spread. Dividing by $n - 1$ (Bessel's correction) fixes that bias.
- **What is correlation?** Covariance rescaled to lie between -1 and 1: $\rho_{ij} = \frac{C_{ij}}{\sqrt{C_{ii}}\sqrt{C_{jj}}}$. It measures the *strength* of a linear relationship regardless of units.
- **What is the multivariate normal distribution?** The bell curve in $d$ dimensions, described by a mean vector $\mu$ and covariance matrix $\Sigma$. Its density is $p(x) = \frac{1}{\sqrt{(2\pi)^d \det \Sigma}} \exp\left(-\tfrac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)$.

## Resources
- [StatQuest (YouTube channel): search "Covariance, Clearly Explained" and "Pearson's Correlation, Clearly Explained"](https://www.youtube.com/@statquest)
- [Wikipedia: Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix)
- [Wikipedia: Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
- [Mathematics for Machine Learning, section 6.4-6.5 (free PDF)](https://mml-book.github.io/)
- [NumPy: linear algebra (`numpy.linalg`)](https://numpy.org/doc/stable/reference/routines.linalg.html)

## Rules
- `import numpy as np` only.
- You may **not** use `numpy.cov` (the tests check). `np.mean`, matrix products, `np.linalg.det` and `np.linalg.inv` are fine.
- Error messages must match exactly.
- Watch the data layout. Task 0 uses `(n, d)`: one row per data point. The `MultiNormal` class uses `(d, n)`: one **column** per data point.

## Tasks

### 0. Mean and Covariance
**File:** `0-mean_cov.py` · **Signature:** `mean_cov(X)`

`X` has shape `(n, d)`. Return `(mean, cov)`: `mean` with shape `(1, d)` and the sample covariance `cov` with shape `(d, d)`.
- `X` not a 2D NumPy array: `TypeError("X must be a 2D numpy.ndarray")`.
- `n < 2`: `ValueError("X must contain multiple data points")`.
```
X = 100000 samples from a 3D normal with mean [12, 30, 10]
    and covariance [[36, -30, 15], [-30, 100, -20], [15, -20, 25]]
mean_cov(X)  ->  mean ~ [[12. 30. 10.]], cov ~ that covariance matrix
```

### 1. Correlation
**File:** `1-correlation.py` · **Signature:** `correlation(C)`

`C` is a `(d, d)` covariance matrix. Return the `(d, d)` correlation matrix.
- `C` not a NumPy array: `TypeError("C must be a numpy.ndarray")`.
- Not 2D and square: `ValueError("C must be a 2D square matrix")`.
```
correlation(np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]]))
  ->  [[ 1.  -0.5  0.5] [-0.5  1.  -0.4] [ 0.5 -0.4  1. ]]
```

### 2. Initialize
**File:** `multinormal.py` · **Class:** `MultiNormal(data)`

`data` has shape `(d, n)`. Set public attributes `mean` (shape `(d, 1)`) and `cov` (shape `(d, d)`, the sample covariance).
- Not a 2D NumPy array: `TypeError("data must be a 2D numpy.ndarray")`.
- `n < 2`: `ValueError("data must contain multiple data points")`.
```
mn = MultiNormal(<same distribution as task 0, transposed to shape (3, n)>)
mn.mean.shape, mn.cov.shape  ->  (3, 1), (3, 3)
```

### 3. PDF
**File:** `multinormal.py` · **Method:** `pdf(self, x)`

`x` has shape `(d, 1)`. Return the density at `x` as a single number.
- Not a NumPy array: `TypeError("x must be a numpy.ndarray")`.
- Wrong shape: `ValueError("x must have the shape ({d}, 1)")` with the real `d` filled in, e.g. `x must have the shape (3, 1)`.
```
mn.pdf(<a (3, 1) point near the mean>)  ->  a small positive float (matches scipy.stats.multivariate_normal.pdf)
```

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 00-math/08-multivariate_prob
```
