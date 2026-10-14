# Hints: Dimensionality Reduction

Open one hint at a time. Try for a while before opening the next.

## 0. PCA

<details><summary>Hint 1</summary>

You need two things: the principal directions, and how much variance each one carries. One NumPy linear-algebra routine gives you both at once.
</details>

<details><summary>Hint 2</summary>

Singular value decomposition. The rows of the third matrix it returns are the directions, already sorted from most to least important. The variance along each is proportional to its singular value squared.
</details>

<details><summary>Hint 3</summary>

Turn the squared singular values into a running total, divided by the grand total, so it climbs from near 0 to exactly 1. The number of components you need is the first position where that running fraction reaches `var` (careful with off-by-one: position 0 means one component). Then take that many directions, as columns.
</details>

## 1. PCA v2

<details><summary>Hint 1</summary>

This time the data isn't centered for you, and you're told how many components to keep instead of how much variance.
</details>

<details><summary>Hint 2</summary>

Center first (subtract each column's mean), then reuse the same decomposition idea as task 0.
</details>

<details><summary>Hint 3</summary>

Once you have the top `ndim` directions as the columns of a `(d, ndim)` matrix, transforming the data is a single matrix product of the centered data with that matrix.
</details>

## 2. P_init

<details><summary>Hint 1</summary>

Most of this is setting up arrays of the right shape. The interesting part is `D`.
</details>

<details><summary>Hint 2</summary>

Expand $\|x_i - x_j\|^2 = \|x_i\|^2 + \|x_j\|^2 - 2\,x_i \cdot x_j$. Each piece can be computed for all pairs at once with broadcasting and one matrix product, no loops.
</details>

<details><summary>Hint 3</summary>

Perplexity is 2 raised to the entropy, so the entropy is the base-2 logarithm of the perplexity. Floating-point error can make tiny negative distances; clipping at zero keeps things clean.
</details>

## 3. Entropy

<details><summary>Hint 1</summary>

Two steps: build a probability distribution from the distances, then measure how spread out it is.
</details>

<details><summary>Hint 2</summary>

Each affinity is the exponential of minus beta times the distance; divide by the sum so they add to 1.
</details>

<details><summary>Hint 3</summary>

Entropy in bits uses log base 2: the negative sum of each probability times its log2. Check yourself: with `beta = 0` every probability is equal, and the entropy is log2 of how many there are.
</details>

## 4. P affinities

<details><summary>Hint 1</summary>

For each point you're solving "find beta so that entropy equals H". Entropy goes down as beta goes up (a sharper Gaussian means a less spread-out distribution). That one-way relationship is what makes binary search work.
</details>

<details><summary>Hint 2</summary>

Keep a lower and an upper bound for beta, starting with "none known". If the entropy is too high, beta must grow: raise the lower bound to the current beta and move up (double it if there's no upper bound yet, otherwise go halfway to the upper bound). Too low: the mirror image. Stop when you're within `tol`.
</details>

<details><summary>Hint 3</summary>

Row *i* of `D` includes the zero distance from point *i* to itself, which must be left out before computing `HP` and left as zero in `P`. At the end, symmetrize: add `P` to its transpose and divide by `2n`.
</details>

## 5. Q affinities

<details><summary>Hint 1</summary>

Same pairwise squared distance trick as task 2, but on `Y`.
</details>

<details><summary>Hint 2</summary>

Apply the Student-t kernel (one over one-plus-distance) to every entry, then force the diagonal to zero, because a point isn't its own neighbour.
</details>

<details><summary>Hint 3</summary>

`Q` is just `num` normalized by its total sum (one number for the whole matrix, not row by row).
</details>

## 6. Gradients

<details><summary>Hint 1</summary>

Look up the gradient formula in the t-SNE paper (section 3.3). It's a sum over all *j* of a weight times the vector from $y_j$ to $y_i$.
</details>

<details><summary>Hint 2</summary>

The weight for each pair is $(p_{ij} - q_{ij})$ times `num[i, j]`. Build that whole `(n, n)` weight matrix at once.
</details>

<details><summary>Hint 3</summary>

$\sum_j w_{ij}(y_i - y_j)$ splits into two parts: $y_i$ times the row-sum of the weights, minus the weight matrix applied to `Y`. A loop over points is fine if vectorizing feels hard at first, but the vectorized version is only a couple of operations.
</details>

## 7. Cost

<details><summary>Hint 1</summary>

This is the KL divergence formula, summed over every entry of the matrices.
</details>

<details><summary>Hint 2</summary>

The diagonal is zero, and log(0) is a problem. Raise both matrices to at least `1e-12` before taking logs.
</details>

## 8. t-SNE

<details><summary>Hint 1</summary>

This file is mostly glue: your PCA, then your P affinities, then a loop that calls your gradients.
</details>

<details><summary>Hint 2</summary>

Momentum needs memory of the previous `Y`. Keep the current and previous positions, and after each update shift them along.
</details>

<details><summary>Hint 3</summary>

Count iterations from 1 so that "every 100" lines up with the printouts. Exaggerate `P` before the loop and undo it right after iteration 100. Re-center after every update by subtracting the mean of each column.
</details>
