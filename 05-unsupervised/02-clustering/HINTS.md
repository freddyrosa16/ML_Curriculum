# Hints: Clustering

Open one hint at a time. Try for a while before opening the next.

## 0. Initialize K-means

<details><summary>Hint 1</summary>

You need a lower and an upper bound for every dimension (column) of the data.
</details>

<details><summary>Hint 2</summary>

NumPy's uniform sampler accepts arrays for its low and high bounds and broadcasts them against the requested size, so one call can produce all k centroids at once.
</details>

## 1. K-means

<details><summary>Hint 1</summary>

The hard part is "distance from every point to every centroid" without looping over points. Think about the shape `(n, k)`: one row per point, one column per centroid.
</details>

<details><summary>Hint 2</summary>

Insert a new axis into the data and into the centroids so that subtracting them broadcasts to `(n, k, d)`. Reduce over the last axis to get distances, then pick the column of the smallest distance in each row.
</details>

<details><summary>Hint 3</summary>

Your two loops are: one over iterations, one over clusters to recompute means (and handle empty clusters). Keep a copy of the old centroids before updating so you can compare and stop early. Make sure `clss` matches the final centroids when you return.
</details>

## 2. Variance

<details><summary>Hint 1</summary>

Reuse the "all point-to-centroid distances" idea from k-means.
</details>

<details><summary>Hint 2</summary>

For each point keep only the smallest squared distance, then add them all up.
</details>

## 3. Optimize k

<details><summary>Hint 1</summary>

This is a loop over k that calls your k-means and your variance, collecting results as you go.
</details>

<details><summary>Hint 2</summary>

`d_vars` is relative to the first k tried, so the first entry is always 0 and the rest grow as clusters are added.
</details>

<details><summary>Hint 3</summary>

Watch the validation rules: `kmax` defaults to the number of points, and `kmin` must be strictly smaller than `kmax`.
</details>

## 4. Initialize GMM

<details><summary>Hint 1</summary>

All three outputs are "starting guesses": equal weights, k-means centres, and the simplest possible covariance.
</details>

<details><summary>Hint 2</summary>

To build k stacked identity matrices without a loop, look at how NumPy's tile or broadcasting can repeat one `(d, d)` matrix k times.
</details>

## 5. PDF

<details><summary>Hint 1</summary>

Write down the multivariate normal density formula: a normalizing constant (involving the determinant of S) times an exponential of a quadratic form (involving the inverse of S).
</details>

<details><summary>Hint 2</summary>

The quadratic form $(x-m)^T S^{-1} (x-m)$ must be computed for every row at once. Computing the full `(n, n)` matrix and taking its diagonal is wasteful (and forbidden here).
</details>

<details><summary>Hint 3</summary>

Multiply the centered data by the inverse covariance, then multiply that *element-wise* by the centered data again and sum across each row. That gives one number per point. Finish with the floor at `1e-300`.
</details>

## 6. Expectation

<details><summary>Hint 1</summary>

For each cluster, "prior times likelihood" gives an unnormalized score for every point. That's your one loop.
</details>

<details><summary>Hint 2</summary>

Stack the scores into a `(k, n)` array. Dividing each column by its sum gives the responsibilities (Bayes' rule).
</details>

<details><summary>Hint 3</summary>

That same column sum is the probability of each point under the whole mixture. The log-likelihood is the sum of their logs.
</details>

## 7. Maximization

<details><summary>Hint 1</summary>

Everything is a weighted version of the ordinary formulas: weighted count, weighted mean, weighted covariance.
</details>

<details><summary>Hint 2</summary>

The total responsibility of a cluster (sum across points) acts as its "effective number of points". The new weight is that divided by n. The new means for all clusters at once are a single matrix product of `g` with `X`, divided by those totals.
</details>

<details><summary>Hint 3</summary>

For each cluster's covariance (your one loop): center the data on that cluster's new mean, weight each centered point by its responsibility, and take the outer-product sum, divided by the cluster's total weight.
</details>

## 8. EM

<details><summary>Hint 1</summary>

This is a loop that calls expectation and maximization alternately, tracking the previous log-likelihood.
</details>

<details><summary>Hint 2</summary>

Order matters: run an expectation right after initializing (that's "iteration 0"), then each further iteration does maximization followed by expectation. That way the `g` and `l` you return always match the returned parameters.
</details>

<details><summary>Hint 3</summary>

Decide when to print before deciding whether to stop, so the final, converged log-likelihood still gets printed.
</details>

## 9. BIC

<details><summary>Hint 1</summary>

Loop over k, run EM, store everything, and compute the BIC for each k from the formula in the README.
</details>

<details><summary>Hint 2</summary>

Count parameters carefully: a symmetric `d × d` covariance only has d(d+1)/2 free numbers, and the weights lose one degree of freedom because they must sum to 1.
</details>

<details><summary>Hint 3</summary>

Finding the minimum gives you a position in your list. Convert it back to a k by adding `kmin`.
</details>

## 10. Hello, sklearn!

<details><summary>Hint 1</summary>

Create a KMeans estimator with the right number of clusters and fit it to X. The fitted estimator exposes the centres and labels as attributes ending in an underscore.
</details>

## 11. GMM with sklearn

<details><summary>Hint 1</summary>

Same pattern with GaussianMixture. Weights, means and covariances are fitted attributes; cluster labels come from predicting on X, and the estimator has a method for BIC.
</details>

## 12. Agglomerative

<details><summary>Hint 1</summary>

Two SciPy functions do the clustering: one builds the linkage (merge history) with the Ward method, the other cuts it into flat clusters.
</details>

<details><summary>Hint 2</summary>

When cutting, tell SciPy to treat your `dist` as a distance threshold. The dendrogram function has an option to colour branches below that same threshold.
</details>
