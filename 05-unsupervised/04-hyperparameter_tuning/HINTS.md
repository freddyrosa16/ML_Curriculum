# Hints: Hyperparameter Tuning

Open one hint at a time. Try for a while before opening the next.

## 0. Initialize Gaussian Process

<details><summary>Hint 1</summary>

The kernel needs the squared distance between every row of `X1` and every row of `X2`, giving an `(m, n)` table.
</details>

<details><summary>Hint 2</summary>

Same trick as pairwise distances in the clustering and t-SNE projects: either broadcast a difference over a new axis, or expand $\|a - b\|^2$ into three terms and use one matrix product.
</details>

<details><summary>Hint 3</summary>

Once you have the squared distances, the RBF formula is applied element-wise. In `__init__`, `K` is just the kernel of the training inputs with themselves.
</details>

## 1. Gaussian Process Prediction

<details><summary>Hint 1</summary>

You need three kernel matrices: training vs. training (you already have it), training vs. new points, and new vs. new.
</details>

<details><summary>Hint 2</summary>

Plug them into the mean and covariance formulas from the README's Concepts section. NumPy's linear-algebra module can invert `K`.
</details>

<details><summary>Hint 3</summary>

The mean comes out as a column; flatten it to `(s,)`. You only want the variance of each point on its own, which lives on the diagonal of the predicted covariance matrix.
</details>

## 2. Update Gaussian Process

<details><summary>Hint 1</summary>

The new point arrives as shape `(1,)` but `self.X` is `(t, 1)`. Make the shapes compatible before joining them.
</details>

<details><summary>Hint 2</summary>

Join along the first axis (stack as a new row), do the same for `Y`, and rebuild `K` with your kernel.
</details>

## 3. Initialize Bayesian Optimization

<details><summary>Hint 1</summary>

Most of this is storing values. The GP class comes from your `2-gp.py` via `__import__`.
</details>

<details><summary>Hint 2</summary>

Evenly spaced points including both ends is exactly what NumPy's linspace does. Then add an axis so the shape is `(ac_samples, 1)`.
</details>

## 4. Bayesian Optimization: Acquisition

<details><summary>Hint 1</summary>

Ask the GP for its mean and variance at every candidate, then turn the variance into a standard deviation.
</details>

<details><summary>Hint 2</summary>

The current best is the smallest value in the GP's `Y` when minimizing (largest when maximizing). Compute the improvement, then `Z`, then EI from the formula, all vectorized over candidates. SciPy's `norm` gives you the CDF and PDF.
</details>

<details><summary>Hint 3</summary>

At already-sampled points the standard deviation is 0 (or a tiny negative variance from rounding). Guard against dividing by zero and set EI to 0 there. The next point is the candidate where EI is largest.
</details>

## 5. Bayesian Optimization

<details><summary>Hint 1</summary>

The loop is: acquire, check for a repeat, evaluate, update.
</details>

<details><summary>Hint 2</summary>

Keep a record of the points you've proposed. Comparing NumPy arrays with Python's `in` doesn't do what you'd expect; compare values explicitly (and allow for floating-point closeness).
</details>

<details><summary>Hint 3</summary>

At the end, the best point is whichever stored `Y` is smallest (or largest). Use its index to pick both the `X` and the `Y`, and keep the `(1,)` shapes.
</details>

## 6. Tune a real model with Optuna

<details><summary>Hint 1</summary>

Optuna wants an objective function that takes a single `trial` argument and returns a number. Your data has to get in some other way: a function that builds and returns the objective (a closure) is a clean pattern.
</details>

<details><summary>Hint 2</summary>

Inside the objective, ask the trial for each hyperparameter with its `suggest_int`, `suggest_float` (use the log-scale option for learning rates and regularization strengths) and `suggest_categorical` methods. Then train and return validation accuracy.
</details>

<details><summary>Hint 3</summary>

`MLPClassifier` has built-in early stopping options. Scaling the inputs first (a scikit-learn pipeline with a standard scaler) makes training much more reliable. After `optimize`, the study object holds the best value, best parameters and the list of trials.
</details>
