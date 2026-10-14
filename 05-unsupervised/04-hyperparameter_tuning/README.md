# Hyperparameter Tuning

**Status:** Original (Gaussian process and Bayesian optimization in NumPy). The last task, which tuned a Keras model with GPyOpt, now tunes a scikit-learn model with Optuna (GPyOpt is no longer maintained).

## Why this matters

Every model has knobs (learning rate, layer sizes, regularization) and each training run can be expensive. Bayesian optimization picks the next setting to try *cleverly*, using everything learned from previous runs, so you find good settings in far fewer runs than grid or random search. You'll build the core yourself: a Gaussian process that models the unknown function, and the Expected Improvement rule that decides where to look next. Then you'll use Optuna, the tool people actually use.

## Concepts to understand first

- **What is a Gaussian process (GP)?** A probability distribution over *functions*. Instead of fitting one curve, it keeps every plausible curve, weighted by how well it fits the points you've seen. At any new x it gives you a mean (best guess) and a variance (how unsure it is).
- **What does the kernel do?** It says how related two inputs are. The RBF kernel $k(x, x') = \sigma_f^2 \exp\!\left(-\frac{\|x - x'\|^2}{2l^2}\right)$ says nearby points have similar values. The length scale $l$ sets how far "nearby" reaches; $\sigma_f$ sets how far the function swings from its mean.
- **How does a GP predict?** With training inputs `X`, outputs `Y`, and new points `X_s`: mean $= K_s^T K^{-1} Y$ and covariance $= K_{ss} - K_s^T K^{-1} K_s$, where $K = k(X, X)$, $K_s = k(X, X_s)$, $K_{ss} = k(X_s, X_s)$. At a training point the variance drops to 0 (the GP is "noiseless").
- **What is an acquisition function?** A cheap score for "how worth it is it to evaluate the expensive function here next?" It balances *exploitation* (where the mean looks good) and *exploration* (where uncertainty is high).
- **What is Expected Improvement (EI)?** The average amount by which a new sample would beat your current best, counting "no improvement" as 0. For minimization, with mean $\mu$, standard deviation $\sigma$, current best $f^+$ and exploration bonus $\xi$: $\mathrm{imp} = f^+ - \mu - \xi$, $Z = \mathrm{imp}/\sigma$, $\mathrm{EI} = \mathrm{imp}\,\Phi(Z) + \sigma\,\phi(Z)$, where $\Phi$ and $\phi$ are the standard normal CDF and PDF. Where $\sigma = 0$, EI is 0.
- **Variance vs. standard deviation?** The GP's `predict` returns the *variance*. EI needs the *standard deviation*, its square root. Mixing them up is a classic bug.
- **What does Optuna add?** It runs the loop for you: it asks a "trial" for hyperparameter values, you train and report a score, and its sampler (TPE by default, or a Gaussian-process sampler) proposes the next values.

## Resources

- [Distill: A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)
- [Distill: Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/)
- [Rasmussen & Williams, Gaussian Processes for Machine Learning (free book), chapter 2](https://gaussianprocess.org/gpml/)
- [Snoek, Larochelle & Adams (2012): Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944)
- [Optuna documentation](https://optuna.readthedocs.io/en/stable/)
- [scikit-learn: MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

## Rules

- Tasks 0-5: `import numpy as np` only, plus `from scipy.stats import norm` in tasks 4-5 and your own earlier files via `__import__`.
- Each class file contains the whole class so far (task 1's file has everything from task 0 plus `predict`, and so on).
- Task 6: anything goes (Optuna, scikit-learn, PyTorch). It needs `optuna`, which isn't in the root requirements: `uv pip install optuna`.
- Start each file with `#!/usr/bin/env python3`; document every module, class and method.

All examples use the black-box function $f(x) = \sin(5x) + 2\sin(-2x)$ on the interval $[-\pi, 2\pi]$, with two starting points `X_init` of shape `(2, 1)` drawn uniformly from that interval after `np.random.seed(0)` (they come out as about 2.03085 and 3.59891), and `Y_init = f(X_init)`.

## Tasks

### 0. Initialize Gaussian Process

**File:** `0-gp.py` · **Class:** `GaussianProcess`

- `__init__(self, X_init, Y_init, l=1, sigma_f=1)`: store `X_init` as `self.X` and `Y_init` as `self.Y` (both `(t, 1)`), plus `self.l` and `self.sigma_f`. Store the covariance of the training inputs as `self.K`.
- `kernel(self, X1, X2)`: RBF kernel between `X1` `(m, 1)` and `X2` `(n, 1)`. **Returns** `(m, n)`.

```
>>> gp = GaussianProcess(X_init, Y_init, l=0.6, sigma_f=2)
>>> gp.X is X_init
True
>>> gp.K
array([[4.     , 0.13151],
       [0.13151, 4.     ]])
```

### 1. Gaussian Process Prediction

**File:** `1-gp.py` · **Method:** `predict(self, X_s)`

- `X_s`: `(s, 1)` points to predict at.
- **Returns** `(mu, sigma)`: both `(s,)`, the mean and the **variance** at each point.

```
>>> mu, sigma = gp.predict(np.array([[0.], [1.5], [2.]]))
>>> mu
array([0.00326, 0.67281, 0.93343])
>>> sigma
array([3.99996, 2.16988, 0.01049])
```
Notice how the variance is almost `sigma_f**2 = 4` far from the data and nearly 0 right next to the training point at 2.03.

### 2. Update Gaussian Process

**File:** `2-gp.py` · **Method:** `update(self, X_new, Y_new)`

- `X_new`, `Y_new`: each `(1,)`, a new sample point and its value.
- Append them to `self.X` and `self.Y` and recompute `self.K`. Returns nothing.

```
>>> gp.update(np.array([1.0]), f(np.array([1.0])))
>>> gp.X.shape, gp.K.shape
((3, 1), (3, 3))
```

### 3. Initialize Bayesian Optimization

**File:** `3-bayes_opt.py` · **Class:** `BayesianOptimization`

- `__init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True)`:
  - `self.f`: the black-box function.
  - `self.gp`: a `GaussianProcess` (from `2-gp.py`) built from `X_init`, `Y_init`, `l`, `sigma_f`.
  - `self.X_s`: `(ac_samples, 1)`, evenly spaced candidate points from `bounds[0]` to `bounds[1]` inclusive.
  - `self.xsi`: the exploration bonus. `self.minimize`: `True` to minimize, `False` to maximize.

```
>>> bo = BayesianOptimization(f, X_init, Y_init, (-np.pi, 2 * np.pi), 50, l=2, sigma_f=3, xsi=0.05)
>>> bo.X_s.shape, bo.xsi, bo.minimize
((50, 1), 0.05, True)
```

### 4. Bayesian Optimization: Acquisition

**File:** `4-bayes_opt.py` · **Method:** `acquisition(self)`

- Compute Expected Improvement at every candidate in `self.X_s` (see Concepts). For maximization, the improvement is flipped: $\mathrm{imp} = \mu - f^+ - \xi$ where $f^+$ is the best (largest) value so far.
- **Returns** `(X_next, EI)`: `X_next` `(1,)` is the candidate with the largest EI; `EI` is `(ac_samples,)`.

```
>>> bo = BayesianOptimization(f, X_init, Y_init, (-np.pi, 2 * np.pi), 50, l=0.6, sigma_f=2, xsi=0.05)
>>> X_next, EI = bo.acquisition()
>>> X_next, EI.shape
(array([4.16742]), (50,))
```

### 5. Bayesian Optimization

**File:** `5-bayes_opt.py` · **Method:** `optimize(self, iterations=100)`

- Repeat up to `iterations` times: pick the next point with `acquisition`, evaluate `self.f` there, and update the GP.
- If the proposed point has already been sampled, stop early.
- **Returns** `(X_opt, Y_opt)`, each `(1,)`: the best point found and its value.

```
>>> bo = BayesianOptimization(f, X_init, Y_init, (-np.pi, 2 * np.pi), 50, l=0.6, sigma_f=2)
>>> X_opt, Y_opt = bo.optimize(50)
>>> X_opt, Y_opt
(array([0.8976]), array([-2.92478]))
```
Plot `f`, the samples in `bo.gp.X`, and the GP mean ± 2 standard deviations to see exploration and exploitation in action.

### 6. Tune a real model with Optuna (bonus, slow)

**File:** `6-optuna.py` · **Function:** `run_study(n_trials=30, seed=0, report_path="bayes_opt.txt")`

Tune a small model on scikit-learn's built-in digits dataset (no download needed):
- Load the data with `sklearn.datasets.load_digits` and hold out a validation split (use `seed` for the split).
- Model: your choice. A scikit-learn `MLPClassifier` is the simplest; a tiny PyTorch network is fine too.
- Tune **at least 5 hyperparameters** (for example number of layers, units per layer, L2 strength, learning rate, batch size).
- Use **one** metric to optimize: validation accuracy. Create the study to maximize it, and pass `seed` to the sampler so runs are repeatable.
- Use early stopping during training.
- After the search, write a short report to `report_path`: the best accuracy, the best hyperparameters and the number of trials.
- **Returns** the finished `optuna.Study`.
- Extras (not tested): save the best model of each trial to a file whose name includes its hyperparameters; plot the optimization history (Optuna has matplotlib plotting helpers) and save it as an image; try Optuna's Gaussian-process sampler and compare it to the default TPE sampler.

```
>>> study = run_study(n_trials=30)
>>> study.best_value
0.97...
>>> len(study.best_params)
5
```

### Write-up (optional)

The original course asked for a blog post. Write one covering: what a Gaussian process is, how Bayesian optimization works, the model and hyperparameters you tuned and why, why you chose your metric, and what you learned from the results. Include your convergence plot.

## Data

Tasks 0-5 use the formula above, no data. Task 6 uses scikit-learn's bundled digits dataset (1,797 8×8 images), which ships with scikit-learn.

## Check your work

From the `ml-curriculum` folder:

```bash
pytest 05-unsupervised/04-hyperparameter_tuning            # everything
pytest 05-unsupervised/04-hyperparameter_tuning -m "not slow"   # skip the Optuna run
```
