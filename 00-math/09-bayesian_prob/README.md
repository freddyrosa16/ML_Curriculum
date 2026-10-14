# Bayesian Probability

**Status:** Original

## Why this matters
Bayes' rule is how you update a belief when new evidence arrives, and it's the backbone of a whole branch of ML: Bayesian optimization, Gaussian processes, Naive Bayes classifiers, and the way we reason about uncertainty in models. In this project you take a real-style question (a drug trial where 26 of 130 patients had a side effect) and compute, step by step, how plausible each possible side-effect rate is.

## Concepts to understand first
- **What is Bayes' rule?** $P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$. In words: *posterior* = *likelihood* x *prior* / *evidence*.
- **What is a prior?** What you believe before seeing the data: here, a probability for each candidate rate $p$ of the side effect. A uniform prior says "every candidate is equally likely to start with".
- **What is a likelihood?** How probable the observed data is *if* a given hypothesis were true. With $x$ side effects among $n$ patients and rate $p$, it's the binomial probability $\binom{n}{x} p^x (1-p)^{n-x}$. (Careful: it's a function of $p$, and it does *not* need to sum to 1 over the hypotheses.)
- **What is the intersection (joint probability)?** $P(\text{data} \cap p) = P(\text{data} \mid p)\,P(p)$: likelihood times prior, for each hypothesis.
- **What is the marginal probability (the evidence)?** The total probability of the data, summed over every hypothesis: $P(\text{data}) = \sum_p P(\text{data} \mid p)\,P(p)$. It's the normalizing constant.
- **What is the posterior?** The updated belief: $P(p \mid \text{data})$ = intersection / marginal. It sums to 1 across the hypotheses.
- **What changes with a continuous prior? (bonus)** If $p$ can be any number in $[0, 1]$ and the prior is uniform, the posterior is a *Beta distribution*, $\text{Beta}(x + 1, n - x + 1)$. The probability that $p$ lies between $p_1$ and $p_2$ is the difference of its CDF at the two points.

## Resources
- [3Blue1Brown: Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- [StatQuest (YouTube channel): search "Bayes' Theorem, Clearly Explained" and "Probability is not Likelihood"](https://www.youtube.com/@statquest)
- [Seeing Theory: Bayesian inference (interactive)](https://seeing-theory.brown.edu/bayesian-inference/index.html)
- [Think Bayes, 2nd edition (free online book)](https://allendowney.github.io/ThinkBayes2/)
- [Wikipedia: Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) (for the bonus)
- [SciPy: `scipy.special.betainc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betainc.html) (for the bonus)

## Rules
- Tasks 0-3: `import numpy as np` only. Task 100 (bonus): `from scipy import special` only.
- Each file must work on its own. Tasks 1-3 need the earlier functions, so either copy them into the file or load the earlier file with Python's built-in `__import__` function.
- Error messages must match exactly. Check errors in the order they're listed.
- `P` is a 1D array of candidate rates; `Pr` is the prior probability of each candidate (same shape as `P`).

## Shared validation (tasks 0-3)
- `n` not a positive int: `ValueError("n must be a positive integer")`
- `x` not an int that is at least 0: `ValueError("x must be an integer that is greater than or equal to 0")`
- `x > n`: `ValueError("x cannot be greater than n")`
- `P` not a 1D NumPy array: `TypeError("P must be a 1D numpy.ndarray")`
- tasks 1-3 only: `Pr` not a NumPy array with the same shape as `P`: `TypeError("Pr must be a numpy.ndarray with the same shape as P")`
- any value of `P` outside [0, 1]: `ValueError("All values in P must be in the range [0, 1]")`, and the same message with `Pr` for the prior (tasks 1-3)
- tasks 1-3 only: `Pr` doesn't sum to 1 (use a tolerant comparison): `ValueError("Pr must sum to 1")`

## Tasks

### 0. Likelihood
**File:** `0-likelihood.py` · **Signature:** `likelihood(x, n, P)`

Return an array with the probability of seeing `x` side effects in `n` patients for each rate in `P`.
```
P = np.linspace(0, 1, 11)       # 0.0, 0.1, ..., 1.0
likelihood(26, 130, P)  ->  [0.00e+00 2.71e-04 8.72e-02 3.07e-03 ... 0.00e+00]
```

### 1. Intersection
**File:** `1-intersection.py` · **Signature:** `intersection(x, n, P, Pr)`

Return the joint probability of the data and each rate.
```
Pr = np.ones(11) / 11
intersection(26, 130, P, Pr)  ->  [0.00e+00 2.47e-05 7.93e-03 2.79e-04 ... 0.00e+00]
```

### 2. Marginal Probability
**File:** `2-marginal.py` · **Signature:** `marginal(x, n, P, Pr)`

Return the total probability of the data (a single number).
```
marginal(26, 130, P, Pr)  ->  0.008229580791426582
```

### 3. Posterior
**File:** `3-posterior.py` · **Signature:** `posterior(x, n, P, Pr)`

Return the posterior probability of each rate given the data.
```
posterior(26, 130, P, Pr)  ->  [0.00e+00 3.00e-03 9.63e-01 3.40e-02 ... 0.00e+00]   (sums to 1)
```

### 100. Continuous Posterior (bonus)
**File:** `100-continuous.py` · **Signature:** `posterior(x, n, p1, p2)`

With a uniform prior over all rates in [0, 1], return the probability that the true rate lies in `[p1, p2]`.
- Same `n`, `x` checks as above.
- `p1` or `p2` not a float in [0, 1]: `ValueError("p1 must be a float in the range [0, 1]")` (or `p2`).
- `p2 <= p1`: `ValueError("p2 must be greater than p1")`.
```
posterior(26, 130, 0.17, 0.23)  ->  0.6098093274896035
```

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 00-math/09-bayesian_prob
```
