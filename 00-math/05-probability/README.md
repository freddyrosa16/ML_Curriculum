# Probability

**Status:** Original

## Why this matters
Machine learning is mostly about uncertainty: a classifier outputs probabilities, a loss function is often a negative log-likelihood, and generative models are probability distributions you can sample from. Here you build four classic distributions from scratch, with no libraries, so you understand exactly what a PMF, PDF and CDF compute.

## Concepts to understand first
- **What do $P(A \mid B)$, $P(A \cap B)$, $P(A \cup B)$ and $P(A \cap B')$ mean?** "A given B" (probability of A once you know B happened), "A and B", "A or B", and "A and not B".
- **What is a random variable, and what makes it discrete or continuous?** A number whose value depends on chance. Discrete ones take countable values (number of emails per hour); continuous ones take any value in a range (a person's height).
- **PMF vs PDF?** A *probability mass function* gives $P(X = k)$ for a discrete variable. A *probability density function* is the continuous version: the probability of an exact value is 0, so you get probabilities from the *area* under the PDF over a range.
- **What is a CDF?** $F(x) = P(X \le x)$: the probability of getting $x$ or less. It rises from 0 to 1. For discrete variables, it's the running sum of the PMF.
- **What are mean, variance, standard deviation, percentile?** Mean: the average value. Variance: the average squared distance from the mean; standard deviation is its square root. The $p$-th percentile is the value with $p\%$ of outcomes below it (the 50th percentile is the median).
- **What do these four distributions model?** *Poisson($\lambda$)*: how many events happen in a fixed time window when they occur at average rate $\lambda$. *Exponential($\lambda$)*: how long until the next such event. *Normal($\mu, \sigma$)*: the bell curve, sums of many small effects. *Binomial($n, p$)*: number of successes in $n$ independent yes/no trials with success probability $p$.
- **What is a z-score?** How many standard deviations a value is from the mean: $z = (x - \mu)/\sigma$.
- **How do you estimate a distribution's parameters from data?** Match the formulas to the sample: e.g. a Poisson's mean equals $\lambda$, so the sample mean estimates $\lambda$. For a binomial, mean $= np$ and variance $= np(1-p)$, two equations you can solve for $p$ and $n$.

## Resources
- [Seeing Theory: a visual introduction to probability](https://seeing-theory.brown.edu/)
- [Khan Academy: Statistics and probability](https://www.khanacademy.org/math/statistics-probability)
- [StatQuest (YouTube channel): search "Probability is not Likelihood" and "The Normal Distribution"](https://www.youtube.com/@statquest)
- [Mathematics for Machine Learning, chapter 6: Probability and Distributions (free PDF)](https://mml-book.github.io/)
- [Wikipedia: Poisson](https://en.wikipedia.org/wiki/Poisson_distribution), [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution), [Normal](https://en.wikipedia.org/wiki/Normal_distribution), [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution) (formula reference)

## Rules
- **No imports at all** in your four class files: no `math`, no NumPy. Write factorials, $e$ and $\pi$ yourself (`e = 2.7182818285`, `pi = 3.1415926536` are close enough). The tests check this.
- Error messages must match exactly (the tests check them).
- "`data` not given" means `data is None`. An empty list is *given* (and too short), so it's an error, not a fallback to defaults.
- The tests generate data with NumPy, then call `.tolist()` so your class always receives a plain Python list.

## Tasks

### 0. Initialize Poisson
**File:** `poisson.py` · **Class:** `Poisson(data=None, lambtha=1.)`

Store `self.lambtha` as a float.
- If `data` is `None`: use `lambtha`. If it's not positive, raise `ValueError("lambtha must be a positive value")`.
- Otherwise estimate `lambtha` from `data`. If `data` isn't a list, raise `TypeError("data must be a list")`; if it has fewer than two values, raise `ValueError("data must contain multiple values")`.
```
Poisson(<100 samples, seed 0, np.random.poisson(5., 100)>).lambtha  ->  4.84
Poisson(lambtha=5).lambtha                                          ->  5.0
```

### 1. Poisson PMF
**File:** `poisson.py` · **Method:** `pmf(self, k)`

Return $P(X = k)$. Convert a non-integer `k` to `int`; return `0` for negative `k`.
```
Poisson(lambtha=5).pmf(9)  ->  0.036265577412911795
```

### 2. Poisson CDF
**File:** `poisson.py` · **Method:** `cdf(self, k)`

Return $P(X \le k)$, with the same rules for `k`.
```
Poisson(lambtha=5).cdf(9)  ->  0.9681719426208609
```

### 3. Initialize Exponential
**File:** `exponential.py` · **Class:** `Exponential(data=None, lambtha=1.)`

Same rules and error messages as `Poisson`, but think carefully about how $\lambda$ relates to the data here: the data are *waiting times*, not counts.
```
Exponential(<seed 0, np.random.exponential(0.5, 100)>).lambtha  ->  2.1771114730906937
Exponential(lambtha=2).lambtha                                  ->  2.0
```

### 4. Exponential PDF
**File:** `exponential.py` · **Method:** `pdf(self, x)`

Return the density at time `x` (0 for negative `x`).
```
Exponential(lambtha=2).pdf(1)  ->  0.2706705664650693
```

### 5. Exponential CDF
**File:** `exponential.py` · **Method:** `cdf(self, x)`

Return $P(X \le x)$ (0 for negative `x`).
```
Exponential(lambtha=2).cdf(1)  ->  0.8646647167674654
```

### 6. Initialize Normal
**File:** `normal.py` · **Class:** `Normal(data=None, mean=0., stddev=1.)`

Store `self.mean` and `self.stddev` as floats.
- If `data` is `None`: use the arguments; if `stddev` isn't positive raise `ValueError("stddev must be a positive value")`.
- Otherwise compute them from `data` (use the **population** standard deviation: divide by the number of points). Same data errors as before.
```
Normal(<seed 0, np.random.normal(70, 10, 100)>)  ->  mean 70.59808015534485, stddev 10.078822447165797
```

### 7. Normalize Normal
**File:** `normal.py` · **Methods:** `z_score(self, x)`, `x_value(self, z)`

Convert a value to its z-score, and a z-score back to a value.
```
n = Normal(mean=70, stddev=10)
n.z_score(90)  ->  2.0
n.x_value(2)   ->  90.0
```

### 8. Normal PDF
**File:** `normal.py` · **Method:** `pdf(self, x)`
```
Normal(mean=70, stddev=10).pdf(90)  ->  0.005399096651147344
```

### 9. Normal CDF
**File:** `normal.py` · **Method:** `cdf(self, x)`

The normal CDF has no simple closed form; it's written with the *error function* $\text{erf}$: $F(x) = \frac{1}{2}\left(1 + \text{erf}\left(\frac{x - \mu}{\sigma\sqrt{2}}\right)\right)$. Without imports, approximate erf with the first five terms of its Taylor series (see Wikipedia's "Error function" page). The tests check accuracy within one standard deviation of the mean (to 0.001).
```
Normal(mean=70, stddev=10).cdf(70)  ->  0.5
Normal(mean=70, stddev=10).cdf(80)  ->  about 0.841
```

### 10. Initialize Binomial
**File:** `binomial.py` · **Class:** `Binomial(data=None, n=1, p=0.5)`

Store `self.n` as an int and `self.p` as a float.
- If `data` is `None`: if `n` isn't positive raise `ValueError("n must be a positive value")`; if `p` isn't strictly between 0 and 1 raise `ValueError("p must be greater than 0 and less than 1")`.
- Otherwise estimate from `data` (same data errors as before): first estimate `p` from the mean and variance, then compute `n` and **round** it (not `int()`), then recompute `p` from the rounded `n`.
```
Binomial(<seed 0, np.random.binomial(50, 0.6, 100)>)  ->  n 50, p 0.606
Binomial(n=50, p=0.6)                                 ->  n 50, p 0.6
```

### 11. Binomial PMF
**File:** `binomial.py` · **Method:** `pmf(self, k)`

Convert non-integer `k` to `int`. Return 0 if `k` is negative or greater than `n`.
```
Binomial(n=50, p=0.6).pmf(30)  ->  0.114558552829524
```

### 12. Binomial CDF
**File:** `binomial.py` · **Method:** `cdf(self, k)`

Convert non-integer `k` to `int`. Return 0 for negative `k`, and 1 when `k` is `n` or more (every outcome is at most `n`).
```
Binomial(n=50, p=0.6).cdf(30)  ->  0.5535236207894576
```

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 00-math/05-probability
```
