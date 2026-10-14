# Hints: Bayesian Probability

Open one hint at a time.

## 0. Likelihood
<details><summary>Hint 1</summary>The likelihood for each candidate rate p is the binomial probability of exactly x successes in n trials. Because P is an array, NumPy can compute all candidates in one expression, no loop needed.</details>
<details><summary>Hint 2</summary>You need "n choose x", built from factorials. NumPy 2 has no factorial function of its own, but a factorial is just the product of 1 through k. Look up NumPy's product and range functions, and use floats to avoid integer overflow for n = 130.</details>
<details><summary>Hint 3</summary>Validate before computing: check that n is an int and positive, then x, then x against n, then the type and dimensions of P, then that P's values are in range.</details>

## 1. Intersection
<details><summary>Hint 1</summary>The joint probability of "the data AND this rate" is the likelihood times the prior, element by element.</details>
<details><summary>Hint 2</summary>Comparing floats with exact equality is risky: three values of 1/3 may not add to exactly 1.0. NumPy has a function for "close enough" comparisons. Use it for the "Pr must sum to 1" check.</details>
<details><summary>Hint 3</summary>x = 0 is valid (no side effects at all), so make sure your x check allows it.</details>

## 2. Marginal Probability
<details><summary>Hint 1</summary>The marginal is what you get when you add up the intersection over every hypothesis.</details>

## 3. Posterior
<details><summary>Hint 1</summary>Bayes' rule: posterior equals intersection divided by marginal. You've already written both.</details>
<details><summary>Hint 2</summary>Sanity check: your result should sum to 1, and its biggest value should be at the candidate rate closest to x / n.</details>

## 100. Continuous Posterior (bonus)
<details><summary>Hint 1</summary>With a uniform prior, the posterior over the rate is a Beta distribution. Its two parameters are x + 1 and n - x + 1.</details>
<details><summary>Hint 2</summary>"The probability the rate lies between p1 and p2" is the Beta CDF at p2 minus the Beta CDF at p1.</details>
<details><summary>Hint 3</summary>The Beta CDF is the *regularized incomplete beta function*, which `scipy.special` provides. Read its docs to see the argument order.</details>
