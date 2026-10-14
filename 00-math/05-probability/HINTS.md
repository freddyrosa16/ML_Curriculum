# Hints: Probability

Open one hint at a time.

## 0. Initialize Poisson
<details><summary>Hint 1</summary>Check `data is None` first, because an empty list is "falsy" but still counts as given.</details>
<details><summary>Hint 2</summary>The mean of a Poisson distribution is exactly lambda, so the best estimate is the average of the data.</details>

## 1. Poisson PMF
<details><summary>Hint 1</summary>The formula is lambda to the k, times e to the minus lambda, divided by k factorial.</details>
<details><summary>Hint 2</summary>With no imports you need your own factorial: a running product from 1 to k (and 0 factorial is 1). Define e as a constant.</details>

## 2. Poisson CDF
<details><summary>Hint 1</summary>"k or fewer events" is the sum of the PMF for every count from 0 up to k.</details>
<details><summary>Hint 2</summary>You can reuse your `pmf` method inside `cdf`.</details>

## 3. Initialize Exponential
<details><summary>Hint 1</summary>Lambda is a *rate* (events per unit time). The data are waiting *times* between events. If events come 2 per second on average, how long is the average wait?</details>
<details><summary>Hint 2</summary>The mean of an exponential distribution is 1 over lambda, so lambda is the reciprocal of the sample mean.</details>

## 4. Exponential PDF
<details><summary>Hint 1</summary>The density is lambda times e to the (minus lambda times x), for x at least 0.</details>

## 5. Exponential CDF
<details><summary>Hint 1</summary>The probability of waiting *longer* than x is e to the (minus lambda times x). The CDF is the complement of that.</details>

## 6. Initialize Normal
<details><summary>Hint 1</summary>Compute the mean first, then the average of the squared differences from it, then take the square root (raising to the power 0.5 needs no import).</details>
<details><summary>Hint 2</summary>Divide by the number of points, not by one less, to match the expected values.</details>

## 7. Normalize Normal
<details><summary>Hint 1</summary>A z-score says "how many standard deviations from the mean". Going back is the same relationship solved for x.</details>

## 8. Normal PDF
<details><summary>Hint 1</summary>Two parts: a normalising constant, 1 over (sigma times the square root of 2 pi), times e to the (minus one half z squared).</details>
<details><summary>Hint 2</summary>You already have a z-score method. Reuse it.</details>

## 9. Normal CDF
<details><summary>Hint 1</summary>First rescale: compute (x minus mean) divided by (sigma times the square root of 2). Call that value t. Then the CDF is one half of (1 plus erf of t).</details>
<details><summary>Hint 2</summary>The Taylor series for erf is 2 over the square root of pi, times a series in t with odd powers and alternating signs. Look at the "Taylor series" section of Wikipedia's "Error function" page and use its first five terms.</details>
<details><summary>Hint 3</summary>The first five terms use powers 1, 3, 5, 7, 9 with denominators 1, 3, 10, 42, 216.</details>

## 10. Initialize Binomial
<details><summary>Hint 1</summary>For a binomial distribution, the mean is n times p and the variance is n times p times (1 minus p). Divide the variance by the mean: what's left depends only on p.</details>
<details><summary>Hint 2</summary>Once you have a first estimate of p, n follows from the mean. n must be a whole number, so round it. Then recompute p from the mean and the rounded n, so that n times p still equals the sample mean.</details>
<details><summary>Hint 3</summary>Use the population variance (divide by the number of points), just like task 6.</details>

## 11. Binomial PMF
<details><summary>Hint 1</summary>The formula is "n choose k" times p to the k times (1 minus p) to the (n minus k).</details>
<details><summary>Hint 2</summary>"n choose k" is n factorial divided by (k factorial times (n minus k) factorial). Write a small factorial helper.</details>

## 12. Binomial CDF
<details><summary>Hint 1</summary>As with Poisson, the CDF is a running sum of the PMF from 0 to k.</details>
<details><summary>Hint 2</summary>Handle the two out-of-range cases before summing: below 0 gives 0, at or above n gives 1.</details>
