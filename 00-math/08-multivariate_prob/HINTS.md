# Hints: Multivariate Probability

Open one hint at a time.

## 0. Mean and Covariance
<details><summary>Hint 1</summary>The mean is the average down each column (axis 0). Keep it 2D with shape (1, d); NumPy's mean has an option to keep the reduced dimension.</details>
<details><summary>Hint 2</summary>Center the data by subtracting the mean from every row (broadcasting does this for you).</details>
<details><summary>Hint 3</summary>The covariance matrix is the centered data's transpose times the centered data, divided by n minus 1. Check the shapes: (d, n) times (n, d) gives (d, d).</details>

## 1. Correlation
<details><summary>Hint 1</summary>Each entry is the covariance divided by the product of the two standard deviations. The standard deviations are the square roots of the diagonal of C.</details>
<details><summary>Hint 2</summary>Get the vector of standard deviations, then build the matrix of all pairwise products with an outer product. Divide C by that element-wise.</details>

## 2. Initialize
<details><summary>Hint 1</summary>The layout is flipped compared to task 0: data points are columns. So average across axis 1, and keep the result as a (d, 1) column.</details>
<details><summary>Hint 2</summary>After centering, the covariance is the centered data times its transpose, divided by n minus 1: (d, n) times (n, d).</details>

## 3. PDF
<details><summary>Hint 1</summary>Break the formula into three pieces: the difference x minus the mean, the normalising constant in front, and the exponent.</details>
<details><summary>Hint 2</summary>The exponent's middle part is a "sandwich": the transposed difference, times the inverse covariance, times the difference. With shapes (1, d), (d, d), (d, 1), the result is a 1x1 array.</details>
<details><summary>Hint 3</summary>The constant uses the determinant of the covariance and (2 pi) raised to d. To return a plain number instead of a 1x1 array, look at the array method that extracts a single element.</details>
