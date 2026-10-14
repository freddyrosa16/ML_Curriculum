# Hints: Advanced Linear Algebra

Open one hint at a time.

## 0. Determinant
<details><summary>Hint 1</summary>Do the validation in order: is it a list? is every row a list? is it empty? is it the special [[]] case? is it square? Each check has its own error.</details>
<details><summary>Hint 2</summary>Recursion is the natural fit. Base cases: 1x1 (the single entry) and 2x2 (the ad minus bc formula). Otherwise, expand along the first row.</details>
<details><summary>Hint 3</summary>For entry j of the first row: build the smaller matrix by dropping the first row and dropping column j from every remaining row, call yourself on it, multiply by the entry, and alternate the sign with j. Add all those up.</details>

## 1. Minor
<details><summary>Hint 1</summary>For every position (i, j), build the matrix without row i and without column j, and take its determinant (reuse task 0).</details>
<details><summary>Hint 2</summary>Watch the 1x1 case: removing its only row and column leaves nothing, and the task defines the answer as [[1]].</details>

## 2. Cofactor
<details><summary>Hint 1</summary>Take the minor matrix and flip the sign of every entry where row index plus column index is odd.</details>

## 3. Adjugate
<details><summary>Hint 1</summary>Transpose the cofactor matrix. You wrote a transpose in the first linear algebra project.</details>

## 4. Inverse
<details><summary>Hint 1</summary>Compute the determinant first. If it's zero, there's no inverse.</details>
<details><summary>Hint 2</summary>Otherwise divide every entry of the adjugate by the determinant. Build a new list rather than editing a shared one.</details>

## 5. Definiteness
<details><summary>Hint 1</summary>Validate in this order: type, number of dimensions, square, symmetric (a matrix equals its own transpose). NumPy has a function that compares two arrays for equality.</details>
<details><summary>Hint 2</summary>Definiteness comes from the signs of the eigenvalues. NumPy's `linalg` module has eigenvalue functions, including one meant specifically for symmetric matrices.</details>
<details><summary>Hint 3</summary>Floating-point math can turn an eigenvalue that should be exactly 0 into something like -0.0000000000000002, which would wrongly make a semi-definite matrix look indefinite. Round the eigenvalues (or compare with a small tolerance) before checking signs. Check the strict cases (all positive, all negative) before the semi-definite ones.</details>
