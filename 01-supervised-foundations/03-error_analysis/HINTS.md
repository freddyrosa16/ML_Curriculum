# Hints: Error Analysis

## 0. Create confusion
<details><summary>Hint 1</summary>
Think about one cell `(i, j)`: it counts examples where the label row has a 1 in column `i` *and* the prediction row has a 1 in column `j`. Multiplying those two indicator values gives 1 exactly for such examples.
</details>
<details><summary>Hint 2</summary>
Summing that product over all examples, for every pair `(i, j)` at once, is exactly what a matrix product does. Work out which operand needs to be transposed so the result is `(classes, classes)` with true classes on the rows.
</details>

## 1. Sensitivity
<details><summary>Hint 1</summary>
The true positives for every class sit on the diagonal. NumPy has a function that pulls out the diagonal as a vector.
</details>
<details><summary>Hint 2</summary>
TP + FN is "every example whose true class is i", which is a sum along a row. Which axis gives row sums?
</details>

## 2. Precision
<details><summary>Hint 1</summary>
Same diagonal. TP + FP is "every example *predicted* as class i": that's a column, not a row.
</details>

## 3. Specificity
<details><summary>Hint 1</summary>
Specificity needs TN and FP, neither of which is a simple row or column. Build them from things you can compute: the total count, row sums, column sums and the diagonal.
</details>
<details><summary>Hint 2</summary>
FP is the column sum minus the diagonal. TN is everything that is in neither class i's row nor its column. Careful: if you subtract the row and the column from the total, you've removed the diagonal cell twice.
</details>

## 4. F1 score
<details><summary>Hint 1</summary>
Reuse tasks 1 and 2, then apply the harmonic mean formula element-wise.
</details>

## 5. Dealing with error
<details><summary>Hint 1</summary>
Sort the approaches into two groups first: which ones make the model *fit the training data better* (fight bias), and which ones make it *generalize better* (fight variance)?
</details>
<details><summary>Hint 2</summary>
A bigger network or longer training helps a model that underfits but tends to make overfitting worse. More data and regularization do the opposite. Changing the architecture can help with either problem.
</details>
<details><summary>Hint 3</summary>
When both problems are present, everything that targets either one is fair game. When neither is present, you're done.
</details>

## 6. Compare and contrast
<details><summary>Hint 1</summary>
Error rate = 1 minus (sum of the diagonal / total). Compute it for both matrices (your NumPy skills help here).
</details>
<details><summary>Hint 2</summary>
Compare two gaps: human error → training error (avoidable bias) and training error → validation error (variance). The larger gap is the most important issue.
</details>
