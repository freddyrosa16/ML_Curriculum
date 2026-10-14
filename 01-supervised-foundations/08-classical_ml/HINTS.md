# Hints: Classical ML

## 0. Train/test split
<details><summary>Hint 1</summary>
`sklearn.model_selection` has a function that does exactly this. Read its parameters: one controls the size, one the randomness, and one keeps the class proportions.
</details>

## 1. Cross-validation
<details><summary>Hint 1</summary>
You need two pieces: a splitter object that makes stratified, shuffled folds, and a helper that runs a model over those folds and returns one score per fold.
</details>
<details><summary>Hint 2</summary>
The helper returns a NumPy array of scores; its mean and standard deviation are what you return.
</details>

## 2. The normal equation
<details><summary>Hint 1</summary>
First build the matrix $A$: a column of ones glued to the left of `X`. NumPy has functions for stacking arrays side by side.
</details>
<details><summary>Hint 2</summary>
The normal equation is a linear system "matrix times unknown = vector". `np.linalg` has a function that solves such a system directly, without forming the inverse.
</details>

## 3. Linear regression
<details><summary>Hint 1</summary>
Create the estimator, call `fit`, and return it (`fit` returns the estimator itself).
</details>
<details><summary>Hint 2</summary>
`sklearn.metrics` has the mean squared error. RMSE is its square root, which brings the error back to the units of `y`.
</details>

## 4. Logistic regression
<details><summary>Hint 1</summary>
Same pattern as task 3, with `C` and `max_iter` passed to the constructor.
</details>

## 5. Pipeline
<details><summary>Hint 1</summary>
A `Pipeline` is built from a list of `(name, estimator)` pairs. The names matter: tests and task 8 refer to them.
</details>

## 6. SVM kernels
<details><summary>Hint 1</summary>
Loop over the kernel names, build a fresh scaled SVC for each, fit on the training data, and score on the test data. Every estimator's `score` method gives accuracy for classifiers.
</details>

## 7. Gradient boosting
<details><summary>Hint 1</summary>
Both models follow the familiar construct-fit-return pattern. XGBoost's classifier has a scikit-learn-style interface, so it fits the same way.
</details>
<details><summary>Hint 2</summary>
Putting the `import` statement inside the function means it only runs when the function is called.
</details>

## 8. Grid search
<details><summary>Hint 1</summary>
`GridSearchCV` takes an estimator, a dict of parameter lists, a `cv` splitter and a scoring name. The estimator here is the whole pipeline.
</details>
<details><summary>Hint 2</summary>
Because the SVC is a step called `"model"`, its `C` is reached as `model__C` (two underscores).
</details>

## 9. Compare models
<details><summary>Hint 1</summary>
Put the models in a dict from name to (unfitted) estimator, then loop over it doing the same two things for each: cross-validate on the training set, then fit and test.
</details>
<details><summary>Hint 2</summary>
Wrap the XGBoost import and creation in a `try`/`except` so a missing library just leaves that entry out. Catch a broad exception: a missing OpenMP library raises something other than `ImportError`.
</details>
