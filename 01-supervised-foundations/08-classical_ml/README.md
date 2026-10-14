# Classical Machine Learning with scikit-learn

**Status:** New. Holberton only covered trees; this fills the gap with the models every ML engineer is expected to know.

## Why this matters
Before reaching for a neural network, a good engineer tries a simple model: it's faster, easier to explain, and on tabular data it often wins. scikit-learn gives every model the same `fit` / `predict` / `score` interface, so once you know the workflow (split, scale, cross-validate, tune, compare) you can try ten models in ten lines. You'll also connect linear regression back to the linear algebra you already know.

## Concepts to understand first
- **Why split into train and test?** The test set is a stand-in for future data. If you tune anything while looking at it, it stops being a fair judge. *Stratifying* the split keeps the class proportions the same in both parts.
- **What is k-fold cross-validation?** Cut the training data into `k` folds; train on `k - 1` of them and score on the one left out, `k` times. The mean score is a much more stable estimate than one split, and the spread tells you how much to trust it.
- **What is linear regression, and the normal equation?** Predict $\hat y = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$ by minimizing the squared error. Stacking a column of ones in front of `X` to get a matrix $A$, the best $\theta$ solves $A^\top A\,\theta = A^\top y$, a closed-form answer with no gradient descent needed (great for small problems, too slow for huge ones).
- **What is logistic regression?** Your single neuron from the classification project: a linear score passed through a sigmoid, trained with cross-entropy. scikit-learn adds L2 regularization by default; the `C` parameter is the *inverse* of its strength.
- **Why scale features, and why inside a Pipeline?** Models that use distances or gradients (SVMs, logistic regression, k-NN) are thrown off when one feature is in thousands and another in fractions. A `Pipeline` bundles "scale, then model" into one estimator, so during cross-validation the scaler is fitted on each training fold only: no information leaks from the validation fold.
- **What is an SVM, and what's a kernel?** A support vector machine finds the boundary with the widest margin between classes. A kernel lets it draw curved boundaries by implicitly mapping points into a higher-dimensional space. `linear` draws a straight line, `poly` a polynomial curve, `rbf` smooth blobs.
- **What is gradient boosting?** Build trees one after another, each fitting the errors the previous ones left. Histogram-based implementations (scikit-learn's `HistGradientBoosting*`, XGBoost, LightGBM) bin the features first, which makes them very fast. They're the go-to models for tabular competitions.
- **What is hyperparameter search?** Settings like `C` or tree depth aren't learned from data. `GridSearchCV` tries every combination you list, scores each with cross-validation, and refits the best one on all the training data.

## Resources
- [scikit-learn: Getting started](https://scikit-learn.org/stable/getting_started.html) and [user guide](https://scikit-learn.org/stable/user_guide.html)
- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) and [Pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [An Introduction to Statistical Learning with Python (free)](https://www.statlearning.com/), chapters 3 (linear regression), 4 (classification), 5 (resampling), 8 (trees and boosting), 9 (SVMs)
- [StatQuest: Support Vector Machines Part 1](https://www.youtube.com/watch?v=efR1C6CvhmE) and [Gradient Boost Part 1](https://www.youtube.com/watch?v=3CC4N4z3GJc)
- [XGBoost documentation: Introduction to boosted trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
- [Mathematics for Machine Learning (free book), section 9.2: parameter estimation for linear regression](https://mml-book.github.io/)

## Rules
- `numpy`, `scikit-learn` and `xgboost`. **Task 2 is NumPy only** (no scikit-learn).
- Always pass the given `seed` as `random_state` wherever randomness is involved, so results are reproducible.
- Later tasks should reuse earlier ones (e.g. `make_scaled_model = __import__('5-pipeline').make_scaled_model`).
- **macOS note:** XGBoost needs the OpenMP runtime. If `import xgboost` fails with a `libomp` error, run `brew install libomp`. Until then, the XGBoost test is skipped.

## Data
Only datasets bundled with scikit-learn (no downloads): `load_breast_cancer`, `load_diabetes`, and the generators `make_moons` / `make_circles` from `sklearn.datasets`.

## Tasks

### 0. Train/test split
**File:** `0-split.py` · **Function:** `split_data(X, y, test_size=0.2, seed=0)`

Return `(X_train, X_test, y_train, y_test)`, split so the class proportions are the same in both parts (stratified), using `seed`.
```
>>> X, y = load_breast_cancer(return_X_y=True)
>>> X_train, X_test, y_train, y_test = split_data(X, y)
>>> X_train.shape, X_test.shape, round(y_train.mean(), 3), round(y_test.mean(), 3)
((455, 30), (114, 30), 0.626, 0.632)
```

### 1. Cross-validation
**File:** `1-cross_validate.py` · **Function:** `cv_score(model, X, y, k=5, seed=0)`

Return `(mean, std)` of the accuracy over `k` stratified folds, shuffled with `seed`.
```
>>> cv_score(LogisticRegression(max_iter=5000), X, y)
(0.9490..., 0.0195...)
```

### 2. The normal equation (NumPy only)
**File:** `2-normal_equation.py` · **Function:** `normal_equation(X, y)`

`X` is `(m, n)`, `y` is `(m,)`. Return `theta` of shape `(n + 1,)`: the intercept first, then one weight per feature, from the normal equation. Solve the linear system rather than computing a matrix inverse (it's faster and more accurate).
```
>>> X, y = load_diabetes(return_X_y=True)
>>> normal_equation(X, y)[:3]
array([ 152.1335,  -10.0099, -239.8156])
```

### 3. Linear regression
**File:** `3-linear_regression.py` · **Functions:** `fit_linear_regression(X, y)` and `rmse(model, X, y)`

Return a fitted `sklearn.linear_model.LinearRegression`, and the root mean squared error of a model on `(X, y)` as a float. Its intercept and coefficients should match your task 2 result.
```
>>> model = fit_linear_regression(X, y)
>>> model.intercept_, model.coef_[:2]
(152.1335, array([ -10.0099, -239.8156]))
>>> rmse(model, X, y)
53.476...
```

### 4. Logistic regression
**File:** `4-logistic_regression.py` · **Function:** `fit_logistic_regression(X, y, C=1.0)`

Return a fitted `LogisticRegression` with inverse regularization strength `C`. Give it enough iterations to converge on unscaled data (e.g. `max_iter=5000`), or it will warn.
```
>>> clf = fit_logistic_regression(X_train, y_train)      # breast cancer split from task 0
>>> clf.score(X_test, y_test)
0.947...
>>> clf.predict_proba(X_test[:2]).round(3)
array([[0.953, 0.047],
       [1.   , 0.   ]])
```

### 5. Scaling with a Pipeline
**File:** `5-pipeline.py` · **Function:** `make_scaled_model(model)`

Return an **unfitted** `Pipeline` with two steps: a `StandardScaler` named `"scaler"`, then `model` named `"model"`.
```
>>> SVC().fit(X_train, y_train).score(X_test, y_test)
0.912...
>>> make_scaled_model(SVC()).fit(X_train, y_train).score(X_test, y_test)
0.964...
```

### 6. SVM kernels
**File:** `6-svm.py` · **Function:** `compare_kernels(X_train, y_train, X_test, y_test, kernels=("linear", "poly", "rbf"))`

For each kernel, fit a scaled (task 5) `SVC` with that kernel and default settings, and return a dict mapping kernel name to test accuracy.
```
>>> X, y = make_circles(400, noise=0.1, factor=0.4, random_state=0)
>>> X_train, X_test, y_train, y_test = split_data(X, y)
>>> compare_kernels(X_train, y_train, X_test, y_test)
{'linear': 0.4625, 'poly': 0.6375, 'rbf': 1.0}
```
Think about why a straight line can't do better than a coin flip on two concentric circles.

### 7. Gradient boosting
**File:** `7-gradient_boosting.py` · **Functions:** `fit_hist_gb(X, y, seed=0)` and `fit_xgboost(X, y, seed=0)`

- `fit_hist_gb`: return a fitted `HistGradientBoostingClassifier` (default settings, `random_state=seed`).
- `fit_xgboost`: return a fitted `xgboost.XGBClassifier` with 200 trees, max depth 4, learning rate 0.1 and `random_state=seed`. Import xgboost inside the function, so the rest of the file works even if xgboost isn't installed.

### 8. Grid search
**File:** `8-grid_search.py` · **Function:** `tune_svm(X, y, seed=0)`

Search over a scaled RBF `SVC` (task 5) with `GridSearchCV`, trying `C` in `[0.1, 1, 10, 100]` and `gamma` in `["scale", 0.01, 0.1, 1]`, scored by accuracy on 5 stratified folds shuffled with `seed`. Return the **fitted** `GridSearchCV` object. (Hint for the parameter names: a pipeline step's parameters are addressed as `<step name>__<parameter>`.)
```
>>> search = tune_svm(X_train, y_train)          # make_moons(400, noise=0.25, random_state=0) split
>>> search.best_params_, round(search.best_score_, 3)
({'model__C': 1, 'model__gamma': 'scale'}, 0.925)
```

### 9. Compare models on real data
**File:** `9-compare_models.py` · **Function:** `compare_models(seed=0)`

Load the breast cancer dataset, split it with task 0 (`test_size=0.2`, `seed`), and evaluate these models:
- `"logistic_regression"`: scaled `LogisticRegression(max_iter=5000)`
- `"svm_rbf"`: scaled `SVC(kernel="rbf")`
- `"random_forest"`: `RandomForestClassifier(n_estimators=200, random_state=seed)`
- `"hist_gradient_boosting"`: `HistGradientBoostingClassifier(random_state=seed)`
- `"xgboost"`: the task 7 settings, **only if** xgboost can be imported (catch the error otherwise)

For each, compute the 5-fold CV score on the training set (task 1), then fit on the whole training set and score on the test set. Return `{name: {"cv_mean": ..., "cv_std": ..., "test_accuracy": ...}}`.

```
>>> results = compare_models()
>>> {name: round(r["test_accuracy"], 3) for name, r in results.items()}
{'logistic_regression': 0.982, 'svm_rbf': 0.965, 'random_forest': 0.947,
 'hist_gradient_boosting': 0.939, 'xgboost': 0.965}
```
Notice that the simplest model is competitive here, and that CV means and test scores don't always agree on the ranking: with 114 test examples, one prediction is almost 1%.

### 10. Write-up (bonus)
Pick a regression dataset (`load_diabetes`, or `fetch_california_housing`, which downloads about 400 KB the first time) and compare linear regression, a random forest and gradient boosting with cross-validated RMSE. Write a short note on which you'd ship and why.

## Check your work
```bash
pytest 01-supervised-foundations/08-classical_ml
```
