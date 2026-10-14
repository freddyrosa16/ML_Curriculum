# Error Analysis

**Status:** Original

## Why this matters
"94% accuracy" can hide a model that never detects the rare class you actually care about. A confusion matrix shows *which* mistakes a model makes, and metrics like precision, recall and F1 put numbers on them. Comparing training error, validation error and human performance then tells you what to fix next: more capacity, more data, or regularization. This is the single most useful habit when a model "isn't working".

## Concepts to understand first
- **What is a confusion matrix?** A table where row `i`, column `j` counts examples whose true class is `i` and predicted class is `j`. The diagonal holds the correct predictions; everything off the diagonal is a mistake, and its position tells you which classes get mixed up.
- **What are TP, FP, FN, TN for one class?** Pick a class and call it "positive". True positives were positive and predicted positive. False positives were predicted positive but weren't. False negatives were positive but got predicted as something else. True negatives are everything else.
- **What is sensitivity (recall)?** Of all the real positives, what fraction did we catch? $\frac{TP}{TP + FN}$. A cancer screen needs high recall.
- **What is precision?** Of everything we *called* positive, what fraction really was? $\frac{TP}{TP + FP}$. A spam filter needs high precision, so real mail isn't thrown away.
- **What is specificity?** Of all the real negatives, what fraction did we correctly leave alone? $\frac{TN}{TN + FP}$.
- **What is the F1 score?** The harmonic mean of precision and recall, $2\frac{PR}{P + R}$. It's high only when *both* are high.
- **What are bias and variance?** *Bias* is error the model makes even on its training data (it's too simple, or undertrained): it shows up as the gap between human-level error and training error. *Variance* is the extra error on new data (it memorized the training set): it shows up as the gap between training error and validation error.
- **Why compare to human-level performance?** It's a stand-in for the best error you can realistically hope for (the "Bayes error"). If humans get 14% wrong, a model at 15% has little bias left to remove.

## Resources
- [StatQuest: The Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o) and [Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8)
- [Google ML Crash Course: Classification (accuracy, precision, recall)](https://developers.google.com/machine-learning/crash-course/classification)
- [Wikipedia: Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) - the big table of every metric
- [Andrew Ng: Machine Learning Yearning (free draft), chapters on bias and variance](https://github.com/ajaymache/machine-learning-yearning)
- [Andrew Ng, Structuring Machine Learning Projects (Coursera, free to audit)](https://www.coursera.org/learn/machine-learning-projects)
- [scikit-learn user guide: classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

## Rules
- Only `import numpy as np`. No loops.
- Every confusion matrix here has true labels on the **rows** and predictions on the **columns**.
- Functions return one value per class, as a `(classes,)` array.

## Tasks

### 0. Create confusion
**File:** `0-create_confusion.py` · **Function:** `create_confusion_matrix(labels, logits)`

`labels` and `logits` are both one-hot arrays of shape `(m, classes)`: the true classes and the predicted classes. Return the `(classes, classes)` confusion matrix as floats.

```
>>> labels = np.eye(3)[[0, 1, 2, 1]]     # true classes 0, 1, 2, 1
>>> logits = np.eye(3)[[0, 2, 2, 1]]     # predicted 0, 2, 2, 1
>>> create_confusion_matrix(labels, logits)
array([[1., 0., 0.],
       [0., 1., 1.],
       [0., 0., 1.]])
```

For tasks 1-4, use this matrix in your own experiments:
```
>>> confusion = np.array([[50, 3, 2], [5, 40, 5], [0, 10, 85]])
```

### 1. Sensitivity
**File:** `1-sensitivity.py` · **Function:** `sensitivity(confusion)`
```
>>> sensitivity(confusion)
array([0.9091, 0.8   , 0.8947])
```

### 2. Precision
**File:** `2-precision.py` · **Function:** `precision(confusion)`
```
>>> precision(confusion)
array([0.9091, 0.7547, 0.9239])
```

### 3. Specificity
**File:** `3-specificity.py` · **Function:** `specificity(confusion)`
```
>>> specificity(confusion)
array([0.9655, 0.9133, 0.9333])
```

### 4. F1 score
**File:** `4-f1_score.py` · **Function:** `f1_score(confusion)`

You may import your functions from tasks 1 and 2 (with `__import__('1-sensitivity')`, etc.).
```
>>> f1_score(confusion)
array([0.9091, 0.7767, 0.9091])
```

### 5. Dealing with error (short answer)
**File:** `5-error_handling` (plain text, no extension)

For each scenario below, which approaches help? Write one line per scenario, in order. If several approaches help, list their letters in alphabetical order separated by commas with no spaces (e.g. `A,C`).

Scenarios:
1. High bias, high variance
2. High bias, low variance
3. Low bias, high variance
4. Low bias, low variance

Approaches:
- A. Train longer
- B. Try a different architecture
- C. Get more data
- D. Build a bigger (deeper) network
- E. Use regularization
- F. Nothing

### 6. Compare and contrast (short answer)
**File:** `6-compare_and_contrast` (plain text)

A 4-class classifier produced these confusion matrices. Human-level error on this task is about 14%.

Training set (1000 examples):
```
[[226  28  14  32]
 [ 31 170  24  25]
 [ 18  22 183  27]
 [ 24  28  22 126]]
```
Validation set (500 examples):
```
[[112  15   9  14]
 [ 16  84  13  12]
 [  8  12  92  13]
 [ 13  13  12  62]]
```
What is the most important issue? Write a single letter:
- A. High bias
- B. High variance
- C. Nothing

(Work out the training and validation error rates first.)

### 7. Write-up (bonus)
Write a short note explaining precision vs recall with a real example where each one matters more, and how you'd use training/validation/human error to decide what to work on next.

## Check your work
```bash
pytest 01-supervised-foundations/03-error_analysis
```
The tests for tasks 5 and 6 compare a fingerprint (hash) of your answer, so they don't reveal the answer.
