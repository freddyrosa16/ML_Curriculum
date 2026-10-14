# Decision Trees, Random Forests and Isolation Forests

**Status:** Original

## Why this matters
On tabular data (spreadsheets, databases, logs), tree-based models are often the strongest thing you can use, beating neural networks. Building one from scratch teaches you recursion, how a model chooses a split, why a single tree overfits, and how averaging many random trees fixes that. The same machinery, turned around, finds outliers: points that are easy to isolate are suspicious.

## Concepts to understand first
- **What is a decision tree?** A flowchart of yes/no questions like "is feature 2 > 1.7?". Each internal node asks one question and sends an example left or right; each leaf holds a prediction. Predicting means walking from the root to a leaf.
- **Depth, nodes and leaves?** The root has depth 0; a child of a node at depth `k` has depth `k + 1`. The tree's depth is the largest depth of any node. Internal nodes always have exactly two children here.
- **How is a tree grown?** Recursively. At a node, pick a feature and a threshold, split the node's examples into two groups, and repeat on each group until a stopping rule says "make this a leaf" (too deep, too few examples, or all examples have the same class). A leaf predicts the most common class among its examples.
- **What is Gini impurity?** A measure of how mixed a group is: $G = 1 - \sum_k p_k^2$, where $p_k$ is the fraction of class $k$. A pure group has $G = 0$. A good split makes the two children as pure as possible, measured by the size-weighted average of their impurities.
- **Why can a tree be seen as boxes?** Each leaf corresponds to a box in feature space: every split along the path adds a lower or upper bound on one feature. Predicting is finding which box a point falls in.
- **What is a random forest?** Many trees, each grown with randomness (here: random features and thresholds), voting on the answer. Each tree overfits in its own way, and the vote averages those errors out.
- **What is an isolation forest?** Grow fully random trees on unlabeled data and record how deep each point ends up. Outliers sit far from everyone else, so random splits isolate them quickly: their average depth is small.

## Resources
- [StatQuest: Decision and Classification Trees, Clearly Explained](https://www.youtube.com/watch?v=_L39rN6gz7Y)
- [StatQuest: Random Forests Part 1](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- [An Introduction to Statistical Learning (free), chapter 8: tree-based methods](https://www.statlearning.com/)
- [scikit-learn user guide: Decision Trees](https://scikit-learn.org/stable/modules/tree.html) and [Isolation Forest](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest)
- [Breiman (2001): Random Forests](https://link.springer.com/article/10.1023/A:1010933404324)
- [Liu, Ting & Zhou (2008): Isolation Forest](https://ieeexplore.ieee.org/document/4781136)
- [Python docs: classes and inheritance](https://docs.python.org/3/tutorial/classes.html#inheritance)

## Rules
- Only `import numpy as np`.
- Each `N-build_decision_tree.py` file is a full, standalone copy of the three classes so far, plus that task's additions. Tasks 9 and 10 import from `8-build_decision_tree.py`; task 11 imports from `10-isolation_tree.py` (use `__import__('8-build_decision_tree').Decision_Tree`, etc.).
- Recursion is encouraged. Avoid loops over individuals when a NumPy operation can do it.
- Data layout: `explanatory` is `(n_individuals, n_features)`; `target` is `(n_individuals,)` of class labels.

## The three classes you'll build on
Write these yourself in `0-build_decision_tree.py`, then carry them forward.

**`Node`**: `Node(feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0)`. Store each argument as an attribute of the same name, and also set `is_leaf = False` and `sub_population = None` (later: a boolean mask of which training individuals reach this node).

**`Leaf(Node)`**: `Leaf(value, depth=None)`. Call the parent constructor with no arguments, then set `value`, `is_leaf = True` and `depth`.

**`Decision_Tree`**: `Decision_Tree(max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None)`. Attributes: `rng` (a NumPy random generator made with `np.random.default_rng(seed)`), `root` (the given root, or a new `Node(is_root=True)` if none), `explanatory = None`, `target = None`, `max_depth`, `min_pop`, `split_criterion`, and `predict = None`.

**A test tree** you can build by hand for tasks 0-6 (the tests use it too). Call it `example_0`:
- the root splits on feature 0 at threshold 0.5 (depth 0);
- its left child is a leaf with value 0 at depth 1;
- its right child is a node splitting on feature 1 at threshold 30000 (depth 1), whose left child is a leaf with value 0 and right child a leaf with value 1, both at depth 2.

Convention used everywhere: an individual goes to the **left** child when its feature value is **greater than** the threshold, and to the right child otherwise.

## Tasks

### 0. Depth of a decision tree
**File:** `0-build_decision_tree.py`

Add `Node.max_depth_below()` (the largest depth of any node in this node's subtree, itself included), `Leaf.max_depth_below()` (its own depth), and `Decision_Tree.depth()` (the root's `max_depth_below()`).
```
>>> example_0().depth()
2
```
A perfect binary tree with leaves at depth 5 has depth 5.

### 1. Number of nodes
**File:** `1-build_decision_tree.py`

Add `count_nodes_below(self, only_leaves=False)` to `Node` and `Leaf`, and `Decision_Tree.count_nodes(only_leaves=False)`. Count every node (root, internal nodes and leaves), or only the leaves.
```
>>> T = example_0()
>>> T.count_nodes(), T.count_nodes(only_leaves=True)
(5, 3)
```
A perfect tree of depth 4 has 31 nodes and 16 leaves.

### 2. Let's print our tree
**File:** `2-build_decision_tree.py`

Add `__str__` to all three classes (the tree's is the root's), plus two helpers on `Node`, `left_child_add_prefix(text)` and `right_child_add_prefix(text)`. The format:
- a root's line is `root [feature=F, threshold=T]`; another node's line is `-> node [feature=F, threshold=T]`; a leaf's line is `-> leaf [value=V]`;
- under a node come its left child's text block, then its right child's;
- in a child's block, the first line gets the prefix `"    +--"` (4 spaces, `+--`); the remaining lines get `"    |  "` (4 spaces, a bar, 2 spaces) for the left child, or 7 spaces for the right child.

```
>>> print(example_0())
root [feature=0, threshold=0.5]
    +---> leaf [value=0]
    +---> node [feature=1, threshold=30000]
           +---> leaf [value=0]
           +---> leaf [value=1]
```
The tests ignore trailing spaces at the end of lines and a trailing newline.

### 3. Towards the predict function (1): the leaves
**File:** `3-build_decision_tree.py`

Add `get_leaves_below()` to `Node` and `Leaf` (a list of all leaves in the subtree, left to right), and `Decision_Tree.get_leaves()`.
```
>>> [leaf.value for leaf in example_0().get_leaves()]
[0, 0, 1]
```

### 4. Towards the predict function (2): the bounds
**File:** `4-build_decision_tree.py`

Add `Node.update_bounds_below()`, `Leaf.update_bounds_below()` (does nothing) and `Decision_Tree.update_bounds()` (calls it on the root). After the call, every node below the root has two dicts, `lower` and `upper`, mapping a feature index to a bound:
- the root gets `lower = {0: -np.inf}` and `upper = {0: np.inf}`;
- each child starts with copies of its parent's dicts; the **left** child's `lower` for the parent's feature becomes the parent's threshold, and the **right** child's `upper` for that feature becomes the threshold.

```
>>> T = example_0(); T.update_bounds()
>>> [(leaf.lower, leaf.upper) for leaf in T.get_leaves()]
[({0: 0.5}, {0: inf}),
 ({0: -inf, 1: 30000}, {0: 0.5}),
 ({0: -inf}, {0: 0.5, 1: 30000})]
```

### 5. Towards the predict function (3): the indicator function
**File:** `5-build_decision_tree.py`

Add `Node.update_indicator()`, which sets `self.indicator` to a function. Given `A` of shape `(n_individuals, n_features)`, `self.indicator(A)` returns a boolean array of shape `(n_individuals,)`: True for the individuals inside the node's box, meaning for every feature `k` in the bounds, `A[:, k] > lower[k]` and `A[:, k] <= upper[k]` (strict on the lower side, to match the "greater goes left" convention). No loop over individuals.
```
>>> T = example_0(); T.update_bounds()
>>> leaf = T.get_leaves()[2]; leaf.update_indicator()
>>> leaf.indicator(np.array([[0, 22000], [0, 44000], [1, 22000]]))
array([ True, False, False])
```

### 6. The predict function
**File:** `6-build_decision_tree.py`

Add two ways to predict:
- `pred(x)` on `Node`, `Leaf` and `Decision_Tree`: predict one individual `x` (shape `(n_features,)`) by walking down from the root recursively.
- `Decision_Tree.update_predict()`: update the bounds, compute every leaf's indicator, then set `self.predict` to a function that takes `A` `(n_individuals, n_features)` and returns all predictions at once, using the leaves' indicators.
```
>>> T = example_0(); T.update_predict()
>>> A = np.array([[1, 22000], [1, 44000], [0, 22000], [0, 44000]])
>>> T.predict(A), [T.pred(x) for x in A]
(array([0, 0, 1, 0]), [0, 0, 1, 0])
```

### 7. Training decision trees
**File:** `7-build_decision_tree.py`

Add to `Decision_Tree`:
- `np_extrema(arr)`: returns `(min, max)` of an array.
- `random_split_criterion(node)`: repeatedly draw a feature with `self.rng.integers(0, n_features)` until that feature's values over the node's sub-population are not all equal. Then draw `x = self.rng.uniform()` and use the threshold `(1 - x) * min + x * max` of that feature over the sub-population. Return `(feature, threshold)`. (Follow this exactly so your trees match the tests.)
- `fit(explanatory, target, verbose=0)`: if `split_criterion` is `"random"` use `random_split_criterion` (in task 8 anything else means Gini). Store the data, set the root's `sub_population` to all True, grow the tree with `fit_node(self.root)`, then call `update_predict()`. If `verbose == 1`, print:
  ```
    Training finished.
      - Depth                     : 15
      - Number of nodes           : 43
      - Number of leaves          : 22
      - Accuracy on training data : 1.0
  ```
- `fit_node(node)`: pick the split, compute the left population (in the node's sub-population and feature value > threshold) and the right one (the rest of the sub-population). A child becomes a **leaf** if the node is at depth `max_depth - 1`, or the child's population has at most `min_pop` individuals, or all its targets are equal. Otherwise it becomes a node that you fit recursively. Handle the **left child completely (including its whole subtree) before the right child**.
- `get_leaf_child(node, sub_population)`: a `Leaf` one level deeper whose value is the most common target in the sub-population.
- `get_node_child(node, sub_population)`: a `Node` one level deeper holding the sub-population.
- `accuracy(test_explanatory, test_target)`: fraction of correct predictions.

Example: iris (`sklearn.datasets.load_iris`), holding out 15 individuals chosen by `np.random.default_rng(0).choice(150, 15, replace=False)`:
```
>>> T = Decision_Tree(split_criterion="random", max_depth=20, seed=0)
>>> T.fit(X_train, y_train, verbose=1)      # depth 15, 43 nodes, 22 leaves, train accuracy 1.0
>>> T.accuracy(X_test, y_test)
0.9333333333333333
```

### 8. Using Gini impurity
**File:** `8-build_decision_tree.py`

Add a smarter split:
- `possible_thresholds(node, feature)`: the midpoints between consecutive distinct values of that feature in the node's sub-population.
- `Gini_split_criterion_one_feature(node, feature)`: try every possible threshold and return `[best_threshold, smallest_weighted_gini]` (the average Gini impurity of the two children, weighted by their sizes).
- `Gini_split_criterion(node)`: return `(feature, threshold)` for the feature with the smallest impurity.

Make `fit` use it when `split_criterion` is anything other than `"random"` (e.g. `"Gini"`). On the iris example above, the Gini tree has depth 5, 13 nodes and 7 leaves, much smaller than the random tree, with the same test accuracy.

### 9. Random forests
**File:** `9-random_forest.py` · **Class:** `Random_Forest`

`Random_Forest(n_trees=100, max_depth=10, min_pop=1, seed=0)` stores its arguments, and has:
- `fit(explanatory, target, n_trees=100, verbose=0)`: train `n_trees` random-split `Decision_Tree`s, tree `i` with `seed = self.seed + i`, and keep each tree's `predict` function. If `verbose == 1`, print the mean depth, mean number of nodes, mean number of leaves, mean training accuracy of the trees, and the forest's training accuracy.
- `predict(explanatory)`: for each individual, the most common prediction among the trees.
- `accuracy(test_explanatory, test_target)`.

```
>>> F = Random_Forest(max_depth=10, seed=0)
>>> F.fit(X_train, y_train, n_trees=20)
>>> F.accuracy(X_test, y_test)
0.9333333333333333
```

### 10. Isolation random tree
**File:** `10-isolation_tree.py` · **Class:** `Isolation_Random_Tree`

`Isolation_Random_Tree(max_depth=10, seed=0, root=None)`: like `Decision_Tree` but for unlabeled data (no target). Reuse `Node` and `Leaf` from task 8. It needs `__str__`, `depth`, `count_nodes`, `get_leaves`, `update_bounds`, `update_predict`, `np_extrema`, `random_split_criterion` (same as task 7), `get_leaf_child`, `get_node_child`, `fit_node` and `fit(explanatory, verbose=0)`. Differences:
- a leaf's value is **its own depth**, so `predict` returns the depth at which each individual lands;
- a child becomes a leaf only if the node is at depth `max_depth - 1` or the child's population has at most `min_pop = 1` individual.

### 11. Isolation random forest
**File:** `11-isolation_forest.py` · **Class:** `Isolation_Random_Forest`

`Isolation_Random_Forest(n_trees=100, max_depth=10, min_pop=1, seed=0)` with:
- `fit(explanatory, n_trees=100, verbose=0)`: train `n_trees` isolation trees, tree `i` with `seed = self.seed + i`;
- `predict(explanatory)`: the mean depth over the trees, for each individual;
- `suspects(explanatory, n_suspects)`: return `(rows, depths)`: the `n_suspects` individuals with the smallest mean depth (most suspicious first) and their mean depths.

```
>>> # 200 points from a tight 2D Gaussian (std 0.3) plus two far-away points (3, 3) and (-3, 2.5)
>>> IRF = Isolation_Random_Forest(max_depth=15, seed=0)
>>> IRF.fit(points, n_trees=50)
>>> IRF.suspects(points, 2)
(array([[ 3. ,  3. ], [-3. ,  2.5]]), array([2.22, 2.56]))
```

### 12. Write-up (bonus)
Plot the decision regions of a random tree, a Gini tree and a random forest on a 2D dataset (for example, 10 Gaussian clouds whose centers sit evenly on a circle). Write a short post on why the forest's regions look smoother.

## Check your work
```bash
pytest 01-supervised-foundations/07-decision_tree
```
