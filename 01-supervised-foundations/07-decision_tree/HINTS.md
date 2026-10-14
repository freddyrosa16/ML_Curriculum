# Hints: Decision Trees

## 0. Depth
<details><summary>Hint 1</summary>
Think recursively: the deepest node in my subtree is either me, or the deepest node in one of my children's subtrees.
</details>
<details><summary>Hint 2</summary>
A leaf is the base case (it already answers with its own depth), so a `Node` only has to ask its children and take the maximum.
</details>

## 1. Number of nodes
<details><summary>Hint 1</summary>
Same recursive shape as task 0, but adding instead of taking a maximum.
</details>
<details><summary>Hint 2</summary>
The only question is whether the current node counts itself. A leaf always counts as 1; an internal node counts itself only when you're not counting leaves only.
</details>

## 2. Printing
<details><summary>Hint 1</summary>
Build the text bottom-up: a node's text is its own line, followed by each child's text with prefixes added. The children produce their own (already prefixed) text recursively.
</details>
<details><summary>Hint 2</summary>
The prefix helpers split a block of text into lines, give the first line one prefix and all other lines another, then join them back. Be careful with the empty string that splitting produces after a trailing newline.
</details>
<details><summary>Hint 3</summary>
The bar `|` under a left child is what connects it visually to the right sibling printed below it. The right child is last, so its continuation lines only need spaces.
</details>

## 3. Leaves
<details><summary>Hint 1</summary>
A leaf's list of leaves is a list containing itself. A node's list is its left child's list followed by its right child's.
</details>

## 4. Bounds
<details><summary>Hint 1</summary>
This one goes top-down: a node first hands bounds to its children, then asks each child to continue.
</details>
<details><summary>Hint 2</summary>
Copy the parent's dicts before changing them (dicts are shared by reference, so without a copy every node would end up with the same dict).
</details>
<details><summary>Hint 3</summary>
Going left means "feature greater than threshold", which tightens the lower bound. Going right tightens the upper bound.
</details>

## 5. Indicator
<details><summary>Hint 1</summary>
For each feature that appears in `lower`, compare the whole column of `A` to the bound at once: that gives one boolean per individual. Do the same for `upper`.
</details>
<details><summary>Hint 2</summary>
Stack those boolean arrays and require all of them to be True for each individual. NumPy's "all" along the right axis does it.
</details>
<details><summary>Hint 3</summary>
Assign a function (a nested `def` or a lambda) to `self.indicator`. It should capture the node's bounds, not compute an answer right away.
</details>

## 6. Predict
<details><summary>Hint 1</summary>
For `pred`: compare the individual's value for the node's feature to the threshold and recurse into the matching child. A leaf just returns its value.
</details>
<details><summary>Hint 2</summary>
For the vectorized version: every individual is inside exactly one leaf's box. If each leaf contributes "indicator times value", adding all the contributions together gives every individual's prediction.
</details>

## 7. Training
<details><summary>Hint 1</summary>
Populations are boolean masks over all training individuals. Combining masks with element-wise "and" / "not" gives the children's populations without copying any data.
</details>
<details><summary>Hint 2</summary>
The three leaf conditions are about the *child*: its depth limit (the parent is at `max_depth - 1`), the number of True values in its mask, and how many distinct targets it contains.
</details>
<details><summary>Hint 3</summary>
For the leaf's value, NumPy's `unique` can return counts as well as values; the most common value is the one with the biggest count. The order of random draws matters for matching the tests: split the node, then build and fully fit the left child, then the right.
</details>

## 8. Gini
<details><summary>Hint 1</summary>
Possible thresholds: sort the distinct values (NumPy's `unique` does that), then average each value with the next one.
</details>
<details><summary>Hint 2</summary>
For one threshold, you need the class proportions on each side. Doing this for all thresholds and all classes at once is possible with a 3D boolean array (individuals × thresholds × classes), but a loop over thresholds is fine to start with.
</details>
<details><summary>Hint 3</summary>
Gini of a side is 1 minus the sum of squared class proportions; the split's score is the two sides' Gini weighted by their share of individuals. Keep the threshold with the smallest score, then the feature with the smallest best score.
</details>

## 9. Random forest
<details><summary>Hint 1</summary>
Each tree is just your `Decision_Tree` with a different seed. Keep the trees' `predict` functions in a list.
</details>
<details><summary>Hint 2</summary>
Stack all trees' predictions into a 2D array (trees × individuals), then take the most common value in each column.
</details>

## 10. Isolation tree
<details><summary>Hint 1</summary>
Most methods are copies of `Decision_Tree`'s. The differences are in what a leaf stores and when a child becomes a leaf, since there is no target.
</details>

## 11. Isolation forest
<details><summary>Hint 1</summary>
Prediction is an average of the trees' depth predictions instead of a vote.
</details>
<details><summary>Hint 2</summary>
`np.argsort` gives the indices that would sort the mean depths. The first few are your suspects.
</details>
