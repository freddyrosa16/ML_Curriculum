# Hints: Linear Algebra

Open one hint at a time. Try for a while before opening the next.

## 0. Slice Me Up
<details><summary>Hint 1</summary>A slice has a start, a stop and an optional step. Either end can be left out, meaning "from the beginning" or "to the end".</details>
<details><summary>Hint 2</summary>The stop index is excluded. "2nd through 6th" in human counting means indices 1 to 5 in Python counting.</details>
<details><summary>Hint 3</summary>For "the last five", a negative start index counts from the end of the list, so you don't need to know the length.</details>

## 1. Trim Me Down
<details><summary>Hint 1</summary>Each row is a list, and you already know how to slice a list.</details>
<details><summary>Hint 2</summary>Loop over the rows once. For each row, take the slice that covers the 3rd and 4th entries, and add it to your result list.</details>

## 2. Size Me Please
<details><summary>Hint 1</summary>The length of the outer list is the first number in the shape. What's the length of its first element?</details>
<details><summary>Hint 2</summary>Keep "diving in" to the first element, recording each length, until the thing you're holding is no longer a list.</details>
<details><summary>Hint 3</summary>The built-in `isinstance` tells you whether something is a list. A `while` loop or recursion both work.</details>

## 3. Flip Me Over
<details><summary>Hint 1</summary>If the input has shape (rows, cols), the output has shape (cols, rows).</details>
<details><summary>Hint 2</summary>Row `i` of the result is made of the `i`-th element of every input row, in order.</details>

## 4. Line Up
<details><summary>Hint 1</summary>Check the lengths first and bail out early with `None` if they differ.</details>
<details><summary>Hint 2</summary>Walk both lists side by side (by index, or with the built-in `zip`) and collect the sums in a fresh list.</details>

## 5. Across The Planes
<details><summary>Hint 1</summary>Two checks: the number of rows must match, and the number of columns must match.</details>
<details><summary>Hint 2</summary>It's task 4 applied once per row.</details>

## 6. Howdy Partner
<details><summary>Hint 1</summary>What does `+` do between two Python lists? Does it change either list?</details>

## 7. Gettin' Cozy
<details><summary>Hint 1</summary>Axis 0 means "put the rows of the second matrix under the first": column counts must match. Axis 1 means "glue each pair of rows together": row counts must match.</details>
<details><summary>Hint 2</summary>The tests change your result and check the inputs didn't change. If your result holds the very same row objects as the input, that fails. Each row you put in the result should be a copy.</details>
<details><summary>Hint 3</summary>A full slice of a list (no start, no stop) gives you a shallow copy of it.</details>

## 8. Ridin' Bareback
<details><summary>Hint 1</summary>The product of an (m x n) and an (n x p) matrix has shape (m x p). If the two n's differ, return `None`.</details>
<details><summary>Hint 2</summary>Each output entry (i, j) is the dot product of row i of the first matrix and column j of the second.</details>
<details><summary>Hint 3</summary>The classic approach uses three nested loops: rows of the first, columns of the second, and the shared dimension you sum over.</details>

## 9. Let The Butcher Slice It
<details><summary>Hint 1</summary>NumPy arrays take one slice per dimension, separated by a comma: rows first, then columns.</details>
<details><summary>Hint 2</summary>A lone colon in a position means "everything along this axis".</details>
<details><summary>Hint 3</summary>"Bottom-right 3x3" means the last 3 rows and the last 3 columns. Negative starts make this tidy.</details>

## 10. I'll Use My Scale
<details><summary>Hint 1</summary>Every NumPy array already knows its own shape. Look through the array attributes in the NumPy docs.</details>

## 11. The Western Exchange
<details><summary>Hint 1</summary>NumPy has both a function and an attribute that transpose an array. For more than 2 dimensions, the default transpose reverses the order of the axes.</details>

## 12. Bracin' The Elements
<details><summary>Hint 1</summary>With NumPy arrays, the normal arithmetic operators already work element by element.</details>
<details><summary>Hint 2</summary>Return the four results packed in a tuple, in the order sum, difference, product, quotient. Broadcasting takes care of the case where the second input is a plain number.</details>

## 13. Cat's Got Your Tongue
<details><summary>Hint 1</summary>Search the NumPy docs for "concatenate". It takes the arrays as a sequence plus an axis.</details>

## 14. Saddle Up
<details><summary>Hint 1</summary>NumPy has a dedicated matrix-multiplication function, and Python has an operator for it too. Note that `*` is element-wise, not matrix multiplication.</details>

## 100. Slice Like A Ninja (bonus)
<details><summary>Hint 1</summary>Indexing an array with a tuple of slice objects, one per axis, is the same as writing the colons by hand.</details>
<details><summary>Hint 2</summary>Start with "take everything" for every axis, then replace the entries for the axes listed in the dictionary.</details>
<details><summary>Hint 3</summary>The built-in `slice` accepts the same 1 to 3 arguments as the tuples in `axes`. Unpacking a tuple into a call passes its items as separate arguments.</details>

## 101. The Whole Barn (bonus)
<details><summary>Hint 1</summary>First compare the shapes (you wrote a shape function in task 2). Different shapes means `None`.</details>
<details><summary>Hint 2</summary>Recursion fits naturally: if the two things are lists, add them piece by piece by calling yourself on each pair; if they're numbers, just add them.</details>

## 102. Squashed Like Sardines (bonus)
<details><summary>Hint 1</summary>The shapes must have the same number of dimensions and match on every axis except the one you concatenate along.</details>
<details><summary>Hint 2</summary>Recursion again: walk down the nesting while the depth is less than `axis`, pairing up matching sub-lists. When you reach depth `axis`, join the two lists end to end.</details>
<details><summary>Hint 3</summary>To return a truly new matrix, the pieces you join should be deep copies, not the original nested lists.</details>
