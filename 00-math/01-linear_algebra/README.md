# Linear Algebra

**Status:** Original

## Why this matters
Every piece of data in ML ends up as a vector or a matrix: an image is a grid of numbers, a batch of images is a stack of grids, and a neural network layer is one matrix multiplication. Getting fluent with shapes, slicing and matrix products now saves you hours of "shape mismatch" errors later. You'll first do it by hand with plain lists, then see how NumPy does the same thing in one line.

## Concepts to understand first
- **What is a vector? A matrix?** A vector is an ordered list of numbers, like `[1, 2, 3]`. A matrix is a rectangular grid of numbers: a list of rows, each row the same length.
- **What is a shape?** The size along each dimension. A matrix with 2 rows and 3 columns has shape `(2, 3)`. A stack of 10 such matrices has shape `(10, 2, 3)`.
- **What is an axis?** A direction you can move along. Axis 0 goes down the rows, axis 1 goes across the columns. "Concatenate along axis 0" means stack vertically.
- **What is a slice?** Picking a contiguous piece of a list or array with `start:stop:step`. The stop index is *not* included, and negative indices count from the end.
- **What is a transpose?** Flipping a matrix over its diagonal, so row `i` becomes column `i`. A `(2, 3)` matrix becomes `(3, 2)`.
- **What are element-wise operations?** Doing the operation pair by pair: `[1, 2] + [3, 4] = [4, 6]`. Both inputs need the same shape (or a shape NumPy can broadcast).
- **What is the dot product, and matrix multiplication?** The dot product multiplies matching entries of two vectors and sums them: $[1,2]\cdot[3,4] = 1\cdot3 + 2\cdot4 = 11$. In a matrix product $AB$, entry $(i, j)$ is the dot product of row $i$ of $A$ with column $j$ of $B$, so $A$'s column count must equal $B$'s row count.
- **What is broadcasting?** NumPy's rule for combining arrays of different shapes by "stretching" size-1 dimensions, e.g. adding a single number to every entry of a matrix.
- **Why NumPy?** It stores numbers in compact arrays and runs loops in fast C code, so one NumPy call can replace a nested Python loop and run 100x faster.

## Resources
- [3Blue1Brown: Essence of Linear Algebra (videos 1-4)](https://www.3blue1brown.com/topics/linear-algebra)
- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [NumPy: indexing and slicing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy: broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Mathematics for Machine Learning, chapter 2 (free PDF)](https://mml-book.github.io/)
- [Python docs: the built-in `slice`](https://docs.python.org/3/library/functions.html#slice)

## Rules
- Tasks 0-8 and 101-102: **plain Python only**, no imports.
- Tasks 9-14 and 100: `import numpy as np` is the only import allowed.
- Tasks 0, 9, 10, 11, 12, 13, 14: no loops (`for`, `while`, comprehensions) and no `if`. Task 10 also: no `try/except`. The tests check this.
- Task 1: at most one `for` loop, no `if`.
- When a task says "return a new list/matrix", don't modify the inputs, and don't hand back the input itself.

## Tasks

### 0. Slice Me Up
**File:** `0-slice_me_up.py` (a script, not a function)

Make a list `arr` containing `9, 8, 2, 3, 9, 4, 1, 0, 3`. Using slicing only, create three variables:
- `arr1`: the first two numbers
- `arr2`: the last five numbers
- `arr3`: the 2nd through 6th numbers

Print each one with a short label. The tests import your file and read `arr1`, `arr2`, `arr3`.

Expected printout:
```
The first two numbers of the array are: [9, 8]
The last five numbers of the array are: [9, 4, 1, 0, 3]
The 2nd through 6th numbers of the array are: [8, 2, 3, 9, 4]
```

### 1. Trim Me Down
**File:** `1-trim_me_down.py` (a script)

Make a variable `matrix` holding these three rows: `1, 3, 9, 4, 5, 8` / `2, 4, 7, 3, 4, 0` / `0, 3, 4, 6, 1, 5`. Build a list `the_middle` that holds the 3rd and 4th columns of `matrix` (as a 2D list, one row per original row) and print it. One `for` loop at most.

Expected printout:
```
The middle columns of the matrix are: [[9, 4], [7, 3], [4, 6]]
```

### 2. Size Me Please
**File:** `2-size_me_please.py` · **Signature:** `matrix_shape(matrix)`

Return the shape of a nested list as a list of integers. Assume every element in the same dimension has the same shape, and that the matrix may have any number of dimensions.
```
matrix_shape([[1, 2], [3, 4]])          ->  [2, 2]
matrix_shape(<2 x 3 x 5 nested list>)   ->  [2, 3, 5]
```

### 3. Flip Me Over
**File:** `3-flip_me_over.py` · **Signature:** `matrix_transpose(matrix)`

Return the transpose of a 2D list as a new list. The input is never empty.
```
matrix_transpose([[1, 2, 3], [4, 5, 6]])  ->  [[1, 4], [2, 5], [3, 6]]
```

### 4. Line Up
**File:** `4-line_up.py` · **Signature:** `add_arrays(arr1, arr2)`

Add two lists of numbers element-wise and return a new list. If they aren't the same length, return `None`.
```
add_arrays([1, 2, 3, 4], [5, 6, 7, 8])  ->  [6, 8, 10, 12]
add_arrays([1, 2], [1, 2, 3])           ->  None
```

### 5. Across The Planes
**File:** `5-across_the_planes.py` · **Signature:** `add_matrices2D(mat1, mat2)`

Add two 2D lists element-wise and return a new matrix, or `None` if their shapes differ.
```
add_matrices2D([[1, 2], [3, 4]], [[5, 6], [7, 8]])  ->  [[6, 8], [10, 12]]
add_matrices2D([[1, 2, 3], [4, 5, 6]], [[1, 2]])    ->  None
```

### 6. Howdy Partner
**File:** `6-howdy_partner.py` · **Signature:** `cat_arrays(arr1, arr2)`

Return a new list with `arr2` joined onto the end of `arr1`.
```
cat_arrays([1, 2, 3, 4, 5], [6, 7, 8])  ->  [1, 2, 3, 4, 5, 6, 7, 8]
```

### 7. Gettin' Cozy
**File:** `7-gettin_cozy.py` · **Signature:** `cat_matrices2D(mat1, mat2, axis=0)`

Concatenate two 2D lists along `axis` (0 = stack rows, 1 = join side by side). Return a new matrix, or `None` if the shapes don't fit. "New" means changing the result must not change the inputs, so copy the rows too.
```
cat_matrices2D([[1, 2], [3, 4]], [[5, 6]])              ->  [[1, 2], [3, 4], [5, 6]]
cat_matrices2D([[1, 2], [3, 4]], [[7], [8]], axis=1)    ->  [[1, 2, 7], [3, 4, 8]]
cat_matrices2D([[1, 2], [3, 4]], [[5, 6]], axis=1)      ->  None
```

### 8. Ridin' Bareback
**File:** `8-ridin_bareback.py` · **Signature:** `mat_mul(mat1, mat2)`

Multiply two 2D lists. Return a new matrix, or `None` if the inner sizes don't match.
```
mat_mul([[1, 2], [3, 4], [5, 6]], [[1, 2, 3, 4], [5, 6, 7, 8]])
  ->  [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]
```

### 9. Let The Butcher Slice It
**File:** `9-let_the_butcher_slice_it.py` (a script)

Import NumPy and make `matrix`, a 4x6 array holding the numbers 1 to 24 in order, row by row. Using slicing only, create:
- `mat1`: the middle two rows
- `mat2`: the middle two columns
- `mat3`: the bottom-right 3x3 square

Print each with a label. Expected values:
```
mat1 -> [[ 7  8  9 10 11 12]
         [13 14 15 16 17 18]]
mat2 -> [[ 3  4] [ 9 10] [15 16] [21 22]]
mat3 -> [[10 11 12] [16 17 18] [22 23 24]]
```

### 10. I'll Use My Scale
**File:** `10-ill_use_my_scale.py` · **Signature:** `np_shape(matrix)`

Return the shape of a NumPy array as a tuple of ints.
```
np_shape(np.array([1, 2, 3, 4, 5, 6]))  ->  (6,)
np_shape(np.array([]))                  ->  (0,)
np_shape(np.zeros((2, 2, 5)))           ->  (2, 2, 5)
```

### 11. The Western Exchange
**File:** `11-the_western_exchange.py` · **Signature:** `np_transpose(matrix)`

Return the transpose of a NumPy array (any number of dimensions) as an array.
```
np_transpose(np.array([[1, 2, 3], [4, 5, 6]]))  ->  [[1 4] [2 5] [3 6]]
np_transpose(np.zeros((2, 3, 4))).shape         ->  (4, 3, 2)
```

### 12. Bracin' The Elements
**File:** `12-bracin_the_elements.py` · **Signature:** `np_elementwise(mat1, mat2)`

Return a tuple `(sum, difference, product, quotient)`, all element-wise. `mat2` may also be a plain number.
```
a = np.array([[11, 22, 33], [44, 55, 66]]); b = np.array([[1, 2, 3], [4, 5, 6]])
np_elementwise(a, b)[0]  ->  [[12 24 36] [48 60 72]]
np_elementwise(a, b)[3]  ->  [[11. 11. 11.] [11. 11. 11.]]
```

### 13. Cat's Got Your Tongue
**File:** `13-cats_got_your_tongue.py` · **Signature:** `np_cat(mat1, mat2, axis=0)`

Concatenate two NumPy arrays along `axis` and return a new array.
```
np_cat(a, b)          ->  4x3 array (b's rows below a's)
np_cat(a, b, axis=1)  ->  [[11 22 33 1 2 3] [44 55 66 4 5 6]]
```

### 14. Saddle Up
**File:** `14-saddle_up.py` · **Signature:** `np_matmul(mat1, mat2)`

Matrix-multiply two NumPy arrays.
```
np_matmul(np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
  ->  [[11 14 17 20] [23 30 37 44] [35 46 57 68]]
```

### 100. Slice Like A Ninja (bonus)
**File:** `100-slice_like_a_ninja.py` · **Signature:** `np_slice(matrix, axes={})`

`axes` maps an axis number to a tuple describing a slice on that axis (the same arguments you'd give Python's built-in `slice`). Axes not mentioned are kept whole. Return the sliced array; don't modify the input.
```
m = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
np_slice(m, axes={1: (1, 3)})                         ->  [[2 3] [7 8]]
np_slice(<3x2x5 array of 1..30>, axes={0: (2,), 2: (None, None, -2)})
  ->  [[[ 5  3  1] [10  8  6]]  [[15 13 11] [20 18 16]]]
```

### 101. The Whole Barn (bonus)
**File:** `101-the_whole_barn.py` · **Signature:** `add_matrices(mat1, mat2)`

Like task 5, but for nested lists with **any** number of dimensions. Return `None` if the shapes differ.
```
add_matrices([1, 2, 3], [4, 5, 6])                  ->  [5, 7, 9]
add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]])    ->  [[6, 8], [10, 12]]
add_matrices(<2x3x2x2 list>, <2x2x2x2 list>)        ->  None
```

### 102. Squashed Like Sardines (bonus)
**File:** `102-squashed_like_sardines.py` · **Signature:** `cat_matrices(mat1, mat2, axis=0)`

Like task 7, but for nested lists with any number of dimensions. All dimensions except `axis` must match, otherwise return `None`.
```
cat_matrices([[1, 2], [3, 4]], [[5], [6]], axis=1)   ->  [[1, 2, 5], [3, 4, 6]]
cat_matrices(<2x3x4 list>, <2x3x2 list>, axis=2)     ->  a 2x3x6 list
cat_matrices(<2x3x4 list>, <2x3x2 list>, axis=0)     ->  None
```

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 00-math/01-linear_algebra
```
