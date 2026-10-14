# Advanced Linear Algebra

**Status:** Original

## Why this matters
Determinants tell you whether a matrix can be inverted. Inverses solve systems of equations (linear regression has a closed-form solution built on one). Eigenvalues and *definiteness* tell you whether a function curves up like a bowl, which is what decides whether an optimizer is at a minimum, a maximum or a saddle point. Computing these by hand, recursively, makes the definitions stick.

## Concepts to understand first
- **What is a determinant?** A single number computed from a square matrix. Geometrically, it's how much the matrix scales area (2D) or volume (3D). Zero means the matrix squashes space flat, so it can't be undone.
- **How do you compute one by hand?** For a 2x2 matrix with rows $(a, b)$ and $(c, d)$: $ad - bc$. For bigger matrices, use *cofactor (Laplace) expansion*: go along the first row, and for each entry multiply it by the determinant of the smaller matrix left when you delete that entry's row and column, alternating signs $+, -, +, \dots$
- **What is a minor?** The minor of entry $(i, j)$ is the determinant of the matrix with row $i$ and column $j$ deleted. The *minor matrix* holds the minor of every entry.
- **What is a cofactor?** A minor with a sign attached: $C_{ij} = (-1)^{i+j} M_{ij}$. The signs form a checkerboard starting with $+$ in the top-left corner.
- **What is the adjugate?** The transpose of the cofactor matrix.
- **What is an inverse, and how does the adjugate give it?** $A^{-1}$ is the matrix with $A A^{-1} = I$. One formula: $A^{-1} = \frac{1}{\det A}\,\text{adj}(A)$. If $\det A = 0$ the matrix is *singular* and has no inverse.
- **What are eigenvalues and eigenvectors?** An eigenvector $v$ of $A$ is a direction that $A$ only stretches: $Av = \lambda v$. The stretch factor $\lambda$ is the eigenvalue.
- **What is definiteness?** For a symmetric matrix, look at the signs of its eigenvalues: all $> 0$ means *positive definite*; all $\ge 0$ means *positive semi-definite*; all $< 0$, *negative definite*; all $\le 0$, *negative semi-definite*; mixed signs, *indefinite*.

## Resources
- [3Blue1Brown: The determinant](https://www.3blue1brown.com/lessons/determinant)
- [3Blue1Brown: Inverse matrices, column space and null space](https://www.3blue1brown.com/lessons/inverse-matrices)
- [3Blue1Brown: Eigenvectors and eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues)
- [Khan Academy: Linear algebra](https://www.khanacademy.org/math/linear-algebra)
- [Wikipedia: Minor (linear algebra)](https://en.wikipedia.org/wiki/Minor_(linear_algebra)) (covers cofactors and Laplace expansion)
- [Wikipedia: Definite matrix](https://en.wikipedia.org/wiki/Definite_matrix)
- [Mathematics for Machine Learning, chapter 4: Matrix Decompositions (free PDF)](https://mml-book.github.io/)

## Rules
- Tasks 0-4: **no imports**. Matrices are plain lists of lists. You may reuse your own earlier tasks: since the file names start with a digit, load them with Python's built-in `__import__` function, passing the file name without `.py`.
- Task 5: `import numpy as np` only.
- Error messages must match exactly.
- For tasks 1-4: if `matrix` isn't a list of lists, raise `TypeError("matrix must be a list of lists")`. If it isn't square, or is the empty matrix `[[]]`, raise `ValueError("matrix must be a non-empty square matrix")`.

## Tasks

### 0. Determinant
**File:** `0-determinant.py` · **Signature:** `determinant(matrix)`

Return the determinant of a square list of lists.
- Not a list of lists (including `[]`): `TypeError("matrix must be a list of lists")`.
- Not square: `ValueError("matrix must be a square matrix")`.
- `[[]]` is the 0x0 matrix; its determinant is 1.
```
determinant([[]])                               ->  1
determinant([[5]])                              ->  5
determinant([[1, 2], [3, 4]])                   ->  -2
determinant([[5, 7, 9], [3, 1, 8], [6, 2, 4]])  ->  192
determinant([[1, 2, 3], [4, 5, 6]])             ->  ValueError: matrix must be a square matrix
```

### 1. Minor
**File:** `1-minor.py` · **Signature:** `minor(matrix)`

Return the minor matrix. The minor matrix of a 1x1 matrix is `[[1]]`.
```
minor([[1, 2], [3, 4]])                    ->  [[4, 3], [2, 1]]
minor([[5, 7, 9], [3, 1, 8], [6, 2, 4]])   ->  [[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
```

### 2. Cofactor
**File:** `2-cofactor.py` · **Signature:** `cofactor(matrix)`
```
cofactor([[1, 2], [3, 4]])                   ->  [[4, -3], [-2, 1]]
cofactor([[5, 7, 9], [3, 1, 8], [6, 2, 4]])  ->  [[-12, 36, 0], [-10, -34, 32], [47, -13, -16]]
```

### 3. Adjugate
**File:** `3-adjugate.py` · **Signature:** `adjugate(matrix)`
```
adjugate([[1, 2], [3, 4]])                   ->  [[4, -2], [-3, 1]]
adjugate([[5, 7, 9], [3, 1, 8], [6, 2, 4]])  ->  [[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
```

### 4. Inverse
**File:** `4-inverse.py` · **Signature:** `inverse(matrix)`

Return the inverse as a list of lists, or `None` if the matrix is singular.
```
inverse([[5]])            ->  [[0.2]]
inverse([[1, 2], [3, 4]]) ->  [[-2.0, 1.0], [1.5, -0.5]]
inverse([[1, 1], [1, 1]]) ->  None
```

### 5. Definiteness
**File:** `5-definiteness.py` · **Signature:** `definiteness(matrix)`

`matrix` is a NumPy array. If it isn't one, raise `TypeError("matrix must be a numpy.ndarray")`. Return one of `"Positive definite"`, `"Positive semi-definite"`, `"Negative semi-definite"`, `"Negative definite"`, `"Indefinite"`. If the matrix isn't a valid square, symmetric 2D matrix, return `None`.
```
definiteness(np.array([[5, 1], [1, 1]]))                       ->  'Positive definite'
definiteness(np.array([[2, 4], [4, 8]]))                       ->  'Positive semi-definite'
definiteness(np.array([[1, 2], [2, 1]]))                       ->  'Indefinite'
definiteness(np.array([[2, 1, 1], [1, 2, -1], [1, -1, 2]]))    ->  'Positive semi-definite'
definiteness(np.array([[1, 2], [1, 2]]))                       ->  None
```

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 00-math/07-advanced_linear_algebra
```
