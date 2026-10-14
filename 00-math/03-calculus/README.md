# Calculus

**Status:** Original (the multiple-choice questions became short-answer questions with numeric answers)

## Why this matters
Training a neural network means nudging millions of numbers in the direction that lowers the error, and "the direction that lowers the error" is a derivative. Sums and products are how ML formulas are written down, and integrals show up whenever probability does. You don't need to be a calculus wizard, but you do need to be comfortable reading this notation and doing simple derivatives by hand.

## Concepts to understand first
- **What do $\sum$ and $\prod$ mean?** $\sum_{i=1}^{n} a_i$ means "add up $a_i$ for $i = 1, 2, \dots, n$". $\prod$ is the same idea with multiplication. They're just compact for-loops written in math.
- **What is a series, and what are some common ones?** A series is the sum of a sequence. Many common series have a *closed form*, a formula that gives the total without adding term by term. The famous one: $1 + 2 + \dots + n = \frac{n(n+1)}{2}$. There are similar formulas for sums of squares and cubes.
- **What is a derivative?** The instantaneous rate of change: how much $f(x)$ changes per tiny change in $x$. Geometrically, the slope of the tangent line.
- **What are the common derivative rules?** Power rule: $(x^n)' = n x^{n-1}$. $(e^x)' = e^x$. $(\ln x)' = 1/x$. Constants disappear, and derivatives of sums are sums of derivatives.
- **What are the product and chain rules?** Product: $(fg)' = f'g + fg'$. Chain: $(f(g(x)))' = f'(g(x)) \cdot g'(x)$, the "outer derivative times inner derivative". The chain rule is literally what backpropagation computes.
- **What is a partial derivative?** For a function of several variables, the derivative with respect to one of them while treating the others as constants. Written $\partial f / \partial x$.
- **What is an indefinite integral?** The reverse of a derivative (an antiderivative): a function whose derivative gives back the original. It includes an arbitrary constant $+C$, since constants vanish when you differentiate.
- **What are definite and double integrals?** $\int_a^b f(x)\,dx$ is the signed area under $f$ between $a$ and $b$: find an antiderivative $F$, then compute $F(b) - F(a)$. A double integral integrates twice, once per variable, giving a volume.

## Resources
- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus)
- [Khan Academy: Differential Calculus](https://www.khanacademy.org/math/differential-calculus)
- [Khan Academy: Integral Calculus](https://www.khanacademy.org/math/integral-calculus)
- [Khan Academy: Partial derivatives (Multivariable Calculus)](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives)
- [Mathematics for Machine Learning, chapter 5: Vector Calculus (free PDF)](https://mml-book.github.io/)
- [Paul's Online Math Notes: Calculus I](https://tutorial.math.lamar.edu/Classes/CalcI/CalcI.aspx)

## Rules
- Short-answer tasks: create the named `.txt` file containing **one number only**. Use a decimal (at least 2 decimal places if it isn't whole) or a fraction like `3/4`. No units, no symbols like `e` or `ln`.
- Coding tasks (9, 10, 17): plain Python, **no imports**.
- Task 9: no loops of any kind and no built-in `sum`.

## Tasks

Here $\ln$ is the natural log and $e \approx 2.71828$.

### 0. Sigma is for Sum
**File:** `0-sigma_is_for_sum.txt` · Compute $\sum_{i=2}^{5} i$

### 1. The Sum of It All
**File:** `1-seegma.txt` · Compute $\sum_{k=1}^{4} (3k - 2)$

### 2. Pi is for Product
**File:** `2-pi_is_for_product.txt` · Compute $\prod_{i=1}^{5} i$

### 3. Watch the Starting Index
**File:** `3-pee.txt` · Compute $\prod_{i=0}^{10} i$

### 4. Hello, Derivatives
**File:** `4-hello_derivatives.txt` · For $f(x) = x^4 + 3x^3 - 5x + 1$, find $f'(2)$.

### 5. Log on Fire
**File:** `5-log_on_fire.txt` · For $f(x) = x \ln x$, find $f'(e)$. (Product rule.)

### 6. Chain Reaction
**File:** `6-voltaire.txt` · For $f(x) = \ln(x^2)$, find $f'(4)$. (Chain rule.)

### 7. Partial Truths
**File:** `7-partial_truths.txt` · For $f(x, y) = e^{xy}$, find $\frac{\partial f}{\partial y}$ at $(x, y) = (2, 0)$.

### 8. All Together
**File:** `8-all-together.txt` · For $f(x, y) = x^3 y^2 + 2xy$, find $\frac{\partial^2 f}{\partial y\,\partial x}$ (differentiate with respect to $x$, then $y$) at $(1, 1)$.

### 9. Our life is the sum total of all the decisions we make every day
**File:** `9-sum_total.py` · **Signature:** `summation_i_squared(n)`

Return $\sum_{i=1}^{n} i^2$ as an `int`. If `n` isn't a positive integer, return `None`. No loops, no `sum`: it must be instant even for `n = 1_000_000`.
```
summation_i_squared(5)    ->  55
summation_i_squared(0)    ->  None
summation_i_squared("5")  ->  None
```

### 10. Derive happiness in oneself from a good day's work
**File:** `10-matisse.py` · **Signature:** `poly_derivative(poly)`

A polynomial is a list of coefficients where index = power: $x^3 + 3x + 5$ is `[5, 3, 0, 1]`. Return the derivative in the same format.
- If `poly` isn't a non-empty list, return `None`.
- If the derivative is zero everywhere (e.g. a constant), return `[0]`.
```
poly_derivative([5, 3, 0, 1])  ->  [3, 0, 3]
poly_derivative([5])           ->  [0]
poly_derivative([])            ->  None
```

### 11. Integrate a Power
**File:** `11-integral.txt` · Let $F(x) = \int x^3\,dx$ with constant $C = 0$. What is $F(2)$?

### 12. Integrate an Exponential
**File:** `12-integral.txt` · Let $G(y) = \int e^{2y}\,dy$ with constant $C = 0$. What is $G(0)$?

### 13. Definitely an Area
**File:** `13-definite.txt` · Compute $\int_{0}^{3} u^2\,du$

### 14. Area Under 1/v
**File:** `14-definite.txt` · Compute $\int_{1}^{e} \frac{1}{v}\,dv$

### 15. Which Variable?
**File:** `15-definite.txt` · The integral $\int_{0}^{5} x\,dy$ is a function of $x$. What is its value when $x = 2$?

### 16. Double Trouble
**File:** `16-double.txt` · Compute $\int_{1}^{2} \int_{0}^{3} \frac{x^2}{y}\,dx\,dy$ (the inner integral is over $x$). Give at least 2 decimal places.

### 17. Integrate
**File:** `17-integrate.py` · **Signature:** `poly_integral(poly, C=0)`

Same polynomial format as task 10. Return the integral, with `C` as the constant term.
- Coefficients that are whole numbers must be `int`s (so `3` not `3.0`).
- Return the smallest list possible: drop trailing zeros.
- If `poly` isn't a non-empty list, or `C` isn't an `int`, return `None`.
```
poly_integral([5, 3, 0, 1])       ->  [0, 5, 1.5, 0, 0.25]
poly_integral([5, 3, 0, 1], C=2)  ->  [2, 5, 1.5, 0, 0.25]
poly_integral([0])                ->  [0]
poly_integral([1, 2], C=1.5)      ->  None
```

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 00-math/03-calculus
```
A wrong short answer fails without revealing the right one. Rework it on paper.
