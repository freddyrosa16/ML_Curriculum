# Hints: Calculus

Open one hint at a time.

## 0. Sigma is for Sum
<details><summary>Hint 1</summary>Write out every term: the index starts at the bottom number and stops at the top number, inclusive.</details>

## 1. The Sum of It All
<details><summary>Hint 1</summary>Plug in k = 1, 2, 3, 4 one at a time and add the results.</details>
<details><summary>Hint 2</summary>Or split it: the sum of 3k is 3 times the sum of k, and the sum of a constant 2 over 4 terms is 4 times 2.</details>

## 2. Pi is for Product
<details><summary>Hint 1</summary>Multiply every value of i from 1 to 5. This product has a special name and its own symbol (an exclamation mark).</details>

## 3. Watch the Starting Index
<details><summary>Hint 1</summary>Look at the very first term before doing any multiplying.</details>

## 4. Hello, Derivatives
<details><summary>Hint 1</summary>Differentiate term by term with the power rule. The constant term contributes nothing.</details>
<details><summary>Hint 2</summary>Find the derivative as a formula first, then substitute x = 2.</details>

## 5. Log on Fire
<details><summary>Hint 1</summary>This is a product of two functions, x and ln x. Use the product rule.</details>
<details><summary>Hint 2</summary>The derivative of ln x is 1/x, and ln e = 1.</details>

## 6. Chain Reaction
<details><summary>Hint 1</summary>Outer function: ln(something). Inner function: x squared. Chain rule: derivative of outer (evaluated at the inner) times derivative of inner.</details>
<details><summary>Hint 2</summary>Alternative route: a log rule lets you pull the exponent out front first, which makes the derivative very short.</details>

## 7. Partial Truths
<details><summary>Hint 1</summary>When differentiating with respect to y, x is just a constant, like the number 2.</details>
<details><summary>Hint 2</summary>The derivative of e raised to (a times y), with respect to y, is a times e raised to (a times y).</details>

## 8. All Together
<details><summary>Hint 1</summary>Two steps. First take the partial derivative with respect to x, treating y as a constant.</details>
<details><summary>Hint 2</summary>Then take the partial derivative of that result with respect to y, treating x as a constant. Finally substitute (1, 1).</details>

## 9. Sum total
<details><summary>Hint 1</summary>A loop is forbidden, so you need a formula that gives the sum of the first n squares directly. Look up "sum of squares formula".</details>
<details><summary>Hint 2</summary>The formula involves n, n + 1 and 2n + 1, divided by 6. Use integer division so the result stays an `int` (the product is always divisible by 6).</details>
<details><summary>Hint 3</summary>Validate first: `isinstance` checks the type, then check that n is at least 1.</details>

## 10. Derive happiness in oneself from a good day's work
<details><summary>Hint 1</summary>The coefficient at index i belongs to the x to the power i term. What does the power rule do to that term?</details>
<details><summary>Hint 2</summary>Each term's new coefficient is its old coefficient times its power, and it moves down one index. The constant term (index 0) disappears.</details>
<details><summary>Hint 3</summary>Handle the special cases separately: bad input gives `None`, and a derivative that is all zeros becomes the single-element list containing 0.</details>

## 11. Integrate a Power
<details><summary>Hint 1</summary>Reverse power rule: raise the exponent by one and divide by the new exponent.</details>

## 12. Integrate an Exponential
<details><summary>Hint 1</summary>Ask: what function has a derivative equal to e to the 2y? Try e to the 2y itself, see what the chain rule gives, and correct by a constant factor.</details>
<details><summary>Hint 2</summary>Remember e to the 0 is 1.</details>

## 13. Definitely an Area
<details><summary>Hint 1</summary>Find an antiderivative F of u squared, then compute F(3) minus F(0).</details>

## 14. Area Under 1/v
<details><summary>Hint 1</summary>Which function has derivative 1/v? Evaluate it at the upper limit, then subtract its value at the lower limit.</details>

## 15. Which Variable?
<details><summary>Hint 1</summary>You're integrating with respect to y, so x is a constant here. What's the integral of a constant from 0 to 5?</details>

## 16. Double Trouble
<details><summary>Hint 1</summary>Work from the inside out. The inner integral is over x, so 1/y acts as a constant there.</details>
<details><summary>Hint 2</summary>After the inner integral you're left with a number times 1/y. Integrate that over y from 1 to 2, using the same antiderivative as task 14.</details>
<details><summary>Hint 3</summary>The exact answer is a whole number times ln 2. Turn it into a decimal with a calculator.</details>

## 17. Integrate
<details><summary>Hint 1</summary>Integration shifts each coefficient up one index and divides it by its new power. The constant C goes in at index 0.</details>
<details><summary>Hint 2</summary>A float knows whether it has a fractional part: look up the float method that tells you if it's a whole number, and convert those to `int`.</details>
<details><summary>Hint 3</summary>"Smallest list possible" means removing zeros from the end, but never removing the last remaining element.</details>
