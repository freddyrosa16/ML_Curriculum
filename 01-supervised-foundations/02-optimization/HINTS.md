# Hints: Optimization

## 0. Normalization constants
<details><summary>Hint 1</summary>
You want one number per feature, and features are columns. NumPy's reductions take an `axis` argument: which axis do you collapse to get one value per column?
</details>
<details><summary>Hint 2</summary>
NumPy has built-in mean and standard deviation functions. The default standard deviation (dividing by `m`, not `m - 1`) is the one the tests expect.
</details>

## 1. Normalize
<details><summary>Hint 1</summary>
Broadcasting does the work: an `(m, nx)` array minus an `(nx,)` array subtracts the vector from every row.
</details>

## 2. Shuffle data
<details><summary>Hint 1</summary>
Don't shuffle X and Y separately; they'd end up in different orders. Generate one random ordering of the row indices and apply it to both.
</details>
<details><summary>Hint 2</summary>
`np.random.permutation` given an integer `m` returns a shuffled list of `0..m-1`. Indexing an array with a list of indices reorders its rows.
</details>

## 3. Mini-batch
<details><summary>Hint 1</summary>
Shuffle once, at the start, by calling your task 2 function. The tests seed NumPy and expect exactly one shuffle.
</details>
<details><summary>Hint 2</summary>
Batch `k` covers rows from `k * batch_size` up to (not including) `(k + 1) * batch_size`. Slicing past the end of an array just stops at the end, which takes care of the smaller final batch for free.
</details>
<details><summary>Hint 3</summary>
Python's `range` with a step lets you walk the starting row of each batch directly.
</details>

## 4. Moving average
<details><summary>Hint 1</summary>
Keep a running value that starts at 0. For each data point, blend the old running value with the new point using `beta` and `1 - beta`.
</details>
<details><summary>Hint 2</summary>
The bias correction is applied to what you *report*, not to the running value you carry forward. At step `t` (counting from 1), divide by `1 - beta**t`.
</details>

## 5. Momentum
<details><summary>Hint 1</summary>
Two lines of math: first update the moment from the old moment and the gradient, then move the variable using the new moment. The concepts section has both equations.
</details>

## 6. Momentum upgraded
<details><summary>Hint 1</summary>
Everything you need is in the docs page for `torch.optim.SGD`. An optimizer takes the parameters it should update as its first argument.
</details>

## 7. RMSProp
<details><summary>Hint 1</summary>
Same shape as momentum, but the moving average is of the *squared* gradient (element-wise).
</details>
<details><summary>Hint 2</summary>
The step divides the raw gradient by the square root of the new second moment, with epsilon added *after* the square root.
</details>

## 8. RMSProp upgraded
<details><summary>Hint 1</summary>
In `torch.optim.RMSprop`, `lr` is the learning rate, `alpha` is the smoothing constant (our `beta2`), and `eps` is epsilon. Leave momentum and the "centered" option at their defaults.
</details>

## 9. Adam
<details><summary>Hint 1</summary>
Update both moments first (one like momentum, one like RMSProp). Then make bias-corrected copies of each, using `t` as the exponent.
</details>
<details><summary>Hint 2</summary>
Use the corrected copies for the step, but return the uncorrected moments: those are what the next call needs.
</details>

## 10. Adam upgraded
<details><summary>Hint 1</summary>
`torch.optim.Adam` takes the two betas together as a tuple.
</details>

## 11. Learning rate decay
<details><summary>Hint 1</summary>
"Stepwise" means the rate only changes every `decay_step` steps. Integer (floor) division of the step count by `decay_step` tells you how many decays have happened.
</details>

## 12. Learning rate decay upgraded
<details><summary>Hint 1</summary>
Browse the list of schedulers in the `torch.optim` docs. One of them lets you supply any function of the step count that returns a *multiplier* for the starting learning rate.
</details>
<details><summary>Hint 2</summary>
Your multiplier is the task 11 formula divided by the starting rate, so it equals 1 at step 0.
</details>

## 13. Batch normalization
<details><summary>Hint 1</summary>
Each column (one neuron) is normalized using that column's mean and variance across the batch. Which axis is the batch?
</details>
<details><summary>Hint 2</summary>
Add epsilon to the variance *inside* the square root, then scale by gamma and shift by beta.
</details>

## 14. Batch normalization upgraded
<details><summary>Hint 1</summary>
`nn.Sequential` chains modules in order. `nn.Linear` has an argument to turn off its bias.
</details>
<details><summary>Hint 2</summary>
`torch.nn.init` contains in-place initializers (their names end with an underscore). Find the Xavier/Glorot one that uses a normal distribution and apply it to the linear layer's weight.
</details>
