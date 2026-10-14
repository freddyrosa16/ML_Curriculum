# Hints: Classification

Open one hint at a time. Try for a while before opening the next.

## 0. Neuron
<details><summary>Hint 1</summary>
Check types before values, because the value check (`< 1`) would crash or mislead on a non-number. Think about whether `True` or `1.0` should count as an integer for your check.
</details>
<details><summary>Hint 2</summary>
The test seeds NumPy's global random generator and then expects your weights to be the very first numbers drawn. So draw the weights exactly once, with the standard normal function, in the shape the task asks for.
</details>
<details><summary>Hint 3</summary>
The shape `(1, nx)` matters: the weights form a row, so that later a row times a `(nx, m)` matrix gives one output per example.
</details>

## 1. Privatize Neuron
<details><summary>Hint 1</summary>
Two leading underscores trigger Python's "name mangling": the attribute gets renamed internally so outside code can't reach it by the plain name.
</details>
<details><summary>Hint 2</summary>
The `@property` decorator turns a method into something you read like an attribute. If you define only the getter and no setter, assigning to it raises `AttributeError` automatically.
</details>

## 2. Neuron forward propagation
<details><summary>Hint 1</summary>
Write the math first: the weighted sum for every example at once is a matrix product of the weights with the inputs, plus the bias. Check the shapes: `(1, nx)` times `(nx, m)` gives `(1, m)`.
</details>
<details><summary>Hint 2</summary>
The sigmoid uses the exponential of the negative of that sum. NumPy's exponential works element-wise on whole arrays, so no loop is needed.
</details>
<details><summary>Hint 3</summary>
The test checks that the returned array is the same object as the getter's value. Store the result first, then return what you stored.
</details>

## 3. Neuron cost
<details><summary>Hint 1</summary>
The cost is an average over the `m` examples. Where does `m` live in the shape of `Y`?
</details>
<details><summary>Hint 2</summary>
The two terms of the formula handle the two cases: when the label is 1 only the first term survives, when it's 0 only the second does. Multiplying arrays element-wise and summing is all you need.
</details>
<details><summary>Hint 3</summary>
Don't forget the minus sign in front: logs of numbers between 0 and 1 are negative, and the cost should be positive.
</details>

## 4. Evaluate neuron
<details><summary>Hint 1</summary>
Reuse your own methods: run the forward pass, then compute the cost on the result.
</details>
<details><summary>Hint 2</summary>
Turning probabilities into 0/1 labels is a comparison against 0.5. NumPy has a function that picks between two values based on a condition, element by element. Make sure the result is integers, not booleans or floats.
</details>

## 5. Neuron gradient descent
<details><summary>Hint 1</summary>
The beauty of sigmoid plus cross-entropy: the derivative of the cost with respect to the weighted sum $z$ simplifies to just "prediction minus label". Watch 3Blue1Brown's backprop calculus video or work it out on paper.
</details>
<details><summary>Hint 2</summary>
From that error signal, the weight gradient is the average over examples of (error times input). In matrix form that's a product of the error with the inputs, with one of them transposed so the shapes line up to match `W`. The bias gradient is the average error.
</details>
<details><summary>Hint 3</summary>
Then step against the gradient, scaled by `alpha`. If your weights come out with the wrong shape, compare the shape of your gradient to `(1, nx)`.
</details>

## 6. Train neuron
<details><summary>Hint 1</summary>
Validate in the order listed. Same question as task 0 about what counts as the right type.
</details>
<details><summary>Hint 2</summary>
Each iteration is: forward pass, then one gradient descent step using the activations you just computed. After the loop, evaluate.
</details>

## 7. Upgrade train neuron
<details><summary>Hint 1</summary>
The printed iterations are 0, step, 2·step, ..., and the final one. That means your loop has to run one extra time so it can report the cost *after* the last update, without doing an extra update.
</details>
<details><summary>Hint 2</summary>
Keep a list of the costs you report (and the iteration numbers they belong to) so you can plot them at the end with matplotlib.
</details>
<details><summary>Hint 3</summary>
Only check `step` if verbose or graph is on, and do it after the iteration and alpha checks.
</details>

## 8-9. NeuralNetwork and privatizing it
<details><summary>Hint 1</summary>
Each layer has a weight matrix shaped (nodes in this layer, nodes in the previous layer). The output layer has 1 node and the hidden layer is its "previous layer".
</details>
<details><summary>Hint 2</summary>
Creation order matters for the seeded test: the hidden layer's random weights are drawn before the output layer's.
</details>

## 10. NeuralNetwork forward propagation
<details><summary>Hint 1</summary>
It's task 2 twice: the hidden layer's output becomes the input of the output layer.
</details>
<details><summary>Hint 2</summary>
The bias `b1` is a column `(nodes, 1)`. Broadcasting stretches it across all `m` columns for you.
</details>

## 11-12. Cost and evaluate
<details><summary>Hint 1</summary>
These are the same as tasks 3 and 4, applied to the output layer.
</details>

## 13. NeuralNetwork gradient descent
<details><summary>Hint 1</summary>
Start at the output: its error signal and gradients look exactly like task 5, except the "inputs" to the output layer are the hidden activations.
</details>
<details><summary>Hint 2</summary>
To pass the error back to the hidden layer, send it backwards through the output weights (a transpose shows up), then multiply element-wise by the sigmoid's derivative at the hidden layer. The sigmoid's derivative can be written using only its own output: $a(1-a)$.
</details>
<details><summary>Hint 3</summary>
Compute all gradients before you change any weights. If you update `W2` first, the hidden layer's error will be computed with the wrong weights.
</details>

## 14-15. Training the network
<details><summary>Hint 1</summary>
These are tasks 6 and 7 again, now calling your two-layer forward pass and gradient descent.
</details>

## 16. DeepNeuralNetwork
<details><summary>Hint 1</summary>
To validate `layers`, check it's a list and not empty, then check every element is a positive integer. Python's built-ins `all` and `isinstance` help.
</details>
<details><summary>Hint 2</summary>
In your single loop, the "previous layer size" is `nx` for the first layer and the previous entry of `layers` after that. Build the dictionary keys from the layer number as strings: `'W' + number`.
</details>
<details><summary>Hint 3</summary>
He initialization multiplies a standard normal matrix by the square root of (2 divided by the previous layer's size).
</details>

## 17. Privatize DeepNeuralNetwork
<details><summary>Hint 1</summary>
Same idea as tasks 1 and 9. Inside the class you can keep using the private names.
</details>

## 18. DeepNeuralNetwork forward propagation
<details><summary>Hint 1</summary>
Think of the cache as a ladder: `A0` is the input, and each rung `A(l)` is built from `A(l-1)` with that layer's weights and bias.
</details>
<details><summary>Hint 2</summary>
The output is simply the last rung. Return it along with the whole cache.
</details>

## 19-20. Cost and evaluate
<details><summary>Hint 1</summary>
Unchanged from the earlier classes, just using the deep forward pass.
</details>

## 21. DeepNeuralNetwork gradient descent
<details><summary>Hint 1</summary>
Loop backwards from the last layer to the first. At each layer you need the error signal for that layer and the activations of the layer before it (both in or derivable from the cache).
</details>
<details><summary>Hint 2</summary>
Before updating a layer's weights, use those (old) weights to compute the error signal for the previous layer. Then update. That way one loop handles everything in the right order.
</details>
<details><summary>Hint 3</summary>
The pattern at every layer: weight gradient = (error × previous activations transposed) / m; bias gradient = average error across examples, keeping it a column; previous error = (weights transposed × error) × sigmoid derivative of previous activations.
</details>

## 22-23. Training the deep network
<details><summary>Hint 1</summary>
Tasks 6 and 7 once more. Your forward pass now returns two things; gradient descent needs the cache.
</details>

## 24. One-hot encode
<details><summary>Hint 1</summary>
Start with a matrix of zeros of shape `(classes, m)`. Each column needs exactly one 1, in the row given by that example's label.
</details>
<details><summary>Hint 2</summary>
NumPy's "fancy indexing" lets you set many cells at once by giving it an array of row indices and an array of column indices of the same length.
</details>
<details><summary>Hint 3</summary>
Guard the invalid cases at the top and return `None` for each.
</details>

## 25. One-hot decode
<details><summary>Hint 1</summary>
In each column, the label is the row where the 1 is. Which NumPy function returns the position of the largest value, and along which axis do you need it?
</details>

## 26. Persistence is key
<details><summary>Hint 1</summary>
`pickle` can save an entire Python object, private attributes and all. Files must be opened in binary mode for pickle.
</details>
<details><summary>Hint 2</summary>
A `@staticmethod` belongs to the class but doesn't receive `self`, which is what you want when there's no object yet. The `os.path` module can tell you whether a file exists.
</details>

## 27. Update DeepNeuralNetwork
<details><summary>Hint 1</summary>
Softmax: exponentiate each score, then divide by the sum of the exponentials *in that column* (each column is one example).
</details>
<details><summary>Hint 2</summary>
The nice property from task 5 survives: softmax plus categorical cross-entropy gives an output error signal of "prediction minus one-hot label". So gradient descent barely changes.
</details>
<details><summary>Hint 3</summary>
For the one-hot prediction, compare each value to the maximum of its column. Watch which axis you take the max along.
</details>

## 28. All the activations
<details><summary>Hint 1</summary>
Forward pass: pick sigmoid or tanh for hidden layers based on the stored activation. Output stays softmax.
</details>
<details><summary>Hint 2</summary>
Backward pass: you need the derivative of the chosen activation, written in terms of its output. For sigmoid it's $a(1-a)$. Look up (or derive) the equivalent for tanh.
</details>
