# Hints: Autodiff

Open one hint at a time. The micrograd video is the best "hint" of all: pause it before each new piece and try first.

## 0. The Value object and the forward pass
<details><summary>Hint 1</summary>Python lets a class define what `+`, `*` and friends mean through "dunder" (double underscore) methods. Look up Python's data model docs for the arithmetic ones.</details>
<details><summary>Hint 2</summary>When the left operand is a plain number (as in 1 + a), Python tries the number's method first, fails, and then calls the *reflected* method on your Value (the ones whose names start with r). Inside your methods, wrap a plain number in a Value before using it.</details>
<details><summary>Hint 3</summary>You only need to really implement addition, multiplication and power. Subtraction is adding a negation, negation is multiplying by -1, and division is multiplying by the -1 power.</details>

## 1. backward() for + and *
<details><summary>Hint 1</summary>Each node needs two things: the nodes it was made from, and a small function that knows how to push its own gradient back to those nodes. Store both when the node is created.</details>
<details><summary>Hint 2</summary>For addition, the local derivative with respect to each input is 1. For multiplication, the local derivative with respect to one input is the *other* input's data. Multiply by the output node's gradient (the chain rule).</details>
<details><summary>Hint 3</summary>To order the nodes, do a depth-first search from the output: visit all of a node's inputs first, then append the node to a list. Keep a set of visited nodes. Set the output's gradient to 1, then walk that list backwards calling each node's push-back function.</details>

## 2. Using a value more than once
<details><summary>Hint 1</summary>In a + a, both "inputs" are the same object. If each input's gradient is *assigned*, the second assignment wipes out the first.</details>
<details><summary>Hint 2</summary>Gradients from different paths should add together, so the push-back functions should increase the input's gradient rather than set it.</details>

## 3. Gradients for every operation
<details><summary>Hint 1</summary>Each of these has one input, so its push-back function updates just one gradient: incoming gradient times local derivative.</details>
<details><summary>Hint 2</summary>Local derivatives: the power rule for a constant exponent; 1 minus the squared output for tanh; the output itself for exp; for relu, 1 where the output is positive and 0 otherwise.</details>
<details><summary>Hint 3</summary>If the finite-difference test fails only for one operation, test that operation alone on a single input and compare your gradient with the formula by hand.</details>

## 4. Neuron, Layer, MLP
<details><summary>Hint 1</summary>A neuron's forward pass is: start from the bias, add each weight times its input, then apply tanh. Because Value supports mixing with plain numbers, the inputs can be ordinary floats.</details>
<details><summary>Hint 2</summary>An MLP with sizes [4, 4, 1] and 3 inputs needs layers 3 to 4, 4 to 4 and 4 to 1. Joining the input size onto the front of the list makes the pairs easy to see.</details>
<details><summary>Hint 3</summary>`parameters()` of a layer is all its neurons' parameters, flattened into one list; an MLP's is all its layers'. Put `zero_grad()` in a base class so all three share it.</details>

## 5. Train it
<details><summary>Hint 1</summary>The loss must be a single Value built from the predictions, so that one backward call reaches every parameter.</details>
<details><summary>Hint 2</summary>Zero the gradients *before* calling backward on each step. Otherwise each step's gradients add to the last step's.</details>
<details><summary>Hint 3</summary>The update changes each parameter's `data` (not the Value object itself) by a small step in the direction that lowers the loss: the opposite sign of its gradient. Record the loss's `data` each step.</details>
