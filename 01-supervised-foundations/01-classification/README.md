# Classification with Neural Networks (NumPy)

**Status:** Original

## Why this matters
Every deep learning library does the same thing under the hood: multiply by weights, squash with an activation, measure the error, and nudge the weights downhill. In this project you build that machinery yourself, starting with one neuron and ending with a deep network that reads handwritten digits. Once you've written backpropagation by hand, PyTorch stops being magic.

## Concepts to understand first
- **What is a neuron?** A function that takes a weighted sum of its inputs, adds a bias, and passes the result through an activation: $a = \sigma(Wx + b)$. Weights say how much each input matters; the bias shifts the threshold.
- **What does the sigmoid do?** $\sigma(z) = \frac{1}{1 + e^{-z}}$ squashes any number into $(0, 1)$, so the output can be read as "probability this example is class 1".
- **Why do we put examples in columns?** In this project `X` has shape `(nx, m)`: one row per feature, one column per example. Then one matrix product `W @ X` handles all `m` examples at once (vectorization), which is far faster than a Python loop.
- **What is a cost function?** A single number saying how wrong the model is across the whole dataset. For binary classification we use the logistic (cross-entropy) cost: $J = -\frac{1}{m}\sum \big[y\log a + (1-y)\log(1-a)\big]$. It is small when confident predictions are right and huge when confident predictions are wrong.
- **What is gradient descent?** Picture the cost as a hilly landscape over all possible weights. The gradient points uphill, so you take a small step the opposite way: $W \leftarrow W - \alpha \frac{\partial J}{\partial W}$. The learning rate $\alpha$ is the step size.
- **What is backpropagation?** The chain rule applied layer by layer, from the output back to the input. Each layer reuses the error signal from the layer after it, so you get every gradient in one backward sweep.
- **Why does a hidden layer help?** One neuron can only draw a straight line (a hyperplane) between classes. Stacking layers with non-linear activations lets the network bend that boundary into any shape it needs.
- **What are one-hot vectors and softmax?** For 10 digit classes, the label 3 becomes `[0,0,0,1,0,0,0,0,0,0]`. Softmax turns the network's 10 raw scores into 10 probabilities that sum to 1, so the output matches that format.
- **Why initialize weights randomly (and scaled)?** If every weight starts the same, every neuron learns the same thing. He initialization scales random weights by $\sqrt{2/n_{prev}}$ so signals don't blow up or vanish as they pass through many layers.

## Resources
- [3Blue1Brown: Neural Networks (chapters 1-4)](https://www.3blue1brown.com/topics/neural-networks) - the best visual intuition for neurons, gradient descent and backprop
- [Andrej Karpathy: The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [CS231n: Optimization and backpropagation notes](https://cs231n.github.io/optimization-2/)
- [Dive into Deep Learning: Multilayer perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/index.html)
- [StatQuest: Neural Networks Pt. 1 - Inside the Black Box](https://www.youtube.com/watch?v=CqOfi41LfDw)
- [Michael Nielsen: Neural Networks and Deep Learning (free book), chapters 1-2](http://neuralnetworksanddeeplearning.com/)
- [Python docs: `pickle`](https://docs.python.org/3/library/pickle.html) and [`@property`](https://docs.python.org/3/library/functions.html#property)

## Rules
- Only `import numpy as np` (plus `matplotlib.pyplot` for the "upgraded train" tasks 7, 15, 23 and later, and `pickle`/`os` for task 26 onward).
- No loops over examples. You may loop over **layers** (tasks 16+) and over **iterations** (training).
- Each task file is a full, standalone copy of the class so far (e.g. `3-neuron.py` contains everything from `2-neuron.py` plus the new method).
- Raise exceptions in the order the task lists them, with exactly the messages given.
- Shapes matter: `X` is `(nx, m)`, binary labels `Y` are `(1, m)`, one-hot labels are `(classes, m)`.

## Data
The `data/` folder already contains:
- `MNIST.npz` - handwritten digits 0-9. Keys: `X_train` (50000, 28, 28), `Y_train` (50000,), `X_valid`, `Y_valid`, `X_test`, `Y_test` (10000 each). Pixel values are already scaled to [0, 1].
- `Binary_Dev.npz` - a binary "is it a 1 or a 0?" dev set. Keys: `X` (2115, 28, 28) and `Y` (1, 2115).

`Binary_Train.npz` (38 MB, keys `X` (12665, 28, 28) and `Y` (1, 12665)) is too big to ship. Download it into `data/` from
`https://github.com/JRMC-PR/holbertonschool-machine_learning/raw/main/supervised_learning/data/Binary_Train.npz`.
(Alternatively, build your own: it is every 0 and 1 from MNIST's training and validation sets combined, with label 1 for the digit 1 and 0 for the digit 0.)

To feed images to your network, flatten each 28x28 image to 784 numbers and transpose, so `X` has shape `(784, m)`.

The tests use tiny synthetic data, so you don't need the datasets to pass them. Use the real data to watch your networks actually learn.

---

## Tasks

### Part A: a single neuron

### 0. Neuron
**File:** `0-neuron.py` · **Class:** `Neuron`

`Neuron(nx)` builds one neuron for binary classification with `nx` input features.
- If `nx` is not an `int`: raise `TypeError("nx must be an integer")`.
- If `nx < 1`: raise `ValueError("nx must be a positive integer")`.
- Public attributes:
  - `W`: weights, shape `(1, nx)`, drawn from a standard normal distribution (`np.random.randn`).
  - `b`: bias, starts at `0`.
  - `A`: activated output, starts at `0`.

```
>>> np.random.seed(0)
>>> neuron = Neuron(2)
>>> neuron.W
array([[1.76405235, 0.40015721]])
>>> neuron.b, neuron.A
(0, 0)
>>> Neuron(0)
ValueError: nx must be a positive integer
```

### 1. Privatize Neuron
**File:** `1-neuron.py`

Same as task 0, but make `W`, `b`, `A` private (`__W`, `__b`, `__A`) and expose each through a read-only property (a getter, no setter). Assigning `neuron.A = 10` must now raise `AttributeError`.

### 2. Neuron forward propagation
**File:** `2-neuron.py` · **Method:** `forward_prop(self, X)`

`X` has shape `(nx, m)`. Compute the neuron's output with a sigmoid activation, store it in `__A`, and return it.

```
>>> X = np.array([[1., 2., 3.], [.5, -1., 2.]])
>>> np.random.seed(0); neuron = Neuron(2)
>>> neuron.forward_prop(X)
array([[0.87700, 0.95799, 0.99773]])
```

### 3. Neuron cost
**File:** `3-neuron.py` · **Method:** `cost(self, Y, A)`

`Y` is `(1, m)` true labels, `A` is `(1, m)` activated outputs. Return the logistic regression cost (a float). To avoid `log(0)`, use `1.0000001 - A` instead of `1 - A`.

```
>>> Y = np.array([[1, 0, 1]])
>>> neuron.cost(Y, neuron.forward_prop(X))
1.1014498118920475
```

### 4. Evaluate neuron
**File:** `4-neuron.py` · **Method:** `evaluate(self, X, Y)`

Return a tuple `(prediction, cost)`: `prediction` is a `(1, m)` array of ints, 1 where the output is `>= 0.5` and 0 otherwise.

```
>>> neuron.evaluate(X, Y)
(array([[1, 1, 1]]), 1.1014498118920475)
```

### 5. Neuron gradient descent
**File:** `5-neuron.py` · **Method:** `gradient_descent(self, X, Y, A, alpha=0.05)`

Do one step of gradient descent on the logistic cost, updating `__W` and `__b`. Returns nothing.

```
>>> A = neuron.forward_prop(X)
>>> neuron.gradient_descent(X, Y, A, 0.5)
>>> neuron.W, neuron.b
(array([[1.46627, 0.57083]]), -0.1387925904917963)
```

### 6. Train neuron
**File:** `6-neuron.py` · **Method:** `train(self, X, Y, iterations=5000, alpha=0.05)`

Run `iterations` rounds of forward prop + gradient descent, then return `self.evaluate(X, Y)`.
- `iterations` not an int → `TypeError("iterations must be an integer")`; `< 1` → `ValueError("iterations must be a positive integer")`.
- `alpha` not a float → `TypeError("alpha must be a float")`; `<= 0` → `ValueError("alpha must be positive")`.

```
>>> np.random.seed(0); neuron = Neuron(2)
>>> neuron.train(X, Y, iterations=10, alpha=0.5)
(array([[1, 0, 1]]), 0.22168200016468909)
```

### 7. Upgrade train neuron
**File:** `7-neuron.py` · **Method:** `train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100)`

Same as task 6, plus:
- If `verbose`, print `Cost after {i} iterations: {cost}` at iteration 0, every `step` iterations, and after the last iteration.
- If `graph`, plot those costs against the iteration number (blue line, x label `iteration`, y label `cost`, title `Training Cost`) once training is done.
- Only when `verbose` or `graph` is on, validate `step`: not an int → `TypeError("step must be an integer")`; `<= 0` or `> iterations` → `ValueError("step must be positive and <= iterations")`.

```
>>> np.random.seed(0); neuron = Neuron(2)
>>> neuron.train(X, Y, iterations=10, alpha=0.5, graph=False, step=5)
Cost after 0 iterations: 1.1014498118920475
Cost after 5 iterations: 0.3228034571814655
Cost after 10 iterations: 0.22168200016468909
(array([[1, 0, 1]]), 0.22168200016468909)
```

### Part B: one hidden layer

### 8. NeuralNetwork
**File:** `8-neural_network.py` · **Class:** `NeuralNetwork`

`NeuralNetwork(nx, nodes)`: one hidden layer with `nodes` neurons, and one output neuron.
- `nx` checks as in task 0. `nodes` not an int → `TypeError("nodes must be an integer")`; `< 1` → `ValueError("nodes must be a positive integer")`.
- Public attributes, created in this order: `W1` `(nodes, nx)` random normal, `b1` zeros `(nodes, 1)`, `A1 = 0`, `W2` `(1, nodes)` random normal, `b2 = 0`, `A2 = 0`.

```
>>> np.random.seed(0); nn = NeuralNetwork(2, 3)
>>> nn.W1.shape, nn.b1.shape, nn.W2.shape
((3, 2), (3, 1), (1, 3))
```

### 9. Privatize NeuralNetwork
**File:** `9-neural_network.py`

Make all six attributes private with getters only (as in task 1).

### 10. NeuralNetwork forward propagation
**File:** `10-neural_network.py` · **Method:** `forward_prop(self, X)`

Sigmoid on both layers. Update `__A1` and `__A2`, return `(A1, A2)`.

```
>>> A1, A2 = nn.forward_prop(X)
>>> A1.shape, A2
((3, 3), array([[0.64927, 0.67756, 0.66727]]))
```

### 11. NeuralNetwork cost
**File:** `11-neural_network.py` · **Method:** `cost(self, Y, A)` - same formula as task 3.

### 12. Evaluate NeuralNetwork
**File:** `12-neural_network.py` · **Method:** `evaluate(self, X, Y)` - returns `(prediction, cost)` like task 4, using `A2`.

```
>>> nn.evaluate(X, Y)
(array([[1, 1, 1]]), 0.6561365315185336)
```

### 13. NeuralNetwork gradient descent
**File:** `13-neural_network.py` · **Method:** `gradient_descent(self, X, Y, A1, A2, alpha=0.05)`

One step of gradient descent on all four parameters (backpropagate through the sigmoid hidden layer). Returns nothing.

### 14. Train NeuralNetwork
**File:** `14-neural_network.py` · **Method:** `train(self, X, Y, iterations=5000, alpha=0.05)` - same rules and errors as task 6.

### 15. Upgrade train NeuralNetwork
**File:** `15-neural_network.py` · **Method:** `train(..., verbose=True, graph=True, step=100)` - same rules as task 7.

```
>>> np.random.seed(0); nn = NeuralNetwork(2, 3)
>>> nn.train(X, Y, iterations=10, alpha=0.5, graph=False, step=5)
Cost after 0 iterations: 0.6561365315185336
Cost after 5 iterations: 0.6235683782732213
Cost after 10 iterations: 0.5905786277741336
(array([[1, 1, 1]]), 0.5905786277741336)
```

### Part C: any number of layers

### 16. DeepNeuralNetwork
**File:** `16-deep_neural_network.py` · **Class:** `DeepNeuralNetwork`

`DeepNeuralNetwork(nx, layers)`, where `layers` is a list with the number of nodes in each layer (the last entry is the output layer).
- `nx` checks as before. If `layers` is not a non-empty list of positive ints → `TypeError("layers must be a list of positive integers")`.
- Public attributes: `L` (number of layers), `cache` (empty dict, will hold every layer's output), `weights` (dict).
- `weights` holds `W1 ... WL` and `b1 ... bL`. Initialize `Wl` with He initialization: standard normal of shape `(nodes in layer l, nodes in layer l-1)` multiplied by $\sqrt{2 / \text{nodes in layer } l-1}$ (layer 0 is the input, with `nx` nodes). Biases are zero column vectors. Create them in order `W1, b1, W2, b2, ...`.
- You may use only one loop.

```
>>> np.random.seed(0); deep = DeepNeuralNetwork(2, [3, 1])
>>> deep.L, deep.cache
(2, {})
>>> deep.weights['W2']
array([[ 0.77572, -0.12358, -0.08428]])
```

### 17. Privatize DeepNeuralNetwork
**File:** `17-deep_neural_network.py` - `L`, `cache`, `weights` become private with getters.

### 18. DeepNeuralNetwork forward propagation
**File:** `18-deep_neural_network.py` · **Method:** `forward_prop(self, X)`

Store `X` in `cache['A0']`, then each layer's sigmoid output in `cache['A1']`, `cache['A2']`, ... Return `(output, cache)`. One loop only.

```
>>> A, cache = deep.forward_prop(X)
>>> A
array([[0.62306, 0.64720, 0.63843]])
>>> sorted(cache)
['A0', 'A1', 'A2']
```

### 19. DeepNeuralNetwork cost
**File:** `19-deep_neural_network.py` · **Method:** `cost(self, Y, A)` - same formula as before.

### 20. Evaluate DeepNeuralNetwork
**File:** `20-deep_neural_network.py` · **Method:** `evaluate(self, X, Y)` - returns `(prediction, cost)`.

```
>>> deep.evaluate(X, Y)
(array([[1, 1, 1]]), 0.6545115124766185)
```

### 21. DeepNeuralNetwork gradient descent
**File:** `21-deep_neural_network.py` · **Method:** `gradient_descent(self, Y, cache, alpha=0.05)`

One step of gradient descent, updating `weights` in place. One loop only.

### 22. Train DeepNeuralNetwork
**File:** `22-deep_neural_network.py` · **Method:** `train(self, X, Y, iterations=5000, alpha=0.05)` - same rules as task 6.

### 23. Upgrade train DeepNeuralNetwork
**File:** `23-deep_neural_network.py` · **Method:** `train(..., verbose=True, graph=True, step=100)` - same rules as task 7.

```
>>> np.random.seed(0); deep = DeepNeuralNetwork(2, [3, 1])
>>> deep.train(X, Y, iterations=10, alpha=0.5, graph=False, step=5)
Cost after 0 iterations: 0.6545115124766185
Cost after 5 iterations: 0.6177849395007321
Cost after 10 iterations: 0.5831786338690446
(array([[1, 1, 1]]), 0.5831786338690446)
```

### Part D: many classes

### 24. One-hot encode
**File:** `24-one_hot_encode.py` · **Function:** `one_hot_encode(Y, classes)`

`Y` is a `(m,)` array of integer labels, `classes` the number of classes. Return a `(classes, m)` one-hot matrix, or `None` if the input is invalid (not an array, not an int, empty, or a label `>= classes`).

```
>>> one_hot_encode(np.array([2, 0, 1, 2]), 3)
array([[0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [1., 0., 0., 1.]])
>>> one_hot_encode(np.array([2, 0, 3]), 3) is None
True
```

### 25. One-hot decode
**File:** `25-one_hot_decode.py` · **Function:** `one_hot_decode(one_hot)`

Turn a `(classes, m)` one-hot matrix back into a `(m,)` array of labels. Return `None` if the input is not a 2D numpy array.

```
>>> one_hot_decode(one_hot_encode(np.array([2, 0, 1, 2]), 3))
array([2, 0, 1, 2])
```

### 26. Persistence is key
**File:** `26-deep_neural_network.py`

Add to the binary `DeepNeuralNetwork` (from task 23):
- `save(self, filename)`: pickle the whole object to `filename`, adding `.pkl` if the name doesn't already end with it.
- `load(filename)` (a `@staticmethod`): return the unpickled object, or `None` if the file doesn't exist.

```
>>> deep.save('my_net')          # writes my_net.pkl
>>> DeepNeuralNetwork.load('my_net.pkl').L
2
>>> DeepNeuralNetwork.load('nope.pkl') is None
True
```

### 27. Update DeepNeuralNetwork
**File:** `27-deep_neural_network.py`

Turn the network into a multiclass classifier (keep `save`/`load`):
- Hidden layers still use sigmoid; the **output layer uses softmax**, so each column of the output sums to 1.
- `Y` is now one-hot, shape `(classes, m)`. The cost becomes the categorical cross-entropy: $-\frac{1}{m}\sum Y \log A$.
- `evaluate` returns a one-hot `(classes, m)` prediction (1 at the most likely class of each column) and the cost.

```
>>> Y = one_hot_encode(np.array([0, 2, 1]), 3)
>>> np.random.seed(0); deep = DeepNeuralNetwork(2, [4, 3])
>>> A, _ = deep.forward_prop(X)
>>> A.sum(axis=0)
array([1., 1., 1.])
>>> deep.evaluate(X, Y)
(array([[1, 0, 1],
        [0, 1, 0],
        [0, 0, 0]]), 1.1179072822887548)
```

Try it on MNIST: a network like `[128, 64, 10]` trained for a few hundred iterations with `alpha=0.5` or so should reach well over 80% validation accuracy.

### 28. All the activations
**File:** `28-deep_neural_network.py`

Add a constructor argument `activation='sig'` choosing the hidden-layer activation: `'sig'` (sigmoid) or `'tanh'`. Anything else → `ValueError("activation must be 'sig' or 'tanh'")`. Store it privately with an `activation` getter. Forward prop and gradient descent must use the chosen activation for hidden layers (output is still softmax).

```
>>> DeepNeuralNetwork(2, [4, 3], activation='tanh').activation
'tanh'
>>> DeepNeuralNetwork(2, [3], activation='relu')
ValueError: activation must be 'sig' or 'tanh'
```

### 29. Write-up (bonus)
Write a short blog post or note explaining what a neuron, a cost function and backpropagation are, as if to a friend who codes but hasn't done ML. Include one diagram you drew yourself.

---

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 01-supervised-foundations/01-classification
```
Add `-m "not slow"` to skip the one test that trains on real MNIST data.
