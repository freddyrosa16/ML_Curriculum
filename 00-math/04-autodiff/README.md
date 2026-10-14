# Autodiff: build a tiny autograd engine

**Status:** New

## Why this matters
Every deep learning library (PyTorch, JAX, TensorFlow) is built around one idea: record the math as you do it, then walk the record backwards with the chain rule to get every gradient automatically. That's *backpropagation*. Here you build it yourself for single numbers, in about 100 lines of plain Python, then train a small neural network with it. After this, `loss.backward()` in PyTorch will never feel like magic again.

This project follows the ideas in Andrej Karpathy's micrograd lecture. Watch it first (or alongside), then write your own version without copying his code.

## Concepts to understand first
- **What is a computational graph?** Any formula can be drawn as a graph: the inputs are leaves, each operation (+, *, tanh...) is a node, and the final output is the root. For $L = (a \cdot b + c) \cdot f$, the multiply node feeds an add node, which feeds another multiply.
- **What is the chain rule, in graph terms?** If $L$ depends on $y$, and $y$ depends on $x$, then $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$. In the graph: the gradient arriving at a node, times that node's *local* derivative, is what gets passed back to its inputs.
- **What are the local derivatives of the basic operations?** For $y = a + b$: both inputs get the incoming gradient unchanged. For $y = a \cdot b$: $a$ gets the incoming gradient times $b$, and $b$ gets it times $a$. For $y = \tanh(x)$: local derivative $1 - y^2$. For $y = e^x$: local derivative $e^x$, which is $y$ itself.
- **Why do gradients accumulate (+=) instead of being overwritten?** If a value is used in two places, a small change in it affects the output through *both* paths, so the contributions add. Overwriting is the classic micrograd bug.
- **What is a topological sort, and why do we need one?** An ordering of the nodes where every node comes after all the nodes it depends on. Walking that order *in reverse* guarantees a node has received all of its gradient before it passes gradient on to its inputs.
- **What is a numerical gradient (finite differences)?** Nudge an input by a tiny $h$ and see how the output moves: $\frac{f(x+h) - f(x-h)}{2h}$. It's slow but simple, so it's the standard way to check that a backprop implementation is right. The tests do exactly this.
- **What is a neuron?** It computes a weighted sum of its inputs plus a bias, then squashes the result with a nonlinearity: $\tanh(w \cdot x + b)$. A layer is several neurons that see the same inputs; an MLP (multi-layer perceptron) is layers stacked so each one's outputs are the next one's inputs.
- **What is gradient descent?** Compute the loss, backprop to get each parameter's gradient, then move every parameter a small step *against* its gradient (`step size = learning rate`). Repeat. Don't forget to reset the gradients to zero before each backward pass, or they pile up from previous steps.

## Resources
- [Andrej Karpathy: The spelled-out intro to neural networks and backpropagation: building micrograd (video)](https://www.youtube.com/watch?v=VMj-3S1tku0) (the main resource for this project)
- [karpathy/micrograd on GitHub](https://github.com/karpathy/micrograd) (look at it *after* you finish, to compare)
- [3Blue1Brown: Backpropagation calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus)
- [CS231n: Backpropagation, intuitions](https://cs231n.github.io/optimization-2/)
- [Chris Olah: Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)
- [Dive into Deep Learning: Automatic Differentiation](https://d2l.ai/chapter_preliminaries/autograd.html)

## Rules
- Plain Python only. The only imports allowed are `math` (in `engine.py`) and `random` (in `nn.py`). No NumPy, no PyTorch.
- `nn.py` gets the `Value` class by importing it from your `engine` module.
- Write it yourself. It's fine to pause the video and try each piece before watching how Karpathy does it.

## Tasks

### 0. The Value object and the forward pass
**File:** `engine.py` · **Class:** `Value(data)`

A `Value` wraps one number.
- Attributes: `data` (stored as a `float`) and `grad` (starts at `0.0`).
- A readable `repr` that shows the data, e.g. `Value(data=2.0, grad=0.0)`.
- Operators that return new `Value`s: `+`, `*`, `-` (both binary and unary), `/`, and `**` with a plain int or float exponent.
- Plain numbers must work on **either side**: `a + 1`, `1 + a`, `3 * a`, `5 - a`, `4 / a`.
- Methods `tanh()`, `exp()`, `relu()` (relu is $\max(0, x)$).
- Each new `Value` should remember which `Value`s it was made from (you'll need this for task 1).
```
a = Value(2.0); b = Value(-3.0)
(a * b + 1).data      ->  -5.0
(4 / a).data          ->  2.0
Value(-2.0).relu()    ->  Value(data=0.0, grad=0.0)
```

### 1. backward() for + and *
**File:** `engine.py` · **Method:** `Value.backward()`

Calling `out.backward()` fills in `.grad` on every `Value` that `out` depends on, with $\frac{\partial\, \text{out}}{\partial\, \text{that value}}$.
- `out.grad` itself becomes `1.0`.
- Visit the nodes in reverse topological order and apply each node's local chain rule.
- Get it working for `+` and `*` first.
```
a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
L = (a * b + c) * f
L.backward()
a.grad, b.grad, c.grad, f.grad  ->  6.0, -4.0, -2.0, 4.0
```

### 2. Using a value more than once
**File:** `engine.py`

Make sure a `Value` used several times gets the **sum** of the gradients from every place it's used.
```
a = Value(3.0); (a + a).backward(); a.grad  ->  2.0
a = Value(3.0); (a * a).backward(); a.grad  ->  6.0
```

### 3. Gradients for every operation
**File:** `engine.py`

Give `**`, `tanh`, `exp` and `relu` their backward rules. If you built `-` and `/` out of `+`, `*` and `**` (a neat trick), they get gradients for free. The tests compare your gradients with finite differences on composite expressions like $\tanh(xy + x^2) \cdot 3 - y/x$ and the sigmoid $1 / (1 + e^{-x})$.
```
x = Value(0.3); y = x.tanh(); y.backward(); x.grad  ->  0.9151...   (= 1 - tanh(0.3)^2)
```

### 4. Neuron, Layer, MLP
**File:** `nn.py` · **Classes:** `Neuron(nin)`, `Layer(nin, nout)`, `MLP(nin, nouts)`

- `Neuron(nin)`: `nin` weights and one bias, each a `Value` initialised with `random.uniform(-1, 1)`. Calling it on a list of numbers returns $\tanh(w \cdot x + b)$ as a `Value`.
- `Layer(nin, nout)`: `nout` neurons. Calling it returns a list of `nout` Values, or a single `Value` when `nout == 1`.
- `MLP(nin, nouts)`: e.g. `MLP(3, [4, 4, 1])` is 3 inputs, two hidden layers of 4 and one output. Calling it runs the layers in order.
- All three have `parameters()` (a flat list of every weight and bias `Value`) and `zero_grad()` (sets every parameter's `grad` to 0). A small shared base class keeps this tidy.
```
random.seed(0)
model = MLP(3, [4, 4, 1])
len(model.parameters())        ->  41
model([2.0, 3.0, -1.0])        ->  Value(data=<a number between -1 and 1>, grad=0.0)
```

### 5. Train it
**File:** `nn.py` · **Signature:** `train(model, xs, ys, steps=100, lr=0.05)`

Run plain gradient descent:
1. forward: predict every example in `xs`
2. loss: the **sum of squared errors** between predictions and `ys`
3. zero the gradients, then `loss.backward()`
4. update: move every parameter by `-lr * grad`

Return the list of loss values (plain floats), one per step.

Toy dataset (from the micrograd lecture): inputs `[2, 3, -1]`, `[3, -1, 0.5]`, `[0.5, 1, 1]`, `[1, 1, -1]` with targets `1, -1, -1, 1`.
```
random.seed(0)
model = MLP(3, [4, 4, 1])
losses = train(model, xs, ys, steps=100, lr=0.05)
losses[0], losses[-1]   ->  about 7.7, then below 0.05
```

### 6. Write-up (optional)
In a short note or blog post, explain backprop to a friend using a hand-drawn graph of $L = (a \cdot b + c) \cdot f$, showing the gradient flowing backwards through each node. Bonus: add a `sigmoid()` method and train the same network on a tiny 2D classification problem of your own.

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 00-math/04-autodiff
```
