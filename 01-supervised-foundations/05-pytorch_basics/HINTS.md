# Hints: PyTorch Basics

## 0. Tensors
<details><summary>Hint 1</summary>
PyTorch has functions that build a tensor from a NumPy array. Some share memory with the array, some copy; either is fine here.
</details>
<details><summary>Hint 2</summary>
Most tensor-creation functions accept `dtype` and `device` arguments directly, so you don't need a separate conversion step. `torch.float32` and `torch.int64` are the dtypes you want.
</details>

## 1. Autograd
<details><summary>Hint 1</summary>
Autograd only tracks tensors that were created with gradient tracking switched on. Make `x` into such a tensor (a 0-dimensional one is fine).
</details>
<details><summary>Hint 2</summary>
Call `f` on that tensor, then ask the result to run the backward pass. Afterwards the input tensor's `.grad` holds the derivative.
</details>
<details><summary>Hint 3</summary>
`.item()` turns a one-element tensor into a plain Python number.
</details>

## 2. Create a layer
<details><summary>Hint 1</summary>
A dictionary from activation names to activation classes in `torch.nn` keeps this tidy and makes the "unknown name" check easy.
</details>
<details><summary>Hint 2</summary>
`torch.nn.init` has in-place initializers (names ending in `_`) for Xavier uniform and for zeros. Apply them to the linear layer's `.weight` and `.bias`.
</details>
<details><summary>Hint 3</summary>
With an activation you need two modules in a row; without one, the linear layer alone is enough.
</details>

## 3. Forward propagation
<details><summary>Hint 1</summary>
Always call the parent constructor first in an `nn.Module` subclass, before assigning any layers.
</details>
<details><summary>Hint 2</summary>
A plain Python list of layers is invisible to PyTorch (its parameters won't be found by the optimizer or saved in the `state_dict`). `nn.ModuleList` or `nn.Sequential` registers them properly.
</details>
<details><summary>Hint 3</summary>
Each layer's input size is the previous layer's output size, starting from `nx`.
</details>

## 4. Accuracy
<details><summary>Hint 1</summary>
The predicted class is the index of the largest score in each row. Which dimension is that?
</details>
<details><summary>Hint 2</summary>
Comparing two tensors gives a boolean tensor. Booleans can't be averaged directly; convert to float first.
</details>

## 5. Loss
<details><summary>Hint 1</summary>
`torch.nn.functional` has a cross-entropy function that takes raw logits and class indices. Check its argument order in the docs: it's not the same as this function's.
</details>

## 6. Train op
<details><summary>Hint 1</summary>
`model.parameters()` gives the optimizer everything it should update.
</details>
<details><summary>Hint 2</summary>
The five steps of a training step, in order: zero the gradients, forward pass, compute the loss, backward pass, optimizer step. Forgetting the first one makes gradients pile up across steps.
</details>

## 7. Train
<details><summary>Hint 1</summary>
Move both the model and the data to the same device. A model on MPS can't read a tensor on the CPU.
</details>
<details><summary>Hint 2</summary>
The loop structure is the same as your NumPy upgraded `train`: run `iterations + 1` times, report on the right iterations, and only do an update when it isn't the last pass.
</details>
<details><summary>Hint 3</summary>
For the reported metrics, switch the model to eval mode and wrap the forward passes in the context manager that disables gradient tracking. A model's `state_dict()` is a dict of tensors; move each one to the CPU before saving.
</details>

## 8. Evaluate
<details><summary>Hint 1</summary>
`torch.load` takes a `map_location` argument so a checkpoint saved from a GPU can load on a CPU.
</details>
<details><summary>Hint 2</summary>
Rebuild the same architecture from the saved settings, then load the saved parameters into it with the module method made for that.
</details>
