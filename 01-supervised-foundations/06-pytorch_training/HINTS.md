# Hints: PyTorch Training

## 0. Sequential
<details><summary>Hint 1</summary>
Build a Python list of modules in a loop, then hand them all to `nn.Sequential` (unpacking the list into arguments).
</details>
<details><summary>Hint 2</summary>
`nn.Dropout` wants the probability of *dropping*. Use the loop index to skip dropout after the last layer.
</details>
<details><summary>Hint 3</summary>
Any `nn.Module` lets you set a new attribute on it after creation, so you can attach the `config` dict before returning.
</details>

## 1. Subclassing
<details><summary>Hint 1</summary>
Register layers with `nn.ModuleList` so PyTorch can find their parameters. The order you create the linear layers in decides the order of `parameters()`.
</details>
<details><summary>Hint 2</summary>
`nn.Identity` is a do-nothing module, handy when a layer has no activation. One dropout module can be reused at several places in `forward`, since it has no parameters.
</details>

## 2. Optimize
<details><summary>Hint 1</summary>
Look at the constructor arguments of `torch.optim.Adam`: one takes the two betas together, another is the L2 penalty.
</details>
<details><summary>Hint 2</summary>
The loss is a module in `torch.nn`, created with no arguments.
</details>

## 3. One-hot
<details><summary>Hint 1</summary>
`torch.nn.functional` has a one-hot function. It needs an integer (int64) tensor, so convert the input first.
</details>
<details><summary>Hint 2</summary>
Check what that function does when you don't pass the number of classes, and what dtype it returns.
</details>

## 4. Train
<details><summary>Hint 1</summary>
Wrap the two tensors in a `TensorDataset`, and that in a `DataLoader` with the batch size and shuffle flag. Looping over the loader yields `(x_batch, y_batch)` pairs.
</details>
<details><summary>Hint 2</summary>
Put the model in training mode before the epoch. For each batch: move it to the model's device, then do the same five-step training step as in pytorch_basics.
</details>
<details><summary>Hint 3</summary>
For the epoch average, keep running totals of (batch loss × batch size), correct predictions and examples seen. Divide at the end of the epoch. Use `.item()` so you store numbers, not tensors that keep the whole graph alive.
</details>

## 5. Validate
<details><summary>Hint 1</summary>
Validation is the same batch loop without the backward pass and optimizer step, in eval mode and inside the no-gradient context. A helper function that runs one epoch either way saves a lot of repetition.
</details>

## 6. Early stopping
<details><summary>Hint 1</summary>
You need two pieces of state that live across epochs: the best validation loss so far (start it at infinity) and a counter of epochs without improvement.
</details>
<details><summary>Hint 2</summary>
Do the check at the end of an epoch, after recording that epoch's metrics, so the history includes the epoch you stopped on.
</details>

## 7. Learning rate decay
<details><summary>Hint 1</summary>
The scheduler that accepts any formula multiplies the optimizer's *starting* learning rate. So set the optimizer's learning rate to `alpha` first (it's stored in each of `optimizer.param_groups`), then create the scheduler.
</details>
<details><summary>Hint 2</summary>
Record the learning rate at the start of the epoch, and step the scheduler at the end.
</details>

## 8. Save the best
<details><summary>Hint 1</summary>
You already detect "new best validation loss" for early stopping. That's exactly the moment to save.
</details>

## 9. Save and load a model
<details><summary>Hint 1</summary>
`torch.save` will happily pickle a whole module. On loading, read the docs for `torch.load`'s `weights_only` argument and its default.
</details>

## 10. Save and load weights
<details><summary>Hint 1</summary>
Save the result of the model's `state_dict()` method. To load, read the dict back and pass it to the model's method for loading a state dict.
</details>

## 11. Save and load a config
<details><summary>Hint 1</summary>
The standard library `json` module can write a dict to an open file and read it back. JSON turns Python `None` into `null` and back again, so the activations survive the round trip.
</details>

## 12. Test
<details><summary>Hint 1</summary>
One forward pass over all the data, in eval mode with gradients off. Loss from the functional cross-entropy, accuracy from comparing the arg-max to the labels.
</details>

## 13. Predict
<details><summary>Hint 1</summary>
The model outputs logits. Softmax along the class dimension turns them into probabilities. Move the result to the CPU before returning.
</details>
