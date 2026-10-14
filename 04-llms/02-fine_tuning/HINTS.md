# Hints: Fine-tuning with LoRA

Open one hint at a time.

## 0. Load a model
<details><summary>Hint 1</summary>
Hugging Face has "Auto" classes that read a model's config and pick the right architecture for you: one for tokenizers, and one for causal language models.
</details>
<details><summary>Hint 2</summary>
Tokenizers expose `pad_token` and `eos_token` as attributes, and you can assign to them.
</details>
<details><summary>Hint 3</summary>
For counting, every parameter tensor has `numel()` and `requires_grad`. Decide when to include each tensor based on the `trainable_only` flag.
</details>

## 1. Prompt template
<details><summary>Hint 1</summary>
Build the prompt piece by piece: instruction section, then (maybe) input section, then the response header.
</details>
<details><summary>Hint 2</summary>
`dict.get` with a default handles a missing key. Something like `None or ""` handles a `None` value.
</details>
<details><summary>Hint 3</summary>
Compare against the exact strings in the README examples, character by character. The newlines are where people slip up.
</details>

## 2. Tokenize with loss masking
<details><summary>Hint 1</summary>
Calling a tokenizer on a string returns a dict-like object with `input_ids`. There's a keyword argument that stops it from adding BOS/EOS tokens on its own.
</details>
<details><summary>Hint 2</summary>
Build the labels as "one −100 per prompt token" followed by the response ids. Truncate everything at the very end, so all three lists stay the same length.
</details>
<details><summary>Hint 3</summary>
For collate, find the longest sequence first. Each list then needs `longest − its length` extra entries, and each field has its own pad value.
</details>

## 3. LoRA by hand
<details><summary>Hint 1</summary>
Wrap the tensors in `nn.Parameter` so they register as trainable. To freeze the base layer, switch off `requires_grad` on each of its parameters.
</details>
<details><summary>Hint 2</summary>
Check the shapes. For x of shape (N, in), you need to go in → r → out. Which transposes of A and B make that chain line up?
</details>
<details><summary>Hint 3</summary>
For `merge`, create a new `nn.Linear` of the same size and copy the merged weight and the bias into it inside `torch.no_grad()`. Don't touch `base.weight` in place.
</details>

## 4. peft
<details><summary>Hint 1</summary>
peft has two pieces: a config object that describes the adapter, and a function that wraps your model using that config.
</details>
<details><summary>Hint 2</summary>
The config's `target_modules` are matched against module *names*. Print `model` to see names like `q_proj`. Set the task type for causal language modeling.
</details>
<details><summary>Hint 3</summary>
The fraction is just two sums over `model.parameters()`: one with a filter and one without.
</details>

## 5. Training loop
<details><summary>Hint 1</summary>
Give the optimizer only the parameters that require gradients. Otherwise it wastes memory on frozen weights.
</details>
<details><summary>Hint 2</summary>
`random.Random(seed).sample` picks a batch without repeats. Every tensor in the collated dict has to move to the device.
</details>
<details><summary>Hint 3</summary>
Hugging Face models compute the loss for you when you pass `labels`. Unpack the batch dict straight into the model call and read `.loss` from the output.
</details>

## 6. Generate, save, load
<details><summary>Hint 1</summary>
Models have a `generate` method. Tokenize the prompt into PyTorch tensors on the model's device first, and pass `pad_token_id` to avoid a warning.
</details>
<details><summary>Hint 2</summary>
`generate` returns the prompt *plus* the new tokens. You know how many prompt tokens there were, so slice them off before decoding.
</details>
<details><summary>Hint 3</summary>
A peft model already knows how to save just its adapter (the same method name as for full models). Loading goes through a class method on `PeftModel` that takes the base model and the folder.
</details>
