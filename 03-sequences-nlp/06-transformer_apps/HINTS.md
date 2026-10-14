# Hints: Transformer Applications

## 0. Dataset
<details><summary>Hint 1</summary>
Load the dataset once in a Python shell and look at one row before writing the class. Where are the Portuguese and English strings?
</details>
<details><summary>Hint 2</summary>
`load_dataset` takes the dataset id, the config name and a `split`. Split strings support slicing. Turn the rows into plain `(pt, en)` tuples so the rest of your code doesn't care where they came from.
</details>
<details><summary>Hint 3</summary>
A fast tokenizer loaded with `AutoTokenizer.from_pretrained` has a method that trains a new tokenizer from an iterator of strings. Give it all the Portuguese sentences for one and all the English sentences for the other.
</details>

## 1. Encode tokens
<details><summary>Hint 1</summary>
The tokenizer's `encode` method normally adds its own [CLS] / [SEP] tokens. Find the argument that turns that off.
</details>
<details><summary>Hint 2</summary>
The start and end ids aren't in the tokenizer's vocabulary. They're the next two numbers after its `vocab_size`, one pair per language.
</details>

## 2. Encode and pad
<details><summary>Hint 1</summary>
Encoding the dataset is one pass over the pairs with your `encode` method.
</details>
<details><summary>Hint 2</summary>
PyTorch has a utility in `torch.nn.utils.rnn` that pads a list of 1-D tensors of different lengths into one 2-D tensor. Check its `batch_first` and `padding_value` arguments.
</details>

## 3. Pipeline
<details><summary>Hint 1</summary>
A `DataLoader` can wrap a plain Python list. Each element is one example.
</details>
<details><summary>Hint 2</summary>
By default the `DataLoader` tries to stack examples, which fails for different lengths. The `collate_fn` argument lets you decide how a list of examples becomes a batch.
</details>

## 4. Create masks
<details><summary>Hint 1</summary>
A padding mask is just "is this token 0?", converted to float, with two size-1 axes inserted so it broadcasts over heads and query positions.
</details>
<details><summary>Hint 2</summary>
The look-ahead mask is the upper triangle of a square of ones, not including the diagonal. Look up `torch.triu` and its `diagonal` argument.
</details>
<details><summary>Hint 3</summary>
`torch.maximum` broadcasts: a `(seq, seq)` look-ahead mask and a `(batch, 1, 1, seq)` padding mask combine into `(batch, 1, seq, seq)`. Use the input tensor's `.device` when creating new tensors.
</details>

## 5. Train
<details><summary>Hint 1</summary>
Get `5-transformer.py` working first: from a Python shell in this folder, importing it should give you your class from 05.
</details>
<details><summary>Hint 2</summary>
`F.cross_entropy` expects `(N, classes)` logits and `(N,)` labels, so flatten the batch and sequence axes together. It has an argument for an index to ignore, which is your padding id.
</details>
<details><summary>Hint 3</summary>
Teacher forcing is two slices of the target: everything but the last token goes into the decoder, everything but the first token is what it should predict. Build the masks from the decoder *input*, not the full target. Call the scheduler's `step` once per batch, after the optimizer's.
</details>

## 6. Translate
<details><summary>Hint 1</summary>
Wrap everything in `torch.no_grad()` and put the model in eval mode.
</details>
<details><summary>Hint 2</summary>
Each loop iteration runs the whole model on the decoder input so far and only looks at the scores for the *last* position.
</details>
<details><summary>Hint 3</summary>
Before decoding back to text, drop your start/end ids: they're outside the tokenizer's vocabulary.
</details>
