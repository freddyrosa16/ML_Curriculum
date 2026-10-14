# Hints: GPT from scratch

Open one hint at a time. Give each one a real try before opening the next.

## 0. Character tokenizer
<details><summary>Hint 1</summary>
The vocabulary needs a fixed order, or the ids would change between runs. Which built-in turns a collection of characters into a fixed order?
</details>
<details><summary>Hint 2</summary>
Build both lookup dicts from the same enumeration, so that id → char and char → id are exact inverses.
</details>
<details><summary>Hint 3</summary>
A tensor's elements aren't plain Python ints. Convert each id before looking it up. A dict lookup on a missing key already raises the error the task asks for.
</details>

## 1. BPE building blocks
<details><summary>Hint 1</summary>
Adjacent pairs are just each element together with the element after it. Python has a neat way to walk two "views" of the same list side by side.
</details>
<details><summary>Hint 2</summary>
For merging, walk the list with an index you move yourself. You need to jump 2 positions on a match and 1 otherwise, which is why a plain for-loop over elements is awkward here.
</details>
<details><summary>Hint 3</summary>
Watch the last position: only check for a pair when there *is* a next element.
</details>

## 2. BPE tokenizer
<details><summary>Hint 1</summary>
Training is a loop: count pairs → choose the most common → merge it → record the merge and the new token's bytes. Each new token's bytes are just its two parents' bytes stuck together.
</details>
<details><summary>Hint 2</summary>
Encoding must replay merges in the order they were learned, because later merges are built from earlier ones. Among the pairs in your current sequence, which one has the smallest merge id?
</details>
<details><summary>Hint 3</summary>
Stop encoding when none of the current pairs is in `merges`. For decoding, think "bytes first, text last", and look at the `errors=` argument of `bytes.decode`.
</details>

## 3. Batches
<details><summary>Hint 1</summary>
You need `batch_size` random start positions. What's the largest start that still leaves room for `block_size + 1` tokens?
</details>
<details><summary>Hint 2</summary>
`torch.randint` accepts a `generator` argument. Slicing gives one window per start, and `torch.stack` turns a list of windows into a batch.
</details>
<details><summary>Hint 3</summary>
`y`'s window is the same as `x`'s, but starts one position later.
</details>

## 4. Bigram
<details><summary>Hint 1</summary>
An `nn.Embedding(V, V)` *is* a table of next-token logits: row i holds the scores for what follows token i.
</details>
<details><summary>Hint 2</summary>
`F.cross_entropy` expects logits shaped (N, C) and targets shaped (N,). Flatten batch and time together.
</details>
<details><summary>Hint 3</summary>
For generation, only the last time step's logits matter. Softmax them, draw one sample with `torch.multinomial`, and concatenate along the time dimension.
</details>

## 5. Causal self-attention
<details><summary>Hint 1</summary>
Reuse what you built in 03-sequences-nlp/05-attention. The new part is the mask: a lower-triangular matrix of ones (`torch.tril`) shows which positions each position may see.
</details>
<details><summary>Hint 2</summary>
Register the mask with `register_buffer` so it moves to the right device with the model and isn't treated as a parameter. Slice it to the current T in forward.
</details>
<details><summary>Hint 3</summary>
For multiple heads, reshape from (B, T, C) to (B, heads, T, head_size) so one batched matrix multiply handles every head at once. Scale the scores by 1/√head_size, mask with −∞, softmax over the last dimension, then reverse the reshape.
</details>

## 6. GPT
<details><summary>Hint 1</summary>
Position embeddings take the positions 0..T−1 as input. Create that range on the same device as `idx`.
</details>
<details><summary>Hint 2</summary>
`nn.Module.apply` visits every submodule, which is an easy way to run your weight initialization once over the whole model.
</details>
<details><summary>Hint 3</summary>
For top-k, find the k-th largest logit in each row (`torch.topk`) and set everything smaller to −∞ before the softmax. Remember to crop the context to the last `block_size` tokens on every step.
</details>

## 7. Training
<details><summary>Hint 1</summary>
The training step is the same five beats as always: get a batch → forward → zero the gradients → backward → optimizer step.
</details>
<details><summary>Hint 2</summary>
`model.training` tells you which mode the model was in before. Save it, switch to eval, then switch back. The `torch.no_grad` decorator keeps evaluation cheap.
</details>
<details><summary>Hint 3</summary>
Write a small inner function that evaluates and appends to the history. Call it at the top of the loop when `step % eval_interval == 0`, and once more after the loop with `max_iters`.
</details>
