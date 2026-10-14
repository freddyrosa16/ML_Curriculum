# GPT from scratch

**Status:** New

## Why this matters
In `03-sequences-nlp/05-attention` you built a full Transformer. A GPT is a cut-down version of it: only the decoder stack, trained to guess the next token. Every chatbot you've used starts out this way. Building a tiny one yourself (tokenizer included) takes the mystery out of LLMs: an LLM is a next-token classifier run in a loop.

## Concepts to understand first
- **What is a token, and why not just use characters or words?** A token is the unit the model reads and predicts. Characters give a tiny vocabulary but long sequences. Whole words give short sequences but a huge vocabulary, plus a problem with words the model has never seen. Byte-pair encoding (BPE) sits in between.
- **How does byte-pair encoding work?** Start from raw bytes (256 possible tokens). Find the most frequent adjacent pair, give it a new id, and replace it everywhere. Repeat until the vocabulary is the size you want. Encoding new text replays the learned merges in the order they were learned.
- **What is language modeling?** Given tokens $x_1..x_t$, predict a probability distribution over $x_{t+1}$. Training minimizes cross-entropy between that distribution and the real next token. A model that guesses uniformly over $V$ tokens has loss $\ln V$, which makes a handy sanity check.
- **What is a bigram model?** The simplest language model: the next token depends only on the current one. It's a $V \times V$ table of logits, and it's the baseline your GPT has to beat.
- **What makes self-attention "causal"?** Position $t$ may only look at positions $\le t$. Otherwise the model could peek at the answer during training. You enforce this by setting the attention scores for future positions to $-\infty$ before the softmax.
- **What is a decoder-only Transformer block?** LayerNorm → causal multi-head attention → residual add, then LayerNorm → MLP (expand 4×, GELU, project back) → residual add. GPT stacks these blocks on token + position embeddings and adds a final linear layer that produces vocabulary logits.
- **How does generation work, and what do temperature and top-k do?** Feed the context in, take the logits at the last position, sample a token, append it, repeat. Dividing the logits by a temperature below 1 makes sampling more confident (and above 1, more random). Top-k keeps only the k most likely tokens before sampling. With top-k = 1 you always pick the most likely token, which is greedy decoding.

## Resources
- [Andrej Karpathy: Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY): this project follows the same path. Watch it, then build your own version without looking.
- [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) and [minbpe](https://github.com/karpathy/minbpe): BPE explained step by step.
- [nanoGPT](https://github.com/karpathy/nanoGPT): a compact, readable reference GPT. Read it *after* you finish.
- [Jay Alammar: The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Hugging Face LLM Course: Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

## Rules
- Tasks 0–2: plain Python, no imports needed.
- Tasks 3–7: `import torch`, `torch.nn as nn` and `torch.nn.functional as F` (plus `math` and `matplotlib` where noted). No Hugging Face, and no `nn.MultiheadAttention` or `nn.TransformerDecoder`: you write attention yourself.
- Files may import earlier tasks with `__import__("1-bpe")` and similar.
- Code must work on CPU, and run on MPS/CUDA when they're available.

## Data
Tiny Shakespeare: about 1 MB of plain text (40k lines) from Karpathy's char-rnn repo.
Download it from `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` (with your browser, `curl -o` or `urllib.request.urlretrieve`) and save it as `data/input.txt` inside this project folder. The `data/` folder is yours to create. The slow test reads it from there, or downloads a temporary copy if it's missing.

## Tasks

### 0. Character tokenizer
File: `0-char_tokenizer.py`. Class `CharTokenizer(text)`.
- The vocabulary is the sorted set of distinct characters in `text`.
- Attributes: `stoi` (dict char → id), `itos` (dict id → char), `vocab_size` (int).
- `encode(s)` → list of ints. An unknown character raises `KeyError`.
- `decode(ids)` → string. Must accept a list of ints or a 1-D tensor.

```
>>> tok = CharTokenizer("hello world")
>>> tok.vocab_size
8
>>> tok.encode("hello")
[3, 2, 4, 4, 5]
>>> tok.decode([3, 2, 4, 4, 5])
'hello'
```

### 1. BPE building blocks
File: `1-bpe.py`.
- `get_stats(ids)` → dict mapping each adjacent pair `(a, b)` to how many times it appears.
- `merge(ids, pair, idx)` → new list where every occurrence of `pair` is replaced by `idx`, scanning left to right with no overlaps.

```
>>> get_stats([1, 2, 3, 1, 2])
{(1, 2): 2, (2, 3): 1, (3, 1): 1}
>>> merge([1, 2, 3, 1, 2], (1, 2), 99)
[99, 3, 99]
>>> merge([7, 7, 7], (7, 7), 256)
[256, 7]
```

### 2. BPE tokenizer
File: `2-bpe_tokenizer.py`. Class `BPETokenizer()`, byte-level like GPT-2's (minus the regex pre-splitting).
- `merges`: dict `(a, b)` → new id. Starts empty.
- `vocab`: dict id → `bytes`. Starts as the 256 single bytes.
- `train(text, vocab_size)`: works on the UTF-8 bytes of `text`. Learns `vocab_size - 256` merges, each one the most frequent pair at that point; new ids are 256, 257, … in order. Raises `ValueError` if `vocab_size < 256`. Stops early if no pairs are left.
- `encode(text)` → list of ids. Applies the learned merges to new text, always choosing the *earliest-learned* merge available.
- `decode(ids)` → string. Joins the byte strings and decodes UTF-8, replacing invalid bytes rather than crashing.

```
>>> tok = BPETokenizer()
>>> tok.encode("hi")
[104, 105]
>>> tok.train("aaabdaaabac", 257)
>>> tok.merges
{(97, 97): 256}
>>> tok.vocab[256]
b'aa'
>>> tok.decode(tok.encode("café ☕"))
'café ☕'
```

### 3. Batches for next-token prediction
File: `3-data.py`. `data` is a 1-D `torch.long` tensor of token ids.
- `train_val_split(data, frac=0.9)` → `(train, val)`: the first `int(frac * len(data))` tokens and the rest. No shuffling, because order matters in text.
- `get_batch(data, block_size, batch_size, generator=None, device="cpu")` → `(x, y)`, each of shape `(batch_size, block_size)`. Each row is a random window of `data`, and `y` is the same window shifted one token to the right. Draw the random start positions with `generator` so results are reproducible.

```
>>> x, y = get_batch(torch.arange(50), block_size=8, batch_size=4, generator=torch.Generator().manual_seed(0))
>>> x.shape, y.shape
(torch.Size([4, 8]), torch.Size([4, 8]))
>>> bool((y == x + 1).all())
True
```

### 4. Bigram baseline
File: `4-bigram.py`. Class `BigramLM(vocab_size)`, an `nn.Module`.
- `forward(idx, targets=None)` → `(logits, loss)`. `idx` has shape `(B, T)` and `logits` has shape `(B, T, vocab_size)`. `loss` is the mean cross-entropy against `targets`, or `None` when no targets are given.
- `generate(idx, max_new_tokens)` → `(B, T + max_new_tokens)`: samples one token at a time from the softmax of the last position. No gradients needed.

```
>>> m = BigramLM(10)
>>> logits, loss = m(torch.zeros((2, 5), dtype=torch.long))
>>> logits.shape, loss
(torch.Size([2, 5, 10]), None)
>>> m.generate(torch.zeros((1, 1), dtype=torch.long), 7).shape
torch.Size([1, 8])
```
Train it on Tiny Shakespeare for a few thousand steps and write down the validation loss (around 2.5 for characters). That's the number to beat.

### 5. Causal self-attention
File: `5-attention.py`. Class `CausalSelfAttention(n_embd, n_head, block_size, dropout=0.0)`.
- Multi-head scaled dot-product attention with a causal mask, followed by an output projection (and dropout).
- Raises `ValueError` if `n_embd` is not divisible by `n_head`.
- `forward(x)` takes `x` of shape `(B, T, n_embd)` with `T ≤ block_size` and returns the same shape.
- Changing tokens at positions ≥ t must not change the output at positions < t.

```
>>> att = CausalSelfAttention(n_embd=16, n_head=4, block_size=8)
>>> att(torch.randn(2, 6, 16)).shape
torch.Size([2, 6, 16])
```

### 6. GPT
File: `6-gpt.py`.
- `Block(n_embd, n_head, block_size, dropout=0.0)`: a pre-norm Transformer block (see Concepts). Input and output shape are `(B, T, n_embd)`.
- `GPT(vocab_size, block_size, n_embd=64, n_head=4, n_layer=4, dropout=0.0)`:
  - token embedding + learned position embedding → `n_layer` blocks → final LayerNorm → linear layer to `vocab_size`.
  - Initialize Linear/Embedding weights from a normal distribution with std 0.02 and biases at zero, so a fresh model's loss is close to $\ln V$.
  - `forward(idx, targets=None)` → `(logits, loss)`, like the bigram model. Raises `ValueError` if `T > block_size`.
  - `generate(idx, max_new_tokens, temperature=1.0, top_k=None)` → `(B, T + max_new_tokens)`. Only feed the last `block_size` tokens to the model. Divide the last logits by `temperature`, keep only the `top_k` largest if `top_k` is set, then sample.

```
>>> m = GPT(vocab_size=20, block_size=8, n_embd=32, n_head=4, n_layer=2)
>>> idx = torch.randint(0, 20, (4, 8))
>>> logits, loss = m(idx, idx)
>>> logits.shape, round(loss.item(), 1)      # ln(20) ≈ 3.0
(torch.Size([4, 8, 20]), 3.0)
>>> m.generate(torch.zeros((2, 1), dtype=torch.long), 20).shape
torch.Size([2, 21])
```

### 7. Train it and plot the loss
File: `7-train.py`. May also import `matplotlib`.
- `get_device()` → `"cuda"`, `"mps"` or `"cpu"`, in that order of preference.
- `estimate_loss(model, splits, block_size, batch_size, eval_iters=20, generator=None, device="cpu")`: `splits` is a dict like `{"train": tensor, "val": tensor}`. Returns a dict with the same keys, each holding the mean loss over `eval_iters` random batches, as a Python float. No gradients. Put the model in eval mode while measuring, then restore whatever mode it was in before.
- `train(model, train_data, val_data, block_size, batch_size=32, max_iters=1000, lr=1e-3, eval_interval=100, eval_iters=20, device=None, seed=0)` → `history`, a dict with keys `"step"`, `"train"` and `"val"` (lists of the same length). Use AdamW. Evaluate at step 0, every `eval_interval` steps, and once more after the last step (recorded as step `max_iters`). `device=None` means use `get_device()`. Seed a `torch.Generator` with `seed` for batch sampling.
- `plot_history(history, path)`: saves a train/val loss plot to `path`. Use the non-interactive `Agg` backend.

```
>>> hist = train(model, train_data, val_data, block_size=16, max_iters=200, eval_interval=50, device="cpu")
>>> hist["step"]
[0, 50, 100, 150, 200]
>>> plot_history(hist, "loss.png")
```

**Putting it together (not tested, but the reason you're here):**
1. Char-level: train a GPT with `block_size=128, n_embd=128, n_head=4, n_layer=4, dropout=0.1` for about 3000 steps on Tiny Shakespeare. It takes a few minutes on an M-series Mac with `mps`. Validation loss should drop well below your bigram baseline, to around 1.6. Generate 500 characters at a couple of temperatures and compare.
2. BPE: train your `BPETokenizer` with `vocab_size=512` on the text and train the same GPT on BPE ids. Compare the samples. (Losses aren't directly comparable across tokenizers. Why not?)
3. Save the loss curves with `plot_history`.

### 8. Write-up (bonus)
Write a short post: what surprised you, your loss curves, and a sample of generated "Shakespeare" at temperature 0.8 and 1.2.

## Check your work
```bash
pytest 04-llms/01-gpt_from_scratch -m "not slow"   # fast checks
pytest 04-llms/01-gpt_from_scratch                 # includes a short Shakespeare training run
```
