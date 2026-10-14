# Attention and the Transformer

**Status:** Modernized. Same tasks as the original, rebuilt in PyTorch (`nn.Module`) instead of TensorFlow/Keras. Positional encoding stays NumPy.

## Why this matters
"Attention Is All You Need" (2017) introduced the Transformer, and every modern language model (GPT, Claude, Llama, BERT) is built from the blocks you'll write here. You'll first add attention to an RNN translator (the 2014 idea that started it all), then build a full Transformer piece by piece. When you're done, you'll be able to read an LLM's architecture diagram and know what every box does.

## Concepts to understand first
- **What problem did attention solve?** A plain seq2seq RNN squeezes a whole input sentence into one fixed hidden vector, which loses detail on long sentences. Attention lets the decoder look back at *all* encoder hidden states and decide, at each step, which ones matter.
- **What are queries, keys and values?** Think of a lookup. The query is what you're looking for, each key is a label on a stored item, and each value is the item itself. You score the query against every key, turn the scores into weights with softmax, and return the weighted average of the values.
- **What's the difference between additive (Bahdanau) and dot-product attention?** Additive attention scores with a small neural net: $v^\top \tanh(W s + U h)$. Dot-product attention scores with $q \cdot k$, which is faster because it's one matrix multiplication.
- **Why divide by $\sqrt{d_k}$?** Dot products of long random vectors grow with their length. Big scores push softmax into putting nearly all weight on one item, where gradients vanish. Scaling keeps the scores in a reasonable range.
- **What is a mask?** A tensor that marks positions attention must ignore. A *padding mask* hides the filler tokens added to make sentences the same length. A *look-ahead mask* hides future tokens so the decoder can't cheat by reading the answer. In this project, **1 (or True) means "hide this position"**: you set those scores to a huge negative number before the softmax, so their weights become 0.
- **What is multi-head attention?** Run several attentions in parallel, each on a smaller slice of the vector, then join the results. Different heads can learn different relationships (one tracks syntax, another tracks which noun "it" refers to).
- **Why do Transformers need positional encoding?** Attention by itself treats the input as a *set*: shuffle the words and you get the same outputs, just shuffled. Adding a position-dependent pattern of sines and cosines to each embedding tells the model where each word sits.
- **What are residual connections and layer norm for?** A residual adds a sub-layer's input back to its output ($x + \text{sublayer}(x)$), giving gradients a shortcut through deep stacks. Layer norm re-centers and re-scales each token's vector. Together they make deep Transformers trainable.

## Resources
- [Jay Alammar: Visualizing a Neural Machine Translation Model (seq2seq with attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [3Blue1Brown: Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention)
- [Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762) (section 3 is the blueprint for tasks 4–11)
- [Bahdanau et al. (2014), Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Harvard NLP: The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) (the paper, line by line in PyTorch; read it *after* trying each task)
- [Dive into Deep Learning: Attention Mechanisms and Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
- [Andrej Karpathy: Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Rules
- Task 4: NumPy only.
- Everything else: `torch` and `torch.nn` (plus `math` / `numpy` if useful). Every class is an `nn.Module`.
- **Don't use** `nn.MultiheadAttention`, `nn.Transformer*`, or `F.scaled_dot_product_attention` in your solutions. The tests use them to check your answers.
- Attribute names matter. The tests look inside your modules, so use the names given below.
- Masks: 1/True = hide. A mask can be `None`. Masks must broadcast against the score tensor.
- Dropout: PyTorch modules don't take a `training` argument. Use `model.train()` / `model.eval()` instead (that's how the original's `training` flag translates).
- Unless stated otherwise, every `nn.Linear` has a bias (the default).

## Tasks

### 0. RNN Encoder
**File:** `0-rnn_encoder.py`
**Class:** `RNNEncoder(nn.Module)` with `__init__(self, vocab, embedding, units, batch)`

- Attributes: `batch`, `units`, `embedding` (an `nn.Embedding` from `vocab` words to `embedding` dims) and `gru` (an `nn.GRU` with `units` hidden units that takes batch-first input).
- `initialize_hidden_state(self)`: returns zeros of shape `(batch, units)`.
- `forward(self, x, initial)`: `x` is token ids `(batch, input_seq_len)`, `initial` is `(batch, units)`. Returns `outputs` `(batch, input_seq_len, units)` and `hidden` `(batch, units)`, the final hidden state. Note that PyTorch's GRU wants its state as `(num_layers, batch, units)`, so adjust shapes on the way in and out.

```python
>>> enc = RNNEncoder(1024, 128, 256, 32)
>>> outputs, hidden = enc(torch.randint(0, 1024, (32, 10)), enc.initialize_hidden_state())
>>> outputs.shape, hidden.shape
(torch.Size([32, 10, 256]), torch.Size([32, 256]))
```

### 1. Self Attention (Bahdanau / additive)
**File:** `1-self_attention.py`
**Class:** `SelfAttention(nn.Module)` with `__init__(self, units)`

- Attributes: `W` (applied to the previous decoder state), `U` (applied to the encoder hidden states), both `nn.Linear(units, units)`, and `V`, an `nn.Linear(units, 1)` that turns each tanh vector into one score.
- `forward(self, s_prev, hidden_states)`: `s_prev` is `(batch, units)`, `hidden_states` is `(batch, input_seq_len, units)`.
  - score for each input position = `V(tanh(W(s_prev) + U(hidden_state)))`
  - `weights` = softmax of the scores over the input positions
  - `context` = weighted sum of the hidden states
- Returns `context` `(batch, units)` and `weights` `(batch, input_seq_len, 1)`.

```python
>>> att = SelfAttention(256)
>>> context, weights = att(torch.rand(32, 256), torch.rand(32, 10, 256))
>>> context.shape, weights.shape, weights.sum(dim=1)[0]
(torch.Size([32, 256]), torch.Size([32, 10, 1]), tensor([1.0000]))
```

### 2. RNN Decoder
**File:** `2-rnn_decoder.py`
**Class:** `RNNDecoder(nn.Module)` with `__init__(self, vocab, embedding, units, batch)`

- Attributes: `embedding` (`nn.Embedding(vocab, embedding)`), `gru` (batch-first `nn.GRU` with `units` hidden units; work out its input size), `F` (`nn.Linear(units, vocab)`) and `attention` (your `SelfAttention(units)`).
- `forward(self, x, s_prev, hidden_states)`: decodes **one** step.
  - `x`: the previous target word ids, `(batch, 1)`
  - `s_prev`: the previous decoder state, `(batch, units)`. Use it both to compute attention **and** as the GRU's initial hidden state.
  - `hidden_states`: encoder outputs, `(batch, input_seq_len, units)`
  - Join the attention context in front of the embedded word (context first), run the GRU for one step, and project to the vocabulary.
- Returns `y` `(batch, vocab)` (raw scores, no softmax) and `s` `(batch, units)`, the new state.

```python
>>> dec = RNNDecoder(2048, 128, 256, 32)
>>> y, s = dec(torch.randint(0, 2048, (32, 1)), torch.rand(32, 256), torch.rand(32, 10, 256))
>>> y.shape, s.shape
(torch.Size([32, 2048]), torch.Size([32, 256]))
```

### 3. Concept check (optional)
No code. Sketch the seq2seq-with-attention model from tasks 0–2 on paper: what flows where at decoding step t? Why does the decoder need `s_prev` twice?

### 4. Positional Encoding
**File:** `4-positional_encoding.py`
**Function:** `def positional_encoding(max_seq_len, dm):` (NumPy only)

Return a NumPy array of shape `(max_seq_len, dm)`. For position `pos` and column `k`, the angle is $pos / 10000^{2\lfloor k/2 \rfloor / dm}$. Even columns hold the sine of the angle and odd columns hold the cosine, so each (even, odd) pair shares one angle.

```python
>>> PE = positional_encoding(30, 512)
>>> PE.shape
(30, 512)
>>> positional_encoding(4, 6).round(4)[1]
array([0.8415, 0.5403, 0.0464, 0.9989, 0.0022, 1.    ])
```
Bonus: no Python loops.

### 5. Scaled Dot Product Attention
**File:** `5-sdp_attention.py`
**Function:** `def sdp_attention(Q, K, V, mask=None):`

- `Q`: `(..., seq_len_q, dk)`, `K`: `(..., seq_len_v, dk)`, `V`: `(..., seq_len_v, dv)`. The `...` means any number of leading batch dimensions (task 6 passes 4-D tensors).
- `mask`: `None` or a tensor that broadcasts to `(..., seq_len_q, seq_len_v)`; 1/True positions get a score of about −1e9 before the softmax.
- Returns `output` `(..., seq_len_q, dv)` and `weights` `(..., seq_len_q, seq_len_v)`.
- Formula: $\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$.

```python
>>> output, weights = sdp_attention(torch.rand(50, 10, 256), torch.rand(50, 15, 256), torch.rand(50, 15, 512))
>>> output.shape, weights.shape
(torch.Size([50, 10, 512]), torch.Size([50, 10, 15]))
```

### 6. Multi Head Attention
**File:** `6-multihead_attention.py`
**Class:** `MultiHeadAttention(nn.Module)` with `__init__(self, dm, h)`

- Attributes: `h`, `dm`, `depth` (= `dm // h`), and four `nn.Linear(dm, dm)` layers: `Wq`, `Wk`, `Wv` (project Q, K, V) and `linear` (the final output projection). `dm` is divisible by `h`.
- `forward(self, Q, K, V, mask)`: inputs are `(batch, seq_len_q, dm)`, `(batch, seq_len_v, dm)`, `(batch, seq_len_v, dm)`.
  - Project, split the last dimension into `h` heads of size `depth`, move the head axis next to the batch axis, run your `sdp_attention`, merge the heads back, and apply `linear`.
- Returns `output` `(batch, seq_len_q, dm)` and `weights` `(batch, h, seq_len_q, seq_len_v)`.
- Head `k` must use dimensions `k*depth` to `(k+1)*depth - 1` of the projected vectors (that's how PyTorch's built-in version does it, and the test compares against it).

```python
>>> mha = MultiHeadAttention(512, 8)
>>> x = torch.rand(50, 15, 512)
>>> output, weights = mha(x, x, x, None)
>>> output.shape, weights.shape
(torch.Size([50, 15, 512]), torch.Size([50, 8, 15, 15]))
```

### 7. Transformer Encoder Block
**File:** `7-transformer_encoder_block.py`
**Class:** `EncoderBlock(nn.Module)` with `__init__(self, dm, h, hidden, drop_rate=0.1)`

- Attributes: `mha` (`MultiHeadAttention`), `dense_hidden` (`nn.Linear` dm→hidden, used with ReLU), `dense_output` (hidden→dm), `layernorm1`, `layernorm2` (`nn.LayerNorm(dm, eps=1e-6)`), `dropout1`, `dropout2` (`nn.Dropout(drop_rate)`).
- `forward(self, x, mask=None)`: `x` is `(batch, input_seq_len, dm)`. Two sub-layers, each followed by dropout, a residual add and a layer norm ("post-norm", as in the 2017 paper):
  1. self-attention (Q = K = V = `x`)
  2. feed-forward network (dense_hidden → ReLU → dense_output)
- Returns `(batch, input_seq_len, dm)`.

```python
>>> block = EncoderBlock(512, 8, 2048)
>>> block(torch.rand(32, 10, 512)).shape
torch.Size([32, 10, 512])
```

### 8. Transformer Decoder Block
**File:** `8-transformer_decoder_block.py`
**Class:** `DecoderBlock(nn.Module)` with `__init__(self, dm, h, hidden, drop_rate=0.1)`

- Attributes: `mha1`, `mha2`, `dense_hidden`, `dense_output`, `layernorm1`–`layernorm3`, `dropout1`–`dropout3`.
- `forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None)`: three post-norm sub-layers:
  1. masked self-attention on `x` with `look_ahead_mask`
  2. attention where the **queries** come from step 1's result and the keys/values come from `encoder_output`, with `padding_mask`
  3. feed-forward network
- Returns `(batch, target_seq_len, dm)`.

```python
>>> block = DecoderBlock(512, 8, 2048)
>>> block(torch.rand(32, 15, 512), torch.rand(32, 10, 512)).shape
torch.Size([32, 15, 512])
```

### 9. Transformer Encoder
**File:** `9-transformer_encoder.py`
**Class:** `Encoder(nn.Module)` with `__init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1)`

- Attributes: `N`, `dm`, `embedding` (`nn.Embedding(input_vocab, dm)`), `positional_encoding` (a float32 tensor of shape `(max_seq_len, dm)` built from task 4; register it as a *buffer* so it moves with `.to(device)`), `blocks` (an `nn.ModuleList` of `N` `EncoderBlock`s), `dropout`.
- `forward(self, x, mask=None)`: `x` is token ids `(batch, input_seq_len)`. Embed, multiply by $\sqrt{dm}$, add the positional encoding for the first `input_seq_len` positions, apply dropout, then run every block. Returns `(batch, input_seq_len, dm)`.

```python
>>> enc = Encoder(6, 512, 8, 2048, 10000, 1000)
>>> enc(torch.randint(0, 10000, (32, 10))).shape
torch.Size([32, 10, 512])
```

### 10. Transformer Decoder
**File:** `10-transformer_decoder.py`
**Class:** `Decoder(nn.Module)` with `__init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1)`

Same attributes and input pipeline as the encoder, but with `DecoderBlock`s.
`forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None)` returns `(batch, target_seq_len, dm)`.

### 11. Transformer Network
**File:** `11-transformer.py`
**Class:** `Transformer(nn.Module)` with `__init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_input, max_seq_target, drop_rate=0.1)`

- Attributes: `encoder`, `decoder`, `linear` (`nn.Linear(dm, target_vocab)`).
- `forward(self, inputs, target, encoder_mask=None, look_ahead_mask=None, decoder_mask=None)`: encode the inputs, decode the target using the encoder output, and project to vocabulary scores. Returns `(batch, target_seq_len, target_vocab)`.

```python
>>> model = Transformer(6, 512, 8, 2048, 10000, 12000, 1000, 1500)
>>> model(torch.randint(0, 10000, (32, 10)), torch.randint(0, 12000, (32, 15))).shape
torch.Size([32, 15, 12000])
```

You'll train this exact class on real translation data in the next project, `06-transformer_apps`.

## Check your work
From `ml-curriculum/`:
```bash
pytest 03-sequences-nlp/05-attention
```
