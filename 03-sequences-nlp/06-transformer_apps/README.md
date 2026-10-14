# Transformer Applications: Machine Translation

**Status:** Modernized. The original used TensorFlow Datasets (`ted_hrlr_translate`) and `tf.data`. Now it's Hugging Face `datasets` + tokenizers, a PyTorch `DataLoader`, PyTorch masks, and the Transformer **you** built in `05-attention`.

## Why this matters
In project 05 you built a Transformer that produced random numbers. Here you make it do the job it was invented for: translating Portuguese to English. You'll go through the steps every real NLP project needs: load a dataset, train a sub-word tokenizer, turn sentences into padded batches of ids, build masks, and run a training loop with the learning-rate warm-up from the original paper. Training is slow on a laptop, and seeing *why* is part of the lesson.

## Concepts to understand first
- **What is a sub-word tokenizer?** It splits text into pieces between characters and words: "unbelievable" → "un", "##believ", "##able". Common words stay whole, rare words are built from parts, so the vocabulary stays small (here 8,192 pieces) while any word can still be written.
- **Why train a new tokenizer instead of using BERT's?** A tokenizer's vocabulary should match your data. `train_new_from_iterator` keeps the algorithm and special tokens of a pretrained tokenizer but learns a fresh, smaller vocabulary from your sentences.
- **What are start and end tokens?** Extra ids that mark "sentence begins here" and "sentence ends here". The decoder starts generating from the start token and stops when it produces the end token. Here they get the two ids just past the tokenizer's vocabulary: `vocab_size` and `vocab_size + 1`.
- **What is padding, and why do batches need it?** Sentences have different lengths but a tensor must be rectangular, so shorter sentences are filled with 0s up to the longest one in the batch. Masks then make sure attention and the loss ignore those 0s.
- **What is teacher forcing?** During training the decoder is fed the *true* previous words (target shifted right by one) and asked to predict the next word at every position at once. The look-ahead mask stops it from peeking at the word it's supposed to predict.
- **Why the strange learning-rate schedule?** Transformers are unstable at the start of training. The paper's schedule warms up (increasing the learning rate linearly for `warmup_steps`) and then decays it like $1/\sqrt{step}$: $lr = d_{model}^{-0.5}\cdot\min(step^{-0.5},\ step \cdot warmup^{-1.5})$.
- **What is greedy decoding?** At inference time there's no target to feed. You start with the start token, pick the most likely next token, append it, and repeat until the end token. (Beam search keeps several candidates instead of one.)

## Resources
- [Hugging Face LLM Course, chapter 6: The Tokenizers library (training a new tokenizer from an old one)](https://huggingface.co/learn/llm-course/chapter6/2)
- [Hugging Face `datasets` quickstart](https://huggingface.co/docs/datasets/quickstart)
- [Dataset card: Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- [Harvard NLP: The Annotated Transformer (sections on batches, masking, training and greedy decoding)](https://nlp.seas.harvard.edu/annotated-transformer/)
- [PyTorch docs: `torch.optim.lr_scheduler.LambdaLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html)
- [Vaswani et al. (2017), Attention Is All You Need, section 5 (training)](https://arxiv.org/abs/1706.03762)

## Rules
- `torch`, `numpy`, `datasets` (Hugging Face) and `transformers` (for `AutoTokenizer` only, not their models).
- The model is **your** `Transformer` from `05-attention`. No `nn.Transformer`.
- Masks use the same convention as project 05: 1 = hide.
- Device-agnostic: pick CUDA, then MPS, then CPU.
- As in the original, tasks 0–3 grow the same `Dataset` class: each file is a complete copy of the class with the new piece added.

## Data
- **Dataset:** [`Helsinki-NLP/opus-100`](https://huggingface.co/datasets/Helsinki-NLP/opus-100), config `"en-pt"`. Each row looks like `{"translation": {"en": "...", "pt": "..."}}`. There's a 1M-pair `train` split (about 90 MB, downloaded once and cached) and a 2,000-pair `validation` split. Use only the first `n_train` training pairs: the `datasets` library accepts split strings like `"train[:50000]"`.
- The original project's TED talks dataset (`ted_hrlr_translate`) isn't available in a current, script-free form on the Hugging Face Hub, so we use OPUS-100. Its sentences are shorter and messier (lots of subtitles), which is fine for learning.
- **Base tokenizers:** `neuralmind/bert-base-portuguese-cased` (Portuguese) and `google-bert/bert-base-uncased` (English). Only the tokenizer files are downloaded (a few MB), not the models.
- Everything is cached under `~/.cache/huggingface/`.

## Tasks

### 0. Dataset
**File:** `0-dataset.py`
**Class:** `Dataset` with `__init__(self, n_train=50_000)` and `tokenize_dataset(self, data)`

- `__init__` sets:
  - `data_train`: list of `(pt, en)` string pairs from the first `n_train` training rows
  - `data_valid`: list of `(pt, en)` pairs from the validation split
  - `tokenizer_pt`, `tokenizer_en`: from `tokenize_dataset(self.data_train)`
- `tokenize_dataset(data)`: `data` is a list of `(pt, en)` pairs. Start from the two pretrained tokenizers above and train new ones on your sentences with a maximum vocabulary of `2**13`. Returns `tokenizer_pt, tokenizer_en`.

```python
>>> data = Dataset(n_train=2000)
>>> data.data_train[2]
('Eu amava-a.', 'I did love her.')
>>> data.tokenizer_en.vocab_size <= 2**13
True
```

### 1. Encode tokens
**File:** `1-dataset.py`
Add `encode(self, pt, en)`: turn a sentence pair into two NumPy integer arrays of token ids, **without** the tokenizer's own special tokens. Instead, put a start token at the front (id `vocab_size` of that language's tokenizer) and an end token at the back (id `vocab_size + 1`).

```python
>>> pt_tokens, en_tokens = data.encode("Eu amava-a.", "I did love her.")
>>> en_tokens[0] == data.tokenizer_en.vocab_size, en_tokens[-1] == data.tokenizer_en.vocab_size + 1
(True, True)
```

### 2. Encode the dataset and pad batches
**File:** `2-dataset.py`
(This replaces the original's `tf_encode` wrapper, which PyTorch doesn't need.)
- Update `__init__` so `data_train` and `data_valid` hold encoded `(pt_tokens, en_tokens)` pairs.
- Add `pad_batch(self, batch)` (it can be a `@staticmethod`): `batch` is a list of `(pt_tokens, en_tokens)` pairs. Return two `torch.long` tensors of shape `(batch_size, longest_in_batch)`, padded with 0 on the right. This is the `collate_fn` you'll give the `DataLoader`.

```python
>>> pt, en = data.pad_batch([(np.array([9, 5, 10]), np.array([8, 5, 6, 7, 9])),
...                          (np.array([9, 5, 6, 7, 8, 10]), np.array([8, 9]))])
>>> pt
tensor([[ 9,  5, 10,  0,  0,  0],
        [ 9,  5,  6,  7,  8, 10]])
>>> en
tensor([[8, 5, 6, 7, 9],
        [8, 9, 0, 0, 0]])
```

### 3. Pipeline
**File:** `3-dataset.py`
- New constructor: `__init__(self, batch_size, max_len, n_train=50_000)`. Store `max_len`.
- Add `filter_max_len(self, pt_tokens, en_tokens)`: `True` if **both** have at most `max_len` tokens.
- After encoding, drop pairs that fail the filter, then replace `data_train` with a `DataLoader` (shuffled, `collate_fn=self.pad_batch`) and `data_valid` with an unshuffled `DataLoader`.

```python
>>> data = Dataset(32, 40, n_train=2000)
>>> pt, en = next(iter(data.data_train))
>>> pt.shape[0], pt.shape[1] <= 40
(32, True)
```

### 4. Create masks
**File:** `4-create_masks.py`
**Function:** `def create_masks(inputs, target):`

- `inputs`: `(batch, seq_len_in)` token ids; `target`: `(batch, seq_len_out)` token ids. Padding is id 0.
- Returns `encoder_mask, combined_mask, decoder_mask` (float tensors, 1.0 = hide):
  - `encoder_mask`: `(batch, 1, 1, seq_len_in)`, 1.0 where `inputs` is padding (for the encoder's self-attention)
  - `combined_mask`: `(batch, 1, seq_len_out, seq_len_out)`, the element-wise maximum of the look-ahead mask and the target's padding mask (for the decoder's first attention)
  - `decoder_mask`: `(batch, 1, 1, seq_len_in)`, the input padding mask again (for the decoder's second attention, which looks at the encoder output)
- Create the masks on the same device as the inputs.

```python
>>> enc, comb, dec = create_masks(torch.tensor([[5, 6, 7, 0]]), torch.tensor([[8, 9, 0]]))
>>> enc
tensor([[[[0., 0., 0., 1.]]]])
>>> comb[0, 0]
tensor([[0., 1., 1.],
        [0., 0., 1.],
        [0., 0., 1.]])
```

### 5. Train
**Files:** `5-transformer.py` and `5-train.py`

`5-transformer.py` must make your `Transformer` class from `05-attention` importable from this folder. Either copy your code in, or (better, so fixes stay in one place) add the `05-attention` folder to Python's import path and re-export the class. Look up `sys.path` and `pathlib.Path(__file__)`. Build the path from the file's own location, not your current directory.

`5-train.py` must contain:
- `lr_schedule(step, dm, warmup_steps=4000)`: the learning rate for training step `step` (starting at 1), using the formula in the concepts section.
- `masked_loss(real, pred)`: `real` is `(batch, seq)` target ids, `pred` is `(batch, seq, vocab)` logits. Mean cross-entropy over the non-padding tokens only. Returns a scalar tensor.
- `masked_accuracy(real, pred)`: fraction of non-padding tokens where the highest-scoring id is correct.
- `train_transformer(N, dm, h, hidden, max_len, batch_size, epochs, n_train=50_000)`:
  - builds `Dataset(batch_size, max_len, n_train)` and a `Transformer` with vocab sizes `vocab_size + 2` (for the start/end tokens) and `max_len` for both sequence lengths
  - trains with Adam (`betas=(0.9, 0.98)`, `eps=1e-9`), learning rate from `lr_schedule` (PyTorch's `LambdaLR` multiplies a base lr of 1.0 by your function), and teacher forcing (decoder input = target without its last token, labels = target without its first token)
  - every 50 batches prints `Epoch {epoch}, Batch {batch}: Loss {loss}, Accuracy {accuracy}` and after each epoch prints `Epoch {epoch}: Loss {loss}, Accuracy {accuracy}` (running averages over the epoch)
  - stores the dataset on the model as `model.data` (handy for task 6) and returns the trained model

```python
>>> lr_schedule(4000, 512)
0.0006987712429686843
>>> model = train_transformer(4, 128, 8, 512, 40, 64, 2)     # your numbers will differ
Epoch 1, Batch 0: Loss 9.1..., Accuracy 0.0...
Epoch 1, Batch 50: Loss 8.3..., Accuracy 0.04...
...
```
Expect minutes per epoch on a laptop CPU. A small model (N=2–4, dm=128) with 50k pairs is a good start. On a Mac, MPS speeds things up.

### 6. Translate (bonus)
**File:** `6-translate.py`
**Function:** `def translate(model, data, sentence, max_len=40):`

Translate one Portuguese string with greedy decoding: encode it, start the decoder input with the English start token, repeatedly run the model and append the highest-scoring next token until it predicts the end token or reaches `max_len`. Return the decoded English string (without start/end tokens). Try a few sentences from `data_valid` and compute their BLEU score with your code from project 03.

## Check your work
From `ml-curriculum/`:
```bash
pytest 03-sequences-nlp/06-transformer_apps -m "not slow"   # fast, no downloads
pytest 03-sequences-nlp/06-transformer_apps                 # also downloads data and trains a tiny model
```
