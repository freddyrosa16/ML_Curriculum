# Word Embeddings

**Status:** Original. Two small updates: the Keras embedding task now produces a PyTorch `nn.Embedding`, and the ELMo multiple-choice question is now a short written answer. Bag of words and TF-IDF are written by hand instead of with scikit-learn.

## Why this matters
Neural networks only understand numbers, so the first step in any NLP model is turning words into vectors. This project walks the history: counting words (bag of words), weighting them by how informative they are (TF-IDF), then *learning* dense vectors where similar words sit close together (word2vec, fastText). That last idea, the embedding layer, is the first layer of every Transformer and LLM you'll build later.

## Concepts to understand first
- **What is a bag of words?** Represent a sentence by how many times each vocabulary word appears in it, ignoring order. "dog bites man" and "man bites dog" get the same vector, which shows both the simplicity and the limitation.
- **What is TF-IDF?** Term frequency × inverse document frequency. A word that appears in every sentence ("is", "the") tells you little, so it gets down-weighted. A word that appears in only a few sentences gets boosted. Each sentence's vector is then scaled to length 1.
- **What's wrong with one-hot / count vectors?** They're huge (one dimension per word) and every pair of words is equally different: "cat" is as far from "kitten" as from "carburetor".
- **What is a word embedding?** A short dense vector (say 100 numbers) for each word, learned so that words used in similar contexts get similar vectors. "You shall know a word by the company it keeps."
- **How does word2vec learn?** It trains a tiny network on a fake task. CBOW predicts a word from its neighbors; skip-gram predicts the neighbors from the word. The hidden-layer weights become the embeddings. *Negative sampling* makes training fast by contrasting the true neighbor with a few random "wrong" words.
- **What does fastText add?** It builds each word's vector from its character n-grams ("comp", "omput", …), so it can produce a sensible vector even for words it never saw, such as typos and rare forms.
- **What is an embedding layer in PyTorch?** A lookup table: row k is the vector for token id k. You can start it from pretrained vectors and keep training it with the rest of your model.
- **What makes ELMo (and BERT) different?** They're *contextual*: the vector for "bank" depends on the sentence it's in. word2vec gives each word one fixed vector.

## Resources
- [Jay Alammar: The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)
- [StatQuest: Word Embedding and Word2Vec, Clearly Explained](https://www.youtube.com/watch?v=viZrOnJclY0)
- [Jurafsky & Martin, *Speech and Language Processing* (free draft), chapter "Vector Semantics and Embeddings"](https://web.stanford.edu/~jurafsky/slp3/) (covers TF-IDF and word2vec)
- [Gensim docs: Word2Vec model tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
- [Gensim docs: FastText model tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html)
- [PyTorch docs: `nn.Embedding` (see `from_pretrained`)](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [Mikolov et al. (2013), Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Peters et al. (2018), Deep contextualized word representations (ELMo)](https://arxiv.org/abs/1802.05365)

## Rules
- Tasks 0–1: NumPy and the standard library (`re` is useful). **No scikit-learn and no gensim.** You may use scikit-learn's `CountVectorizer` / `TfidfVectorizer` afterwards to check your answers; your output should match them.
- Tasks 2 and 4: `gensim`.
- Task 3: `torch` (no Keras or TensorFlow).

## Tasks

### 0. Bag of words
**File:** `0-bag_of_words.py`
**Function:** `def bag_of_words(sentences, vocab=None):`

- `sentences`: list of strings
- `vocab`: list of words to count, in that order. If `None`, use every word in `sentences`, sorted alphabetically.
- Tokenizing: lowercase the sentence, then a "word" is any run of **two or more** letters, digits or underscores. Punctuation splits words, so "children's" gives "children" (the lone "s" is too short).
- Returns `embeddings, features`:
  - `embeddings`: integer array of shape `(s, f)`; entry `[i, j]` counts how many times word `j` appears in sentence `i`
  - `features`: NumPy array of the `f` words, in column order

```python
>>> sentences = ["Holberton school is Awesome!", "Machine learning is awesome", "NLP is the future!",
...              "The children are our future", "Our children's children are our grandchildren",
...              "The cake was not very good", "No one said that the cake was not very good",
...              "Life is beautiful"]
>>> E, F = bag_of_words(sentences)
>>> E.shape
(8, 24)
>>> list(F[:6])
['are', 'awesome', 'beautiful', 'cake', 'children', 'future']
>>> E[4]      # "Our children's children are our grandchildren"
array([1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0])
```

### 1. TF-IDF
**File:** `1-tf_idf.py`
**Function:** `def tf_idf(sentences, vocab=None):`

Same inputs and outputs as task 0, but with float TF-IDF weights. Use the same formula scikit-learn uses by default, so you can check against it:
- term frequency $tf$ = the raw count of the word in the sentence
- $idf = \ln\frac{1 + n}{1 + df} + 1$, where $n$ is the number of sentences and $df$ is the number of sentences containing the word
- multiply, then scale each row to unit length (L2 norm). Rows that are all zero stay zero.

```python
>>> vocab = ["awesome", "learning", "children", "cake", "good", "none", "machine"]
>>> E, F = tf_idf(sentences, vocab)
>>> E.round(4)[:2]
array([[1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
       [0.5098, 0.6083, 0.    , 0.    , 0.    , 0.    , 0.6083]])
```

### 2. Train word2vec
**File:** `2-word2vec.py`
**Function:** `def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5, cbow=True, epochs=5, seed=0, workers=1):`

Create, build and train a gensim `Word2Vec` model, and return it.
- `sentences`: list of tokenized sentences (lists of words)
- `vector_size`: embedding size; `min_count`: ignore words rarer than this; `window`: how many neighbors on each side count as context; `negative`: number of negative samples
- `cbow`: `True` → CBOW, `False` → skip-gram
- `epochs`, `seed`, `workers`: passed through

```python
>>> from gensim.test.utils import common_texts
>>> common_texts[:2]
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
>>> w2v = word2vec_model(common_texts, min_count=1)
>>> w2v.wv["computer"].shape
(100,)
```

### 3. word2vec → PyTorch
**File:** `3-gensim_to_torch.py`
**Function:** `def gensim_to_torch(model):`

Convert a trained gensim word2vec model into a **trainable** `torch.nn.Embedding`. Row `k` must be the vector of the word whose index is `k` in gensim (`model.wv.key_to_index`). Training the PyTorch layer later must not change gensim's vectors.

```python
>>> emb = gensim_to_torch(w2v)
>>> emb
Embedding(12, 100)
>>> emb.weight.requires_grad
True
```

### 4. fastText
**File:** `4-fasttext.py`
**Function:** `def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5, cbow=True, epochs=5, seed=0, workers=1):`

Same as task 2, but with gensim's `FastText`. Notice that it can return a vector for a word that isn't in the vocabulary.

```python
>>> ft = fasttext_model(common_texts, min_count=1)
>>> ft.wv["computer"].shape, "computers" in ft.wv.key_to_index, ft.wv["computers"].shape
((100,), False, (100,))
```

### 5. ELMo (short answer)
**File:** `5-elmo.md`

In your own words (at least a short paragraph): when you train an ELMo model, which parts are trained? Think about the character-level layer, the bidirectional LSTM layers, and the per-layer weights used to combine them for a downstream task. How do ELMo's embeddings differ from word2vec's?

## Check your work
From `ml-curriculum/`:
```bash
pytest 03-sequences-nlp/04-word_embeddings
```
Tests use gensim's built-in `common_texts` corpus, so nothing is downloaded.
