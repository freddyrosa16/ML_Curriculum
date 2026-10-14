# Hints: Word Embeddings

## 0. Bag of words
<details><summary>Hint 1</summary>
Split the job in two: first turn each sentence into a list of tokens, then count. Get tokenizing right before anything else, and print the tokens of "Our children's children are our grandchildren" to check.
</details>
<details><summary>Hint 2</summary>
The `re` module can find every match of a pattern in a string. You want "a word boundary, two or more word characters, a word boundary". Look up `\b` and `\w`. A set gives you the unique words; sort it for the vocabulary.
</details>
<details><summary>Hint 3</summary>
A dictionary from word to column number turns "which column is this word?" into one lookup. Start from a zero matrix of shape (sentences, vocabulary) and add one for each token you find in the vocabulary.
</details>

## 1. TF-IDF
<details><summary>Hint 1</summary>
Your bag-of-words counts are already the "tf" part. Reuse task 0.
</details>
<details><summary>Hint 2</summary>
Document frequency for a word is the number of rows where its count is above zero. That's a comparison followed by a sum down the columns. Then apply the idf formula to the whole vector at once.
</details>
<details><summary>Hint 3</summary>
Broadcasting multiplies each column by its idf. For the row normalization, `np.linalg.norm` with an `axis` and `keepdims` gives you one length per row. Guard against dividing an all-zero row by zero.
</details>

## 2. Train word2vec
<details><summary>Hint 1</summary>
Look at the gensim `Word2Vec` constructor in the docs. Most of your arguments map straight onto its parameters, some with slightly different names.
</details>
<details><summary>Hint 2</summary>
The `sg` parameter picks the training algorithm: 0 for CBOW, 1 for skip-gram. Convert the `cbow` flag accordingly.
</details>
<details><summary>Hint 3</summary>
If you pass the sentences to the constructor, gensim builds the vocabulary and trains immediately. Alternatively, create the model empty, then call the vocabulary-building method and the training method yourself. Do one or the other, not both, or you'll train twice.
</details>

## 3. word2vec → PyTorch
<details><summary>Hint 1</summary>
The trained vectors live in a NumPy matrix on the model's `wv` attribute, already in index order.
</details>
<details><summary>Hint 2</summary>
`nn.Embedding` has a class method for building a layer from an existing weight matrix. Read its `freeze` argument carefully.
</details>
<details><summary>Hint 3</summary>
To make sure gensim's array isn't shared with PyTorch, create a new tensor (a copy) rather than wrapping the NumPy memory directly. Use float32.
</details>

## 4. fastText
<details><summary>Hint 1</summary>
It's task 2 again with a different gensim class. Compare the two constructors' signatures in the docs.
</details>
<details><summary>Hint 2</summary>
Try looking up a word that isn't in the training set on both models. What happens with each, and why?
</details>

## 5. ELMo
<details><summary>Hint 1</summary>
Read the ELMo paper's section 3 (or a blog summary). Draw the stack: characters → word vectors → LSTM layer 1 → LSTM layer 2.
</details>
<details><summary>Hint 2</summary>
Separate *pre-training* (the language model learns everything) from *using* ELMo in another task (what gets learned there?).
</details>
