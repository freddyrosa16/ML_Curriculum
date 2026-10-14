# RAG and evals

**Status:** New

## Why this matters
LLMs don't know your documents and they make things up with confidence. **Retrieval-augmented generation (RAG)** fixes both: find the relevant passages first, paste them into the prompt, and ask the model to answer from them with citations. It's the most common way LLMs get used in real products. The other half of the job is **evaluation**: without numbers you can't tell whether a change to chunking, embeddings or prompts made things better or worse. You'll build both here, from NumPy cosine similarity up.

## Concepts to understand first
- **What is an embedding?** A vector that represents the meaning of a piece of text, so that similar meanings land close together. Sentence-embedding models (like `all-MiniLM-L6-v2`, which gives 384 numbers per text) are trained so that a question and the passage that answers it end up near each other.
- **What is cosine similarity?** $\cos(u, v) = \frac{u \cdot v}{\|u\|\|v\|}$: the cosine of the angle between two vectors, from −1 to 1. If you normalize every vector to length 1 first, it's just a dot product, so searching a whole corpus is one matrix-vector multiply.
- **Why chunk documents?** Embedding models have an input limit, one vector can't capture a long document well, and you want to paste only the relevant part into the prompt. Overlap between chunks keeps a sentence that straddles a boundary from being lost.
- **What are recall@k and MRR?** For each question you know which documents are relevant. **Recall@k**: the fraction of relevant documents that appear in the top k results. **MRR** (mean reciprocal rank): the average of 1/(rank of the first relevant result), with 0 if none shows up. Recall asks "did we find it?" and MRR asks "how high up?".
- **Why number the sources in the prompt?** So the model can cite them ("[2]"), and so you can check the citations: did it cite a source that was actually retrieved, and was it the right one?
- **How do you score a free-text answer?** Exact match (after normalizing case, punctuation and articles) is strict and easy to compute. Token F1 gives partial credit for overlapping words. "Contains" checks whether the gold answer appears anywhere in a long answer. An **LLM-as-judge** asks a model to grade answers against a rubric: flexible, but it has biases of its own, so check it against your own judgment on a sample.

## Resources
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401): the original RAG paper.
- [Sentence Transformers documentation](https://sbert.net/) and its [pretrained models list](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- [Introduction to Information Retrieval, ch. 8: Evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html): precision, recall and ranking metrics.
- [Pinecone: Chunking strategies for LLM applications](https://www.pinecone.io/learn/chunking-strategies/)
- [Hamel Husain: Your AI product needs evals](https://hamel.dev/blog/posts/evals/): practical advice on evaluating LLM systems.
- [Judging LLM-as-a-Judge (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685): where LLM judges work and where they're biased.
- [Hugging Face: chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating): how to prompt an instruction-tuned model correctly.

## Rules
- Tasks 0, 4, 5, 7, 8: standard library only (`re`, `string`, `collections`, `json`, `os`).
- Tasks 1–3: `numpy`. Task 1 also uses `sentence_transformers`, but only inside `load_embedder`. Do the similarity search yourself, with no FAISS or vector databases (try one afterwards if you're curious).
- Task 6: `transformers` and `torch`, only inside `make_local_generator`.
- An **embedder** is any object with a method `encode(list_of_texts, batch_size=...)` that returns an array of shape `(n, d)`. A **generator** is any function `prompt -> str`. Writing your code against these two small interfaces lets the tests use fakes and lets you swap in an API model later.
- Files may import earlier tasks with `__import__("2-search")` and similar.

## Data
`data/corpus.json` maps 12 document ids to short passages on ML topics you've already studied. `data/questions.json` holds 12 questions, each with `relevant_doc_ids` and a short gold `answer`. They're small enough to check by eye. Once everything works, try your own documents (your course notes, a project's README files…).

Models (downloaded automatically by the libraries the first time, then cached):
- Embeddings: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), about 90 MB.
- Generation: `HuggingFaceTB/SmolLM2-360M-Instruct` (about 720 MB), or `HuggingFaceTB/SmolLM2-135M-Instruct` if you want it smaller.
- Optional: any hosted LLM API wrapped as a `prompt -> str` function.

## Tasks

### 0. Chunking
File: `0-chunking.py`.
- `chunk_text(text, chunk_size=200, overlap=50)` → list of strings. Split on whitespace into words. Each chunk has at most `chunk_size` words, and the next chunk starts `chunk_size - overlap` words later. Stop once a chunk reaches the end of the text. Empty text → `[]`. Raise `ValueError` unless `chunk_size > 0` and `0 <= overlap < chunk_size`.
- `chunk_documents(docs, chunk_size=200, overlap=50)` → list of dicts `{"id": "<doc_id>#<i>", "doc_id": ..., "text": ...}`, in document order.

```
>>> chunk_text("w0 w1 w2 w3 w4 w5 w6 w7 w8 w9", chunk_size=4, overlap=1)
['w0 w1 w2 w3', 'w3 w4 w5 w6', 'w6 w7 w8 w9']
>>> chunk_documents({"a": "one two three four five", "b": "six"}, chunk_size=3, overlap=1)
[{'id': 'a#0', 'doc_id': 'a', 'text': 'one two three'}, {'id': 'a#1', 'doc_id': 'a', 'text': 'three four five'}, {'id': 'b#0', 'doc_id': 'b', 'text': 'six'}]
```

### 1. Embeddings
File: `1-embed.py`.
- `load_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=None)` → a `SentenceTransformer`.
- `embed_texts(texts, embedder, batch_size=32)` → `float32` array of shape `(n, d)` where every row has length 1. A row that is all zeros must not turn into NaNs.

```
>>> vecs = embed_texts(["A cat sat on the mat.", "A kitten rested on a rug.", "Stock prices fell."], load_embedder())
>>> vecs.shape
(3, 384)
>>> float(vecs[0] @ vecs[1]) > float(vecs[0] @ vecs[2])
True
```

### 2. Cosine similarity search
File: `2-search.py`.
- `cosine_similarity(queries, matrix)`: `queries` has shape `(d,)` or `(q, d)`, and `matrix` has shape `(n, d)`. Returns shape `(n,)` or `(q, n)`. Don't assume the inputs are already normalized. No Python loops.
- `top_k(query_vec, matrix, k=3)` → `(indices, scores)` for the `k` most similar rows, best first. If `k` is larger than `n`, return all `n`.

```
>>> m = np.array([[1., 0.], [0., 2.], [-3., 0.], [1., 1.]])
>>> cosine_similarity(np.array([2., 0.]), m)
array([ 1.        ,  0.        , -1.        ,  0.70710678])
>>> top_k(np.array([1., 0.2]), m, k=2)[0]
array([0, 3])
```

### 3. A small vector index
File: `3-vector_index.py`. Class `VectorIndex(embedder)`.
- `add(chunks)`: embeds the chunks' `text` fields and stores both the chunks and their vectors. Can be called more than once.
- `search(query, k=3)` → list of `(chunk, score)` tuples, best first, with `score` as a Python float. An empty index returns `[]`.
- `save(path)`: writes the vectors and chunks into the folder `path` (for example a `.npy` file and a `.json` file).
- `VectorIndex.load(path, embedder)` (a classmethod) → an index restored from disk **without re-embedding** anything.
- `len(index)` → number of chunks.

```
>>> index = VectorIndex(load_embedder())
>>> index.add(chunk_documents(corpus))
>>> len(index)
12
>>> chunk, score = index.search("Which LoRA matrix is initialized to zero?", k=3)[0]
>>> chunk["doc_id"]
'lora'
```

### 4. Retrieval metrics
File: `4-retrieval_metrics.py`. `retrieved` is a list (one entry per question) of ranked doc-id lists, and `relevant` is a list of sets of doc ids.
- `recall_at_k(retrieved, relevant, k)` → mean over questions of |top-k ∩ relevant| / |relevant|.
- `mrr(retrieved, relevant)` → mean over questions of 1/rank of the first relevant doc (ranks start at 1), or 0 if none appears.
- `evaluate_retriever(index, questions, k=3)` → `{"recall@<k>": ..., "mrr": ...}`. Search for `k` chunks per question, turn them into a ranked list of doc ids (keep the first occurrence when several chunks come from the same document), and compare against `relevant_doc_ids`.

```
>>> recall_at_k([["a", "b", "c"], ["x", "y", "z"]], [{"b"}, {"z", "q"}], 3)
0.75
>>> mrr([["a", "b", "c"], ["x", "y", "z"], ["m"]], [{"b"}, {"x", "z"}, {"nope"}])
0.5
>>> evaluate_retriever(index, questions, k=3)
{'recall@3': 1.0, 'mrr': 1.0}
```
With MiniLM, this easy 12-document set is retrieved perfectly. Make it harder and watch the numbers move: use smaller chunks (`chunk_size=20`), k = 1, reworded questions that share no keywords with the passage, or extra distractor documents on similar topics.

### 5. Prompt assembly with citations
File: `5-prompt.py`.
- `build_prompt(question, chunks, max_chars=None)` → a string laid out like this:
  1. One instruction paragraph: answer only from the sources, cite them like `[1]`, and say `I don't know` if the answer isn't there.
  2. A blank line, then one line per chunk: `[i] (doc_id) text`, numbered from 1. If `max_chars` is set, cut each chunk's text to that many characters.
  3. A blank line, then `Question: <question>`, and a final line `Answer:`.
- `extract_citations(answer)` → sorted list of the distinct numbers cited as `[n]`.

```
>>> print(build_prompt("What starts at zero?", [{"id": "lora#0", "doc_id": "lora", "text": "B starts at zero."}]))
Answer the question using only the sources below. Cite the sources you use like [1]. If the answer is not in the sources, say "I don't know".

[1] (lora) B starts at zero.

Question: What starts at zero?
Answer:
>>> extract_citations("B is zero [2]. Also [1][2] and [10].")
[1, 2, 10]
```
(Your instruction wording can differ. The structure and the `[i] (doc_id) text` lines are what get checked.)

### 6. Generate an answer
File: `6-generate.py`.
- `generate_answer(question, index, generator, k=3)` → `{"answer": str, "citations": [doc_ids], "sources": [chunks]}`. Retrieve `k` chunks, build the prompt, call `generator(prompt)`, and strip the result. Turn the cited numbers into doc ids, ignoring numbers outside 1..k and listing each doc only once.
- `make_local_generator(model_id="HuggingFaceTB/SmolLM2-360M-Instruct", max_new_tokens=128, device=None)` → a function `prompt -> str`. It wraps the prompt as a single user message with the tokenizer's chat template, decodes greedily, and returns only the new text. Load the model in float32: bfloat16 is very slow on a laptop CPU (same story as in 02-fine_tuning).

```
>>> gen = make_local_generator()
>>> out = generate_answer("Which LoRA matrix is initialized to zero?", index, gen)
>>> out["answer"]
'Matrix B is initialized to zero [1].'
>>> out["citations"]
['lora']
```
(Your model's wording will differ.)

### 7. Answer evaluation
File: `7-evals.py`. `gold` may be a string or a list of acceptable strings. Score against the best one.
- `normalize_answer(s)`: lowercase, remove punctuation, remove the words a/an/the, collapse whitespace.
- `exact_match(pred, gold)` → 1.0 or 0.0, compared after normalizing.
- `token_f1(pred, gold)` → F1 over the normalized tokens (count repeated words properly).
- `answer_contains(pred, gold)` → 1.0 if the normalized gold string appears inside the normalized prediction.
- `run_eval(questions, answer_fn)` → `{"exact_match", "f1", "contains", "n"}`, averaged over the questions. `answer_fn(question) -> str`.

```
>>> exact_match("The elbow method.", "elbow method")
1.0
>>> round(token_f1("elbow", "elbow method"), 3)
0.667
>>> run_eval(questions, lambda q: generate_answer(q, index, gen)["answer"])
{'exact_match': 0.0, 'f1': 0.21, 'contains': 0.58, 'n': 12}    # yours will differ
```
Why is exact match nearly useless for a chatty RAG system? Which metric matches your own sense of "correct" best?

### 8. LLM-as-judge (bonus)
File: `8-llm_judge.py`.
- `build_judge_prompt(question, answer, reference)` → a rubric prompt that includes all three, explains a 1–5 scale, and asks for a final line `Score: N`.
- `parse_score(text)` → int 1–5, or `None`. Prefer the last `Score: N` (or `score = N`). Otherwise use the last standalone digit from 1 to 5.
- `judge(question, answer, reference, generator)` → the parsed score.

```
>>> parse_score("Mostly right.\nScore: 4")
4
>>> parse_score("no number") is None
True
```
Grade 12 answers yourself, then with the judge. How often do you agree? A 360M model is a weak judge, so if you have API access, try a stronger one.

## Check your work
```bash
pytest 04-llms/03-rag_and_evals -m "not slow"   # offline, uses a stub embedder
pytest 04-llms/03-rag_and_evals                 # also downloads MiniLM + SmolLM2-360M-Instruct
```
