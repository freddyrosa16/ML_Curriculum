# Hints: RAG and evals

Open one hint at a time.

## 0. Chunking
<details><summary>Hint 1</summary>
Think in word positions, not characters. Each chunk starts `chunk_size − overlap` words after the previous one.
</details>
<details><summary>Hint 2</summary>
`range` with a step gives you every start position. Slicing past the end of a list is safe in Python.
</details>
<details><summary>Hint 3</summary>
Without a stopping rule you'll emit a final chunk that is only overlap. Stop right after the first chunk whose end reaches the last word.
</details>

## 1. Embeddings
<details><summary>Hint 1</summary>
The embedder does the hard part. Your job is to call its `encode`, turn the result into a NumPy array of the right dtype, and normalize it.
</details>
<details><summary>Hint 2</summary>
`np.linalg.norm` with `axis=1` and `keepdims=True` gives one length per row, shaped so it broadcasts when you divide.
</details>
<details><summary>Hint 3</summary>
Dividing by zero gives NaN. Make sure the denominator is never smaller than a tiny positive number.
</details>

## 2. Cosine search
<details><summary>Hint 1</summary>
Normalize both sides, and then cosine similarity is just a matrix product.
</details>
<details><summary>Hint 2</summary>
Normalizing along the *last* axis works for both a single vector and a batch of vectors, so one helper covers both cases.
</details>
<details><summary>Hint 3</summary>
For top-k, `np.argsort` sorts ascending. How can you get descending order without reversing afterwards? Then slice the first k.
</details>

## 3. Vector index
<details><summary>Hint 1</summary>
Keep two things side by side: a list of chunk dicts and a 2-D array of vectors, where row i belongs to chunk i.
</details>
<details><summary>Hint 2</summary>
To add more rows later, stack the new vectors under the old ones (`np.vstack`). Handle the very first `add`, when there's nothing to stack onto yet.
</details>
<details><summary>Hint 3</summary>
For saving, NumPy has its own file format for arrays, and `json` handles the list of dicts. A classmethod builds an empty instance and then fills in both fields from disk, without touching the embedder.
</details>

## 4. Retrieval metrics
<details><summary>Hint 1</summary>
Do one question at a time, then average. Python sets make "how many of the relevant docs are in the top k" a one-liner.
</details>
<details><summary>Hint 2</summary>
For MRR, walk the ranked list with `enumerate(..., start=1)` and stop at the first hit.
</details>
<details><summary>Hint 3</summary>
In `evaluate_retriever`, several chunks can come from one document. Build the doc-id list in rank order and skip ids you've already added.
</details>

## 5. Prompt assembly
<details><summary>Hint 1</summary>
Build a list of lines and join them with newlines at the end. It's much easier than gluing strings together as you go.
</details>
<details><summary>Hint 2</summary>
`enumerate(chunks, start=1)` gives you the source numbers.
</details>
<details><summary>Hint 3</summary>
For citations, a regular expression that captures the digits between square brackets, plus a set to remove duplicates, is all you need. Remember to convert to int before sorting, or "10" sorts before "2".
</details>

## 6. Generation
<details><summary>Hint 1</summary>
`generate_answer` just chains your earlier pieces: search → build_prompt → generator → extract_citations.
</details>
<details><summary>Hint 2</summary>
Citation numbers start at 1 and list positions start at 0. Check the range before you index.
</details>
<details><summary>Hint 3</summary>
For the local generator, instruction-tuned models expect their chat format: the tokenizer's `apply_chat_template` builds it, and `add_generation_prompt` adds the cue for the assistant's turn. Return a closure that remembers the model and tokenizer.
</details>

## 7. Evals
<details><summary>Hint 1</summary>
Normalize in a fixed order: lowercase → strip punctuation (`string.punctuation`) → remove articles (a regex with word boundaries) → collapse whitespace.
</details>
<details><summary>Hint 2</summary>
For F1, `collections.Counter` supports `&` (the overlap, counting duplicates correctly). Precision is overlap/len(pred) and recall is overlap/len(gold).
</details>
<details><summary>Hint 3</summary>
Write the single-gold version first, then wrap it: turn `gold` into a list if it's a string and take the max score over the list.
</details>

## 8. LLM-as-judge
<details><summary>Hint 1</summary>
A template string with named placeholders keeps the rubric readable.
</details>
<details><summary>Hint 2</summary>
`re.findall` with `re.IGNORECASE` finds every "score: N". If there are several, the model's final answer is the last one.
</details>
<details><summary>Hint 3</summary>
Fall back to standalone digits with word boundaries, so that "10" or "2023" don't count as scores.
</details>
