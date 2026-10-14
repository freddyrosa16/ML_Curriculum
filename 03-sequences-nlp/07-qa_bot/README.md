# QA Bot

**Status:** Modernized. The original used TensorFlow BERT-large plus TF Hub. Now it uses Hugging Face `transformers` with a small PyTorch model (DistilBERT fine-tuned on SQuAD) and `sentence-transformers` for semantic search. The code is structured so the logic can be tested without downloading models.

## Why this matters
This is a mini version of the "chat with your documents" systems everyone is building. Step 1: find the most relevant document (*retrieval*, using sentence embeddings). Step 2: pull the answer out of it (*extractive QA*). That pattern, retrieve and then read, is the core of RAG, which you'll build with an LLM in stage 4. Here you'll see each half on its own, using a real knowledge base (a coding school's student-handbook articles).

## Concepts to understand first
- **What is extractive question answering?** The model doesn't write an answer. It points at one: it returns a start position and an end position inside the reference text, and the answer is the text between them. That's why it can only answer with words that actually appear in the document.
- **What do the start and end logits mean?** For every token, the model gives a score for "the answer starts here" and another for "the answer ends here". The best answer is the pair (start, end) with the highest combined score, as long as it's a sensible span: the end isn't before the start, it isn't too long, and it's inside the document rather than the question.
- **What is SQuAD?** The Stanford Question Answering Dataset: 100k questions about Wikipedia paragraphs, each answered by a span of the paragraph. Models "fine-tuned on SQuAD" have learned this pointing task. SQuAD 1.1 always has an answer, so models trained on it rarely say "I don't know". SQuAD 2.0 adds unanswerable questions.
- **What is a sentence embedding?** A vector for a whole sentence or document, trained so that texts with similar *meaning* get nearby vectors, even with no words in common ("When are PLDs?" ↔ "Peer Learning Days run from 9 to 3").
- **What is cosine similarity?** The cosine of the angle between two vectors: $\frac{a \cdot b}{\|a\|\|b\|}$. It ranges from −1 to 1 and ignores length, so a long document doesn't win just because its vector is bigger. Semantic search returns the document whose embedding has the highest cosine similarity with the question's.
- **What is a stub (or mock)?** A tiny fake version of something slow or external, used in tests. The tests swap in a fake embedder and a fake QA function, so they check your logic in milliseconds without downloading anything. To make that possible, your functions must take the embedder as an argument and look up the QA function by name at call time.

## Resources
- [Hugging Face LLM Course, chapter 7: Question answering](https://huggingface.co/learn/llm-course/chapter7/7) (explains start/end logits and choosing the best span)
- [Hugging Face docs: Question answering task guide](https://huggingface.co/docs/transformers/tasks/question_answering)
- [Model card: distilbert/distilbert-base-cased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad)
- [Sentence-Transformers docs: Semantic Search](https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [Model card: sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Rajpurkar et al. (2016), SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)
- [Jay Alammar: The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)

## Rules
- `numpy`, `torch`, `transformers` (`AutoTokenizer`, `AutoModelForQuestionAnswering`) and `sentence_transformers`, plus the standard library.
- Don't use the `pipeline("question-answering")` helper. Turning logits into an answer is the point of task 0.
- Load each model **once** and reuse it, not on every question (look up `functools.lru_cache`, or load lazily into a module-level variable). Importing your module should *not* download anything; load models the first time they're needed.
- The exit words are `exit`, `quit`, `goodbye`, `bye`, case-insensitive, ignoring surrounding spaces.
- Device: CPU is fine for these models.

## Data
- The knowledge base is in this project: `data/ZendeskArticles/` (91 Markdown articles from a coding school's student help center).
- Models (downloaded on first use, cached in `~/.cache/huggingface/`):
  - QA: [`distilbert/distilbert-base-cased-distilled-squad`](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad) (~260 MB)
  - Embeddings: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (~90 MB)

## Tasks

### 0. Question Answering
**File:** `0-qa.py`
Two functions:

`best_span(start_logits, end_logits, context_mask=None, max_answer_len=30)`
- `start_logits`, `end_logits`: 1-D arrays, one score per token
- `context_mask`: optional list/array of booleans, `True` for tokens that belong to the reference text (answers must lie entirely inside it); `None` means every token is allowed
- Returns the `(start, end)` pair of token indices with the highest `start_logits[start] + end_logits[end]`, subject to `start <= end` and `end - start < max_answer_len`. Returns `None` if no pair is allowed.

`question_answer(question, reference)`
- Tokenize the question and the reference together as a pair, run the QA model, and use `best_span` to pick the answer, allowing only reference tokens (the tokenizer's `sequence_ids()` tells you which tokens came from which text).
- Decode those tokens back to a string. Return `None` if there's no answer: no valid span, or the best span scores lower than the "no answer" score at the very first token ([CLS]).

```python
>>> best_span([0, 1, 5, 0, 0], [0, 0, 1, 6, 0])
(2, 3)
>>> reference = open("data/ZendeskArticles/PeerLearningDays.md").read()
>>> question_answer("When are PLDs?", reference)
'9 : 00 AM to 3 : 00 PM'
```

### 1. Create the loop
**File:** `1-loop.py`
**Function:** `def loop():` (also make the file runnable as a script: `python 1-loop.py`)

Repeatedly prompt the user with `Q: `. For an exit word, print `A: Goodbye` and stop. Otherwise print `A:`. Only the whole input counts as an exit word: "say bye to me" isn't one.

```
$ python 1-loop.py
Q: Hello
A:
Q: How are you?
A:
Q: BYE
A: Goodbye
```

### 2. Answer questions
**File:** `2-qa.py`
**Function:** `def answer_loop(reference):`

The loop from task 1, but each question is answered with task 0's `question_answer` against `reference`. If there's no answer, print `A: Sorry, I do not understand your question.`

Import task 0's function at the top of the file so it's a module-level name called `question_answer` (the tests replace that name with a stub).

```python
>>> answer_loop(open("data/ZendeskArticles/PeerLearningDays.md").read())
Q: When are PLDs?
A: 9 : 00 AM to 3 : 00 PM
Q: EXIT
A: Goodbye
```

### 3. Semantic Search
**File:** `3-semantic_search.py`
**Function:** `def semantic_search(corpus_path, sentence, embedder=None):`

- `corpus_path`: a folder; read every `.md` file in it (ignore other files)
- `sentence`: the query
- `embedder`: a function that takes a list of strings and returns a NumPy array of shape `(n, d)`, one embedding per string. If `None`, use `all-MiniLM-L6-v2` through `sentence-transformers`.
- Embed all documents in one call and the query in another, then return the **text** of the document with the highest cosine similarity to the query. Return `None` if the folder has no Markdown files.

```python
>>> doc = semantic_search("data/ZendeskArticles", "When are PLDs?")
>>> print(doc[:60])
PLD Overview
Peer Learning Days (PLDs) are a time for you
```

### 4. Multi-reference Question Answering
**File:** `4-qa.py`
**Function:** `def question_answer(corpus_path):`

The final bot: for each question, use `semantic_search` to find the best article in `corpus_path`, then extract the answer from that article. Same loop, exit words and "Sorry" message as before.

At the top of the file, import task 3's function as a module-level name `semantic_search` and task 0's `question_answer` under the name `extract_answer` (it can't keep its own name, since this file's function is called `question_answer`). The tests replace both with stubs.

```
>>> question_answer("data/ZendeskArticles")
Q: When are PLDs?
A: 9 : 00 AM to 3 : 00 PM
Q: What are Mock Interviews?
A: Each student in a pair will receive a different script to interview their partner
Q: What does PLD stand for?
A: Overview Peer Learning Days
Q: goodbye
A: Goodbye
```
Try questions that *aren't* answered in any article. What happens, and why? (Hint: think about what SQuAD 1.1 taught the model.)

## Check your work
From `ml-curriculum/`:
```bash
pytest 03-sequences-nlp/07-qa_bot -m "not slow"   # fast, uses stubs
pytest 03-sequences-nlp/07-qa_bot                 # also downloads and runs the real models
```
