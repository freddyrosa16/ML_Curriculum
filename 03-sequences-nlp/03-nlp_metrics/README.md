# NLP Metrics: BLEU

**Status:** Original

## Why this matters
How do you score a translation automatically when there are many correct ways to say the same thing? BLEU was the first widely adopted answer, and it is still reported in nearly every machine translation paper. You'll use it to grade your own translation Transformer in project 06. Building it yourself also shows you its blind spots, which matters when you read claims like "our model gets 40 BLEU".

## Concepts to understand first
- **What is an n-gram?** A run of n consecutive words. In "the cat sat", the unigrams are the 3 words, the bigrams are "the cat" and "cat sat", and there is one trigram.
- **What is n-gram precision?** Of the n-grams in your candidate sentence, what fraction also appear in a reference translation? Higher means more of what you wrote matches what a human wrote.
- **Why "clip" the counts?** Without clipping, "the the the the" would score perfectly against any reference containing "the". Clipping caps each n-gram's credit at the most times it appears in any *single* reference.
- **What is the brevity penalty?** Precision rewards short, safe outputs ("the cat" is 100% precise!). So BLEU multiplies by a penalty when the candidate is shorter than the reference: $BP = e^{1 - r/c}$ if $c \le r$, else 1, where $c$ is the candidate length and $r$ is the reference length closest to $c$ (on a tie, the shorter reference).
- **What does "cumulative" BLEU mean?** Instead of a single n, combine the precisions for n = 1, 2, …, N with a *geometric* mean (equal weights), then apply the brevity penalty. BLEU-4 (N = 4) is the standard.
- **What are BLEU's weaknesses?** It only counts exact word overlap, so synonyms and paraphrases get no credit, and it ignores meaning and grammar beyond short n-grams. Sentence-level BLEU is noisy; it's more reliable over a whole test set. ROUGE (recall-focused, for summaries) and learned metrics like BERTScore and COMET address some of this.

## Resources
- [Papineni et al. (2002), "BLEU: a Method for Automatic Evaluation of Machine Translation"](https://aclanthology.org/P02-1040/) (the original paper; sections 2.1–2.3 are short and readable)
- [Hugging Face Evaluate: BLEU metric card](https://huggingface.co/spaces/evaluate-metric/bleu)
- [Jurafsky & Martin, *Speech and Language Processing* (free draft): the machine translation chapter covers BLEU and chrF](https://web.stanford.edu/~jurafsky/slp3/)
- [Matt Post (2018), "A Call for Clarity in Reporting BLEU Scores" (SacreBLEU)](https://aclanthology.org/W18-6319/)
- [Lin (2004), ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)

## Rules
- `import numpy as np` and the standard library (e.g. `collections`) only.
- **Don't use `nltk`** or any BLEU library. You can use one to check your answers afterwards.
- `references` is a list of reference translations, each a list of words. `sentence` is the candidate, a list of words.

## Tasks

### 0. Unigram BLEU score
**File:** `0-uni_bleu.py`
**Function:** `def uni_bleu(references, sentence):`

Return the unigram BLEU score: clipped unigram precision times the brevity penalty.

```python
>>> references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
>>> sentence = ["there", "is", "a", "cat", "here"]
>>> uni_bleu(references, sentence)
0.6549846024623855
```

### 1. N-gram BLEU score
**File:** `1-ngram_bleu.py`
**Function:** `def ngram_bleu(references, sentence, n):`

Same idea, but with clipped precision over n-grams of size `n`. The brevity penalty still uses word counts, not n-gram counts.

```python
>>> ngram_bleu(references, sentence, 2)
0.6140480648084865
```

### 2. Cumulative N-gram BLEU score
**File:** `2-cumulative_bleu.py`
**Function:** `def cumulative_bleu(references, sentence, n):`

Combine the precisions for every n-gram size from 1 to `n` with an evenly weighted geometric mean, then apply the brevity penalty once.

```python
>>> cumulative_bleu(references, sentence, 4)
0.5475182535069453
```

### 3. Concept check (optional)
No code. Answer in your own words:
1. Name three real applications of NLP and the metric you'd use to evaluate each.
2. When would you prefer ROUGE over BLEU? What does perplexity measure, and why can't it compare a translation to a reference?
3. Find a translation that is clearly correct but gets a low BLEU score against a single reference. What does that tell you?

## Check your work
From `ml-curriculum/`:
```bash
pytest 03-sequences-nlp/03-nlp_metrics
```
