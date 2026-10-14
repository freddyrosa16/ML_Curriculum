# Hints: NLP Metrics (BLEU)

## 0. Unigram BLEU score
<details><summary>Hint 1</summary>
Do the example by hand first. Which of the 5 candidate words appear in some reference? What are the two reference lengths, and which is closest to 5?
</details>
<details><summary>Hint 2</summary>
Three pieces: (1) count each candidate word; (2) for each word, find the highest number of times it appears in any one reference; (3) credit each word with the smaller of those two numbers. The precision is total credit divided by candidate length.
</details>
<details><summary>Hint 3</summary>
`collections.Counter` makes counting easy. For the closest reference length, Python's `min` accepts a `key` function. Make the key a tuple of (distance, length) and ties go to the shorter reference automatically.
</details>

## 1. N-gram BLEU score
<details><summary>Hint 1</summary>
Everything from task 0 still applies. The only change is what you count: n-grams instead of single words.
</details>
<details><summary>Hint 2</summary>
Write a small helper that turns a word list into its list of n-grams. Lists can't be dictionary keys, but tuples can.
</details>
<details><summary>Hint 3</summary>
A sentence of length L has L − n + 1 n-grams, which is your precision's denominator. The brevity penalty is unchanged: it compares word lengths.
</details>

## 2. Cumulative N-gram BLEU score
<details><summary>Hint 1</summary>
Reuse your n-gram precision (without the brevity penalty) for every size from 1 to n.
</details>
<details><summary>Hint 2</summary>
A geometric mean of n numbers is the n-th root of their product. An equivalent, more stable form is exp(mean of the logs).
</details>
<details><summary>Hint 3</summary>
Apply the brevity penalty once, to the combined score, not to each precision.
</details>
