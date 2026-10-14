"""Tests for 03-sequences-nlp/03-nlp_metrics (BLEU)."""
import math

import pytest

from helpers import load_task

REFS = [["the", "cat", "is", "on", "the", "mat"],
        ["there", "is", "a", "cat", "on", "the", "mat"]]
SENT = ["there", "is", "a", "cat", "here"]


def uni():
    return load_task(__file__, "0-uni_bleu.py", "uni_bleu")


def ngram():
    return load_task(__file__, "1-ngram_bleu.py", "ngram_bleu")


def cumulative():
    return load_task(__file__, "2-cumulative_bleu.py", "cumulative_bleu")


# ---------- Task 0 ----------

def test_uni_bleu_example():
    assert uni()(REFS, SENT) == pytest.approx(0.6549846024623855)


def test_uni_bleu_perfect_match_is_one():
    assert uni()(REFS, REFS[1]) == pytest.approx(1.0)


def test_uni_bleu_clips_repeated_words():
    # "the" appears at most twice in any reference, so only 2 of the 7 words count.
    # The closest reference length is 7 (no brevity penalty).
    score = uni()(REFS, ["the"] * 7)
    assert score == pytest.approx(2 / 7)


def test_uni_bleu_brevity_penalty():
    # Every word matches, but the sentence is short: closest reference has length 6.
    score = uni()(REFS, ["the", "cat"])
    assert score == pytest.approx(math.exp(1 - 6 / 2))


def test_uni_bleu_closest_length_tie_prefers_shorter_reference():
    # length 4 is equally close to references of length 3 and 5; BLEU uses the shorter one.
    refs = [["a", "b", "c"], ["a", "b", "c", "d", "e"]]
    assert uni()(refs, ["a", "b", "c", "d"]) == pytest.approx(1.0)


# ---------- Task 1 ----------

def test_ngram_bleu_example():
    assert ngram()(REFS, SENT, 2) == pytest.approx(0.6140480648084865)


def test_ngram_bleu_n1_matches_unigram():
    assert ngram()(REFS, SENT, 1) == pytest.approx(0.6549846024623855)


def test_ngram_bleu_trigram():
    # 3-grams of SENT: (there is a), (is a cat), (a cat here) -> 2 of 3 appear in a reference
    expected = (2 / 3) * math.exp(1 - 6 / 5)
    assert ngram()(REFS, SENT, 3) == pytest.approx(expected)


def test_ngram_bleu_perfect_match():
    assert ngram()(REFS, REFS[0], 4) == pytest.approx(1.0)


# ---------- Task 2 ----------

def test_cumulative_bleu_example():
    assert cumulative()(REFS, SENT, 4) == pytest.approx(0.5475182535069453)


def test_cumulative_bleu_n1_is_unigram():
    assert cumulative()(REFS, SENT, 1) == pytest.approx(0.6549846024623855)


def test_cumulative_bleu_is_geometric_mean():
    # precisions for n=1,2 are 4/5 and 3/4, brevity penalty exp(1 - 6/5)
    expected = math.exp(1 - 6 / 5) * math.sqrt((4 / 5) * (3 / 4))
    assert cumulative()(REFS, SENT, 2) == pytest.approx(expected)


def test_cumulative_bleu_perfect_match():
    assert cumulative()(REFS, REFS[1], 4) == pytest.approx(1.0)
