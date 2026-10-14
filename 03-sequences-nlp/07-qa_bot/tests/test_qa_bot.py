"""Tests for 03-sequences-nlp/07-qa_bot.

Fast tests replace the Hugging Face models with stubs, so they test your logic (span picking,
the loop, ranking by cosine similarity) without downloads. Real-model tests are marked slow.
"""
import pathlib

import numpy as np
import pytest

from helpers import load_task, project_dir

SORRY = "A: Sorry, I do not understand your question."


def corpus_dir():
    return pathlib.Path(project_dir(__file__)) / "data" / "ZendeskArticles"


def feed_input(monkeypatch, lines):
    """Make input() return these lines one by one."""
    it = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def printed_lines(capsys):
    return [line.rstrip() for line in capsys.readouterr().out.splitlines() if line.strip()]


# ---------- Task 0: picking the answer span ----------

def test_best_span_simple():
    best_span = load_task(__file__, "0-qa.py", "best_span")
    start = [0.0, 1.0, 5.0, 0.0, 0.0]
    end = [0.0, 0.0, 1.0, 6.0, 0.0]
    assert tuple(best_span(start, end)) == (2, 3)


def test_best_span_end_never_before_start():
    best_span = load_task(__file__, "0-qa.py", "best_span")
    # the best start is after the best end; a valid span must still have start <= end
    start = [0.0, 0.0, 0.0, 9.0]
    end = [8.0, 1.0, 0.0, 2.0]
    s, e = best_span(start, end)
    assert s <= e
    assert (s, e) == (3, 3)


def test_best_span_respects_context_mask():
    best_span = load_task(__file__, "0-qa.py", "best_span")
    start = [10.0, 0.0, 2.0, 1.0, 0.0]
    end = [10.0, 0.0, 1.0, 3.0, 0.0]
    mask = [False, False, True, True, True]     # tokens 0-1 are the question, not the context
    assert tuple(best_span(start, end, mask)) == (2, 3)


def test_best_span_max_length():
    best_span = load_task(__file__, "0-qa.py", "best_span")
    start = [5.0, 0, 0, 0, 0, 0]
    end = [0, 0, 0, 0, 0, 5.0]
    s, e = best_span(start, end, max_answer_len=3)
    assert e - s < 3


def test_best_span_none_when_nothing_allowed():
    best_span = load_task(__file__, "0-qa.py", "best_span")
    assert best_span([1.0, 2.0], [1.0, 2.0], [False, False]) is None


@pytest.mark.slow
def test_question_answer_real_model():
    question_answer = load_task(__file__, "0-qa.py", "question_answer")
    reference = (corpus_dir() / "PeerLearningDays.md").read_text()
    answer = question_answer("When are PLDs?", reference)
    assert isinstance(answer, str)
    assert "9" in answer and "3" in answer, answer


# ---------- Task 1: the loop ----------

def test_loop_until_exit(monkeypatch, capsys):
    loop = load_task(__file__, "1-loop.py", "loop")
    feed_input(monkeypatch, ["Hello", "How are you?", "BYE", "never read"])
    loop()
    assert printed_lines(capsys) == ["A:", "A:", "A: Goodbye"]


@pytest.mark.parametrize("word", ["exit", "Quit", "GOODBYE", "bye", "  exit  "])
def test_loop_exit_words(monkeypatch, capsys, word):
    loop = load_task(__file__, "1-loop.py", "loop")
    feed_input(monkeypatch, [word])
    loop()
    assert printed_lines(capsys) == ["A: Goodbye"]


def test_loop_exit_word_inside_sentence_does_not_exit(monkeypatch, capsys):
    loop = load_task(__file__, "1-loop.py", "loop")
    feed_input(monkeypatch, ["say bye to me", "exit"])
    loop()
    assert printed_lines(capsys) == ["A:", "A: Goodbye"]


# ---------- Task 2: answer loop over one reference ----------

def test_answer_loop_with_stub(monkeypatch, capsys):
    mod = load_task(__file__, "2-qa.py")
    calls = []

    def fake_qa(question, reference):
        calls.append((question, reference))
        return "on Fridays" if "PLD" in question else None

    monkeypatch.setattr(mod, "question_answer", fake_qa)
    feed_input(monkeypatch, ["When are PLDs?", "What is love?", "Exit"])
    mod.answer_loop("some reference text")
    assert printed_lines(capsys) == ["A: on Fridays", SORRY, "A: Goodbye"]
    assert calls == [("When are PLDs?", "some reference text"),
                     ("What is love?", "some reference text")]


# ---------- Task 3: semantic search ----------

VOCAB = ["cat", "dog", "fish", "bird"]


def keyword_embedder(texts):
    """Stub embedder: count each VOCAB word. Similar topics -> similar vectors."""
    return np.array([[t.lower().split().count(w) for w in VOCAB] for t in texts], dtype=float)


@pytest.fixture
def tiny_corpus(tmp_path):
    (tmp_path / "a_cats.md").write_text("cat cat dog")
    (tmp_path / "b_fish.md").write_text("fish fish fish bird")
    (tmp_path / "c_dogs.md").write_text("dog dog dog cat")
    (tmp_path / "notes.txt").write_text("cat cat cat cat cat")   # not Markdown: ignore it
    return tmp_path


def test_semantic_search_ranks_by_similarity(tiny_corpus):
    semantic_search = load_task(__file__, "3-semantic_search.py", "semantic_search")
    assert semantic_search(tiny_corpus, "fish", embedder=keyword_embedder) == "fish fish fish bird"
    assert semantic_search(tiny_corpus, "dog", embedder=keyword_embedder) == "dog dog dog cat"
    assert semantic_search(tiny_corpus, "cat", embedder=keyword_embedder) == "cat cat dog"


def test_semantic_search_ignores_non_markdown(tiny_corpus):
    semantic_search = load_task(__file__, "3-semantic_search.py", "semantic_search")
    assert semantic_search(tiny_corpus, "cat cat cat", embedder=keyword_embedder) != "cat cat cat cat cat"


def test_semantic_search_uses_cosine_not_dot_product(tmp_path):
    semantic_search = load_task(__file__, "3-semantic_search.py", "semantic_search")
    (tmp_path / "long.md").write_text("cat cat cat cat dog dog dog dog")   # [4, 4]: long but mixed
    (tmp_path / "short.md").write_text("cat")                              # [1, 0]: short but exact
    # dot product would pick long.md (4 > 1); cosine similarity picks short.md (1.0 > 0.71)
    assert semantic_search(tmp_path, "cat", embedder=keyword_embedder) == "cat"


def test_semantic_search_embeds_documents_once(tiny_corpus):
    semantic_search = load_task(__file__, "3-semantic_search.py", "semantic_search")
    calls = []

    def counting_embedder(texts):
        calls.append(len(texts))
        return keyword_embedder(texts)

    semantic_search(tiny_corpus, "fish", embedder=counting_embedder)
    assert sorted(calls) == [1, 3], "embed the 3 documents together in one batch, plus the query"


@pytest.mark.slow
def test_semantic_search_real_model():
    semantic_search = load_task(__file__, "3-semantic_search.py", "semantic_search")
    doc = semantic_search(corpus_dir(), "When are PLDs?")
    assert "Peer Learning Days" in doc


# ---------- Task 4: multi-reference QA ----------

def test_multi_reference_loop_with_stubs(monkeypatch, capsys, tiny_corpus):
    mod = load_task(__file__, "4-qa.py")
    searched = []

    def fake_search(corpus_path, sentence):
        searched.append((pathlib.Path(corpus_path), sentence))
        return "doc about " + sentence

    def fake_extract(question, reference):
        assert reference == "doc about " + question, "answer from the document search found"
        return None if "unknown" in question else question.upper()

    monkeypatch.setattr(mod, "semantic_search", fake_search)
    monkeypatch.setattr(mod, "extract_answer", fake_extract)
    feed_input(monkeypatch, ["what is a cat", "unknown thing", "quit"])
    mod.question_answer(tiny_corpus)
    assert printed_lines(capsys) == ["A: WHAT IS A CAT", SORRY, "A: Goodbye"]
    assert [s for _, s in searched] == ["what is a cat", "unknown thing"]
    assert all(p == tiny_corpus for p, _ in searched)


@pytest.mark.slow
def test_multi_reference_real_models(monkeypatch, capsys):
    mod = load_task(__file__, "4-qa.py")
    feed_input(monkeypatch, ["What does PLD stand for?", "goodbye"])
    mod.question_answer(str(corpus_dir()))
    lines = printed_lines(capsys)
    assert lines[-1] == "A: Goodbye"
    assert "peer learning" in lines[0].lower(), lines[0]
