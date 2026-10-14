"""Tests for 04-llms/03-rag_and_evals.

Fast tests use a stub bag-of-words embedder (defined below), so they run offline.
Tests marked slow download a real sentence-transformers model / a small chat model.
"""
import json
import pathlib
import re
import zlib

import numpy as np
import pytest

from helpers import load_task

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


class StubEmbedder:
    """Hashing bag-of-words: deterministic, offline, good enough for keyword-ish retrieval."""
    STOP = set("a an the of to in on and or is are was what which how does do for with by it its "
               "that this as at be from when into each".split())

    def __init__(self, dim=256):
        self.dim = dim
        self.calls = 0

    def encode(self, texts, batch_size=32, **kwargs):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w not in self.STOP:
                    out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        return out


def corpus():
    return json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))


def questions():
    return json.loads((DATA / "questions.json").read_text(encoding="utf-8"))


def build_index(chunk_size=200, overlap=50):
    chunk_documents = load_task(__file__, "0-chunking.py", "chunk_documents")
    VectorIndex = load_task(__file__, "3-vector_index.py", "VectorIndex")
    index = VectorIndex(StubEmbedder())
    index.add(chunk_documents(corpus(), chunk_size, overlap))
    return index


# ---------- 0. chunking ----------

def test_chunk_text_overlap():
    chunk_text = load_task(__file__, "0-chunking.py", "chunk_text")
    text = " ".join(f"w{i}" for i in range(10))
    assert chunk_text(text, chunk_size=4, overlap=1) == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert chunk_text(text, chunk_size=4, overlap=0) == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]


def test_chunk_text_edges():
    chunk_text = load_task(__file__, "0-chunking.py", "chunk_text")
    assert chunk_text("one two", chunk_size=5, overlap=2) == ["one two"]
    assert chunk_text("", chunk_size=5, overlap=2) == []
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=3, overlap=3)


def test_chunk_text_covers_every_word():
    chunk_text = load_task(__file__, "0-chunking.py", "chunk_text")
    words = [f"x{i}" for i in range(103)]
    chunks = chunk_text(" ".join(words), chunk_size=20, overlap=5)
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(words)
    assert all(len(c.split()) <= 20 for c in chunks)


def test_chunk_documents():
    chunk_documents = load_task(__file__, "0-chunking.py", "chunk_documents")
    docs = {"a": "one two three four five", "b": "six"}
    out = chunk_documents(docs, chunk_size=3, overlap=1)
    assert [c["id"] for c in out] == ["a#0", "a#1", "b#0"]
    assert [c["doc_id"] for c in out] == ["a", "a", "b"]
    assert out[1]["text"] == "three four five"


# ---------- 1. embeddings ----------

def test_embed_texts_normalized():
    embed_texts = load_task(__file__, "1-embed.py", "embed_texts")
    emb = StubEmbedder(dim=32)
    vecs = embed_texts(["gradient descent step", "dropout", "zzz qqq"], emb)
    assert vecs.shape == (3, 32)
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_embed_texts_zero_vector_safe():
    embed_texts = load_task(__file__, "1-embed.py", "embed_texts")
    vecs = embed_texts(["the the the"], StubEmbedder(dim=8))  # only stopwords -> zero vector
    assert np.all(np.isfinite(vecs))


@pytest.mark.slow
def test_real_embedder():
    mod = load_task(__file__, "1-embed.py")
    model = mod.load_embedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    v = mod.embed_texts(["A cat sat on the mat.", "A kitten rested on a rug.", "Stock prices fell."], model)
    assert v.shape == (3, 384)
    assert v[0] @ v[1] > v[0] @ v[2]


# ---------- 2. cosine search ----------

def test_cosine_similarity():
    cosine_similarity = load_task(__file__, "2-search.py", "cosine_similarity")
    m = np.array([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [1.0, 1.0]])
    s = cosine_similarity(np.array([2.0, 0.0]), m)
    assert s.shape == (4,)
    assert np.allclose(s, [1.0, 0.0, -1.0, 1 / np.sqrt(2)])
    s2 = cosine_similarity(np.array([[1.0, 0.0], [0.0, 1.0]]), m)
    assert s2.shape == (2, 4)
    assert np.allclose(s2[1], [0.0, 1.0, 0.0, 1 / np.sqrt(2)])


def test_top_k():
    top_k = load_task(__file__, "2-search.py", "top_k")
    m = np.array([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [1.0, 1.0]])
    idx, scores = top_k(np.array([1.0, 0.2]), m, k=2)
    assert list(idx) == [0, 3]
    assert scores[0] >= scores[1]
    idx, _ = top_k(np.array([1.0, 0.0]), m, k=10)
    assert len(idx) == 4


# ---------- 3. vector index ----------

def test_index_add_and_search():
    index = build_index()
    assert len(index) == 12
    results = index.search("Which LoRA matrix is initialized to zero?", k=3)
    assert len(results) == 3
    chunk, score = results[0]
    assert chunk["doc_id"] == "lora"
    assert isinstance(score, float)
    assert [s for _, s in results] == sorted([s for _, s in results], reverse=True)


def test_index_empty_and_incremental():
    VectorIndex = load_task(__file__, "3-vector_index.py", "VectorIndex")
    index = VectorIndex(StubEmbedder())
    assert len(index) == 0
    assert index.search("anything") == []
    index.add([{"id": "a#0", "doc_id": "a", "text": "cats purr"}])
    index.add([{"id": "b#0", "doc_id": "b", "text": "dogs bark"}])
    assert len(index) == 2
    assert index.search("dogs", k=1)[0][0]["id"] == "b#0"


def test_index_save_load(tmp_path):
    VectorIndex = load_task(__file__, "3-vector_index.py", "VectorIndex")
    index = build_index()
    index.save(str(tmp_path / "idx"))
    emb = StubEmbedder()
    loaded = VectorIndex.load(str(tmp_path / "idx"), emb)
    assert len(loaded) == len(index)
    assert emb.calls == 0  # loading must not re-embed the corpus
    q = "epsilon greedy exploration"
    assert [c["id"] for c, _ in loaded.search(q, 3)] == [c["id"] for c, _ in index.search(q, 3)]


# ---------- 4. retrieval metrics ----------

def test_recall_at_k():
    recall_at_k = load_task(__file__, "4-retrieval_metrics.py", "recall_at_k")
    retrieved = [["a", "b", "c"], ["x", "y", "z"]]
    relevant = [{"b"}, {"z", "q"}]
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx((1 + 0.5) / 2)


def test_mrr():
    mrr = load_task(__file__, "4-retrieval_metrics.py", "mrr")
    retrieved = [["a", "b", "c"], ["x", "y", "z"], ["m"]]
    relevant = [{"b"}, {"x", "z"}, {"nope"}]
    assert mrr(retrieved, relevant) == pytest.approx((1 / 2 + 1 + 0) / 3)


def test_evaluate_retriever():
    evaluate_retriever = load_task(__file__, "4-retrieval_metrics.py", "evaluate_retriever")
    index = build_index(chunk_size=40, overlap=10)  # several chunks per document
    res = evaluate_retriever(index, questions(), k=3)
    assert set(res) == {"recall@3", "mrr"}
    assert 0.75 <= res["recall@3"] <= 1.0
    assert 0.6 <= res["mrr"] <= 1.0


# ---------- 5. prompt assembly ----------

CHUNKS = [
    {"id": "lora#0", "doc_id": "lora", "text": "B starts at zero."},
    {"id": "pca#0", "doc_id": "pca", "text": "PCA keeps variance."},
]


def test_build_prompt():
    build_prompt = load_task(__file__, "5-prompt.py", "build_prompt")
    p = build_prompt("What starts at zero?", CHUNKS)
    assert "[1] (lora) B starts at zero." in p
    assert "[2] (pca) PCA keeps variance." in p
    assert p.index("[1]") < p.index("[2]") < p.index("Question: What starts at zero?")
    assert p.rstrip().endswith("Answer:")
    assert "I don't know" in p


def test_build_prompt_truncates():
    build_prompt = load_task(__file__, "5-prompt.py", "build_prompt")
    p = build_prompt("q", [{"id": "x#0", "doc_id": "x", "text": "abcdefghij"}], max_chars=4)
    assert "[1] (x) abcd" in p and "abcde" not in p


def test_extract_citations():
    extract_citations = load_task(__file__, "5-prompt.py", "extract_citations")
    assert extract_citations("B is zero [2]. Also [1][2] and [10].") == [1, 2, 10]
    assert extract_citations("no citations here") == []


# ---------- 6. generation ----------

def test_generate_answer_with_fake_llm():
    generate_answer = load_task(__file__, "6-generate.py", "generate_answer")
    index = build_index()
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "  Matrix B is initialized to zero [1]. See also [7].  "

    out = generate_answer("Which LoRA matrix is initialized to zero?", index, fake_llm, k=3)
    assert out["answer"] == "Matrix B is initialized to zero [1]. See also [7]."
    assert len(out["sources"]) == 3
    assert out["citations"] == ["lora"]  # [7] is out of range and ignored
    assert "Which LoRA matrix is initialized to zero?" in seen["prompt"]
    assert out["sources"][0]["text"] in seen["prompt"]


@pytest.mark.slow
def test_local_generator_end_to_end():
    mod = load_task(__file__, "6-generate.py")
    gen = mod.make_local_generator("HuggingFaceTB/SmolLM2-360M-Instruct", max_new_tokens=40, device="cpu")
    out = mod.generate_answer("Which LoRA matrix is initialized to zero?", build_index(), gen, k=2)
    assert isinstance(out["answer"], str) and out["answer"]


# ---------- 7. evals ----------

def test_normalize_answer():
    normalize_answer = load_task(__file__, "7-evals.py", "normalize_answer")
    assert normalize_answer("  The  Elbow method!! ") == "elbow method"
    assert normalize_answer("An apple, a day.") == "apple day"


def test_exact_match_and_f1():
    mod = load_task(__file__, "7-evals.py")
    assert mod.exact_match("The elbow method.", "elbow method") == 1.0
    assert mod.exact_match("elbow", "elbow method") == 0.0
    assert mod.exact_match("0.999", ["0.99", "0.999"]) == 1.0
    assert mod.token_f1("elbow", "elbow method") == pytest.approx(2 / 3)
    assert mod.token_f1("totally wrong", "elbow method") == 0.0
    assert mod.token_f1("the elbow method", "elbow method") == 1.0


def test_answer_contains():
    answer_contains = load_task(__file__, "7-evals.py", "answer_contains")
    assert answer_contains("I think it is the elbow method [1].", "elbow method") == 1.0
    assert answer_contains("No idea.", "elbow method") == 0.0


def test_run_eval():
    run_eval = load_task(__file__, "7-evals.py", "run_eval")
    qs = [{"question": "q1", "answer": "blue"}, {"question": "q2", "answer": "red car"}]
    answers = {"q1": "Blue.", "q2": "a red bike"}
    res = run_eval(qs, lambda q: answers[q])
    assert res["n"] == 2
    assert res["exact_match"] == pytest.approx(0.5)
    assert res["f1"] == pytest.approx((1 + 0.5) / 2)
    assert res["contains"] == pytest.approx(0.5)


# ---------- 8. LLM-as-judge (bonus) ----------

def test_judge_prompt_and_parse():
    mod = load_task(__file__, "8-llm_judge.py")
    p = mod.build_judge_prompt("Q?", "my answer", "the reference")
    for piece in ("Q?", "my answer", "the reference", "1", "5", "Score:"):
        assert piece in p
    assert mod.parse_score("Mostly right.\nScore: 4") == 4
    assert mod.parse_score("score = 2") == 2
    assert mod.parse_score("I'd give it a 3 out of 5... final: 3") == 3
    assert mod.parse_score("no number") is None
    assert mod.judge("Q?", "a", "r", lambda prompt: "Fine. Score: 5") == 5
