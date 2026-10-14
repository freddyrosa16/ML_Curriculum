"""Tests for 03-sequences-nlp/04-word_embeddings."""
import pathlib

import numpy as np
import pytest

from helpers import load_task, project_dir

SENTENCES = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The children are our future",
             "Our children's children are our grandchildren",
             "The cake was not very good",
             "No one said that the cake was not very good",
             "Life is beautiful"]

BOW_FEATURES = ['are', 'awesome', 'beautiful', 'cake', 'children', 'future', 'good',
                'grandchildren', 'holberton', 'is', 'learning', 'life', 'machine', 'nlp', 'no',
                'not', 'one', 'our', 'said', 'school', 'that', 'the', 'very', 'was']
BOW = [[0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
       [1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
       [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

VOCAB = ["awesome", "learning", "children", "cake", "good", "none", "machine"]
TFIDF = [[1, 0, 0, 0, 0, 0, 0],
         [0.5098139, 0.60831315, 0, 0, 0, 0, 0.60831315],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0.70710678, 0.70710678, 0, 0],
         [0, 0, 0, 0.70710678, 0.70710678, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]


def common_texts():
    gensim_utils = pytest.importorskip("gensim.test.utils")
    return gensim_utils.common_texts


# ---------- Task 0 ----------

def test_bag_of_words_full_vocab():
    bag_of_words = load_task(__file__, "0-bag_of_words.py", "bag_of_words")
    E, F = bag_of_words(SENTENCES)
    assert list(F) == BOW_FEATURES
    assert E.shape == (8, 24)
    assert np.array_equal(E, BOW)


def test_bag_of_words_given_vocab_keeps_order():
    bag_of_words = load_task(__file__, "0-bag_of_words.py", "bag_of_words")
    E, F = bag_of_words(SENTENCES, ["our", "children", "zebra"])
    assert list(F) == ["our", "children", "zebra"]
    assert np.array_equal(E[:, 0], [0, 0, 0, 1, 2, 0, 0, 0])
    assert np.array_equal(E[:, 1], [0, 0, 0, 1, 2, 0, 0, 0])
    assert not E[:, 2].any()


# ---------- Task 1 ----------

def test_tf_idf_with_vocab():
    tf_idf = load_task(__file__, "1-tf_idf.py", "tf_idf")
    E, F = tf_idf(SENTENCES, VOCAB)
    assert list(F) == VOCAB
    assert E.shape == (8, 7)
    assert np.allclose(E, TFIDF, atol=1e-6)


def test_tf_idf_rows_are_unit_length():
    tf_idf = load_task(__file__, "1-tf_idf.py", "tf_idf")
    E, F = tf_idf(SENTENCES)
    assert list(F) == BOW_FEATURES
    assert np.allclose(np.linalg.norm(E, axis=1), 1)
    # in sentence 0, the rarest words ('holberton', 'school') outweigh the common 'is'
    row = dict(zip(F, E[0]))
    assert row["holberton"] > row["awesome"] > row["is"] > 0


# ---------- Task 2 ----------

def test_word2vec_model_settings():
    texts = common_texts()
    gensim = pytest.importorskip("gensim")
    word2vec_model = load_task(__file__, "2-word2vec.py", "word2vec_model")
    model = word2vec_model(texts, vector_size=20, min_count=1, window=3, negative=4,
                           cbow=False, epochs=3)
    assert isinstance(model, gensim.models.Word2Vec)
    assert model.vector_size == 20 and model.window == 3 and model.negative == 4
    assert model.sg == 1, "cbow=False should mean skip-gram"
    assert model.epochs == 3
    assert model.wv["computer"].shape == (20,)
    assert len(model.wv) == 12


def test_word2vec_defaults_and_min_count():
    texts = common_texts()
    word2vec_model = load_task(__file__, "2-word2vec.py", "word2vec_model")
    model = word2vec_model(texts, min_count=1)
    assert model.vector_size == 100 and model.sg == 0
    # with min_count=3 only words that occur 3+ times survive
    small = word2vec_model(texts, min_count=3)
    assert set(small.wv.index_to_key) == {"system", "user", "trees", "graph"}


def test_word2vec_actually_trains():
    texts = common_texts()
    word2vec_model = load_task(__file__, "2-word2vec.py", "word2vec_model")
    model = word2vec_model(texts, min_count=1, vector_size=10, epochs=5)
    assert model.corpus_count == len(texts)
    assert model.train_count >= 1


# ---------- Task 3 ----------

def test_gensim_to_torch():
    torch = pytest.importorskip("torch")
    texts = common_texts()
    word2vec_model = load_task(__file__, "2-word2vec.py", "word2vec_model")
    gensim_to_torch = load_task(__file__, "3-gensim_to_torch.py", "gensim_to_torch")
    w2v = word2vec_model(texts, min_count=1, vector_size=16)
    emb = gensim_to_torch(w2v)
    assert isinstance(emb, torch.nn.Embedding)
    assert emb.weight.shape == (12, 16)
    assert emb.weight.requires_grad, "the layer should stay trainable"
    idx = w2v.wv.key_to_index["computer"]
    vec = emb(torch.tensor([idx]))[0].detach().numpy()
    assert np.allclose(vec, w2v.wv["computer"], atol=1e-6)


def test_gensim_to_torch_is_a_copy():
    torch = pytest.importorskip("torch")
    texts = common_texts()
    word2vec_model = load_task(__file__, "2-word2vec.py", "word2vec_model")
    gensim_to_torch = load_task(__file__, "3-gensim_to_torch.py", "gensim_to_torch")
    w2v = word2vec_model(texts, min_count=1, vector_size=8)
    before = w2v.wv.vectors.copy()
    emb = gensim_to_torch(w2v)
    with torch.no_grad():
        emb.weight.add_(1.0)
    assert np.allclose(w2v.wv.vectors, before), "training the layer must not change gensim's vectors"


# ---------- Task 4 ----------

def test_fasttext_model():
    texts = common_texts()
    gensim = pytest.importorskip("gensim")
    fasttext_model = load_task(__file__, "4-fasttext.py", "fasttext_model")
    model = fasttext_model(texts, vector_size=25, min_count=1, window=4, negative=3,
                           cbow=False, epochs=2)
    assert isinstance(model, gensim.models.FastText)
    assert model.vector_size == 25 and model.window == 4 and model.negative == 3
    assert model.sg == 1 and model.epochs == 2
    assert model.wv["computer"].shape == (25,)


def test_fasttext_handles_unseen_words():
    texts = common_texts()
    fasttext_model = load_task(__file__, "4-fasttext.py", "fasttext_model")
    model = fasttext_model(texts, min_count=1, vector_size=10)
    assert "computers" not in model.wv.key_to_index
    vec = model.wv["computers"]           # built from character n-grams
    assert vec.shape == (10,) and np.any(vec != 0)


# ---------- Task 5 ----------

def test_elmo_answer_written():
    path = pathlib.Path(project_dir(__file__)) / "5-elmo.md"
    if not path.exists():
        pytest.skip("5-elmo.md not written yet")
    words = path.read_text().split()
    assert len(words) >= 40, "write at least a few sentences"
