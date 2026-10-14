"""Tests for 04-llms/01-gpt_from_scratch."""
import math
import os
import urllib.request

import pytest
import torch

from helpers import load_task

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# ---------- 0. character tokenizer ----------

def test_char_tokenizer_vocab():
    CharTokenizer = load_task(__file__, "0-char_tokenizer.py", "CharTokenizer")
    tok = CharTokenizer("hello world")
    assert tok.vocab_size == 8
    # ids follow sorted character order: ' ' < 'd' < 'e' < ...
    assert tok.stoi[" "] == 0
    assert tok.stoi["w"] == 7
    assert tok.itos[0] == " "


def test_char_tokenizer_roundtrip():
    CharTokenizer = load_task(__file__, "0-char_tokenizer.py", "CharTokenizer")
    tok = CharTokenizer("hello world")
    ids = tok.encode("hello")
    assert ids == [3, 2, 4, 4, 5]
    assert tok.decode(ids) == "hello"
    assert tok.decode(torch.tensor(ids)) == "hello"


def test_char_tokenizer_unknown():
    CharTokenizer = load_task(__file__, "0-char_tokenizer.py", "CharTokenizer")
    tok = CharTokenizer("abc")
    with pytest.raises(KeyError):
        tok.encode("z")


# ---------- 1. BPE building blocks ----------

def test_get_stats():
    get_stats = load_task(__file__, "1-bpe.py", "get_stats")
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert get_stats([5]) == {}


def test_merge():
    merge = load_task(__file__, "1-bpe.py", "merge")
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    # non-overlapping, left to right
    assert merge([7, 7, 7], (7, 7), 256) == [256, 7]
    assert merge([1, 2], (3, 4), 9) == [1, 2]


# ---------- 2. BPE tokenizer ----------

def test_bpe_first_merge():
    BPETokenizer = load_task(__file__, "2-bpe_tokenizer.py", "BPETokenizer")
    tok = BPETokenizer()
    tok.train("aaabdaaabac", 257)
    assert tok.merges == {(97, 97): 256}
    assert tok.vocab[256] == b"aa"
    assert len(tok.vocab) == 257


def test_bpe_roundtrip_and_compression():
    BPETokenizer = load_task(__file__, "2-bpe_tokenizer.py", "BPETokenizer")
    text = "the cat sat on the mat. the cat ate the rat. " * 20
    tok = BPETokenizer()
    tok.train(text, 300)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert len(ids) < len(text.encode("utf-8")) / 3
    assert max(ids) < 300
    # unseen text (including non-ASCII) still round-trips
    other = "a brand new sentence, café ☕"
    assert tok.decode(tok.encode(other)) == other


def test_bpe_untrained_is_bytes():
    BPETokenizer = load_task(__file__, "2-bpe_tokenizer.py", "BPETokenizer")
    tok = BPETokenizer()
    assert tok.encode("hi") == [104, 105]
    assert tok.decode([104, 105]) == "hi"


def test_bpe_small_vocab_error():
    BPETokenizer = load_task(__file__, "2-bpe_tokenizer.py", "BPETokenizer")
    with pytest.raises(ValueError):
        BPETokenizer().train("abc", 100)


# ---------- 3. data ----------

def test_train_val_split():
    train_val_split = load_task(__file__, "3-data.py", "train_val_split")
    data = torch.arange(100)
    tr, va = train_val_split(data, 0.9)
    assert len(tr) == 90 and len(va) == 10
    assert torch.equal(torch.cat([tr, va]), data)


def test_get_batch():
    get_batch = load_task(__file__, "3-data.py", "get_batch")
    data = torch.arange(50)
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, block_size=8, batch_size=4, generator=g)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.long
    assert torch.equal(y, x + 1)  # targets are inputs shifted by one
    assert int(y.max()) <= 49
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    assert torch.equal(get_batch(data, 8, 4, g1)[0], get_batch(data, 8, 4, g2)[0])


# ---------- 4. bigram ----------

def test_bigram_shapes_and_loss():
    BigramLM = load_task(__file__, "4-bigram.py", "BigramLM")
    torch.manual_seed(0)
    m = BigramLM(10)
    idx = torch.randint(0, 10, (2, 5))
    logits, loss = m(idx)
    assert logits.shape == (2, 5, 10) and loss is None
    logits, loss = m(idx, idx)
    assert loss.ndim == 0 and loss.item() > 0


def test_bigram_generate():
    BigramLM = load_task(__file__, "4-bigram.py", "BigramLM")
    torch.manual_seed(0)
    m = BigramLM(10)
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), 7)
    assert out.shape == (1, 8)
    assert int(out[0, 0]) == 0
    assert int(out.max()) < 10


def test_bigram_learns():
    BigramLM = load_task(__file__, "4-bigram.py", "BigramLM")
    torch.manual_seed(0)
    m = BigramLM(4)
    # deterministic sequence 0->1->2->3->0...
    x = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, 0, 1, 2, 3, 0]])
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    for _ in range(200):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


# ---------- 5. causal self-attention ----------

def test_attention_shape():
    CSA = load_task(__file__, "5-attention.py", "CausalSelfAttention")
    torch.manual_seed(0)
    att = CSA(n_embd=16, n_head=4, block_size=8)
    x = torch.randn(2, 6, 16)
    assert att(x).shape == (2, 6, 16)


def test_attention_is_causal():
    CSA = load_task(__file__, "5-attention.py", "CausalSelfAttention")
    torch.manual_seed(0)
    att = CSA(n_embd=16, n_head=4, block_size=8).eval()
    x = torch.randn(1, 8, 16)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, 16)  # change the "future"
    y1, y2 = att(x), att(x2)
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6)
    assert not torch.allclose(y1[:, 5:], y2[:, 5:])


def test_attention_bad_heads():
    CSA = load_task(__file__, "5-attention.py", "CausalSelfAttention")
    with pytest.raises(ValueError):
        CSA(n_embd=10, n_head=3, block_size=8)


# ---------- 6. GPT ----------

def _gpt(V=20, block=8):
    GPT = load_task(__file__, "6-gpt.py", "GPT")
    torch.manual_seed(0)
    return GPT(vocab_size=V, block_size=block, n_embd=32, n_head=4, n_layer=2)


def test_block_shape():
    Block = load_task(__file__, "6-gpt.py", "Block")
    torch.manual_seed(0)
    b = Block(n_embd=16, n_head=2, block_size=8)
    assert b(torch.randn(3, 8, 16)).shape == (3, 8, 16)


def test_gpt_forward_and_initial_loss():
    m = _gpt()
    idx = torch.randint(0, 20, (4, 8))
    logits, loss = m(idx, idx)
    assert logits.shape == (4, 8, 20)
    # a fresh model should be close to uniform: loss ~ ln(vocab_size)
    assert abs(loss.item() - math.log(20)) < 0.3
    logits, loss = m(idx)
    assert loss is None


def test_gpt_too_long_raises():
    m = _gpt()
    with pytest.raises(ValueError):
        m(torch.zeros((1, 9), dtype=torch.long))


def test_gpt_causal():
    m = _gpt().eval()
    a = torch.randint(0, 20, (1, 8))
    b = a.clone()
    b[0, -1] = (a[0, -1] + 1) % 20
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)


def test_gpt_generate_crops_context():
    m = _gpt().eval()
    torch.manual_seed(0)
    out = m.generate(torch.zeros((2, 1), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (2, 21)  # longer than block_size, so context must be cropped
    assert int(out.max()) < 20


def test_gpt_top_k_1_is_greedy():
    m = _gpt().eval()
    start = torch.tensor([[3, 1, 4]])
    out = m.generate(start, max_new_tokens=5, top_k=1)
    # greedy decoding by hand
    idx = start.clone()
    with torch.no_grad():
        for _ in range(5):
            logits, _ = m(idx[:, -8:])
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, idx)


def test_gpt_low_temperature_is_near_greedy():
    m = _gpt().eval()
    start = torch.tensor([[3, 1, 4]])
    greedy = m.generate(start, max_new_tokens=5, top_k=1)
    torch.manual_seed(1)
    cold = m.generate(start, max_new_tokens=5, temperature=1e-4)
    assert torch.equal(greedy, cold)


def test_gpt_overfits_one_batch():
    m = _gpt()
    torch.manual_seed(0)
    x = torch.randint(0, 20, (4, 8))
    y = torch.randint(0, 20, (4, 8))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


# ---------- 7. training ----------

def test_estimate_loss_restores_mode():
    estimate_loss = load_task(__file__, "7-train.py", "estimate_loss")
    m = _gpt()
    m.train()
    data = torch.randint(0, 20, (200,))
    out = estimate_loss(m, {"train": data, "val": data}, block_size=8, batch_size=4, eval_iters=3)
    assert set(out) == {"train", "val"}
    assert all(isinstance(v, float) for v in out.values())
    assert m.training


def test_train_reduces_loss(tmp_path):
    mod = load_task(__file__, "7-train.py")
    GPT = load_task(__file__, "6-gpt.py", "GPT")
    CharTokenizer = load_task(__file__, "0-char_tokenizer.py", "CharTokenizer")
    text = "to be or not to be, that is the question. " * 40
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    torch.manual_seed(0)
    m = GPT(tok.vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2)
    hist = mod.train(m, data[:n], data[n:], block_size=16, batch_size=16, max_iters=200,
                     lr=3e-3, eval_interval=50, eval_iters=5, device="cpu", seed=0)
    assert hist["step"] == [0, 50, 100, 150, 200]
    assert len(hist["train"]) == len(hist["val"]) == 5
    assert hist["val"][-1] < 0.5 * hist["val"][0]
    path = tmp_path / "loss.png"
    mod.plot_history(hist, str(path))
    assert path.exists() and path.stat().st_size > 0


def test_get_device():
    get_device = load_task(__file__, "7-train.py", "get_device")
    assert get_device() in {"cpu", "mps", "cuda"}


@pytest.mark.slow
def test_shakespeare_char_gpt(tmp_path):
    mod = load_task(__file__, "7-train.py")
    GPT = load_task(__file__, "6-gpt.py", "GPT")
    CharTokenizer = load_task(__file__, "0-char_tokenizer.py", "CharTokenizer")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input.txt")
    if not os.path.exists(path):
        path = str(tmp_path / "input.txt")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    assert tok.vocab_size == 65
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    torch.manual_seed(0)
    m = GPT(tok.vocab_size, block_size=64, n_embd=64, n_head=4, n_layer=2)
    hist = mod.train(m, data[:n], data[n:], block_size=64, batch_size=32, max_iters=500,
                     lr=1e-3, eval_interval=100, eval_iters=10, device="cpu", seed=0)
    assert hist["val"][-1] < 2.7
