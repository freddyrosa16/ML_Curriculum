"""Tests for 03-sequences-nlp/06-transformer_apps.

Fast tests use a tiny fake tokenizer, so nothing is downloaded. Tests that download the
dataset/tokenizers or train are marked slow.
"""
import numpy as np
import pytest

from helpers import load_task

torch = pytest.importorskip("torch")
F = torch.nn.functional


class FakeTokenizer:
    """Maps each word to a fixed id. Mimics the bits of a Hugging Face tokenizer we use."""

    def __init__(self, words):
        self.ids = {w: i for i, w in enumerate(words, start=5)}   # 0-4 reserved for specials
        self.vocab_size = 5 + len(words)

    def encode(self, text, add_special_tokens=True):
        ids = [self.ids[w] for w in text.split()]
        return ([2] + ids + [3]) if add_special_tokens else ids


def bare_dataset(filename, **attrs):
    """Make a Dataset object without running __init__ (which would download data)."""
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    Dataset = load_task(__file__, filename, "Dataset")
    ds = Dataset.__new__(Dataset)
    ds.tokenizer_pt = FakeTokenizer(["olá", "mundo", "bom", "dia"])     # vocab_size 9
    ds.tokenizer_en = FakeTokenizer(["hello", "world", "good"])         # vocab_size 8
    for k, v in attrs.items():
        setattr(ds, k, v)
    return ds


# ---------- Task 1: encode ----------

@pytest.mark.parametrize("filename", ["1-dataset.py", "2-dataset.py", "3-dataset.py"])
def test_encode_adds_start_and_end_tokens(filename):
    ds = bare_dataset(filename)
    pt, en = ds.encode("olá mundo", "hello world")
    assert isinstance(pt, np.ndarray) and isinstance(en, np.ndarray)
    assert pt.tolist() == [9, 5, 6, 10]        # start = vocab_size, end = vocab_size + 1
    assert en.tolist() == [8, 5, 6, 9]


def test_encode_does_not_use_tokenizer_specials():
    ds = bare_dataset("1-dataset.py")
    pt, en = ds.encode("bom dia", "good")
    assert 2 not in pt.tolist() and 3 not in en.tolist(), "use add_special_tokens=False"
    assert en.tolist() == [8, 7, 9]


# ---------- Task 2: pad a batch ----------

@pytest.mark.parametrize("filename", ["2-dataset.py", "3-dataset.py"])
def test_pad_batch(filename):
    ds = bare_dataset(filename)
    batch = [(np.array([9, 5, 10]), np.array([8, 5, 6, 7, 9])),
             (np.array([9, 5, 6, 7, 8, 10]), np.array([8, 9]))]
    pt, en = ds.pad_batch(batch)
    assert pt.dtype == torch.long and en.dtype == torch.long
    assert pt.tolist() == [[9, 5, 10, 0, 0, 0], [9, 5, 6, 7, 8, 10]]
    assert en.tolist() == [[8, 5, 6, 7, 9], [8, 9, 0, 0, 0]]


# ---------- Task 3: filter ----------

def test_filter_max_len():
    ds = bare_dataset("3-dataset.py", max_len=4)
    assert ds.filter_max_len(np.arange(4), np.arange(3))
    assert not ds.filter_max_len(np.arange(5), np.arange(3))
    assert not ds.filter_max_len(np.arange(2), np.arange(5))


# ---------- Task 4: masks ----------

def test_create_masks_shapes_and_values():
    create_masks = load_task(__file__, "4-create_masks.py", "create_masks")
    inputs = torch.tensor([[5, 6, 7, 0], [5, 0, 0, 0]])
    target = torch.tensor([[8, 9, 0], [8, 9, 10]])
    enc_mask, combined, dec_mask = create_masks(inputs, target)
    assert enc_mask.shape == (2, 1, 1, 4)
    assert dec_mask.shape == (2, 1, 1, 4)
    assert combined.shape == (2, 1, 3, 3)
    assert enc_mask[:, 0, 0].tolist() == [[0, 0, 0, 1], [0, 1, 1, 1]]
    assert torch.equal(enc_mask, dec_mask)
    # example 0: future positions AND the padded last position are hidden
    assert combined[0, 0].tolist() == [[0, 1, 1], [0, 0, 1], [0, 0, 1]]
    # example 1: no padding, so it's just the look-ahead mask
    assert combined[1, 0].tolist() == [[0, 1, 1], [0, 0, 1], [0, 0, 0]]


def test_create_masks_work_with_attention():
    create_masks = load_task(__file__, "4-create_masks.py", "create_masks")
    inputs = torch.tensor([[5, 6, 0]])
    target = torch.tensor([[8, 9]])
    enc_mask, combined, _ = create_masks(inputs, target)
    scores = torch.zeros(1, 2, 2, 2).masked_fill(combined.bool(), -1e9)
    w = torch.softmax(scores, -1)
    assert torch.allclose(w[0, 0, 0], torch.tensor([1.0, 0.0]))
    assert enc_mask.dtype.is_floating_point or enc_mask.dtype == torch.bool


# ---------- Task 5: the Transformer from project 05 ----------

def load_transformer():
    try:
        return load_task(__file__, "5-transformer.py", "Transformer")
    except (ImportError, AttributeError) as e:
        pytest.skip(f"5-transformer.py couldn't load your Transformer (finish 05-attention?): {e}")


def test_transformer_available():
    Transformer = load_transformer()
    torch.manual_seed(0)
    model = Transformer(1, 16, 2, 32, 30, 40, 12, 12)
    out = model(torch.randint(1, 30, (2, 7)), torch.randint(1, 40, (2, 5)))
    assert out.shape == (2, 5, 40)


def test_lr_schedule():
    lr_schedule = load_task(__file__, "5-train.py", "lr_schedule")
    # warm-up: grows linearly; peak at warmup_steps; then decays like 1/sqrt(step)
    assert lr_schedule(1, 512) == pytest.approx(512 ** -0.5 * 4000 ** -1.5)
    assert lr_schedule(4000, 512) == pytest.approx(512 ** -0.5 * 4000 ** -0.5)
    assert lr_schedule(16000, 512) == pytest.approx(lr_schedule(4000, 512) / 2)
    assert lr_schedule(100, 128, warmup_steps=100) == pytest.approx(128 ** -0.5 * 0.1)
    assert lr_schedule(2000, 512) < lr_schedule(4000, 512) > lr_schedule(8000, 512)


def test_masked_loss_ignores_padding():
    masked_loss = load_task(__file__, "5-train.py", "masked_loss")
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 10)
    real = torch.tensor([[3, 4, 0, 0], [5, 6, 7, 0]])
    expected = F.cross_entropy(pred.reshape(-1, 10), real.reshape(-1), ignore_index=0)
    loss = masked_loss(real, pred)
    assert loss.shape == ()
    assert torch.allclose(loss, expected, atol=1e-6)
    # changing predictions at padded positions must not change the loss
    pred2 = pred.clone()
    pred2[0, 2:] = torch.randn(2, 10)
    assert torch.allclose(masked_loss(real, pred2), loss, atol=1e-6)


def test_masked_accuracy():
    masked_accuracy = load_task(__file__, "5-train.py", "masked_accuracy")
    real = torch.tensor([[1, 2, 0], [3, 0, 0]])
    pred = torch.zeros(2, 3, 5)
    pred[0, 0, 1] = 1     # correct
    pred[0, 1, 4] = 1     # wrong
    pred[1, 0, 3] = 1     # correct
    pred[0, 2, 0] = 1     # padding: predicting 0 there must not count
    acc = masked_accuracy(real, pred)
    assert float(acc) == pytest.approx(2 / 3)


# ---------- Slow: real data, tokenizers and training ----------

@pytest.mark.slow
def test_dataset_real_data():
    Dataset = load_task(__file__, "3-dataset.py", "Dataset")
    data = Dataset(32, 40, n_train=2000)
    assert data.tokenizer_pt.vocab_size <= 2 ** 13
    assert data.tokenizer_en.vocab_size <= 2 ** 13
    pt, en = next(iter(data.data_train))
    assert pt.shape[0] == 32 and pt.shape[1] <= 40 and en.shape[1] <= 40
    assert (pt[:, 0] == data.tokenizer_pt.vocab_size).all()


@pytest.mark.slow
def test_train_and_translate(capsys):
    train_transformer = load_task(__file__, "5-train.py", "train_transformer")
    torch.manual_seed(0)
    model = train_transformer(1, 64, 2, 128, 30, 64, 1, n_train=3000)
    printed = capsys.readouterr().out
    assert "Epoch 1, Batch 0: Loss" in printed and "Epoch 1: Loss" in printed
    assert isinstance(model, torch.nn.Module)
    translate = load_task(__file__, "6-translate.py", "translate")
    data = getattr(model, "data", None)
    if data is None or not hasattr(data, "encode"):
        pytest.skip("attach the Dataset to the model as model.data to test translate")
    assert isinstance(translate(model, data, "obrigado", max_len=30), str)
