"""Tests for 04-llms/02-fine_tuning.

Fast tests build a tiny random Llama-style model and a character-level tokenizer
locally, so they need no network. Tests marked slow download SmolLM2-135M and Alpaca.
"""
import copy
import os
import string

import pytest
import torch
import torch.nn as nn

from helpers import load_task

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")


# ---------- offline fixtures ----------

def make_tokenizer():
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast
    specials = ["<unk>", "<pad>", "</s>"]
    chars = sorted(set(string.printable))
    vocab = {t: i for i, t in enumerate(specials + chars)}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tok.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                                   pad_token="<pad>", eos_token="</s>")


def make_tiny_model(vocab_size, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=256, tie_word_embeddings=False)
    return LlamaForCausalLM(cfg)


@pytest.fixture
def tokenizer():
    return make_tokenizer()


@pytest.fixture
def tiny(tokenizer):
    return make_tiny_model(len(tokenizer))


EXAMPLES = [
    {"instruction": "Say hi.", "input": "", "output": "hi"},
    {"instruction": "Add the numbers.", "input": "2 and 3", "output": "5"},
    {"instruction": "Name a color.", "input": "", "output": "blue"},
    {"instruction": "Repeat the word.", "input": "cat", "output": "cat cat"},
]


# ---------- 0. loading ----------

def test_count_parameters():
    count_parameters = load_task(__file__, "0-load_model.py", "count_parameters")
    lin = nn.Linear(10, 4)
    assert count_parameters(lin) == 44
    lin.bias.requires_grad_(False)
    assert count_parameters(lin, trainable_only=True) == 40
    assert count_parameters(lin) == 44


def test_get_device():
    get_device = load_task(__file__, "0-load_model.py", "get_device")
    assert get_device() in {"cpu", "mps", "cuda"}


@pytest.mark.slow
def test_load_smollm():
    mod = load_task(__file__, "0-load_model.py")
    model, tok = mod.load_model("HuggingFaceTB/SmolLM2-135M", device="cpu")
    assert tok.pad_token is not None
    n = mod.count_parameters(model)
    assert 120e6 < n < 150e6


# ---------- 1. formatting ----------

def test_format_with_input():
    format_example = load_task(__file__, "1-format.py", "format_example")
    prompt, response = format_example(EXAMPLES[1])
    assert prompt == "### Instruction:\nAdd the numbers.\n\n### Input:\n2 and 3\n\n### Response:\n"
    assert response == "5"


def test_format_without_input():
    format_example = load_task(__file__, "1-format.py", "format_example")
    prompt, response = format_example({"instruction": "  Say hi. ", "input": "  ", "output": " hi\n"})
    assert prompt == "### Instruction:\nSay hi.\n\n### Response:\n"
    assert response == "hi"
    # a missing "input" key is treated as empty
    prompt2, _ = format_example({"instruction": "Say hi.", "output": "hi"})
    assert prompt2 == prompt


# ---------- 2. tokenize + mask + collate ----------

def test_tokenize_masks_prompt(tokenizer):
    tokenize_example = load_task(__file__, "2-tokenize.py", "tokenize_example")
    ex = tokenize_example("Q: ", "ab", tokenizer)
    ids = ex["input_ids"]
    assert len(ids) == 3 + 2 + 1  # prompt chars + response chars + eos
    assert ids[-1] == tokenizer.eos_token_id
    assert ex["labels"][:3] == [-100, -100, -100]
    assert ex["labels"][3:] == ids[3:]
    assert ex["attention_mask"] == [1] * 6


def test_tokenize_truncates(tokenizer):
    tokenize_example = load_task(__file__, "2-tokenize.py", "tokenize_example")
    ex = tokenize_example("x" * 10, "y" * 10, tokenizer, max_length=12)
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"]) == 12
    assert ex["labels"][:10] == [-100] * 10


def test_collate_pads():
    collate = load_task(__file__, "2-tokenize.py", "collate")
    batch = [
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
        {"input_ids": [8], "attention_mask": [1], "labels": [8]},
    ]
    out = collate(batch, pad_token_id=0)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert all(t.dtype == torch.long for t in out.values())


# ---------- 3. LoRA by hand ----------

def test_lora_param_counts():
    LoRALinear = load_task(__file__, "3-lora_linear.py", "LoRALinear")
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(64, 32), r=4, alpha=8)
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
    assert trainable == 4 * (64 + 32)
    assert frozen == 64 * 32 + 32
    assert layer.lora_A.shape == (4, 64) and layer.lora_B.shape == (32, 4)
    assert layer.scaling == 2.0


def test_lora_starts_as_identity():
    LoRALinear = load_task(__file__, "3-lora_linear.py", "LoRALinear")
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    layer = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(5, 16)
    assert torch.allclose(layer(x), base(x))
    assert torch.count_nonzero(layer.lora_A) > 0  # A random, B zero


def test_lora_merge_matches():
    LoRALinear = load_task(__file__, "3-lora_linear.py", "LoRALinear")
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    W = base.weight.detach().clone()
    layer = LoRALinear(base, r=2, alpha=6)
    with torch.no_grad():
        layer.lora_B.normal_()
    expected = W + layer.lora_B @ layer.lora_A * 3.0
    assert torch.allclose(layer.merged_weight(), expected, atol=1e-6)
    merged = layer.merge()
    assert isinstance(merged, nn.Linear)
    x = torch.randn(3, 16)
    assert torch.allclose(merged(x), layer(x), atol=1e-5)
    assert torch.equal(base.weight, W)  # merging must not modify the frozen base


def test_lora_only_adapters_get_gradients():
    LoRALinear = load_task(__file__, "3-lora_linear.py", "LoRALinear")
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(4, 4), r=1)
    layer(torch.randn(2, 4)).sum().backward()
    assert layer.base.weight.grad is None
    assert layer.lora_B.grad is not None


def test_lora_bad_rank():
    LoRALinear = load_task(__file__, "3-lora_linear.py", "LoRALinear")
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), r=0)


# ---------- 4. peft ----------

def test_get_lora_model_counts(tiny):
    mod = load_task(__file__, "4-apply_lora.py")
    total_before = sum(p.numel() for p in tiny.parameters())
    model = mod.get_lora_model(tiny, r=4, alpha=8, target_modules=("q_proj", "v_proj"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 2 layers x 2 target modules x r * (in + out), with hidden size 32
    assert trainable == 2 * 2 * 4 * (32 + 32)
    frac = mod.trainable_fraction(model)
    assert frac == pytest.approx(trainable / (total_before + trainable))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in names)


# ---------- 5. training ----------

def _dataset(tokenizer):
    fmt = load_task(__file__, "1-format.py", "format_example")
    tok = load_task(__file__, "2-tokenize.py", "tokenize_example")
    return [tok(*fmt(ex), tokenizer, max_length=64) for ex in EXAMPLES]


def test_train_lora_reduces_loss(tokenizer, tiny):
    get_lora_model = load_task(__file__, "4-apply_lora.py", "get_lora_model")
    train_lora = load_task(__file__, "5-train.py", "train_lora")
    data = _dataset(tokenizer)
    model = get_lora_model(tiny, r=8, alpha=16, dropout=0.0,
                           target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    losses = train_lora(model, data, tokenizer.pad_token_id, steps=60, batch_size=4,
                        lr=1e-2, device="cpu", seed=0)
    assert len(losses) == 60
    # the frozen random embeddings limit how low a tiny model can go, but it must improve
    assert sum(losses[-10:]) / 10 < 0.92 * sum(losses[:10]) / 10
    for n, p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), f"{n} changed but should be frozen"


# ---------- 6. generate + save/load ----------

def test_generate_returns_only_new_text(tokenizer, tiny):
    generate = load_task(__file__, "6-generate.py", "generate")
    a = generate(tiny, tokenizer, "Hello", max_new_tokens=5)
    b = generate(tiny, tokenizer, "Hello", max_new_tokens=5)
    assert isinstance(a, str)
    assert a == b  # greedy decoding is deterministic
    assert len(a) <= 5  # char-level tokenizer: at most one char per new token
    assert "Hello" not in a or len(a) < 5


def test_save_and_load_adapter(tmp_path, tokenizer):
    get_lora_model = load_task(__file__, "4-apply_lora.py", "get_lora_model")
    mod = load_task(__file__, "6-generate.py")
    base = make_tiny_model(len(tokenizer), seed=0)
    fresh_base = copy.deepcopy(base)
    model = get_lora_model(base, r=4, alpha=8)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_()
    model.eval()
    path = tmp_path / "adapter"
    mod.save_adapter(model, str(path))
    files = os.listdir(path)
    assert "adapter_config.json" in files
    weights = [f for f in files if f.startswith("adapter_model")]
    assert weights, "adapter weights were not saved"
    loaded = mod.load_adapter(fresh_base, str(path))
    loaded.eval()
    ids = torch.tensor([tokenizer("abc")["input_ids"]])
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, loaded(ids).logits, atol=1e-5)
        assert not torch.allclose(loaded(ids).logits, make_tiny_model(len(tokenizer))(ids).logits)


# ---------- slow: the real thing ----------

@pytest.mark.slow
def test_smollm_lora_pipeline(tmp_path):
    datasets = pytest.importorskip("datasets")
    load = load_task(__file__, "0-load_model.py")
    fmt = load_task(__file__, "1-format.py", "format_example")
    tok_ex = load_task(__file__, "2-tokenize.py", "tokenize_example")
    get_lora_model = load_task(__file__, "4-apply_lora.py", "get_lora_model")
    train_lora = load_task(__file__, "5-train.py", "train_lora")
    gen = load_task(__file__, "6-generate.py")

    model, tok = load.load_model("HuggingFaceTB/SmolLM2-135M", device="cpu")
    ds = datasets.load_dataset("yahma/alpaca-cleaned", split="train[:64]")
    data = [tok_ex(*fmt(ex), tok, max_length=128) for ex in ds]
    model = get_lora_model(model, r=8, alpha=16)
    losses = train_lora(model, data, tok.pad_token_id, steps=30, batch_size=4, lr=5e-4,
                        device="cpu", seed=0)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    prompt, _ = fmt({"instruction": "Name three primary colors.", "input": "", "output": ""})
    assert isinstance(gen.generate(model, tok, prompt, max_new_tokens=20), str)
    gen.save_adapter(model, str(tmp_path / "adapter"))
    base, _ = load.load_model("HuggingFaceTB/SmolLM2-135M", device="cpu")
    loaded = gen.load_adapter(base, str(tmp_path / "adapter"))
    assert sum(p.numel() for n, p in loaded.named_parameters() if "lora_" in n) > 0
