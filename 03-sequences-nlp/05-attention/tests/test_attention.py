"""Tests for 03-sequences-nlp/05-attention.

Checks shapes, masking behaviour, attention weights summing to 1, and (where PyTorch has a
built-in equivalent) agreement with PyTorch's own implementation.
"""
import numpy as np
import pytest

from helpers import load_task

torch = pytest.importorskip("torch")
F = torch.nn.functional


def seeded(seed=0):
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


# ---------- Task 0: RNN encoder ----------

def test_rnn_encoder_shapes():
    RNNEncoder = load_task(__file__, "0-rnn_encoder.py", "RNNEncoder")
    seeded()
    enc = RNNEncoder(vocab=100, embedding=16, units=32, batch=4)
    assert isinstance(enc, torch.nn.Module)
    assert enc.batch == 4 and enc.units == 32
    assert isinstance(enc.embedding, torch.nn.Embedding)
    assert isinstance(enc.gru, torch.nn.GRU)
    h0 = enc.initialize_hidden_state()
    assert h0.shape == (4, 32) and not h0.any()
    x = torch.randint(0, 100, (4, 7))
    outputs, hidden = enc(x, h0)
    assert outputs.shape == (4, 7, 32)
    assert hidden.shape == (4, 32)
    assert torch.allclose(outputs[:, -1], hidden, atol=1e-6), \
        "the last output of a one-layer GRU is its final hidden state"


def test_rnn_encoder_uses_initial_state():
    RNNEncoder = load_task(__file__, "0-rnn_encoder.py", "RNNEncoder")
    seeded()
    enc = RNNEncoder(50, 8, 16, 2)
    x = torch.randint(0, 50, (2, 5))
    out_a, _ = enc(x, torch.zeros(2, 16))
    out_b, _ = enc(x, torch.ones(2, 16))
    assert not torch.allclose(out_a, out_b)


# ---------- Task 1: additive attention ----------

def test_self_attention_shapes_and_weights():
    SelfAttention = load_task(__file__, "1-self_attention.py", "SelfAttention")
    seeded()
    att = SelfAttention(32)
    for name in ["W", "U", "V"]:
        assert isinstance(getattr(att, name), torch.nn.Linear), name
    assert att.V.out_features == 1
    s_prev, hidden = torch.randn(4, 32), torch.randn(4, 9, 32)
    context, weights = att(s_prev, hidden)
    assert context.shape == (4, 32)
    assert weights.shape == (4, 9, 1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4, 1), atol=1e-5)
    assert (weights >= 0).all()
    assert torch.allclose(context, (weights * hidden).sum(dim=1), atol=1e-5), \
        "the context is the attention-weighted sum of the hidden states"


def test_self_attention_formula():
    SelfAttention = load_task(__file__, "1-self_attention.py", "SelfAttention")
    seeded(1)
    att = SelfAttention(3)
    with torch.no_grad():
        att.W.weight.copy_(torch.eye(3)); att.W.bias.zero_()
        att.U.weight.copy_(torch.eye(3)); att.U.bias.zero_()
        att.V.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]])); att.V.bias.zero_()
    s_prev = torch.zeros(1, 3)
    hidden = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]])
    _, weights = att(s_prev, hidden)
    # scores are tanh(0), tanh(1), tanh(2) -> softmax
    expected = torch.softmax(torch.tanh(torch.tensor([0.0, 1.0, 2.0])), dim=0)
    assert torch.allclose(weights[0, :, 0], expected, atol=1e-6)


# ---------- Task 2: RNN decoder ----------

def test_rnn_decoder_shapes():
    RNNDecoder = load_task(__file__, "2-rnn_decoder.py", "RNNDecoder")
    seeded()
    dec = RNNDecoder(vocab=200, embedding=16, units=32, batch=4)
    assert isinstance(dec.F, torch.nn.Linear) and dec.F.out_features == 200
    assert isinstance(dec.gru, torch.nn.GRU)
    x = torch.randint(0, 200, (4, 1))
    y, s = dec(x, torch.randn(4, 32), torch.randn(4, 6, 32))
    assert y.shape == (4, 200)
    assert s.shape == (4, 32)


def test_rnn_decoder_depends_on_encoder_states():
    RNNDecoder = load_task(__file__, "2-rnn_decoder.py", "RNNDecoder")
    seeded()
    dec = RNNDecoder(50, 8, 16, 2)
    x, s_prev = torch.randint(0, 50, (2, 1)), torch.randn(2, 16)
    y1, _ = dec(x, s_prev, torch.randn(2, 5, 16))
    y2, _ = dec(x, s_prev, torch.randn(2, 5, 16))
    assert not torch.allclose(y1, y2), "attention context must feed into the GRU"


def test_rnn_decoder_uses_s_prev_as_gru_state():
    RNNDecoder = load_task(__file__, "2-rnn_decoder.py", "RNNDecoder")
    seeded()
    dec = RNNDecoder(50, 8, 16, 2)
    x, hidden = torch.randint(0, 50, (2, 1)), torch.randn(2, 5, 16)
    s_prev = torch.randn(2, 16)
    # Freeze the attention so s_prev can only matter through the GRU's initial state.
    with torch.no_grad():
        dec.attention.W.weight.zero_(); dec.attention.W.bias.zero_()
    _, s1 = dec(x, s_prev, hidden)
    _, s2 = dec(x, s_prev * 3, hidden)
    assert not torch.allclose(s1, s2)


# ---------- Task 4: positional encoding (NumPy) ----------

PE_4_6 = [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
          [0.84147098, 0.54030231, 0.04639922, 0.99892298, 0.00215443, 0.99999768],
          [0.90929743, -0.41614684, 0.0926985, 0.99569422, 0.00430886, 0.99999072],
          [0.14112001, -0.9899925, 0.1387981, 0.9903207, 0.00646326, 0.99997911]]


def test_positional_encoding_values():
    positional_encoding = load_task(__file__, "4-positional_encoding.py", "positional_encoding")
    PE = positional_encoding(4, 6)
    assert isinstance(PE, np.ndarray)
    assert PE.shape == (4, 6)
    assert np.allclose(PE, PE_4_6, atol=1e-6)


def test_positional_encoding_big():
    positional_encoding = load_task(__file__, "4-positional_encoding.py", "positional_encoding")
    PE = positional_encoding(30, 512)
    assert PE.shape == (30, 512)
    assert np.all(np.abs(PE) <= 1)
    assert np.allclose(PE[0, 0::2], 0) and np.allclose(PE[0, 1::2], 1)
    assert np.allclose(PE[:, 0::2] ** 2 + PE[:, 1::2] ** 2, 1), \
        "each (sin, cos) pair shares one angle"


# ---------- Task 5: scaled dot-product attention ----------

def test_sdp_attention_matches_pytorch():
    sdp_attention = load_task(__file__, "5-sdp_attention.py", "sdp_attention")
    seeded()
    Q, K, V = torch.randn(2, 5, 8), torch.randn(2, 7, 8), torch.randn(2, 7, 4)
    out, w = sdp_attention(Q, K, V)
    assert out.shape == (2, 5, 4) and w.shape == (2, 5, 7)
    assert torch.allclose(w.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert torch.allclose(out, F.scaled_dot_product_attention(Q, K, V), atol=1e-5)


def test_sdp_attention_masking():
    sdp_attention = load_task(__file__, "5-sdp_attention.py", "sdp_attention")
    seeded()
    Q, K, V = torch.randn(3, 4, 8), torch.randn(3, 6, 8), torch.randn(3, 6, 8)
    mask = torch.zeros(3, 1, 6)
    mask[:, :, 4:] = 1                      # hide the last two keys
    out, w = sdp_attention(Q, K, V, mask)
    assert torch.allclose(w[..., 4:], torch.zeros(3, 4, 2), atol=1e-6)
    assert torch.allclose(w.sum(-1), torch.ones(3, 4), atol=1e-5)
    expected = F.scaled_dot_product_attention(Q, K, V, attn_mask=(mask == 0))
    assert torch.allclose(out, expected, atol=1e-5)


def test_sdp_attention_look_ahead_mask():
    sdp_attention = load_task(__file__, "5-sdp_attention.py", "sdp_attention")
    seeded()
    Q = K = V = torch.randn(1, 5, 4)
    look_ahead = torch.triu(torch.ones(5, 5), diagonal=1)   # 1 above the diagonal = future
    _, w = sdp_attention(Q, K, V, look_ahead)
    assert torch.allclose(w[0].triu(diagonal=1), torch.zeros(5, 5), atol=1e-6)
    assert torch.allclose(w[0, 0], torch.tensor([1.0, 0, 0, 0, 0]), atol=1e-6)


def test_sdp_attention_scaling():
    sdp_attention = load_task(__file__, "5-sdp_attention.py", "sdp_attention")
    Q = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    K = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    V = torch.tensor([[[1.0], [0.0]]])
    _, w = sdp_attention(Q, K, V)
    # raw scores 4 and 0, divided by sqrt(4) = 2 -> softmax([2, 0])
    assert torch.allclose(w[0, 0], torch.softmax(torch.tensor([2.0, 0.0]), 0), atol=1e-6)


# ---------- Task 6: multi-head attention ----------

def _copy_into_torch_mha(mha, dm, h):
    ref = torch.nn.MultiheadAttention(dm, h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.Wq.weight, mha.Wk.weight, mha.Wv.weight]))
        ref.in_proj_bias.copy_(torch.cat([mha.Wq.bias, mha.Wk.bias, mha.Wv.bias]))
        ref.out_proj.weight.copy_(mha.linear.weight)
        ref.out_proj.bias.copy_(mha.linear.bias)
    return ref.eval()


def test_multihead_attention_shapes():
    MHA = load_task(__file__, "6-multihead_attention.py", "MultiHeadAttention")
    seeded()
    mha = MHA(32, 4)
    assert mha.h == 4 and mha.dm == 32 and mha.depth == 8
    for name in ["Wq", "Wk", "Wv", "linear"]:
        layer = getattr(mha, name)
        assert isinstance(layer, torch.nn.Linear) and layer.out_features == 32, name
    Q, KV = torch.randn(2, 5, 32), torch.randn(2, 7, 32)
    out, w = mha(Q, KV, KV, None)
    assert out.shape == (2, 5, 32)
    assert w.shape == (2, 4, 5, 7)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 5), atol=1e-5)


def test_multihead_attention_matches_pytorch():
    MHA = load_task(__file__, "6-multihead_attention.py", "MultiHeadAttention")
    seeded()
    mha = MHA(16, 4)
    ref = _copy_into_torch_mha(mha, 16, 4)
    Q, KV = torch.randn(3, 5, 16), torch.randn(3, 6, 16)
    out, w = mha(Q, KV, KV, None)
    ref_out, ref_w = ref(Q, KV, KV, average_attn_weights=False)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(w, ref_w, atol=1e-5)


def test_multihead_attention_padding_mask_matches_pytorch():
    MHA = load_task(__file__, "6-multihead_attention.py", "MultiHeadAttention")
    seeded(3)
    mha = MHA(16, 2)
    ref = _copy_into_torch_mha(mha, 16, 2)
    Q, KV = torch.randn(2, 4, 16), torch.randn(2, 6, 16)
    pad = torch.zeros(2, 6, dtype=torch.bool)
    pad[0, 5:] = True
    pad[1, 3:] = True
    out, w = mha(Q, KV, KV, pad[:, None, None, :].float())
    ref_out, _ = ref(Q, KV, KV, key_padding_mask=pad)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(w[1, :, :, 3:], torch.zeros(2, 4, 3), atol=1e-6)


# ---------- Tasks 7-8: encoder / decoder blocks ----------

def test_encoder_block_shape_and_layernorm():
    EncoderBlock = load_task(__file__, "7-transformer_encoder_block.py", "EncoderBlock")
    seeded()
    block = EncoderBlock(32, 4, 64).eval()
    for name in ["mha", "dense_hidden", "dense_output", "layernorm1", "layernorm2",
                 "dropout1", "dropout2"]:
        assert hasattr(block, name), name
    out = block(torch.randn(2, 6, 32), None)
    assert out.shape == (2, 6, 32)
    # The block ends with a fresh LayerNorm, so every position is normalized.
    assert torch.allclose(out.mean(-1), torch.zeros(2, 6), atol=1e-5)
    assert torch.allclose(out.std(-1, unbiased=False), torch.ones(2, 6), atol=1e-3)


def test_encoder_block_dropout_only_in_train_mode():
    EncoderBlock = load_task(__file__, "7-transformer_encoder_block.py", "EncoderBlock")
    seeded()
    block = EncoderBlock(32, 4, 64, drop_rate=0.5)
    x = torch.randn(2, 6, 32)
    block.eval()
    assert torch.allclose(block(x), block(x))
    block.train()
    assert not torch.allclose(block(x), block(x))


def test_encoder_block_padding_mask_hides_padded_tokens():
    EncoderBlock = load_task(__file__, "7-transformer_encoder_block.py", "EncoderBlock")
    seeded()
    block = EncoderBlock(16, 2, 32).eval()
    x = torch.randn(1, 5, 16)
    mask = torch.tensor([0, 0, 0, 1, 1.0])[None, None, None, :]   # last two are padding
    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 2, 16)                             # change only the padding
    out1, out2 = block(x, mask), block(x2, mask)
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5)


def test_decoder_block_shape_and_attributes():
    DecoderBlock = load_task(__file__, "8-transformer_decoder_block.py", "DecoderBlock")
    seeded()
    block = DecoderBlock(32, 4, 64).eval()
    for name in ["mha1", "mha2", "dense_hidden", "dense_output", "layernorm1", "layernorm2",
                 "layernorm3", "dropout1", "dropout2", "dropout3"]:
        assert hasattr(block, name), name
    out = block(torch.randn(2, 5, 32), torch.randn(2, 7, 32), None, None)
    assert out.shape == (2, 5, 32)


def test_decoder_block_is_causal_with_look_ahead_mask():
    DecoderBlock = load_task(__file__, "8-transformer_decoder_block.py", "DecoderBlock")
    seeded()
    block = DecoderBlock(16, 2, 32).eval()
    enc = torch.randn(1, 4, 16)
    look_ahead = torch.triu(torch.ones(5, 5), diagonal=1)
    x = torch.randn(1, 5, 16)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 2, 16)          # change the future
    out1, out2 = block(x, enc, look_ahead, None), block(x2, enc, look_ahead, None)
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5), \
        "positions 0-2 must not see positions 3-4"
    assert not torch.allclose(out1[:, 3:], out2[:, 3:])


def test_decoder_block_padding_mask_on_encoder_output():
    DecoderBlock = load_task(__file__, "8-transformer_decoder_block.py", "DecoderBlock")
    seeded()
    block = DecoderBlock(16, 2, 32).eval()
    x = torch.randn(1, 3, 16)
    enc = torch.randn(1, 6, 16)
    enc2 = enc.clone()
    enc2[:, 4:] = torch.randn(1, 2, 16)
    pad = torch.tensor([0, 0, 0, 0, 1, 1.0])[None, None, None, :]
    assert torch.allclose(block(x, enc, None, pad), block(x, enc2, None, pad), atol=1e-5)


# ---------- Tasks 9-10: encoder / decoder ----------

def test_encoder_shapes_and_blocks():
    Encoder = load_task(__file__, "9-transformer_encoder.py", "Encoder")
    seeded()
    enc = Encoder(2, 32, 4, 64, input_vocab=100, max_seq_len=20).eval()
    assert enc.N == 2 and enc.dm == 32
    assert isinstance(enc.blocks, torch.nn.ModuleList) and len(enc.blocks) == 2
    assert tuple(enc.positional_encoding.shape) == (20, 32)
    out = enc(torch.randint(0, 100, (3, 9)), None)
    assert out.shape == (3, 9, 32)


def test_encoder_input_pipeline():
    Encoder = load_task(__file__, "9-transformer_encoder.py", "Encoder")
    positional_encoding = load_task(__file__, "4-positional_encoding.py", "positional_encoding")
    seeded()
    enc = Encoder(0, 8, 2, 16, input_vocab=30, max_seq_len=10).eval()   # zero blocks
    x = torch.randint(0, 30, (2, 6))
    out = enc(x)
    PE = torch.tensor(positional_encoding(10, 8), dtype=torch.float32)
    expected = enc.embedding(x) * np.sqrt(8) + PE[:6]
    assert torch.allclose(out, expected, atol=1e-5), \
        "embed, scale by sqrt(dm), add the positional encoding"


def test_encoder_parameters_are_registered():
    Encoder = load_task(__file__, "9-transformer_encoder.py", "Encoder")
    enc = Encoder(2, 16, 2, 32, 50, 10)
    block_params = sum(p.numel() for p in enc.blocks.parameters())
    assert block_params > 0 and sum(p.numel() for p in enc.parameters()) > block_params


def test_decoder_shapes_and_causality():
    Decoder = load_task(__file__, "10-transformer_decoder.py", "Decoder")
    seeded()
    dec = Decoder(2, 16, 2, 32, target_vocab=60, max_seq_len=12).eval()
    assert dec.N == 2 and len(dec.blocks) == 2
    enc_out = torch.randn(2, 7, 16)
    x = torch.randint(0, 60, (2, 5))
    look_ahead = torch.triu(torch.ones(5, 5), diagonal=1)
    out = dec(x, enc_out, look_ahead, None)
    assert out.shape == (2, 5, 16)
    x2 = x.clone()
    x2[:, 4] = (x2[:, 4] + 1) % 60
    out2 = dec(x2, enc_out, look_ahead, None)
    assert torch.allclose(out[:, :4], out2[:, :4], atol=1e-5)


# ---------- Task 11: transformer ----------

def test_transformer_end_to_end():
    Transformer = load_task(__file__, "11-transformer.py", "Transformer")
    seeded()
    model = Transformer(2, 32, 4, 64, input_vocab=100, target_vocab=120,
                        max_seq_input=20, max_seq_target=25).eval()
    assert isinstance(model.linear, torch.nn.Linear) and model.linear.out_features == 120
    inputs = torch.randint(0, 100, (3, 10))
    target = torch.randint(0, 120, (3, 15))
    look_ahead = torch.triu(torch.ones(15, 15), diagonal=1)
    out = model(inputs, target, None, look_ahead, None)
    assert out.shape == (3, 15, 120)
    # causal: changing the last target token can't change earlier predictions
    target2 = target.clone()
    target2[:, -1] = (target2[:, -1] + 1) % 120
    out2 = model(inputs, target2, None, look_ahead, None)
    assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-5)


def test_transformer_can_overfit_a_tiny_batch():
    Transformer = load_task(__file__, "11-transformer.py", "Transformer")
    seeded()
    model = Transformer(1, 32, 4, 64, 20, 20, 10, 10, drop_rate=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    src = torch.randint(1, 20, (4, 6))
    tgt = torch.flip(src, dims=[1])               # task: reverse the sequence
    dec_in = torch.cat([torch.zeros(4, 1, dtype=torch.long), tgt[:, :-1]], dim=1)
    look_ahead = torch.triu(torch.ones(6, 6), diagonal=1)
    first = None
    for _ in range(150):
        logits = model(src, dec_in, None, look_ahead, None)
        loss = F.cross_entropy(logits.reshape(-1, 20), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < first / 5, "gradients should flow through every part of the model"
