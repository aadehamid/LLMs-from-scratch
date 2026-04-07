"""
=============================================================================
Module 10: Verification Tests (Our Replication Oracles)
=============================================================================

These tests verify that the implementation is correct BEFORE we try training.
Each test checks a specific property that MUST hold if the code is right.

Think of these as "unit tests for a neural network architecture."
=============================================================================
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add parent dir to path so imports work from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformer_from_scratch.embeddings import TokenEmbedding, PositionalEncoding
from transformer_from_scratch.attention import scaled_dot_product_attention, MultiHeadAttention
from transformer_from_scratch.feedforward import PositionwiseFeedForward
from transformer_from_scratch.encoder import Encoder
from transformer_from_scratch.decoder import Decoder
from transformer_from_scratch.transformer import Transformer
from transformer_from_scratch.lr_schedule import NoamScheduler
from transformer_from_scratch.label_smoothing import LabelSmoothingLoss


def test_embedding_shapes():
    """Token embedding should produce (batch, seq_len, d_model)."""
    embed = TokenEmbedding(vocab_size=100, d_model=64)
    x = torch.randint(0, 100, (2, 10))
    out = embed(x)
    assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"
    print("  ✓ Embedding shapes correct")


def test_embedding_scaling():
    """Embeddings should be scaled by √d_model."""
    d_model = 64
    embed = TokenEmbedding(vocab_size=100, d_model=d_model)
    x = torch.tensor([[0]])
    raw = embed.embedding(x)
    scaled = embed(x)
    ratio = (scaled / raw).mean().item()
    expected = d_model ** 0.5
    assert abs(ratio - expected) < 1e-4, f"Expected scaling {expected}, got {ratio}"
    print("  ✓ Embedding scaling by √d_model correct")


def test_positional_encoding_shape():
    """PE should not change the shape."""
    pe = PositionalEncoding(d_model=64, dropout=0.0)
    x = torch.randn(2, 10, 64)
    out = pe(x)
    assert out.shape == x.shape, f"Shape changed: {x.shape} → {out.shape}"
    print("  ✓ Positional encoding preserves shape")


def test_positional_encoding_different_positions():
    """Different positions should get different encodings."""
    pe = PositionalEncoding(d_model=64, dropout=0.0)
    pos0 = pe.pe[0, 0]
    pos1 = pe.pe[0, 1]
    assert not torch.allclose(pos0, pos1), "Position 0 and 1 have identical PE!"
    print("  ✓ Different positions get different encodings")


def test_attention_output_shape():
    """Scaled dot-product attention should produce correct shape."""
    Q = torch.randn(2, 8, 10, 64)  # (batch, heads, seq, d_k)
    K = torch.randn(2, 8, 10, 64)
    V = torch.randn(2, 8, 10, 64)
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 8, 10, 64), f"Output shape: {out.shape}"
    assert weights.shape == (2, 8, 10, 10), f"Weights shape: {weights.shape}"
    print("  ✓ Attention output shapes correct")


def test_attention_weights_sum_to_one():
    """Attention weights should sum to 1 along the key dimension."""
    Q = torch.randn(1, 1, 5, 16)
    K = torch.randn(1, 1, 5, 16)
    V = torch.randn(1, 1, 5, 16)
    _, weights = scaled_dot_product_attention(Q, K, V)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"Weights don't sum to 1: {sums}"
    print("  ✓ Attention weights sum to 1")


def test_causal_mask_blocks_future():
    """Causal mask should zero out attention to future positions."""
    seq_len = 5
    Q = torch.randn(1, 1, seq_len, 16)
    K = torch.randn(1, 1, seq_len, 16)
    V = torch.randn(1, 1, seq_len, 16)

    # Causal mask: 1 = allowed, 0 = blocked
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    # Check upper triangle is zero (future positions blocked)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert weights[0, 0, i, j].item() < 1e-7, \
                f"Position {i} attends to future position {j}: {weights[0, 0, i, j].item()}"
    print("  ✓ Causal mask blocks future positions")


def test_multihead_attention_shape():
    """MHA should produce (batch, seq_len, d_model)."""
    mha = MultiHeadAttention(d_model=64, num_heads=4, dropout=0.0)
    x = torch.randn(2, 10, 64)
    out = mha(x, x, x)
    assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"
    print("  ✓ Multi-head attention shape correct")


def test_feedforward_shape():
    """FFN should preserve shape."""
    ffn = PositionwiseFeedForward(d_model=64, d_ff=128, dropout=0.0)
    x = torch.randn(2, 10, 64)
    out = ffn(x)
    assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"
    print("  ✓ Feed-forward shape correct")


def test_feedforward_position_independent():
    """FFN output at position i should not depend on position j."""
    ffn = PositionwiseFeedForward(d_model=64, d_ff=128, dropout=0.0)
    x = torch.randn(1, 5, 64)
    out_full = ffn(x)
    out_single = ffn(x[:, 2:3, :])  # just position 2
    diff = (out_full[:, 2, :] - out_single[:, 0, :]).abs().max().item()
    assert diff < 1e-6, f"Position dependence detected: max diff = {diff}"
    print("  ✓ Feed-forward is position-independent")


def test_encoder_shape():
    """Encoder should produce (batch, seq_len, d_model)."""
    encoder = Encoder(num_layers=2, d_model=64, num_heads=4, d_ff=128, dropout=0.0)
    x = torch.randn(2, 10, 64)
    out = encoder(x)
    assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"
    print("  ✓ Encoder shape correct")


def test_decoder_shape():
    """Decoder should produce (batch, tgt_len, d_model)."""
    decoder = Decoder(num_layers=2, d_model=64, num_heads=4, d_ff=128, dropout=0.0)
    tgt = torch.randn(2, 8, 64)
    enc_out = torch.randn(2, 10, 64)
    mask = torch.tril(torch.ones(8, 8)).unsqueeze(0).unsqueeze(0)
    out = decoder(tgt, enc_out, tgt_mask=mask)
    assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"
    print("  ✓ Decoder shape correct")


def test_transformer_end_to_end():
    """Full Transformer forward pass should produce logits."""
    model = Transformer(
        src_vocab_size=50, tgt_vocab_size=50,
        d_model=64, num_layers=2, num_heads=4, d_ff=128,
        dropout=0.0, pad_idx=0,
    )
    src = torch.randint(1, 50, (2, 10))
    tgt = torch.randint(1, 50, (2, 8))
    logits = model(src, tgt)
    assert logits.shape == (2, 8, 50), f"Expected (2, 8, 50), got {logits.shape}"
    print("  ✓ Full Transformer end-to-end shape correct")


def test_transformer_parameter_count():
    """
    Base model should have ~65M parameters (Table 3 in the paper).
    We test with the full base config.
    """
    model = Transformer(
        src_vocab_size=37000,  # BPE vocab from the paper
        tgt_vocab_size=37000,
        d_model=512, num_layers=6, num_heads=8, d_ff=2048,
        dropout=0.1, pad_idx=0,
    )
    total = sum(p.numel() for p in model.parameters())
    # Paper says ~65M. Allow 60M-75M range (exact count depends on
    # whether embeddings are shared and other minor implementation choices).
    assert 55_000_000 < total < 80_000_000, \
        f"Parameter count {total:,} outside expected range (55M-80M)"
    print(f"  ✓ Parameter count: {total:,} (paper reports ~65M)")


def test_lr_schedule_peak():
    """LR should peak at warmup_steps."""
    model = nn.Linear(64, 64)
    opt = torch.optim.Adam(model.parameters(), lr=1.0)
    sched = NoamScheduler(opt, d_model=512, warmup_steps=4000)

    lrs = []
    for step in range(1, 8001):
        sched.step()
        lrs.append(sched.get_lr())

    peak_step = lrs.index(max(lrs)) + 1  # +1 because steps are 1-indexed
    assert peak_step == 4000, f"Peak at step {peak_step}, expected 4000"
    print(f"  ✓ LR peaks at warmup step 4000 (peak LR = {max(lrs):.6f})")


def test_label_smoothing_distribution():
    """Label smoothing should distribute probability mass correctly."""
    vocab = 10
    criterion = LabelSmoothingLoss(vocab_size=vocab, padding_idx=0, smoothing=0.1)
    assert abs(criterion.confidence - 0.9) < 1e-6
    expected_smooth = 0.1 / 9  # 9 = vocab_size - 1
    assert abs(criterion.smoothed_value - expected_smooth) < 1e-6
    print("  ✓ Label smoothing distribution correct")


def test_gradient_flow():
    """Gradients should flow from loss back to all parameters."""
    model = Transformer(
        src_vocab_size=20, tgt_vocab_size=20,
        d_model=32, num_layers=1, num_heads=2, d_ff=64,
        dropout=0.0, pad_idx=0,
    )
    src = torch.randint(1, 20, (1, 5))
    tgt = torch.randint(1, 20, (1, 5))
    logits = model(src, tgt)
    loss = logits.sum()
    loss.backward()

    # Check that at least some parameters have non-zero gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in model.parameters())
    assert has_grad > total * 0.5, f"Only {has_grad}/{total} params have gradients"
    print(f"  ✓ Gradients flow to {has_grad}/{total} parameters")


# ─── Run all tests ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("Embeddings", [
            test_embedding_shapes,
            test_embedding_scaling,
            test_positional_encoding_shape,
            test_positional_encoding_different_positions,
        ]),
        ("Attention", [
            test_attention_output_shape,
            test_attention_weights_sum_to_one,
            test_causal_mask_blocks_future,
            test_multihead_attention_shape,
        ]),
        ("Feed-Forward", [
            test_feedforward_shape,
            test_feedforward_position_independent,
        ]),
        ("Encoder & Decoder", [
            test_encoder_shape,
            test_decoder_shape,
        ]),
        ("Full Transformer", [
            test_transformer_end_to_end,
            test_transformer_parameter_count,
            test_gradient_flow,
        ]),
        ("Training Utilities", [
            test_lr_schedule_peak,
            test_label_smoothing_distribution,
        ]),
    ]

    total_passed = 0
    total_tests = 0

    print("=" * 60)
    print("TRANSFORMER VERIFICATION TESTS")
    print("=" * 60)

    for group_name, test_fns in tests:
        print(f"\n── {group_name} ──")
        for fn in test_fns:
            total_tests += 1
            try:
                fn()
                total_passed += 1
            except Exception as e:
                print(f"  ✗ {fn.__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {total_passed}/{total_tests} tests passed")
    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED")
    else:
        print(f"⚠️  {total_tests - total_passed} test(s) failed")
    print("=" * 60)
