"""
=============================================================================
Module 5: Decoder Layer & Decoder Stack
=============================================================================

THE DECODER'S JOB
─────────────────
The decoder generates the OUTPUT sequence one token at a time (e.g., a
German translation). At each step, it:
  1. Looks at what it has generated so far (self-attention)
  2. Looks at the encoder's output to understand the input (cross-attention)
  3. Decides what token to produce next (FFN → linear → softmax)

STRUCTURE OF ONE DECODER LAYER (3 sub-layers, not 2!)
─────────────────────────────────────────────────────
  1. MASKED SELF-ATTENTION
     Like encoder self-attention, but with a CAUSAL MASK that prevents
     each position from attending to FUTURE positions. During training,
     we process all positions in parallel, but position 5 must not "cheat"
     by looking at position 6's target token.

     Example for seq_len=4 (1=allowed, 0=blocked):
       Position:  0  1  2  3
       Pos 0:  [  1  0  0  0 ]   ← can only see itself
       Pos 1:  [  1  1  0  0 ]   ← can see 0, 1
       Pos 2:  [  1  1  1  0 ]   ← can see 0, 1, 2
       Pos 3:  [  1  1  1  1 ]   ← can see everything up to 3

  2. CROSS-ATTENTION (Encoder-Decoder Attention)
     Queries come from the decoder (previous layer's output).
     Keys and Values come from the ENCODER's output.

     This is how the decoder "reads" the input. Every decoder position
     can attend to every encoder position — there's no mask here
     (except possibly padding).

  3. FEED-FORWARD NETWORK
     Same as in the encoder.

All three sub-layers have residual connections + layer normalization.
=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from transformer_from_scratch.attention import MultiHeadAttention
from transformer_from_scratch.feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    One layer of the Transformer decoder.

    Input → Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm → Output
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Sub-layer 1: Masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 2: Multi-Head Cross-Attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 3: Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # One LayerNorm + Dropout for each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (batch, tgt_seq_len, d_model) — decoder input
            encoder_output: (batch, src_seq_len, d_model) — encoder's final output
            src_mask:       mask for source padding (used in cross-attention)
            tgt_mask:       causal mask + target padding (used in self-attention)

        Returns:
            (batch, tgt_seq_len, d_model)
        """
        # ── Sub-layer 1: Masked Self-Attention ──
        # Q = K = V = x, but we pass tgt_mask to block future positions
        attn_output = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # ── Sub-layer 2: Cross-Attention ──
        # Q = decoder state, K = V = encoder output
        # This is where the decoder "reads" the source sentence
        cross_output = self.cross_attention(
            query=x,                 # from decoder
            key=encoder_output,      # from encoder
            value=encoder_output,    # from encoder
            mask=src_mask,           # mask padding in source
        )
        x = self.norm2(x + self.dropout2(cross_output))

        # ── Sub-layer 3: Feed-Forward ──
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class Decoder(nn.Module):
    """
    The full decoder: N=6 identical DecoderLayers stacked.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (batch, tgt_seq_len, d_model) — target embeddings
            encoder_output: (batch, src_seq_len, d_model) — encoder output
            src_mask:       source padding mask
            tgt_mask:       causal + target padding mask

        Returns:
            (batch, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch_size = 2
    src_seq_len = 12
    tgt_seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6

    decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)

    # Simulated inputs
    tgt = torch.randn(batch_size, tgt_seq_len, d_model)         # decoder input
    encoder_out = torch.randn(batch_size, src_seq_len, d_model)  # encoder output

    # Causal mask for decoder self-attention
    causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).unsqueeze(0).unsqueeze(0)

    output = decoder(tgt, encoder_out, tgt_mask=causal_mask)

    print(f"Target input shape:     {tgt.shape}")         # (2, 10, 512)
    print(f"Encoder output shape:   {encoder_out.shape}")  # (2, 12, 512)
    print(f"Decoder output shape:   {output.shape}")       # (2, 10, 512)

    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {num_params:,}")
    print("✓ Decoder module OK")
