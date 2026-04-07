"""
=============================================================================
Module 4: Encoder Layer & Encoder Stack
=============================================================================

THE ENCODER'S JOB
─────────────────
The encoder reads the INPUT sequence (e.g., an English sentence) and builds
a rich representation of it. Each token's representation gets enriched by
information from ALL other tokens through self-attention.

STRUCTURE OF ONE ENCODER LAYER
──────────────────────────────
Each of the N=6 identical encoder layers does:

  1. MULTI-HEAD SELF-ATTENTION
     Every position attends to every other position in the input.
     This is how "bank" learns to mean "river bank" vs "money bank"
     — by looking at surrounding context.

  2. FEED-FORWARD NETWORK
     Processes each position independently (the "thinking" step).

Both sub-layers use:
  - RESIDUAL CONNECTION: output = x + sublayer(x)
    This helps gradients flow during training (like skip connections in ResNet).

  - LAYER NORMALIZATION: normalize the output to have mean≈0, variance≈1
    This stabilizes training by preventing values from exploding or vanishing.

So the actual computation is:
  x = LayerNorm(x + MultiHeadAttention(x, x, x))    # self-attention
  x = LayerNorm(x + FeedForward(x))                  # FFN

THE FULL ENCODER
────────────────
Just N=6 of these layers stacked. The output of layer 1 feeds into layer 2,
which feeds into layer 3, etc. Each layer refines the representation.

By layer 6, each token's vector encodes not just "what word is this" but
"what does this word mean in the full context of this sentence."
=============================================================================
"""

import torch
import torch.nn as nn

from transformer_from_scratch.attention import MultiHeadAttention
from transformer_from_scratch.feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    One layer of the Transformer encoder.

    Input → Self-Attention → Add & Norm → FFN → Add & Norm → Output
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model:   model dimension (512)
            num_heads: attention heads (8)
            d_ff:      feed-forward inner dimension (2048)
            dropout:   dropout rate (0.1)
        """
        super().__init__()

        # Sub-layer 1: Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 2: Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer normalization for each sub-layer
        # LayerNorm normalizes across the d_model dimension (last dim)
        # This is different from BatchNorm which normalizes across the batch
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        (batch_size, seq_len, d_model) — input to this layer
            src_mask: optional mask for padding tokens in the source

        Returns:
            (batch_size, seq_len, d_model) — enriched representations
        """
        # ── Sub-layer 1: Self-Attention with residual + norm ──
        # "Self" because Q=K=V=x (the layer attends to itself)
        attn_output = self.self_attention(x, x, x, mask=src_mask)
        attn_output = self.dropout1(attn_output)

        # Residual connection: add original input back
        # Then normalize. This is "Post-LN" (the original paper's approach).
        # (There's also "Pre-LN" where you normalize BEFORE the sub-layer,
        #  which some later work found more stable, but we follow the paper.)
        x = self.norm1(x + attn_output)

        # ── Sub-layer 2: Feed-Forward with residual + norm ──
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)

        return x


class Encoder(nn.Module):
    """
    The full encoder: N=6 identical EncoderLayers stacked.

    Input embeddings → Layer 1 → Layer 2 → ... → Layer 6 → Output
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_layers: N = 6 in the base model
            d_model:    512
            num_heads:  8
            d_ff:       2048
            dropout:    0.1
        """
        super().__init__()

        # Create N identical layers
        # nn.ModuleList is like a Python list, but PyTorch knows about
        # the parameters inside (so they get saved, moved to GPU, etc.)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm (some implementations add this for stability)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        (batch_size, seq_len, d_model) — embedded + positionally encoded input
            src_mask: optional padding mask

        Returns:
            (batch_size, seq_len, d_model) — the encoder's output representation
        """
        # Pass through each layer sequentially
        for layer in self.layers:
            x = layer(x, src_mask)

        # Final normalization
        return self.norm(x)


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6

    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)
    output = encoder(x)

    print(f"Input shape:  {x.shape}")       # (2, 10, 512)
    print(f"Output shape: {output.shape}")   # (2, 10, 512)

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,}")
    print("✓ Encoder module OK")
