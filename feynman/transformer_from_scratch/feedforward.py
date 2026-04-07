"""
=============================================================================
Module 3: Position-wise Feed-Forward Network
=============================================================================

WHAT THIS IS
────────────
After attention gathers information from across the sequence, we need to
actually PROCESS that information at each position. That's what the
feed-forward network (FFN) does.

It's deceptively simple — just two linear layers with a ReLU in between:

  FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

  - First layer:  d_model (512) → d_ff (2048)   — expand
  - ReLU activation                               — introduce non-linearity
  - Second layer: d_ff (2048) → d_model (512)    — compress back

WHY "POSITION-WISE"?
  The SAME FFN (same weights) is applied independently to each position.
  Position 0 gets the same transformation as position 5. It's like applying
  the same little neural network to each token separately.

  Another way to think about it: this is equivalent to a 1D convolution
  with kernel size 1.

WHY THE EXPANSION TO 4x?
  d_ff = 4 × d_model gives the FFN a bigger "workspace" to compute in.
  The expansion lets the network create richer intermediate representations
  before compressing back. This is a common pattern — bottleneck layers
  work well in many architectures.

NOTE: The weights are SHARED across positions within a layer,
      but DIFFERENT between layers (each of the 6 layers has its own FFN).
=============================================================================
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Also called a "two-layer MLP" or "point-wise feed-forward network."
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: input and output dimension (512)
            d_ff:    inner layer dimension (2048 = 4 × 512)
            dropout: dropout rate (0.1)
        """
        super().__init__()

        # First linear layer: expand from d_model to d_ff
        self.linear1 = nn.Linear(d_model, d_ff)

        # Second linear layer: compress from d_ff back to d_model
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout applied after ReLU (paper Section 5.4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)

        The computation at each position independently:
          d_model=512 ──→ d_ff=2048 ──→ ReLU ──→ dropout ──→ d_model=512
        """
        # Step 1: Expand + ReLU
        # (batch, seq, 512) → (batch, seq, 2048) → ReLU
        x = torch.relu(self.linear1(x))

        # Step 2: Dropout (regularization)
        x = self.dropout(x)

        # Step 3: Compress back
        # (batch, seq, 2048) → (batch, seq, 512)
        x = self.linear2(x)

        return x


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048

    ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)
    output = ffn(x)

    print(f"Input shape:  {x.shape}")      # (2, 10, 512)
    print(f"Output shape: {output.shape}")  # (2, 10, 512)

    # Verify position-wise independence:
    # output at position 0 should depend ONLY on input at position 0
    x2 = x.clone()
    x2[:, 1:, :] = 0  # zero out all positions except 0
    output2 = ffn(x2)
    # Position 0 output should be identical
    diff = (output[:, 0, :] - output2[:, 0, :]).abs().max().item()
    print(f"Position 0 unaffected by other positions: {diff < 1e-6}")
    print("✓ FeedForward module OK")
