"""
=============================================================================
Module 2: Scaled Dot-Product Attention & Multi-Head Attention
=============================================================================

THIS IS THE CORE OF THE TRANSFORMER. Everything else is plumbing around
these two functions.

KEY IDEA
────────
Attention answers the question: "For each position in my sequence, which
OTHER positions should I look at, and how much should I weight them?"

Think of it like a database query:
  - You have a QUERY  (what you're looking for)
  - You have KEYS    (labels on the data)
  - You have VALUES  (the actual data)

You compare your query against all keys to get relevance scores, then
use those scores to take a weighted average of the values.

SCALED DOT-PRODUCT ATTENTION (Section 3.2.1)
─────────────────────────────────────────────
  Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

Step by step:
  1. Q · K^T   — dot product of each query with each key → raw scores
  2. / √d_k    — scale down to prevent softmax from saturating
  3. softmax   — convert scores to probabilities (sum to 1)
  4. · V       — weighted sum of values

WHY SCALE BY √d_k?
  If q and k have components drawn from N(0,1), their dot product has
  variance d_k. For d_k=64, that means scores can easily be ±8 or more,
  pushing softmax into a regime where gradients are nearly zero.
  Dividing by √d_k brings variance back to 1.

MULTI-HEAD ATTENTION (Section 3.2.2)
────────────────────────────────────
Instead of one attention function on d_model-dimensional vectors, we:
  1. Project Q, K, V into h separate "heads" (h=8), each of dimension d_k=64
  2. Run attention independently on each head
  3. Concatenate the h outputs back to d_model dimensions
  4. Apply one final linear projection

WHY MULTIPLE HEADS?
  Different heads can learn to attend to different things:
  - Head 1 might focus on the previous word (local syntax)
  - Head 2 might focus on the subject of the sentence (long-range)
  - Head 3 might focus on punctuation patterns
  This is much richer than a single attention function.
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The fundamental attention operation.

    Args:
        query: (batch, heads, seq_len_q, d_k)
        key:   (batch, heads, seq_len_k, d_k)
        value: (batch, heads, seq_len_k, d_v)   [usually d_v == d_k]
        mask:  broadcastable boolean/float tensor. Positions with True (or 1)
               are BLOCKED (set to -inf before softmax). This is used for:
               - Padding mask: ignore <pad> tokens
               - Causal mask: prevent decoder from seeing future tokens
        dropout: optional dropout layer applied to attention weights

    Returns:
        output:  (batch, heads, seq_len_q, d_v) — the weighted sum of values
        weights: (batch, heads, seq_len_q, seq_len_k) — the attention weights
                 (useful for visualization / debugging)
    """
    d_k = query.size(-1)  # last dimension = key dimension

    # ── Step 1: Compute raw attention scores ──
    # query @ key^T: (batch, heads, seq_q, d_k) × (batch, heads, d_k, seq_k)
    #              → (batch, heads, seq_q, seq_k)
    # Each entry [b, h, i, j] = how much query position i attends to key position j
    scores = torch.matmul(query, key.transpose(-2, -1))

    # ── Step 2: Scale ──
    scores = scores / math.sqrt(d_k)

    # ── Step 3: Apply mask (if provided) ──
    # We set masked positions to -infinity so softmax gives them ~0 weight
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
        # Note: mask==0 means "blocked". This convention varies across
        # implementations. Here we use: 1 = allowed, 0 = blocked.

    # ── Step 4: Softmax → attention weights ──
    # Along the last dimension (key positions), so weights sum to 1
    weights = F.softmax(scores, dim=-1)

    # Optional dropout on attention weights (paper Section 5.4)
    if dropout is not None:
        weights = dropout(weights)

    # ── Step 5: Weighted sum of values ──
    # (batch, heads, seq_q, seq_k) × (batch, heads, seq_k, d_v)
    # → (batch, heads, seq_q, d_v)
    output = torch.matmul(weights, value)

    return output, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Section 3.2.2).

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

    where head_i = Attention(Q · W_i^Q, K · W_i^K, V · W_i^V)

    IMPLEMENTATION NOTE:
    ────────────────────
    We don't literally create h separate linear layers. Instead, we use
    ONE big linear layer of size (d_model → d_model) and then RESHAPE
    the output into (h, d_k). This is mathematically equivalent but
    much more efficient on GPUs because it's a single matrix multiply.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model:   total model dimension (512)
            num_heads: number of attention heads (8)
            dropout:   dropout rate for attention weights
        """
        super().__init__()

        # d_model must be divisible by num_heads
        # because we split d_model evenly across heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head (512/8 = 64)

        # ── Projection matrices ──
        # Each is (d_model → d_model), but we'll reshape to (h, d_k)
        # W^Q, W^K, W^V from the paper
        self.W_q = nn.Linear(d_model, d_model)  # projects queries
        self.W_k = nn.Linear(d_model, d_model)  # projects keys
        self.W_v = nn.Linear(d_model, d_model)  # projects values

        # W^O from the paper — combines the h head outputs back to d_model
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # We'll store attention weights here for visualization
        self.attn_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key:   (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_k, d_model)
            mask:  optional mask tensor

        In SELF-ATTENTION: query = key = value = same tensor
        In CROSS-ATTENTION: query comes from decoder,
                            key & value come from encoder output

        Returns:
            (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # ── Step 1: Linear projections ──
        # (batch, seq_len, d_model) → (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # ── Step 2: Reshape into multiple heads ──
        # (batch, seq_len, d_model) → (batch, seq_len, h, d_k) → (batch, h, seq_len, d_k)
        #
        # The .view() splits the last dimension: d_model → (num_heads, d_k)
        # The .transpose(1, 2) swaps seq_len and num_heads dimensions
        # so each head can process all positions independently.
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Shape now: (batch, num_heads, seq_len, d_k)

        # ── Step 3: Apply attention ──
        attn_output, self.attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        # attn_output: (batch, num_heads, seq_len_q, d_k)

        # ── Step 4: Concatenate heads ──
        # Reverse the reshape: (batch, h, seq_len, d_k) → (batch, seq_len, d_model)
        attn_output = (
            attn_output.transpose(1, 2)                    # (batch, seq_len, h, d_k)
            .contiguous()                                   # ensure memory layout
            .view(batch_size, -1, self.d_model)             # (batch, seq_len, d_model)
        )

        # ── Step 5: Final linear projection (W^O) ──
        output = self.W_o(attn_output)

        return output


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

    # Random input (simulating embedded tokens)
    x = torch.randn(batch_size, seq_len, d_model)

    # Self-attention: Q = K = V = x
    output = mha(x, x, x)
    print(f"Input shape:  {x.shape}")       # (2, 10, 512)
    print(f"Output shape: {output.shape}")   # (2, 10, 512)
    print(f"Attention weights shape: {mha.attn_weights.shape}")  # (2, 8, 10, 10)

    # Test with causal mask (decoder self-attention)
    # 1 = allowed, 0 = blocked (future positions)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    output_masked = mha(x, x, x, mask=causal_mask)
    print(f"\nCausal mask shape: {causal_mask.shape}")
    print(f"Masked output shape: {output_masked.shape}")

    # Verify causal mask works: attention weights should be 0 above diagonal
    weights = mha.attn_weights[0, 0]  # first batch, first head
    upper_tri = weights[0, 1:]  # position 0 attending to positions 1+
    print(f"Attn from pos 0 to future: {upper_tri}")
    print(f"All ~zero: {upper_tri.abs().max().item() < 1e-6}")
    print("✓ Attention module OK")
