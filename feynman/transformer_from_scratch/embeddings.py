"""
=============================================================================
Module 1: Embeddings & Positional Encoding
=============================================================================

WHY THIS EXISTS
───────────────
A Transformer has NO recurrence and NO convolution. That means it has no
built-in notion of "this token comes before that token." We need two things:

  1. TOKEN EMBEDDINGS  — convert each integer token ID into a dense vector
     of size d_model (512 in the base model).

  2. POSITIONAL ENCODING — inject information about WHERE each token sits
     in the sequence. Without this, the model sees a "bag of tokens" and
     can't distinguish "the cat sat on the mat" from "mat the on sat cat the."

The paper uses *sinusoidal* positional encodings (not learned). The intuition:
each dimension of the encoding oscillates at a different frequency, so each
position gets a unique "fingerprint" that the model can use to figure out
relative distances between tokens.

FORMULA (Section 3.5 of the paper):
  PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
  PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))

where:
  - pos  = position in the sequence (0, 1, 2, ...)
  - i    = dimension index (0, 1, 2, ..., d_model/2 - 1)
  - Even dimensions get sin, odd dimensions get cos.

WHY SINUSOIDS?
  For any fixed offset k, PE(pos+k) can be written as a linear function of
  PE(pos). This means the model can easily learn to attend to "3 positions
  back" or "5 positions forward" — relative position is encoded linearly.
=============================================================================
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts integer token IDs → dense vectors of size d_model.

    One important detail from the paper (Section 3.4):
      "In the embedding layers, we multiply those weights by √d_model."

    Why? The embedding vectors are initialized with small random values
    (roughly mean=0, std=1). After this scaling, their magnitude is on the
    same order as the positional encodings, so neither signal drowns out
    the other when they're summed together.
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: how many unique tokens exist (e.g., 37000 for BPE)
            d_model:    dimension of the embedding vectors (512 in base model)
        """
        super().__init__()
        # nn.Embedding is just a lookup table: token_id → vector
        # Internally it's a matrix of shape (vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: integer tensor of shape (batch_size, seq_len)
               Each value is a token ID in [0, vocab_size).

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Look up each token ID in the embedding table, then scale up
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional information to the token embeddings.

    This is NOT a learned parameter — it's a fixed matrix computed once
    and reused. We register it as a "buffer" so PyTorch saves it with the
    model but doesn't try to compute gradients for it.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: embedding dimension (must match TokenEmbedding's d_model)
            max_len: maximum sequence length we'll ever see
            dropout: dropout rate (paper uses 0.1)
                     Applied to (embedding + positional_encoding)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ─── Build the positional encoding matrix ───
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # position = column vector [0, 1, 2, ..., max_len-1]
        # shape: (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term implements: 10000^(2i / d_model)
        # We compute it in log-space for numerical stability:
        #   10000^(2i/d_model) = exp(2i * ln(10000) / d_model)
        #
        # shape: (d_model/2,) — one value per pair of (sin, cos) dimensions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()       # [0, 2, 4, ..., d_model-2]
            * (-math.log(10000.0) / d_model)           # multiply by -ln(10000)/d_model
        )
        # Note the negative sign: exp(-x) = 1/exp(x), so this gives us
        # 1 / 10000^(2i/d_model), which we'll multiply by `position`.

        # Even dimensions: sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd dimensions: cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: (1, max_len, d_model)
        # so we can broadcast-add to any batch of embeddings
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter — no gradients)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (batch_size, seq_len, d_model)
               — typically the output of TokenEmbedding

        Returns:
            Same shape, with positional encoding added and dropout applied.
        """
        seq_len = x.size(1)

        # Slice the pre-computed PE to match the actual sequence length,
        # then add it element-wise to the embeddings.
        # .detach() is a safety measure — we definitely don't want gradients here.
        x = x + self.pe[:, :seq_len, :].detach()

        # The paper applies dropout to (embedding + PE)
        return self.dropout(x)


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    vocab_size = 1000
    d_model = 512
    batch_size = 2
    seq_len = 10

    embed = TokenEmbedding(vocab_size, d_model)
    pe = PositionalEncoding(d_model, dropout=0.0)  # no dropout for testing

    # Fake input: batch of 2 sequences, each 10 tokens long
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape:  {tokens.shape}")   # (2, 10)

    embedded = embed(tokens)
    print(f"After embed:  {embedded.shape}")  # (2, 10, 512)

    encoded = pe(embedded)
    print(f"After PE:     {encoded.shape}")   # (2, 10, 512)

    # Verify PE is different for different positions
    pos0 = pe.pe[0, 0, :4]
    pos1 = pe.pe[0, 1, :4]
    print(f"\nPE at position 0 (first 4 dims): {pos0}")
    print(f"PE at position 1 (first 4 dims): {pos1}")
    print(f"They differ: {not torch.allclose(pos0, pos1)}")
    print("✓ Embeddings module OK")
