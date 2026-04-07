"""
=============================================================================
Module 6: The Complete Transformer
=============================================================================

PUTTING IT ALL TOGETHER
───────────────────────
The Transformer is an ENCODER-DECODER model:

  Source tokens
       │
       ▼
  ┌─────────────┐
  │  Embedding   │ ← token embed + positional encoding
  │  + Pos Enc   │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │   ENCODER   │ ← N=6 layers of {self-attention, FFN}
  │  (6 layers) │
  └──────┬──────┘
         │
         │ encoder_output ──────────┐
         │                          │
  Target tokens (shifted right)     │
       │                            │
       ▼                            │
  ┌─────────────┐                   │
  │  Embedding   │                   │
  │  + Pos Enc   │                   │
  └──────┬──────┘                   │
         │                          ▼
  ┌──────▼──────┐          (cross-attention
  │   DECODER   │ ←──────── reads encoder
  │  (6 layers) │           output here)
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │ Linear Head │ ← project d_model → vocab_size
  │  + Softmax  │
  └─────────────┘
         │
         ▼
  Output probabilities

WEIGHT SHARING (Section 3.4)
────────────────────────────
The paper shares the same weight matrix between:
  1. Source embedding layer
  2. Target embedding layer
  3. The pre-softmax linear layer

This reduces parameters and acts as a regularizer. We implement this by
making them all point to the same nn.Embedding weight matrix.

MASKS
─────
We need two types of masks:

  1. PADDING MASK: When sentences have different lengths, shorter ones are
     padded with a special <pad> token. We mask these so attention ignores them.

  2. CAUSAL MASK: In the decoder, position i must not see positions > i.
     This is a lower-triangular matrix of 1s.
=============================================================================
"""

import torch
import torch.nn as nn

from transformer_from_scratch.embeddings import TokenEmbedding, PositionalEncoding
from transformer_from_scratch.encoder import Encoder
from transformer_from_scratch.decoder import Decoder


class Transformer(nn.Module):
    """
    The complete Transformer model from "Attention Is All You Need."
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        pad_idx: int = 0,
    ):
        """
        Args:
            src_vocab_size: source vocabulary size
            tgt_vocab_size: target vocabulary size
            d_model:        model dimension (512)
            num_layers:     N = 6
            num_heads:      h = 8
            d_ff:           feed-forward inner dim (2048)
            dropout:        dropout rate (0.1)
            max_seq_len:    max sequence length for positional encoding
            pad_idx:        index of the <pad> token (for masking)
        """
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        # ── Embedding layers ──
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # ── Encoder and Decoder stacks ──
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

        # ── Output projection ──
        # Maps d_model → vocab_size for next-token prediction
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # ── Weight sharing (Section 3.4) ──
        # The paper: "we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation."
        #
        # This only works when src and tgt share the same vocabulary
        # (which they do in the paper — they use a shared BPE vocab).
        # This saves ~38M parameters and acts as a regularizer.
        if src_vocab_size == tgt_vocab_size:
            # Share source and target embedding weights
            self.src_embedding.embedding.weight = self.tgt_embedding.embedding.weight
            # Share target embedding and output projection weights
            # Note: nn.Linear.weight is (out_features, in_features) = (vocab, d_model)
            # nn.Embedding.weight is (vocab, d_model) — same layout!
            self.output_projection.weight = self.tgt_embedding.embedding.weight

        # ── Initialize parameters ──
        # Xavier uniform initialization (commonly used for Transformers)
        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize all parameters with Xavier uniform distribution.

        This helps with training stability — the initial values are scaled
        so that the variance of activations stays roughly constant across layers.
        """
        for p in self.parameters():
            if p.dim() > 1:  # skip biases (1D tensors)
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create a mask that hides padding tokens in the source.

        Args:
            src: (batch_size, src_seq_len) — source token IDs

        Returns:
            (batch_size, 1, 1, src_seq_len) — broadcastable mask
            1 = real token (attend to it), 0 = padding (ignore it)
        """
        # src != pad_idx gives True for real tokens, False for padding
        # We add two dimensions for (num_heads, query_positions) broadcasting
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # Shape: (batch, 1, 1, src_seq_len)
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create a combined mask for the decoder:
          1. Padding mask: hide <pad> tokens
          2. Causal mask: hide future tokens

        Args:
            tgt: (batch_size, tgt_seq_len) — target token IDs

        Returns:
            (batch_size, 1, tgt_seq_len, tgt_seq_len) — combined mask
        """
        tgt_seq_len = tgt.size(1)

        # Padding mask: (batch, 1, 1, tgt_seq_len)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # Causal mask: lower-triangular matrix
        # (1, 1, tgt_seq_len, tgt_seq_len)
        causal_mask = torch.tril(
            torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device)
        ).unsqueeze(0).unsqueeze(0)
        # tril = lower triangular: 1s on and below the diagonal, 0s above

        # Combine: both conditions must be satisfied (logical AND via multiply)
        # A position is attended to only if it's:
        #   (a) not padding AND (b) not in the future
        tgt_mask = tgt_pad_mask & (causal_mask.bool())
        # Shape: (batch, 1, tgt_seq_len, tgt_seq_len)

        return tgt_mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass of the Transformer.

        Args:
            src: (batch_size, src_seq_len) — source token IDs
            tgt: (batch_size, tgt_seq_len) — target token IDs
                 (shifted right: starts with <sos>, does NOT include final token)

        Returns:
            (batch_size, tgt_seq_len, tgt_vocab_size) — logits (before softmax)
        """
        # ── Step 1: Create masks ──
        src_mask = self.make_src_mask(src)  # (batch, 1, 1, src_len)
        tgt_mask = self.make_tgt_mask(tgt)  # (batch, 1, tgt_len, tgt_len)

        # ── Step 2: Embed + positionally encode source ──
        src_embedded = self.positional_encoding(self.src_embedding(src))
        # (batch, src_len, d_model)

        # ── Step 3: Encode ──
        encoder_output = self.encoder(src_embedded, src_mask)
        # (batch, src_len, d_model) — rich representation of the input

        # ── Step 4: Embed + positionally encode target ──
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))
        # (batch, tgt_len, d_model)

        # ── Step 5: Decode ──
        decoder_output = self.decoder(
            tgt_embedded,
            encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        # (batch, tgt_len, d_model)

        # ── Step 6: Project to vocabulary ──
        logits = self.output_projection(decoder_output)
        # (batch, tgt_len, tgt_vocab_size)
        # These are raw scores. We apply softmax in the loss function,
        # not here (for numerical stability with cross-entropy loss).

        return logits


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hyperparameters matching the paper's base model
    src_vocab = 1000
    tgt_vocab = 1000
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048

    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    src_len = 12
    tgt_len = 10

    src = torch.randint(1, src_vocab, (batch_size, src_len))  # avoid 0 (pad)
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len))

    logits = model(src, tgt)
    print(f"Source shape: {src.shape}")        # (2, 12)
    print(f"Target shape: {tgt.shape}")        # (2, 10)
    print(f"Output shape: {logits.shape}")     # (2, 10, 1000)

    # Verify masks
    src_mask = model.make_src_mask(src)
    tgt_mask = model.make_tgt_mask(tgt)
    print(f"\nSource mask shape: {src_mask.shape}")  # (2, 1, 1, 12)
    print(f"Target mask shape: {tgt_mask.shape}")    # (2, 1, 10, 10)
    print(f"Target mask (batch 0):\n{tgt_mask[0, 0].int()}")

    print("✓ Transformer module OK")
