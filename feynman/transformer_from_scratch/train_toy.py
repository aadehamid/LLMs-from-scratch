"""
=============================================================================
Module 9: Training on a Toy "Copy" Task
=============================================================================

WHAT IS THE COPY TASK?
──────────────────────
The simplest possible sequence-to-sequence task:
  Input:  [1, 5, 3, 8, 2]
  Output: [1, 5, 3, 8, 2]

The model must learn to copy the input sequence to the output.

WHY THIS TASK?
  - If the Transformer can't learn to COPY, something is fundamentally broken.
  - It requires attention (the decoder must look at the encoder's output).
  - It's fast — we can train on CPU in seconds.
  - It gives us a binary pass/fail verification: did the model learn to copy?

TRAINING SETUP
──────────────
  - Vocab: tokens 0-9 (0 = padding, 1 = SOS, 2-9 = data)
  - Sequences: length 10, random tokens from {2, 3, ..., 9}
  - We prepend a <SOS>=1 token to the target (standard in seq2seq)
  - Small model: d_model=64, 2 layers, 2 heads (trains in seconds)

TEACHER FORCING
───────────────
During training, the decoder receives the CORRECT previous tokens as input
(not its own predictions). This is called "teacher forcing" and is standard
for training seq2seq models. At inference time, we'd feed back the model's
own predictions (autoregressive generation).
=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from transformer_from_scratch.transformer import Transformer
from transformer_from_scratch.lr_schedule import NoamScheduler
from transformer_from_scratch.label_smoothing import LabelSmoothingLoss


def generate_copy_data(
    batch_size: int,
    seq_len: int,
    vocab_low: int = 2,
    vocab_high: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of copy-task data.

    Args:
        batch_size: number of sequences
        seq_len:    length of each sequence
        vocab_low:  minimum token value (2, avoiding 0=pad and 1=SOS)
        vocab_high: maximum token value (exclusive)

    Returns:
        src:        (batch, seq_len) — source tokens
        tgt_input:  (batch, seq_len+1) — decoder input: [SOS, t1, t2, ..., tn]
        tgt_output: (batch, seq_len+1) — expected output: [t1, t2, ..., tn, SOS]
                    (we reuse SOS as EOS for simplicity)
    """
    # Random source tokens
    src = torch.randint(vocab_low, vocab_high, (batch_size, seq_len))

    # Target input: prepend SOS token (1) to the source
    # [SOS, t1, t2, ..., tn]
    sos = torch.ones(batch_size, 1, dtype=torch.long)  # SOS = 1
    tgt_input = torch.cat([sos, src], dim=1)

    # Target output: the source tokens followed by EOS
    # [t1, t2, ..., tn, EOS]
    # We use SOS=1 as EOS too for simplicity
    tgt_output = torch.cat([src, sos], dim=1)

    return src, tgt_input, tgt_output


def train_copy_task():
    """
    Train a small Transformer on the copy task.

    Success criteria:
      - Loss drops below 0.1 within 200 steps
      - Model can perfectly copy a test sequence
    """
    # ── Hyperparameters (small model for fast training) ──
    vocab_size = 10       # tokens 0-9
    d_model = 64          # small for speed (paper uses 512)
    num_layers = 2        # small for speed (paper uses 6)
    num_heads = 2         # small for speed (paper uses 8)
    d_ff = 128            # small for speed (paper uses 2048)
    dropout = 0.0         # no dropout for toy task
    seq_len = 10          # sequence length
    batch_size = 64       # batch size
    num_steps = 1000      # training steps
    warmup_steps = 200    # warmup (proportionally smaller)

    print("=" * 60)
    print("TRAINING: Toy Copy Task")
    print("=" * 60)
    print(f"Model: d_model={d_model}, layers={num_layers}, heads={num_heads}")
    print(f"Task:  copy sequences of length {seq_len}")
    print(f"Vocab: {vocab_size} tokens (0=pad, 1=SOS, 2-9=data)")
    print()

    # ── Create model ──
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        pad_idx=0,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # ── Optimizer ──
    # For the toy task we use a simple constant LR with Adam.
    # The Noam schedule (Module 7) is designed for the full-scale model
    # with d_model=512 and warmup_steps=4000 — it doesn't work well
    # at this tiny scale. In production you'd use:
    #   optimizer = Adam(params, lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    #   scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,             # constant learning rate for toy task
        betas=(0.9, 0.98),   # from the paper
        eps=1e-9,            # from the paper
    )

    # ── Loss: standard cross-entropy for the toy task ──
    # For the full model you'd use LabelSmoothingLoss with smoothing=0.1
    criterion = LabelSmoothingLoss(
        vocab_size=vocab_size,
        padding_idx=0,
        smoothing=0.0,  # no smoothing for toy task (want exact copy)
    )

    # ── Training loop ──
    model.train()
    for step in range(1, num_steps + 1):
        # Generate a fresh batch
        src, tgt_input, tgt_output = generate_copy_data(batch_size, seq_len)

        # Forward pass
        logits = model(src, tgt_input)
        # logits: (batch, tgt_len, vocab_size)
        # tgt_output: (batch, tgt_len)

        # Compute loss
        loss = criterion(logits, tgt_output)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % 100 == 0 or step == 1:
            # Check accuracy: what fraction of predictions match the target?
            predictions = logits.argmax(dim=-1)  # (batch, tgt_len)
            correct = (predictions == tgt_output).float().mean().item()
            print(
                f"Step {step:>4d} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {correct:.2%}"
            )

    # ── Evaluation: can the model copy a test sequence? ──
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # Generate one test batch
        src, tgt_input, tgt_output = generate_copy_data(1, seq_len)

        logits = model(src, tgt_input)
        predictions = logits.argmax(dim=-1)

        src_tokens = src[0].tolist()
        pred_tokens = predictions[0].tolist()
        expected = tgt_output[0].tolist()

        print(f"Source:     {src_tokens}")
        print(f"Expected:   {expected}")
        print(f"Predicted:  {pred_tokens}")
        print(f"Match: {'✓ PASS' if pred_tokens == expected else '✗ FAIL'}")

    # ── Also test with greedy autoregressive decoding ──
    print("\n" + "-" * 60)
    print("AUTOREGRESSIVE GENERATION (no teacher forcing)")
    print("-" * 60)

    with torch.no_grad():
        src, _, tgt_output = generate_copy_data(1, seq_len)

        # Encode the source
        src_mask = model.make_src_mask(src)
        src_embedded = model.positional_encoding(model.src_embedding(src))
        encoder_output = model.encoder(src_embedded, src_mask)

        # Start with just <SOS>
        generated = torch.ones(1, 1, dtype=torch.long)  # [[1]]

        for i in range(seq_len + 1):
            # Create target mask for what we've generated so far
            tgt_mask = model.make_tgt_mask(generated)

            # Decode
            tgt_embedded = model.positional_encoding(model.tgt_embedding(generated))
            decoder_output = model.decoder(
                tgt_embedded, encoder_output, src_mask, tgt_mask
            )

            # Get the last position's prediction
            next_token_logits = model.output_projection(decoder_output[:, -1, :])
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

        src_tokens = src[0].tolist()
        gen_tokens = generated[0, 1:].tolist()  # skip the initial SOS

        print(f"Source:     {src_tokens}")
        print(f"Generated:  {gen_tokens}")
        print(f"Match: {'✓ PASS' if gen_tokens[:seq_len] == src_tokens else '✗ FAIL'}")

    return model


if __name__ == "__main__":
    train_copy_task()
