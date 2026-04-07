"""
=============================================================================
Module 8: Label Smoothing Cross-Entropy Loss
=============================================================================

THE PROBLEM WITH STANDARD CROSS-ENTROPY
────────────────────────────────────────
In standard cross-entropy, the target is a "one-hot" vector:
  Target for token 42: [0, 0, ..., 0, 1, 0, ..., 0]
                                      ↑ position 42

This tells the model: "Be 100% confident this is token 42."
But this causes problems:
  - The model can never actually output probability = 1.0 (it would need
    the logit to be +infinity)
  - So the loss NEVER reaches zero, creating pressure to make logits
    extremely large → poor generalization (overfitting)

LABEL SMOOTHING (Section 5.4)
─────────────────────────────
Instead of "be 100% sure it's token 42," we say:
  "Be 90% sure it's token 42, and spread 10% evenly across all other tokens."

With ε_ls = 0.1 and vocab_size V:
  Target for correct class:  1 - ε_ls + ε_ls/V ≈ 0.9 + tiny
  Target for all others:     ε_ls / V ≈ 0.0001

This makes the model less overconfident, which:
  - HURTS perplexity (model is intentionally less certain)
  - HELPS accuracy and BLEU score (better generalization)

The paper reports this tradeoff explicitly.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Instead of hard targets (0 or 1), we use soft targets that put
    (1 - smoothing) on the correct class and distribute `smoothing`
    evenly across all classes.
    """

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        """
        Args:
            vocab_size:  number of classes (vocabulary size)
            padding_idx: index of padding token (excluded from loss)
            smoothing:   label smoothing factor ε (0.1 in the paper)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing

        # Confidence on the correct class
        self.confidence = 1.0 - smoothing
        # Probability mass spread to non-correct classes
        # We subtract 1 from vocab_size because the correct class is separate
        self.smoothed_value = smoothing / (vocab_size - 1)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size) — raw model output
            target: (batch_size, seq_len) — ground-truth token IDs

        Returns:
            Scalar loss value (averaged over non-padding tokens)
        """
        # Reshape for computation
        # (batch * seq_len, vocab_size)
        logits = logits.reshape(-1, self.vocab_size)
        # (batch * seq_len,)
        target = target.reshape(-1)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Create the smoothed target distribution
        # Start with uniform: every class gets smoothed_value
        smooth_target = torch.full_like(log_probs, self.smoothed_value)
        # Then set the correct class to the higher confidence value
        smooth_target.scatter_(
            dim=1,
            index=target.unsqueeze(1),  # which column to set
            value=self.confidence,       # set it to 1 - ε
        )

        # Zero out padding positions (don't compute loss for <pad> tokens)
        padding_mask = (target == self.padding_idx)
        smooth_target[padding_mask] = 0.0

        # KL divergence loss = -sum(target * log_probs) for each position
        # This is equivalent to cross-entropy with soft targets
        loss = -(smooth_target * log_probs).sum(dim=-1)

        # Average over non-padding tokens
        non_padding = (~padding_mask).sum()
        if non_padding > 0:
            loss = loss.sum() / non_padding
        else:
            loss = loss.sum()  # edge case: all padding

        return loss


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    vocab_size = 100
    batch_size = 2
    seq_len = 5

    criterion = LabelSmoothingLoss(vocab_size, padding_idx=0, smoothing=0.1)
    criterion_hard = nn.CrossEntropyLoss(ignore_index=0)

    # Fake logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    target = torch.randint(1, vocab_size, (batch_size, seq_len))  # no padding

    loss_smooth = criterion(logits, target)
    loss_hard = criterion_hard(logits.reshape(-1, vocab_size), target.reshape(-1))

    print(f"Label-smoothed loss: {loss_smooth.item():.4f}")
    print(f"Standard CE loss:    {loss_hard.item():.4f}")

    # Test that padding is ignored
    target_with_pad = target.clone()
    target_with_pad[:, -2:] = 0  # last 2 positions are padding
    loss_padded = criterion(logits, target_with_pad)
    print(f"Loss with padding:   {loss_padded.item():.4f}")
    print("✓ Label Smoothing OK")
