"""
=============================================================================
Module 7: Noam Learning Rate Schedule
=============================================================================

THE PROBLEM
───────────
Choosing the right learning rate is tricky:
  - Too high → training diverges (loss explodes)
  - Too low  → training is painfully slow
  - Fixed LR → can't adapt as training progresses

THE PAPER'S SOLUTION (Section 5.3)
──────────────────────────────────
A custom schedule with two phases:

  Phase 1 — WARMUP (steps 1 to 4000):
    Learning rate increases LINEARLY from ~0 up to a peak.
    Why? In the beginning, the model's parameters are random. Large gradients
    with a high LR would cause wild updates. Warmup lets the model "settle"
    before cranking up the learning rate.

  Phase 2 — DECAY (steps 4001+):
    Learning rate decreases proportionally to 1/√step.
    Why? As training progresses, we want finer adjustments — big LR changes
    would overshoot the good solution we're converging toward.

FORMULA:
  lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))

Let's unpack this:
  - d_model^(-0.5) = 1/√512 ≈ 0.044 — scales LR inversely with model size
  - min(A, B) switches between:
    - B = step × warmup^(-1.5): linear increase (B is smaller when step < warmup)
    - A = step^(-0.5):          inverse-sqrt decay (A is smaller when step > warmup)
  - At step = warmup_steps, both terms are equal → that's the peak LR

PEAK LR = d_model^(-0.5) × warmup_steps^(-0.5)
         = 1/√512 × 1/√4000 ≈ 0.00070
=============================================================================
"""

import torch


class NoamScheduler:
    """
    The "Noam" learning rate schedule from the Transformer paper.

    Named after Noam Shazeer, one of the paper's authors who proposed it.

    Usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)

        for batch in data:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()   # update LR after each step
            optimizer.zero_grad()

    NOTE: We set the optimizer's base lr=1.0 because the scheduler overwrites
    it completely. The formula computes the absolute LR from scratch at each step.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
    ):
        """
        Args:
            optimizer:    the Adam optimizer
            d_model:      model dimension (512) — used to scale the LR
            warmup_steps: how many steps to linearly increase LR (4000)
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        """Call this after every optimizer.step()."""
        self._step += 1
        lr = self._compute_lr()

        # Set the new learning rate on all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_lr(self) -> float:
        """
        lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
        """
        step = self._step

        # d_model^(-0.5) = 1 / sqrt(d_model)
        scale = self.d_model ** (-0.5)

        # The two competing terms:
        term_a = step ** (-0.5)                              # decay term
        term_b = step * (self.warmup_steps ** (-1.5))        # warmup term

        return scale * min(term_a, term_b)

    def get_lr(self) -> float:
        """Return the current learning rate (for logging)."""
        return self._compute_lr()


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create a dummy model and optimizer
    model = torch.nn.Linear(512, 512)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)

    # Print LR at key steps
    lrs = []
    for step in range(1, 20001):
        scheduler.step()
        if step in [1, 100, 1000, 2000, 4000, 8000, 16000, 20000]:
            lr = scheduler.get_lr()
            print(f"Step {step:>6d}: lr = {lr:.6f}")
            lrs.append((step, lr))

    # Verify: peak should be at warmup_steps=4000
    peak_lr = 512 ** (-0.5) * 4000 ** (-0.5)
    print(f"\nExpected peak LR at step 4000: {peak_lr:.6f}")
    print(f"Actual LR at step 4000:        {lrs[4][1]:.6f}")
    print(f"Match: {abs(lrs[4][1] - peak_lr) < 1e-8}")
    print("✓ LR Schedule OK")
