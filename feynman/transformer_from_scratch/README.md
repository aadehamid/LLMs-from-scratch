# Transformer from Scratch

A heavily-commented, from-scratch implementation of **"Attention Is All You Need"** (Vaswani et al., 2017) in PyTorch. Every line is annotated to teach you the architecture.

## Reading Order

Read these files in sequence — each builds on the previous:

| # | File | What You'll Learn |
|---|------|------------------|
| 1 | `embeddings.py` | Token embeddings, sinusoidal positional encoding, why √d_model scaling |
| 2 | `attention.py` | Scaled dot-product attention, multi-head attention, causal masking |
| 3 | `feedforward.py` | Position-wise FFN, why the 4× expansion |
| 4 | `encoder.py` | Encoder layer (self-attention + FFN + residuals + LayerNorm), full stack |
| 5 | `decoder.py` | Decoder layer (masked self-attn + cross-attn + FFN), causal masking |
| 6 | `transformer.py` | Full encoder-decoder, weight sharing, mask creation |
| 7 | `lr_schedule.py` | Noam warmup-then-decay learning rate schedule |
| 8 | `label_smoothing.py` | Label smoothing loss and why it helps generalization |
| 9 | `train_toy.py` | Training loop on a copy task (verifies everything works) |
| 10 | `run_tests.py` | 17 verification tests: shapes, masks, param count, gradient flow |

## Quick Start

```bash
# Activate the venv
source .venv/bin/activate

# Run all verification tests (17 checks)
PYTHONPATH=.. python run_tests.py

# Train on the copy task (~10 seconds, reaches 100% accuracy)
PYTHONPATH=.. python train_toy.py
```

## Architecture Summary (Base Model)

```
Parameter          Value    Meaning
─────────────────────────────────────────────────
N (layers)         6        Encoder & decoder depth
d_model            512      Embedding / residual dimension
d_ff               2048     Feed-forward inner dimension (4×)
h (heads)          8        Parallel attention heads
d_k = d_v          64       Per-head dimension (512 / 8)
Dropout            0.1      Applied after each sub-layer
Parameters         ~63M     With weight sharing
```

## Verification Results

- ✅ 17/17 shape, mask, parameter, and gradient tests pass
- ✅ Copy task: 100% accuracy (teacher forcing + autoregressive)
- ✅ Parameter count: 63.1M (paper reports ~65M)
- ✅ Causal mask correctly blocks future positions
- ✅ LR schedule peaks at warmup step 4000

## Paper Reference

**"Attention Is All You Need"**
Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017)
https://arxiv.org/abs/1706.03762
