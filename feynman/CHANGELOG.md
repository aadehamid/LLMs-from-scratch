# CHANGELOG

## 2026-04-07 — Transformer from Scratch: Complete Implementation

**Objective:** Build a heavily-commented, from-scratch Transformer implementation for learning purposes.

**What was built:**
- 10 Python modules covering every component of the original Transformer
- Bottom-up reading order: embeddings → attention → FFN → encoder → decoder → full model → training utilities
- 17 verification tests covering shapes, masks, parameter count, gradient flow

**Verification results:**
- ✅ 17/17 tests pass
- ✅ Parameter count: 63.1M (paper: ~65M) — matches after weight sharing fix
- ✅ Copy task: 100% accuracy by step 400 (both teacher-forced and autoregressive)
- ✅ Causal mask blocks future positions correctly
- ✅ LR schedule peaks at warmup step 4000

**Issues encountered and fixed:**
- Python files can't start with digits → renamed from `01_embeddings.py` to `embeddings.py`
- Initial param count was 101M → added weight sharing (Section 3.4) to match paper's ~65M
- Noam LR schedule doesn't work at tiny d_model=64 scale → used constant LR=1e-3 for toy task

**Environment:** Python 3.11 venv via `uv`, PyTorch 2.11, CPU-only
