# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This is the code repository for "Build a Large Language Model (From Scratch)" by Sebastian Raschka. It implements a GPT-like LLM from scratch using PyTorch, covering tokenization, attention mechanisms, model architecture, pretraining, and finetuning.

## Development Commands

### Environment Setup

```bash
# Install dependencies using pip (quickest method)
pip install -r requirements.txt

# Alternative: Using uv (faster package installer)
uv pip install --system -r requirements.txt

# Install bonus materials dependencies
uv pip install --group bonus

# Alternative: Using pixi
pixi install
```

### Running Tests

```bash
# Run all main chapter tests
pytest ch04/01_main-chapter-code/tests.py
pytest ch05/01_main-chapter-code/tests.py
pytest ch06/01_main-chapter-code/tests.py

# Run specific component tests
pytest ch03/02_bonus_efficient-multihead-attention/tests/test_mha_implementations.py
pytest ch04/03_kv-cache/tests.py
pytest ch05/07_gpt_to_llama/tests/tests_rope_and_parts.py

# Test Jupyter notebooks
pytest --nbval ch02/01_main-chapter-code/dataloader.ipynb
pytest --nbval ch03/01_main-chapter-code/multihead-attention.ipynb

# Run package tests
pytest pkg/llms_from_scratch/tests/
```

### Linting and Style Checks

```bash
# Run ruff linter (project uses ruff instead of flake8)
ruff check .

# Ruff configuration in pyproject.toml:
# - Line length: 140 characters
# - Ignores: C406, E226, E402, E702, E703, E722, E731, E741
```

### Training Models

```bash
# Pretrain GPT model (Chapter 5)
python ch05/01_main-chapter-code/gpt_train.py

# Generate text with pretrained model
python ch05/01_main-chapter-code/gpt_generate.py

# Download pretrained GPT weights
python ch05/01_main-chapter-code/gpt_download.py

# Finetune for classification (Chapter 6)
python ch06/01_main-chapter-code/gpt_class_finetune.py

# Finetune for instruction following (Chapter 7)
python ch07/01_main-chapter-code/gpt_instruction_finetuning.py

# Evaluate with Ollama
python ch07/01_main-chapter-code/ollama_evaluate.py
```

## Code Architecture

### Chapter Organization

The repository is structured by book chapters (ch01-ch07, appendices A-E). Each chapter contains:
- `01_main-chapter-code/`: Primary chapter code with main notebook and Python modules
- `02_bonus_*/`: Optional bonus materials and experiments
- Additional numbered directories for specific bonus topics

### Core Model Components

**Token and Position Embeddings** (ch02):
- `GPTDatasetV1`: Sliding window dataset with tokenization using tiktoken
- `create_dataloader_v1`: DataLoader factory with configurable stride and batch size

**Attention Mechanism** (ch03):
- `MultiHeadAttention`: Multi-head self-attention with causal masking
- Implements scaled dot-product attention with dropout
- Uses `register_buffer` for causal mask

**Model Architecture** (ch04):
- `GPTModel`: Complete transformer with token/position embeddings, transformer blocks, and output head
- `TransformerBlock`: Single transformer layer with attention, feedforward, and residual connections
- `FeedForward`: 4x expansion MLP with GELU activation
- `LayerNorm`: Custom layer normalization implementation

**Training Pipeline** (ch05):
- `train_model_simple`: Main training loop with evaluation
- `calc_loss_batch` / `calc_loss_loader`: Loss computation utilities
- `generate_and_print_sample`: Text generation for monitoring training progress

### Reusable Package (pkg/llms_from_scratch)

The `pkg/` directory contains a PyPI-installable package with modules organized by chapter:
- `ch02.py` through `ch07.py`: Chapter-specific utilities
- `generate.py`: Text generation utilities
- `llama3.py`, `qwen3.py`: Alternative model architectures
- `kv_cache/`, `kv_cache_batched/`: Key-value cache implementations
- `utils.py`: Shared utility functions

### previous_chapters.py Pattern

Each chapter includes a `previous_chapters.py` module that imports required components from earlier chapters, allowing notebooks to focus on new concepts without re-implementing foundational code.

### Model Configuration

Models use dictionary-based configuration with keys:
- `vocab_size`: Tokenizer vocabulary size (default: 50257 for GPT-2)
- `context_length`: Maximum sequence length
- `emb_dim`: Embedding dimension
- `n_heads`: Number of attention heads
- `n_layers`: Number of transformer blocks
- `drop_rate`: Dropout probability
- `qkv_bias`: Whether to use bias in QKV projections

## Hardware and Performance

The code is designed to run on conventional laptops (tested on M3 MacBook Air). It automatically uses CUDA if available. GPU acceleration significantly improves runtime for chapters 5-7 (pretraining and finetuning).

## Testing Strategy

- Main chapter code has corresponding `tests.py` files
- Notebooks can be validated with `pytest --nbval`
- CI/CD runs tests on Linux, macOS, and Windows
- Uses PyTorch 2.2.2+ and Python 3.10-3.13

## Dependencies

Core dependencies:
- PyTorch >= 2.2.2
- tiktoken (GPT-2 tokenizer)
- matplotlib (visualization)
- TensorFlow >= 2.16.2 (for loading pretrained weights)
- JupyterLab >= 4.0

Optional bonus dependencies (install with `--group bonus`):
- transformers, safetensors (HuggingFace integration)
- chainlit (UI interfaces)
- sentencepiece (alternative tokenizers)

## Logging Configuration

The repository uses a custom logging configuration (`logger_config.yaml`) with rotating file handlers. Logs are written to `logs/logger.log` with a maximum size of 1MB and 3 backup files.

## Git Branch Strategy

- Current branch: `experiment` (custom development work)
- Contains custom tokenizer implementation in `experiment/tokenizer.py`
- Main development follows the book structure on the `main` branch
