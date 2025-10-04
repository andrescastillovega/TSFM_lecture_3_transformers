## Lecture 3 — NumPy Transformer: Backprop Validation and Training Assignment

This directory contains a NumPy implementation of Transformer components with both forward and backward passes. The implementations are validated against PyTorch to ensure gradient correctness.

### What you have

- `transformer.py`: NumPy implementations for core Transformer components and a full block, each with forward and backward passes.
- `test_backward_passes.py`: Tests that compare NumPy gradients against PyTorch for correctness.
- `main.py`: Minimal entry point.

### Setup

- Python: 3.13 (see `pyproject.toml`).
- Install dependencies:
```bash
cd /Users/sid/lecture-3-code
uv sync
# or create a venv and `pip install -e .`
```

Run the provided gradient checks:
```bash
python /Users/sid/lecture-3-code/test_backward_passes.py
```

---

## Assignment

1) Add causal masking to the attention mechanism in this NumPy implementation for autoregressive language modeling.

2) Use the TinyStories dataset from Assignment 2 and a BPE tokenizer (either the one you trained previously or an existing tokenizer) to prepare tokenized training and validation splits.

3) Train a small Transformer language model end-to-end using only this NumPy implementation. Your training should include:
   - Token embedding and positional information
   - One or more Transformer blocks
   - A language modeling head that produces next-token logits
   - An appropriate training objective for next-token prediction

4) Report: Provide a brief summary of your training configuration, training/validation loss curves, and sample generations.

### Constraints and expectations

- Implement causal masking within this NumPy stack.
- Use TinyStories from Assignment 2 and a BPE tokenizer (yours or an existing one).
- Keep your implementation self-contained in NumPy for the forward and backward passes.
- Ensure gradient checks continue to pass for unmasked cases.
- Provide clear instructions or scripts to reproduce your results.

### Notes

- You may organize additional code (data loading, training loop, etc.) as you see fit in this directory.
- If you add new dependencies (e.g., for tokenization or data handling), document them and how to install.


