# Example Tiny Llama3 Checkpoint

This directory contains the model and tokenizer configuration for a tiny Llama3 model (~1M parameters) optimized for fast convergence testing on genomic sequences. This checkpoint is designed for quick sanity checks and convergence tests.

## Contents

- **config.json**: Model configuration for a tiny Llama3 model (4 layers, 384 hidden size)
- **tokenizer.json**: Fast tokenizer for nucleotide sequences (256 vocab size)
- **tokenizer_config.json**: Tokenizer configuration
- **special_tokens_map.json**: Special tokens mapping (EOS=0, PAD=1, BOS=2, UNK=3)

## Usage

Use this directory as the `model_tag` in your training configurations:

```yaml
# In your hydra config (e.g., L0_convergence configs)
model_tag: ./example_tiny_llama_checkpoint

dataset:
  tokenizer_path: ./example_tiny_llama_checkpoint  # Same directory for tokenizer
```

This eliminates the need for absolute paths and makes configurations portable across different environments.

## Model Parameters

- Layers: 4
- Hidden size: 384
- Attention heads: 6
- Intermediate size: 1536
- Vocabulary size: 256 (nucleotide tokenizer)
- Max position embeddings: 8192

Perfect for fast convergence testing where you want to verify the model can overfit on small datasets.



