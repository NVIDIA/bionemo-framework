# Example Small Llama3 Checkpoint

This directory contains the model and tokenizer configuration for a small Llama3 model (~10M parameters) optimized for genomic sequences. This checkpoint is designed for testing and development purposes, allowing unit tests to run without requiring external paths or complex configuration.

## Contents

- **config.json**: Model configuration for a small Llama3 model (4 layers, 2048 hidden size)
- **tokenizer.json**: Fast tokenizer for nucleotide sequences (256 vocab size)
- **tokenizer_config.json**: Tokenizer configuration
- **special_tokens_map.json**: Special tokens mapping (EOS=0, PAD=1, BOS=2, UNK=3)

## Usage

Use this directory as the `model_tag` in your training configurations:

```yaml
# In your hydra config
model_tag: ./example_small_llama_checkpoint

dataset:
  tokenizer_path: ./example_small_llama_checkpoint  # Same directory for tokenizer
```

This eliminates the need for absolute paths and makes configurations portable across different environments.

## Model Parameters

- Layers: 4
- Hidden size: 2048
- Attention heads: 16
- Intermediate size: 8192
- Vocabulary size: 256 (nucleotide tokenizer)
- Max position embeddings: 8192


