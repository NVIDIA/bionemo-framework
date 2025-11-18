# Llama-3.1-8B Configuration

This directory contains the model configuration for Meta's Llama-3.1-8B architecture (~8B parameters) for genomic sequence pretraining.

## Contents

- **config.json**: Official Llama-3.1-8B architecture configuration

## Usage

Use this directory as the `model_tag` in your training configurations to avoid HuggingFace authentication and internet requirements:

```yaml
# In your hydra config (e.g., train_8b_opengenome2.yaml)
model_tag: ./llama3_8b_config
```

This eliminates the need for:

- Internet access on compute nodes
- HuggingFace authentication tokens
- Downloading configs at runtime

## Model Architecture

- **Layers**: 32
- **Hidden size**: 4096
- **Attention heads**: 32 (with 8 KV heads for Grouped Query Attention)
- **Intermediate size**: 14336
- **Vocabulary size**: 128256 (standard Llama 3.1 tokenizer vocab)
- **Max position embeddings**: 131072 (128K context window)
- **RoPE theta**: 500000.0
- **RoPE scaling**: Llama3 style with 8.0 factor

## Important Notes

1. **This is architecture-only**: No pretrained weights are included or loaded. Training starts from random initialization, which is appropriate for genomic sequence pretraining.

2. **Tokenizer**: You still need to provide your own nucleotide tokenizer via `dataset.tokenizer_path` in your config.

3. **Vocabulary size mismatch**: The standard Llama vocab is 128256, but if your nucleotide tokenizer has a different vocab size (e.g., 256), the model will be automatically resized during initialization.

## Comparison with Tiny Test Model

| Feature         | Tiny (example_checkpoint) | Full (llama3_8b_config) |
| --------------- | ------------------------- | ----------------------- |
| Layers          | 4                         | 32                      |
| Hidden size     | 384                       | 4096                    |
| Attention heads | 6                         | 32 (8 KV)               |
| Parameters      | ~9.6M                     | ~8B                     |
| Purpose         | Fast testing              | Production training     |
| Context length  | 8192                      | 131072                  |
