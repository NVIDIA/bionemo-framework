# Llama-3-8B Genomic Configuration

This directory contains the model configuration for a Llama-3-8B architecture (~8B parameters) optimized for genomic sequence pretraining, matching John's trained model configuration.

## Contents

- **config.json**: Llama-3-8B architecture configuration for genomic data

## Key Configuration Details

This config is specifically tailored for genomic sequences:

- **Vocabulary size**: 256 (nucleotide tokenizer, not standard Llama tokenizer)
- **Context length**: 8192 (standard, not extended 128K)
- **No RoPE scaling**: Uses standard RoPE (rope_theta=500000)
- **Special tokens**: EOS=0, BOS=null (nucleotide tokenizer conventions)

## Usage

Use this directory as the `model_tag` in your training configurations:

```yaml
# In your hydra config (e.g., train_8b_opengenome2.yaml)
model_tag: ./llama3_8b_genomic_config

dataset:
  tokenizer_path: ./example_checkpoint  # Nucleotide tokenizer with vocab_size=256
```

## Comparison with Standard Llama-3.1-8B

| Feature        | Standard Llama-3.1-8B | Genomic Config         |
| -------------- | --------------------- | ---------------------- |
| Vocabulary     | 128256 (text)         | 256 (nucleotides)      |
| Context length | 131072 (128K)         | 8192 (8K)              |
| RoPE scaling   | Llama3 extended       | Standard               |
| Special tokens | Standard Llama        | Nucleotide conventions |

## Important Notes

1. **This is architecture-only**: No pretrained weights are included. Training starts from random initialization.

2. **Tokenizer requirement**: You must provide a nucleotide tokenizer with exactly 256 tokens via `dataset.tokenizer_path`.

3. **Based on genomic training**: This configuration is optimized for genomic sequence pretraining with 8K context length.

## Model Architecture

- **Layers**: 32
- **Hidden size**: 4096
- **Attention heads**: 32 (with 8 KV heads for Grouped Query Attention)
- **Intermediate size**: 14336
- **Vocabulary size**: 256 (nucleotide tokenizer)
- **Max position embeddings**: 8192 (8K context window)
- **RoPE theta**: 500000
- **Total parameters**: ~8B
