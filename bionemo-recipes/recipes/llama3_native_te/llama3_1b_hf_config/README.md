# Llama-3-1B HuggingFace Configuration (Standard, No TE)

Local 1B Llama configuration for genomic training using standard HuggingFace implementation (no Transformer Engine).

## Architecture

- **Layers**: 16
- **Hidden size**: 2048
- **Attention heads**: 32 (with 8 KV heads for GQA)
- **Intermediate size**: 8192
- **Vocabulary**: 256 (nucleotide tokenizer)
- **Context**: 8192
- **Total parameters**: ~1.2B

## Key Features

- **Standard HuggingFace**: Uses `LlamaForCausalLM` (not NVLlama/TE)
- **No gated access**: Local config, no HuggingFace download required
- **Genomic tokenizer**: vocab_size=256, special tokens for nucleotides

## Usage

```yaml
# In your config:
model_tag: ./llama3_1b_hf_config
```

## Purpose

For testing convergence with standard HuggingFace Llama to compare against TE implementation.
