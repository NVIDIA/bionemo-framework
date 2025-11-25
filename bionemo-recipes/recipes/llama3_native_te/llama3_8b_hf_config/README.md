# Llama-3-8B HuggingFace Configuration (Standard, No TE)

This directory contains the model configuration for standard Llama-3-8B architecture (8B parameters) for genomic sequence pretraining, **without Transformer Engine (TE) layers**.

## Key Differences from `llama3_8b_genomic_config`

| Feature      | llama3_8b_genomic_config        | llama3_8b_hf_config (this)     |
| ------------ | ------------------------------- | ------------------------------ |
| Architecture | NVLlamaForCausalLM (TE)         | LlamaForCausalLM (standard HF) |
| Layers       | TransformerLayer (TE fused ops) | Standard LlamaDecoderLayer     |
| Use case     | Production training             | Convergence testing / baseline |

## Purpose

This config is for **convergence testing** to isolate whether training issues are from:

- Transformer Engine implementation, OR
- Genomic collator / data preprocessing

## Configuration Details

- **Vocabulary size**: 256 (nucleotide tokenizer)
- **Context length**: 8192
- **Architecture**: 32 layers, 4096 hidden size, 32 attention heads
- **Special tokens**: EOS=0, BOS=2, PAD=1 (nucleotide conventions)
- **No TE**: Uses standard PyTorch/HuggingFace implementation

## Usage

```yaml
# In hydra config:
model_tag: ./llama3_8b_hf_config

dataset:
  tokenizer_path: ./example_checkpoint  # Nucleotide tokenizer
```

## Notes

1. **No pretrained weights**: Trains from random initialization
2. **Standard Llama**: Uses HuggingFace's LlamaForCausalLM (not NVLlama)
3. **Same architecture as TE version**: Only difference is implementation (TE vs standard)
4. **For comparison**: Use to test if convergence issues are TE-related
