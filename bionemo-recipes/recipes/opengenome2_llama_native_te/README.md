# OpenGenome2 Llama3 Native TE Recipe

Self-contained training recipe for OpenGenome2 genomic language models using a Llama3 architecture
with TransformerEngine layers and FSDP2.

## Features

- **Spike-No-More embedding init** (std=1.0) for training stability
- **Megatron-style scaled init** for residual output layers (proj/fc2)
- **FP32 master weights** with BF16 compute via MixedPrecisionPolicy
- **Weight decay grouping** (skip bias and 1D params, optionally embeddings)
- **Genomic masking** for degenerate bases (N, R, Y, etc.)
- **THD sequence packing** for efficient variable-length training
- **FP8 training** with configurable first/last layer BF16 override

## Quick Start

```bash
# Build the container
docker build -t og2_llama_te .

# Single-node training (8 GPUs)
torchrun --nproc_per_node=8 train_fsdp2.py --config-name og2_7b_thd_gqa \
    checkpoint.ckpt_dir=/output/checkpoints \
    wandb.project=my-project

# Multi-node training (6 nodes x 8 GPUs)
torchrun --nproc_per_node=8 --nnodes=6 --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train_fsdp2.py --config-name og2_7b_thd_gqa \
    checkpoint.ckpt_dir=/output/checkpoints
```

## Hydra Configs

| Config                          | Description                                     |
| ------------------------------- | ----------------------------------------------- |
| `og2_7b_thd_gqa`                | Main 7B GQA config (BF16 + FP32 master weights) |
| `og2_7b_thd_gqa_global_shuffle` | Pre-chunked parquet shard variant               |
| `og2_7b_thd_gqa_fp8`            | FP8 variant with Float8BlockScaling             |
| `L0_sanity`                     | Tiny model for CI/CD testing                    |

## Testing

```bash
pytest -v tests/
```
