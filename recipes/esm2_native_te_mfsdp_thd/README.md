# ESM-2 training with megatron-fsdp and custom pytorch training loop with sequence packing

Build the docker image with the following command:

```bash
docker build -t my_image .
```

## Running training

Run training with

```bash
docker run --rm -it --gpus all my_image torchrun train.py --config-name L0_sanity
```

## Sequence Packing for Efficient Training

This implementation uses **sequence packing** with THD (Total, Height, Depth) format to achieve maximum computational efficiency when training on variable-length protein sequences.

### The Problem with Traditional Padding

Traditional BERT-like models pad all sequences to the same length, leading to significant computational waste:

- **Memory waste**: Padding tokens consume GPU memory but provide no learning signal
- **FLOPS waste**: Every layer processes padding tokens through expensive operations (attention, feed-forward)
- **Scaling issues**: Waste increases with batch size and sequence length variance

For protein sequences with high length variability (50-1000+ amino acids), padding can waste **65-90% of computation**.

### Sequence Length Distribution Analysis

Our analysis of the training dataset reveals significant length variability that makes sequence packing particularly beneficial:

![Sequence Length Distribution](sequence_length_distribution.png)

**Key Statistics**:

- **Mean length**: 325 tokens (after tokenization)
- **Median length**: 286 tokens
- **Standard deviation**: 218 tokens (high variability!)
- **Range**: 50-1024 tokens
- **75th percentile**: 406 tokens
- **95th percentile**: 802 tokens

**Padding Impact**:
With traditional padding to `max_length=1024`:

- **Current efficiency**: Only 31.8% of computation is useful
- **Padding waste**: 68.2% of FLOPS wasted on meaningless tokens
- **Memory waste**: 2/3 of GPU memory stores padding tokens

This distribution perfectly demonstrates why sequence packing is essential for protein language models - the high variance in sequence lengths makes padding extremely inefficient.

### Our Solution: THD Format with Sequence Packing

Instead of padding, we:

1. **Concatenate sequences** without padding tokens
2. **Pack multiple sequences** into efficient batches
3. **Use Transformer Engine w/ Flash Attention** with sequence boundary metadata (`cu_seq_lens`)
4. **Achieve 100% computational efficiency** - every FLOP contributes to learning

### Implementation Details

The `MLMDataCollatorWithFlattening` combines:

- `DataCollatorWithFlattening`: Packs sequences into THD format
- `DataCollatorForLanguageModeling`: Applies MLM masking to packed sequences

**Input**: Variable-length sequences

```python
[
    {"input_ids": [0, 5, 6, 7, 2]},  # 5 tokens
    {"input_ids": [0, 8, 9, 10, 11, 2]},  # 6 tokens
    {"input_ids": [0, 12, 13, 2]},  # 4 tokens
]
```

**Output**: Packed THD format

```python
{
    "input_ids": tensor(
        [[0, 5, 6, 7, 2, 0, 8, 9, 10, 11, 2, 0, 12, 13, 2]]
    ),  # Shape: [1, 15]
    "cu_seq_lens_q": tensor([0, 5, 11, 15]),  # Sequence boundaries
    "labels": tensor([[-100, 13, -100, -100, -100, ...]]),  # MLM targets
}
```

### Performance Benefits

**Memory Efficiency**:

- Traditional BSHD: ~31.8% efficiency (68.2% padding waste)
- Our THD format: 100% efficiency (0% waste)
- **Result**: 3x more real tokens processed per GPU memory unit

**FLOPS Efficiency**:

- Traditional: Hardware FLOPS ≠ Effective FLOPS (due to padding)
- Our approach: Hardware FLOPS = Effective FLOPS (perfect alignment)
- **Result**: 65%+ reduction in computational overhead

**Scaling Benefits**:

- Process 3x more protein sequences in the same memory
- Train on longer sequences without padding penalties
- Better gradient updates (no padding dilution)

### Flash Attention Compatibility

The THD format is optimized for Flash Attention's variable-length sequence processing:

- No attention masks needed (no padding to mask)
- Sequence boundaries provided via `cu_seq_lens`
- Perfect memory access patterns for GPU efficiency

# Lepton script

```bash
#!/bin/bash

# Download the environment setup script from Lepton's GitHub repository, make it executable, and source it to initialize the environment variables.
wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh
chmod +x init.sh
source init.sh

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

PIP_CONSTRAINT= pip install -r requirements.txt

torchrun \
    --rdzv_id ${LEPTON_JOB_NAME} \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --nproc-per-node ${GPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node-rank ${NODE_RANK} \
    train.py --config-name L1_15B_perf_test
```
