# BioNeMo Geneformer-T: Temporal/Neighbor-Aware Fine-tuning

BioNeMo Geneformer-T is a temporal extension of the Geneformer model that enables next-cell prediction training using neighbor information from Single Cell Data Loader (SCDL) datasets.

## Overview

This package implements a temporal training strategy for Geneformer models, where the model learns to predict the next cell in a developmental trajectory or temporal sequence. The key innovation is the use of neighbor information from SCDL datasets to create training pairs of current and next cells.

### Key Features

- **Temporal Training**: Train Geneformer models to predict next cells in developmental trajectories
- **Neighbor-Aware**: Leverages SCDL neighbor information for creating training pairs
- **Masked Language Modeling**: Uses MLM on next cell while preserving current cell representation
- **Custom Attention Masks**: Implements temporal attention masks to prevent information leakage
- **Lightning Integration**: Built on PyTorch Lightning for scalable training
- **Flexible Configuration**: Supports various model sizes and training configurations

## Architecture

The temporal Geneformer model extends the base Geneformer with:

1. **Temporal Dataset**: Creates training samples with current and next cell pairs
2. **Temporal Attention Masks**: Prevents next cell tokens from attending to each other
3. **Masked Language Modeling**: Applies masking only to next cell tokens
4. **Custom Loss Function**: Computes loss only on masked next cell tokens

### Training Strategy

```
[CLS] current_cell_genes [SEP] [MASK] next_cell_genes [MASK] [SEP]
  |                        |                                    |
  └─ No masking           └─ Separator token    └─ Masked tokens (15% prob)
```

## Installation

```bash
# Install the package
pip install -e sub-packages/bionemo-geneformer-t/

# Or install with dependencies
pip install -e sub-packages/bionemo-geneformer-t/[dev]
```

## Quick Start

### 1. Prepare Your Data

Your SCDL dataset should have neighbor information. The expected structure:

```
data/
├── train/
│   ├── data.h5ad          # Single cell data
│   ├── neighbors.json     # Neighbor information
│   ├── medians.json       # Gene expression medians
│   └── geneformer.vocab   # Tokenizer vocabulary
├── val/
│   └── ...
└── test/
    └── ...
```

### 2. Basic Training

```python
from bionemo.geneformer_t.run.train_temporal import train_temporal_geneformer

# Train temporal Geneformer
train_temporal_geneformer(
    data_dir="path/to/your/scdl/data",
    output_dir="path/to/output",
    seq_length=2048,
    mask_prob=0.15,
    micro_batch_size=8,
    num_epochs=10,
    learning_rate=1e-4,
)
```

### 3. Command Line Training

```bash
python -m bionemo.geneformer_t.run.train_temporal \
    --data_dir /path/to/scdl/data \
    --output_dir /path/to/output \
    --seq_length 2048 \
    --mask_prob 0.15 \
    --micro_batch_size 8 \
    --num_epochs 10 \
    --use_wandb \
    --wandb_project temporal-geneformer
```

## Advanced Usage

### Custom Model Configuration

```python
from bionemo.geneformer_t.data.temporal_datamodule import TemporalGeneformerDataModule
from bionemo.geneformer_t.model.temporal_model import TemporalGeneformerModel
from bionemo.llm.model.biobert import BioBertConfig

# Create custom model configuration
config = BioBertConfig(
    vocab_size=25426,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=2048,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

# Create model
model = TemporalGeneformerModel(config=config)

# Create data module
data_module = TemporalGeneformerDataModule(
    data_dir="path/to/data",
    seq_length=2048,
    mask_prob=0.15,
    neighbor_key="next_cell_ids",
    micro_batch_size=8,
)
```

### Using Different Neighbor Keys

```python
# Use different neighbor relationships
data_module = TemporalGeneformerDataModule(
    data_dir="path/to/data",
    neighbor_key="temporal_neighbors",  # or "knn_neighbors", "trajectory_neighbors"
    only_cells_with_neighbors=True,
)
```

### Prediction and Inference

```python
import torch

# Load trained model
model = TemporalGeneformerModel.load_from_checkpoint("path/to/checkpoint.ckpt")

# Predict next cell
current_cell_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])  # [batch_size, seq_len]
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])   # [batch_size, seq_len]

predictions = model.predict_next_cell(
    current_cell_ids=current_cell_ids,
    attention_mask=attention_mask,
    num_predictions=10
)
```

## Data Format

### SCDL Dataset Structure

The temporal Geneformer expects SCDL datasets with neighbor information:

```json
{
  "neighbors": {
    "cell_id_1": ["neighbor_1", "neighbor_2"],
    "cell_id_2": ["neighbor_3", "neighbor_4"],
    ...
  }
}
```

### Supported Neighbor Types

- `next_cell_ids`: Direct temporal successors
- `knn_neighbors`: K-nearest neighbors in expression space
- `trajectory_neighbors`: Neighbors along developmental trajectories
- `pseudotime_neighbors`: Neighbors in pseudotime ordering

## Model Architecture Details

### Temporal Attention Mechanism

The model uses custom attention masks to implement the temporal training strategy:

```python
# Temporal attention mask structure
# 1 = can attend, 0 = cannot attend
mask = [
    [1, 1, 1, 1, 1, 1, 0, 0, 0],  # CLS can attend to current cell + SEP
    [1, 1, 1, 1, 1, 1, 0, 0, 0],  # Current cell tokens
    [1, 1, 1, 1, 1, 1, 0, 0, 0],  # ...
    [1, 1, 1, 1, 1, 1, 0, 0, 0],  # ...
    [1, 1, 1, 1, 1, 1, 0, 0, 0],  # ...
    [1, 1, 1, 1, 1, 1, 0, 0, 0],  # SEP token
    [1, 1, 1, 1, 1, 1, 0, 0, 0],  # Next cell tokens (cannot attend to each other)
    [1, 1, 1, 1, 1, 1, 0, 0, 0],  # ...
    [1, 1, 1, 1, 1, 1, 0, 0, 0],  # ...
]
```

### Loss Computation

Only masked tokens in the next cell contribute to the loss:

```python
# Loss mask structure
# 1 = compute loss, 0 = ignore
loss_mask = [0, 0, 0, 0, 0, 0, 1, 0, 1]  # Only masked next cell tokens
```

## Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_length` | 2048 | Maximum sequence length |
| `mask_prob` | 0.15 | Probability of masking next cell tokens |
| `mask_token_prob` | 0.8 | Probability of using [MASK] token |
| `random_token_prob` | 0.1 | Probability of using random token |
| `neighbor_key` | "next_cell_ids" | Key for neighbor data in SCDL |
| `only_cells_with_neighbors` | True | Only include cells with neighbors |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 512 | Hidden dimension size |
| `num_layers` | 8 | Number of transformer layers |
| `num_attention_heads` | 8 | Number of attention heads |
| `intermediate_size` | 2048 | FFN intermediate size |
| `dropout_prob` | 0.1 | Dropout probability |

## Examples

### Example 1: Basic Training

```python
from bionemo.geneformer_t.run.train_temporal import train_temporal_geneformer

train_temporal_geneformer(
    data_dir="data/pancreas_development",
    output_dir="outputs/temporal_model",
    seq_length=1024,
    mask_prob=0.15,
    micro_batch_size=16,
    num_epochs=20,
    learning_rate=5e-5,
    weight_decay=0.01,
)
```

### Example 2: Multi-GPU Training

```bash
python -m bionemo.geneformer_t.run.train_temporal \
    --data_dir data/large_dataset \
    --output_dir outputs/multi_gpu \
    --accelerator gpu \
    --devices 4 \
    --strategy ddp \
    --micro_batch_size 4 \
    --global_batch_size 64 \
    --precision 16-mixed
```

### Example 3: Custom Configuration

```python
import lightning as L
from bionemo.geneformer_t.data.temporal_datamodule import TemporalGeneformerDataModule
from bionemo.geneformer_t.model.temporal_model import TemporalGeneformerModel
from bionemo.llm.model.biobert import BioBertConfig

# Custom configuration
config = BioBertConfig(
    vocab_size=30000,
    hidden_size=1024,
    num_hidden_layers=16,
    num_attention_heads=16,
    intermediate_size=4096,
)

# Create components
data_module = TemporalGeneformerDataModule(
    data_dir="data/custom_dataset",
    neighbor_key="trajectory_neighbors",
    mask_prob=0.2,
    seq_length=1024,
)

model = TemporalGeneformerModel(config=config)

# Train with Lightning
trainer = L.Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=2,
    strategy="ddp",
    precision="16-mixed",
)

trainer.fit(model, data_module)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `micro_batch_size` or `seq_length`
2. **No Neighbors Found**: Check `neighbor_key` and ensure neighbor data exists
3. **Slow Training**: Increase `num_workers` or use multiple GPUs
4. **Loss Not Decreasing**: Adjust `learning_rate` or `mask_prob`

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Limit batches for debugging
train_temporal_geneformer(
    data_dir="data/small_test",
    output_dir="outputs/debug",
    limit_train_batches=10,
    limit_val_batches=5,
    num_epochs=1,
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{bionemo_geneformer_t,
  title={BioNeMo Geneformer-T: Temporal/Neighbor-Aware Fine-tuning},
  author={NVIDIA Corporation},
  year={2024},
  url={https://github.com/NVIDIA/bionemo-framework}
}
```
