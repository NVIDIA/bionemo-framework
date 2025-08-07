# Evo2 Fine-tuning Script

This directory contains scripts for fine-tuning Evo2 models on downstream tasks.

## Overview

The `finetune_evo2.py` script allows you to fine-tune pre-trained Evo2 models (both Hyena and Mamba architectures) for:
- Sequence-level regression tasks
- Sequence-level classification tasks
- Token-level classification tasks

## Data Format

The script expects CSV files with the following columns:

### For sequence-level tasks:
```csv
sequences,labels
ATCGATCGATCG...,0.85
GCTAGCTAGCTA...,0.92
```

### For token-level tasks:
```csv
sequences,labels,labels_mask
ATCGATCGATCG...,0 1 2 1 0 1 2 1 0 1 2 1,1 1 1 1 1 1 1 1 1 1 1 1
GCTAGCTAGCTA...,2 1 0 1 2 1 0 1 2 1 0 1,1 1 1 1 1 1 1 1 1 1 1 1
```

Where:
- `sequences`: DNA/RNA sequences
- `labels`: Target values (float for regression, int/string for classification)
- `labels_mask` (optional): Binary mask indicating which positions to use for loss calculation in token-level tasks

## Usage Examples

### 1. Sequence-level Regression (e.g., expression prediction)
```bash
python sub-packages/bionemo-evo2/src/bionemo/evo2/scripts/finetune_evo2.py \
  --train-data-path /path/to/train.csv \
  --valid-data-path /path/to/val.csv \
  --task-type regression \
  --restore-from-checkpoint-path /path/to/evo2_checkpoint \
  --config-class Evo2FineTuneSeqConfig \
  --dataset-class InMemorySingleValueDataset \
  --model-type hyena \
  --model-size 7b \
  --mlp-ft-dropout 0.25 \
  --mlp-hidden-size 256 \
  --mlp-target-size 1 \
  --num-steps 500 \
  --experiment-name evo2-expression-prediction \
  --encoder-frozen \
  --lr 0.0005 \
  --result-dir ./results/ \
  --micro-batch-size 8 \
  --max-seq-length 8192 \
  --precision bf16-mixed \
  --num-gpus 4 \
  --num-nodes 1
```

### 2. Sequence-level Classification (e.g., regulatory element classification)
```bash
python sub-packages/bionemo-evo2/src/bionemo/evo2/scripts/finetune_evo2.py \
  --train-data-path /path/to/train.csv \
  --valid-data-path /path/to/val.csv \
  --task-type classification \
  --restore-from-checkpoint-path /path/to/evo2_checkpoint \
  --config-class Evo2FineTuneSeqConfig \
  --dataset-class InMemorySingleValueDataset \
  --model-type mamba \
  --model-size 1b \
  --mlp-ft-dropout 0.25 \
  --mlp-hidden-size 256 \
  --mlp-target-size 10 \
  --num-steps 1000 \
  --experiment-name evo2-regulatory-classification \
  --encoder-frozen \
  --lr 0.0005 \
  --result-dir ./results/ \
  --micro-batch-size 16 \
  --precision bf16-mixed \
  --label-column class_label \
  --accumulate-grad-batches 2 \
  --wandb-entity my-team \
  --wandb-project evo2-finetuning \
  --num-gpus 8 \
  --num-nodes 1
```

### 3. Token-level Classification (e.g., secondary structure prediction)
```bash
python sub-packages/bionemo-evo2/src/bionemo/evo2/scripts/finetune_evo2.py \
  --train-data-path /path/to/train.csv \
  --valid-data-path /path/to/val.csv \
  --config-class Evo2FineTuneTokenConfig \
  --dataset-class InMemoryPerTokenValueDataset \
  --task-type classification \
  --model-type hyena \
  --model-size 7b \
  --cnn-dropout 0.25 \
  --cnn-hidden-size 32 \
  --cnn-num-classes 3 \
  --experiment-name evo2-secondary-structure \
  --num-steps 1500 \
  --val-check-interval 100 \
  --encoder-frozen \
  --lr 0.0005 \
  --result-dir ./results/ \
  --micro-batch-size 4 \
  --max-seq-length 4096 \
  --precision bf16-mixed \
  --label-column structure \
  --labels-mask-column resolved \
  --num-gpus 8 \
  --num-nodes 2
```

### 4. LoRA Fine-tuning (Parameter-efficient)
```bash
python sub-packages/bionemo-evo2/src/bionemo/evo2/scripts/finetune_evo2.py \
  --train-data-path /path/to/train.csv \
  --valid-data-path /path/to/val.csv \
  --task-type regression \
  --restore-from-checkpoint-path /path/to/evo2_checkpoint \
  --lora-finetune \
  --config-class Evo2FineTuneSeqConfig \
  --dataset-class InMemorySingleValueDataset \
  --model-type hyena \
  --model-size 7b \
  --num-steps 500 \
  --experiment-name evo2-lora-finetuning \
  --lr 0.001 \
  --result-dir ./results/ \
  --micro-batch-size 32 \
  --precision bf16-mixed
```

### 5. Using FP8 Precision (for compatible GPUs)
```bash
python sub-packages/bionemo-evo2/src/bionemo/evo2/scripts/finetune_evo2.py \
  --train-data-path /path/to/train.csv \
  --valid-data-path /path/to/val.csv \
  --task-type classification \
  --restore-from-checkpoint-path /path/to/evo2_checkpoint \
  --config-class Evo2FineTuneSeqConfig \
  --dataset-class InMemorySingleValueDataset \
  --model-type hyena \
  --model-size 7b \
  --fp8 \
  --precision fp8 \
  --num-steps 500 \
  --experiment-name evo2-fp8-classification \
  --num-gpus 8
```

## Key Parameters

### Model Configuration
- `--model-type`: Choose between `hyena` or `mamba` architecture
- `--model-size`: Model size (e.g., `1b`, `7b`, `7b_arc_longcontext`)
- `--restore-from-checkpoint-path`: Path to pre-trained Evo2 checkpoint

### Task Configuration
- `--task-type`: `regression` or `classification`
- `--config-class`: Use `Evo2FineTuneSeqConfig` for sequence-level, `Evo2FineTuneTokenConfig` for token-level
- `--dataset-class`: Use `InMemorySingleValueDataset` for sequence-level, `InMemoryPerTokenValueDataset` for token-level
- `--encoder-frozen`: Freeze encoder weights during fine-tuning

### Training Parameters
- `--lr`: Learning rate (typically 1e-4 to 1e-3 for fine-tuning)
- `--micro-batch-size`: Batch size per GPU
- `--accumulate-grad-batches`: Gradient accumulation steps
- `--num-steps`: Total training steps
- `--max-seq-length`: Maximum sequence length (up to 131k for long-context models)

### Model Head Parameters
For sequence-level tasks:
- `--mlp-ft-dropout`: Dropout in MLP head
- `--mlp-hidden-size`: Hidden size of MLP
- `--mlp-target-size`: Output size (1 for regression, num_classes for classification)

For token-level tasks:
- `--cnn-dropout`: Dropout in CNN head
- `--cnn-hidden-size`: Hidden channels in CNN
- `--cnn-num-classes`: Number of output classes per token

### Parallelism Options
- `--num-gpus`: GPUs per node
- `--num-nodes`: Number of nodes
- `--tensor-model-parallel-size`: Tensor parallelism
- `--pipeline-model-parallel-size`: Pipeline parallelism
- `--context-parallel-size`: Context parallelism
- `--sequence-parallel`: Enable sequence parallelism

### Logging and Checkpointing
- `--wandb-project`: W&B project name
- `--wandb-entity`: W&B team/entity
- `--create-tensorboard-logger`: Enable TensorBoard logging
- `--save-top-k`: Number of best checkpoints to keep
- `--val-check-interval`: Validation frequency (in steps)

## Tips for Fine-tuning

1. **Start with frozen encoder**: Use `--encoder-frozen` to only train the task head initially
2. **Use appropriate batch sizes**: Larger models may require smaller batch sizes
3. **Monitor validation metrics**: Adjust learning rate if validation loss plateaus
4. **Consider LoRA**: For large models, LoRA fine-tuning can be more efficient
5. **Sequence length**: Match your training sequence length to your inference needs

## Checkpoint Compatibility

The script is designed to work with:
- Converted Evo2 checkpoints from HuggingFace (use `evo2_convert_to_nemo2` first)
- NeMo2 format Evo2 checkpoints
- Previously fine-tuned Evo2 checkpoints

## Troubleshooting

1. **CUDA OOM**: Reduce `--micro-batch-size` or increase model parallelism
2. **FP8 issues**: Ensure your GPU supports FP8 (compute capability >= 8.9)
3. **Slow training**: Enable `--sequence-parallel` when using tensor parallelism
4. **Data loading**: Ensure CSV files are properly formatted with correct column names
