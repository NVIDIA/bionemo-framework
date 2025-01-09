# Pre-training ESM-2

Pre-trained checkpoints for ESM-2 are available at the 8M, 650M, and 3B model sizes. These models were trained by the
bionemo-framework team to reproduce the original training results from Lin et al, Science (2023), with more recent
UniProt data and leveraging the bionemo training infrastructure. The full [pre-training data](../../datasets/uniprot.md)
and train/test splits are available.

## Model Convergence

Validation perplexity evaluated on the NVIDIA validation set.

<figure markdown="span">
  ![ESM-2 Pre-training Convergence](../assets/images/esm2/esm2_pretrain_convergence.svg){ width="350" }
</figure>

| Model Size     | Perplexity at 500k updates  |
| -------------- | ------ |
| 8M             | 10.26  |
| 650M           | 7.14   |
| 3B             | 6.42   |

## Pre-training recipes

=== "8M"

    ```python
    esm2_8m_ckpt_path = load("esm2/nv_8m:2.0")
    ```

    ### Training Script

    | Training Parameters     | Value  |
    | ----------------------- | ------ |
    | # of GPUs               | 32    |
    | GPU Type                | A100   |
    | Batch size (per device) | 64     |

    ```bash
    train_esm2 \
      --create-tensorboard-logger \
      --resume-if-exists \
      --wandb-project=<wandb-project-name> \
      --save-top-k=10 \
      --train-cluster-path=/data/train_clusters.parquet \  # (1)!
      --train-database-path=/data/train.db \
      --valid-cluster-path=/data/valid_clusters.parquet \
      --valid-database-path=/data/validation.db \
      --num-steps=500_000 \
      --metric-to-monitor-for-checkpoints=val_loss \
      --micro-batch-size=64 \
      --num-nodes=4 \
      --num-gpus=8 \
      --val-check-interval=10000 \
      --limit-val-batches=1.0 \
      --result-dir=/results/esm2_pretrain_8m \
      --experiment-name=esm2_pretrain_8m \
      --num-layers=6 \
      --hidden-size=320 \
      --num-attention-heads=20 \
      --ffn-hidden-size=1280;
    ```

    1. Paths here must be mounted into the `bionemo-framework` docker image.

=== "650M"

    ```python
    esm2_650m_ckpt_path = load("esm2/nv_650m:2.1")
    ```

    ### Training Script

    | Training Parameters     | Value  |
    | ----------------------- | ------ |
    | # of GPUs               | 64    |
    | GPU Type                | H100   |
    | Batch size (per device) | 32     |

    ```bash
    train_esm2 \
      --create-tensorboard-logger \
      --resume-if-exists \
      --wandb-project=<wandb-project-name> \
      --save-top-k=10 \
      --train-cluster-path=/data/train_clusters.parquet \  # (1)!
      --train-database-path=/data/train.db \
      --valid-cluster-path=/data/valid_clusters.parquet \
      --valid-database-path=/data/validation.db \
      --num-steps=500_000 \
      --metric-to-monitor-for-checkpoints=val_loss \
      --micro-batch-size=32 \
      --num-nodes=8 \
      --num-gpus=8 \
      --val-check-interval=10000 \
      --limit-val-batches=1.0 \
      --result-dir=/results/esm2_pretrain_650m \
      --experiment-name=esm2_pretrain_650m \
      --min-seq-length=1024 \
      --max-seq-length=1024 \
      --num-layers=33 \
      --hidden-size=1280 \
      --num-attention-heads=20 \
      --ffn-hidden-size=5120;
    ```

    1. Paths here must be mounted into the `bionemo-framework` docker image.

=== "3B"

    ```python
    esm2_3b_ckpt_path = load("esm2/nv_3b:2.1")
    ```

    ### Training Script

    | Training Parameters     | Value  |
    | ----------------------- | ------ |
    | # of GPUs               | 128    |
    | GPU Type                | H100   |
    | Batch size (per device) | 16     |
    | warmup steps            | 20,000 |

    ```bash
    train_esm2 \
      --create-tensorboard-logger \
      --resume-if-exists \
      --wandb-project=<wandb-project-name> \
      --save-top-k=10 \
      --train-cluster-path=/data/train_clusters.parquet \  # (2)!
      --train-database-path=/data/train.db \
      --valid-cluster-path=/data/valid_clusters.parquet \
      --valid-database-path=/data/validation.db \
      --num-steps=500_000 \
      --warmup-steps=20_000 \  # (1)!
      --metric-to-monitor-for-checkpoints=val_loss \
      --micro-batch-size=16 \
      --num-nodes=16 \
      --num-gpus=8 \
      --val-check-interval=2500 \
      --limit-val-batches=1.0 \
      --result-dir=/results/esm2_pretrain_3b \
      --experiment-name=esm2_pretrain_3b \
      --min-seq-length=1024 \
      --max-seq-length=1024 \
      --num-layers=36 \
      --hidden-size=2560 \
      --num-attention-heads=40 \
      --ffn-hidden-size=10240;
    ```

    1. We had to increase the number of warmup steps 10x over the published training recipe for ESM-2 3B, which was
       likely trained with fp16 precision. This gave us an overall similar initial curve, but avoided convergence issues
       at around 2,000 steps.

    2. Paths here must be mounted into the `bionemo-framework` docker image.
