#!/bin/bash
set -ex

# --- Helper functions --- #
function print_help {
    echo "Usage: $0 <cluster> <user> <num_jobs>"
    echo "NOTE: This script must be run from the root directory of the project."
    echo "Example (local): $0 local myuser 1"
    echo "Example (slurm): $0 slurm myuser 1"
}

# --- Argument Validation --- #
if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments. Expected 3, but got $#."
    print_help
    exit 1
fi

# - provide cluster and user as arguments
CLUSTER=$1
USER=$2
num_jobs=$3

# - hyperparameters
learning_rate=7.5e-5
num_nodes=16
num_gpus=8
train_batch_size=8
val_batch_size=8
effective_batch_size=$((train_batch_size * num_gpus * num_nodes))
num_workers=12

# - run
python -m src.runner pretrain \
    --user $USER \
    --cluster $CLUSTER \
    --exp_name encodon_600m_latest_${learning_rate}_${effective_batch_size} \
    --model_name encodon_600m \
    --data_path /data/ncbi/processed_unfiltered/ \
    --process_item mlm_memmap \
    --dataset_name CodonMemmapDataset \
    --lr $learning_rate \
    --num_gpus $num_gpus \
    --num_nodes $num_nodes \
    --train_batch_size $train_batch_size \
    --val_batch_size $val_batch_size \
    --num_workers $num_workers \
    --bf16 \
    --num_jobs $num_jobs \
    --enable_wandb
