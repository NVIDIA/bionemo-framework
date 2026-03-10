#!/bin/bash
set -ex

# --- Helper functions --- #
function print_help {
    echo "Usage: $0 <cluster> <user> <num_jobs> [local_out_dir]"
    echo "NOTE: This script must be run from the root directory of the project."
    echo "Example (local): $0 local myuser 1 /path/to/output"
    echo "Example (slurm): $0 slurm myuser 1"
}

# --- Argument Validation --- #
if [ "$1" = "local" ]; then
    if [ "$#" -ne 4 ]; then
        echo "Error: For local execution, expected 4 arguments but got $#."
        echo "Arguments should be: <cluster> <user> <num_jobs> <local_out_dir>"
        print_help
        exit 1
    fi
    local_out_dir=$4
else
    if [ "$#" -ne 3 ]; then
        echo "Error: For cluster execution, expected 3 arguments but got $#."
        print_help
        exit 1
    fi
fi

# - provide cluster and user as arguments
CLUSTER=$1
USER=$2
num_jobs=$3

learning_rate=1.0e-5

# Adjust configurations based on cluster type
if [ "$CLUSTER" = "local" ]; then
    num_nodes=1
    num_gpus=1
else
    num_nodes=2
    num_gpus=8
fi

train_batch_size=16
val_batch_size=16
effective_batch_size=$((train_batch_size * num_gpus * num_nodes))
num_workers=12

# Build python command
python_cmd="python -m src.runner pretrain \
    --user $USER \
    --cluster $CLUSTER \
    --exp_name decodon_200m_${learning_rate}_${effective_batch_size}_no_pathogen \
    --model_name decodon_200m \
    --data_path /data/ncbi/processed_unfiltered/ \
    --process_item clm_memmap \
    --dataset_name CodonMemmapDataset \
    --split_name_prefix decodon_nopathogen \
    --organism_tokens_file /data/nopathogen_organism_tokens.txt \
    --taxid_exclusion_file /data/ncbi/taxids_to_remove.json \
    --lr $learning_rate \
    --num_gpus $num_gpus \
    --num_nodes $num_nodes \
    --train_batch_size $train_batch_size \
    --val_batch_size $val_batch_size \
    --num_workers $num_workers \
    --project_name codon-fm \
    --bf16 \
    --vocab_size 22308 \
    --num_jobs $num_jobs \
    --lr_total_iterations 531000 \
    --max_steps 531000 \
    --enable_fsdp \
    --enable_wandb"

# Add local_out_dir argument for local execution
if [ "$CLUSTER" = "local" ]; then
    python_cmd="$python_cmd --local_out_dir $local_out_dir"
fi

# Execute the command
eval $python_cmd