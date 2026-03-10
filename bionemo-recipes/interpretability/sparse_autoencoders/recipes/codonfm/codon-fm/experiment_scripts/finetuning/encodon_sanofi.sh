#!/bin/bash

# This script launches finetuning jobs for Sanofi eval datasets.
# It takes USER, CLUSTER, CHECKPOINT_PATH, MODEL_NAME, FINETUNE_STRATEGY and DATA_PATH as command line arguments.

# --- Helper functions --- #
function print_help {
    echo "Usage: $0 <user> <cluster> <checkpoint_path> <model_name> <finetune_strategy> <data_path> [local_out_dir] [use_cross_attention]"
    echo "NOTE: This script must be run from the root directory of the project."
    echo "Example (local): $0 myuser local /path/to/checkpoint encodon_80m lora /data/sanofi/dataset.csv /path/to/output"
    echo "Example (slurm): $0 myuser slurm /path/to/checkpoint encodon_80m full /data/sanofi/dataset.csv"
    echo "Example (with cross-attention): $0 myuser slurm /path/to/checkpoint encodon_80m full /data/sanofi/dataset.csv \"\" true"
    echo "Finetune strategies: lora, head_only_random, head_only_pretrained, full"
    echo "use_cross_attention: true/false (default: false) - Whether to use downstream cross-attention head"
    echo "local_out_dir is required if cluster is 'local'."
}

# --- Argument Validation --- #

if [ "$#" -eq 0 ]; then
    print_help
    exit 1
fi

if [ "$2" == "local" ] && [ "$#" -lt 7 ] || [ "$2" == "local" ] && [ "$#" -gt 8 ]; then
    echo "Error: For local cluster, 7-8 arguments are required: <user> local <checkpoint_path> <model_name> <finetune_strategy> <data_path> <local_out_dir> [use_cross_attention]"
    exit 1
fi

if [ "$2" != "local" ] && [ "$#" -lt 6 ] || [ "$2" != "local" ] && [ "$#" -gt 7 ]; then
    echo "Error: For non-local cluster, 6-7 arguments are required: <user> <cluster> <checkpoint_path> <model_name> <finetune_strategy> <data_path> [use_cross_attention]"
    exit 1
fi

USER=$1
CLUSTER=$2
CHECKPOINT_PATH=$3
MODEL_NAME=$4
FINETUNE_STRATEGY=$5
DATA_PATH=$6
if [ "$CLUSTER" == "local" ]; then
    LOCAL_OUT_DIR=$7
    USE_CROSS_ATTENTION=${8:-"false"}
    # Validate that local_out_dir is provided and not empty
    if [ -z "$LOCAL_OUT_DIR" ]; then
        echo "Error: local_out_dir is required for local cluster but was not provided or is empty"
        exit 1
    fi
else
    USE_CROSS_ATTENTION=${7:-"false"}
fi

NUM_JOBS=1
# Allow override via environment variable for testing
MAX_STEPS=${TEST_MAX_STEPS:-100000}

# Defaults
NUM_NODES=1
NUM_GPUS=8
LR=""
TRAIN_BATCH_SIZE=""
ENABLE_FSDP=""

case $MODEL_NAME in
    "encodon_80m")
        NUM_NODES=1
        LR="1e-5"
        # Set batch size based on finetuning strategy
        case $FINETUNE_STRATEGY in
            "full")
                TRAIN_BATCH_SIZE=8
                ;;
            "head_only_random"|"head_only_pretrained")
                TRAIN_BATCH_SIZE=8
                ;;
            "lora")
                TRAIN_BATCH_SIZE=8
                ;;
        esac
        ;;
    "encodon_600m")
        NUM_NODES=1
        LR="1e-5"
        TRAIN_BATCH_SIZE=8
        ;;
    "encodon_1b")
        NUM_NODES=1
        LR="1.0e-4"
        # Set batch size based on finetuning strategy
        case $FINETUNE_STRATEGY in
            "full")
                TRAIN_BATCH_SIZE=2
                ;;
            "head_only_random"|"head_only_pretrained")
                TRAIN_BATCH_SIZE=16
                ;;
            "lora")
                TRAIN_BATCH_SIZE=8
                ;;
        esac
        ;;
    *)
        echo "Invalid model_name: $MODEL_NAME"
        exit 1
        ;;
esac

VAL_BATCH_SIZE=$TRAIN_BATCH_SIZE

if [ "$CLUSTER" == "local" ]; then
    NUM_NODES=1
    NUM_GPUS=2
    NUM_JOBS=1
    TRAIN_BATCH_SIZE=4
    VAL_BATCH_SIZE=4
fi

# Derive EXP_NAME from CHECKPOINT_PATH, FINETUNE_STRATEGY and DATA_PATH
DIR=$(dirname "$CHECKPOINT_PATH")
while [[ "$(basename "$DIR")" == *"checkpoint"* ]]; do
  DIR=$(dirname "$DIR")
done
DATASET_NAME=$(basename "$DATA_PATH" .csv)
if [[ "$USE_CROSS_ATTENTION" == "true" ]]; then
    EXP_NAME="$(basename "$DIR")_sanofi_${DATASET_NAME}_${FINETUNE_STRATEGY}_cross_attn"
else
    EXP_NAME="$(basename "$DIR")_sanofi_${DATASET_NAME}_${FINETUNE_STRATEGY}"
fi

# Determine loss type and process_item based on dataset
PROCESS_ITEM="codon_sequence"
# E.Coli_proteins is a classification task, others are regression
if [[ "$DATASET_NAME" == "E_Coli_proteins" ]]; then
    # Classification task
    LOSS_TYPE="classification"
    NUM_CLASSES=3 
else
    # Regression task
    LOSS_TYPE="regression"
    NUM_CLASSES=2  # Default value, not used for regression
fi

CMD=(
    "python" "-m" "src.runner" "finetune"
    "--user" "$USER"
    "--cluster" "$CLUSTER"
    "--exp_name" "$EXP_NAME"
    "--time" "03:55:00"
    "--num_nodes" "$NUM_NODES"
    "--num_gpus" "$NUM_GPUS"
    "--num_jobs" "$NUM_JOBS"
    "--seed" "42"
    "--project_name" "codon-fm-sanofi-finetune"
    "--data_path" "$DATA_PATH"
    "--process_item" "$PROCESS_ITEM"
    "--dataset_name" "CodonBertDataset"
    "--train_batch_size" "$TRAIN_BATCH_SIZE"
    "--val_batch_size" "$VAL_BATCH_SIZE"
    "--max_steps" "$MAX_STEPS"
    "--checkpoint_path" "$CHECKPOINT_PATH"
    "--model_name" "$MODEL_NAME"
    "--loss_type" "$LOSS_TYPE"
    "--num_classes" "$NUM_CLASSES"
    "--label_col" "value"
    "--check_val_every_n_epoch" "1"
    "--log_every_n_steps" "1"
    "--checkpoint_every_n_train_steps" "5"
    "--bf16"
    "--enable_wandb"
)

# Add cross-attention parameters if enabled
if [[ "$USE_CROSS_ATTENTION" == "true" ]]; then
    CMD+=("--use_downstream_head" "--cross_attention_hidden_dim" "512" "--cross_attention_num_heads" "8")
fi

case $FINETUNE_STRATEGY in
    "lora")
        CMD+=("--finetune_strategy" "lora" "--lora" "--lora_alpha" "32.0" "--lora_r" "32" "--lora_dropout" "0.1")
        ;;
    "head_only_random")
        CMD+=("--finetune_strategy" "head_only_random")
        ;;
    "head_only_pretrained")
        CMD+=("--finetune_strategy" "head_only_pretrained")
        ;;
    "full")
        CMD+=("--finetune_strategy" "full")
        ;;
    *)
        echo "Invalid finetune_strategy: $FINETUNE_STRATEGY"
        echo "Supported strategies: lora, head_only_random, head_only_pretrained, full"
        exit 1
        ;;
esac

if [ -n "$LR" ]; then
    CMD+=("--lr" "$LR")
fi

if [ -n "$ENABLE_FSDP" ]; then
    CMD+=("$ENABLE_FSDP")
fi

if [ "$CLUSTER" == "local" ]; then
    CMD+=("--local_out_dir" "$LOCAL_OUT_DIR")
fi

echo "Executing: ${CMD[@]}"
"${CMD[@]}" 