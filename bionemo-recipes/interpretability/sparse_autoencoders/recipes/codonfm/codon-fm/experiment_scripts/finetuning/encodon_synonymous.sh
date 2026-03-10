#!/bin/bash

# This script launches finetuning jobs for synonymous variants.
# It takes USER, CLUSTER, CHECKPOINT_PATH, MODEL_NAME and FINETUNE_STRATEGY as command line arguments.

# --- Helper functions --- #
function print_help {
    echo "Usage: $0 <user> <cluster> <checkpoint_path> <model_name> <finetune_strategy> [local_out_dir]"
    echo "NOTE: This script must be run from the root directory of the project."
    echo "Example (local): $0 myuser local /path/to/checkpoint encodon_80m lora /path/to/output"
    echo "Example (slurm): $0 myuser slurm /path/to/checkpoint encodon_80m full"
    echo "Finetune strategies: lora, head_only_random, head_only_pretrained, full"
    echo "local_out_dir is required if cluster is 'local'."
}

# --- Argument Validation --- #

if [ "$#" -eq 0 ]; then
    print_help
    exit 1
fi

if [ "$2" == "local" ] && [ "$#" -ne 6 ]; then
    echo "Error: For local cluster, exactly 6 arguments are required: <user> local <checkpoint_path> <model_name> <finetune_strategy> <local_out_dir>"
    exit 1
fi

if [ "$2" != "local" ] && [ "$#" -ne 5 ]; then
    echo "Error: For non-local cluster, exactly 5 arguments are required: <user> <cluster> <checkpoint_path> <model_name> <finetune_strategy>"
    exit 1
fi


USER=$1
CLUSTER=$2
CHECKPOINT_PATH=$3
MODEL_NAME=$4
FINETUNE_STRATEGY=$5
if [ "$CLUSTER" == "local" ]; then
    LOCAL_OUT_DIR=$6
fi

DATA_PATH="/data/synonymous_variant_author/data.csv"
NUM_JOBS=1
MAX_STEPS=100000

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
                TRAIN_BATCH_SIZE=32  # Similar to MLM pretraining
                ;;
            "head_only_random"|"head_only_pretrained")
                TRAIN_BATCH_SIZE=32
                ;;
            "lora")
                TRAIN_BATCH_SIZE=32
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
    TRAIN_BATCH_SIZE=8
fi

# Derive EXP_NAME from CHECKPOINT_PATH and FINETUNE_STRATEGY
DIR=$(dirname "$CHECKPOINT_PATH")
while [[ "$(basename "$DIR")" == *"checkpoint"* ]]; do
  DIR=$(dirname "$DIR")
done
EXP_NAME="$(basename "$DIR")_synonymous_variant_${FINETUNE_STRATEGY}_ref_alt_loss"

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
    "--project_name" "codon-fm-synonymous-finetune"
    "--data_path" "$DATA_PATH"
    "--process_item" "mutation_pred_mlm"
    "--dataset_name" "MutationDataset"
    "--train_batch_size" "$TRAIN_BATCH_SIZE"
    "--val_batch_size" "$VAL_BATCH_SIZE"
    "--max_steps" "$MAX_STEPS"
    "--checkpoint_path" "$CHECKPOINT_PATH"
    "--model_name" "$MODEL_NAME"
    "--loss_type" "huber-ref-alt"
    "--label_col" "Abundance_score"
    "--process" "mutation_pred_mlm"
    "--check_val_every_n_epoch" "1"
    "--log_every_n_steps" "10"
    "--bf16"
    "--enable_wandb"
)

case $FINETUNE_STRATEGY in
    "lora")
        CMD+=("--finetune_strategy" "lora" "--lora_alpha" "32.0" "--lora_r" "16" "--lora_dropout" "0.1")
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
