#!/bin/bash

# This script launches finetuning jobs for Sanofi eval datasets.
# It takes USER, CLUSTER, CHECKPOINT_PATH, MODEL_NAME, FINETUNE_STRATEGY and DATA_PATH as command line arguments.

# --- Helper functions --- #
function print_help {
    echo "Usage: $0 <user> <cluster> <checkpoint_path> <model_name> <finetune_strategy> <data_path> [local_out_dir]"
    echo "NOTE: This script must be run from the root directory of the project."
    echo "Example (local): $0 myuser local /path/to/checkpoint encodon_80m lora /data/sanofi/dataset.csv /path/to/output"
    echo "Example (slurm): $0 myuser slurm /path/to/checkpoint encodon_80m full /data/sanofi/dataset.csv"
    echo "Finetune strategies: lora, head_only_random, head_only_pretrained, full"
    echo "local_out_dir is required if cluster is 'local'."
}

# --- Argument Validation --- #

if [ "$#" -eq 0 ]; then
    print_help
    exit 1
fi

if [ "$2" == "local" ] && [ "$#" -ne 8 ]; then
    echo "Error: For local cluster, exactly 8 arguments are required: <user> local <checkpoint_path> <model_name> <finetune_strategy> <data_path> <loss_type> <local_out_dir>"
    exit 1
fi

if [ "$2" != "local" ] && [ "$#" -ne 7 ]; then
    echo "Error: For non-local cluster, exactly 7 arguments are required: <user> <cluster> <checkpoint_path> <model_name> <finetune_strategy> <data_path> <loss_type>"
    exit 1
fi

USER=$1
CLUSTER=$2
CHECKPOINT_PATH=$3
MODEL_NAME=$4
FINETUNE_STRATEGY=$5
DATA_PATH=$6
LOSS_TYPE=$7
if [ "$CLUSTER" == "local" ]; then
    LOCAL_OUT_DIR=$8
fi

MAX_STEPS=500_000

# Defaults
NUM_NODES=8
NUM_GPUS=8
LR=""
TRAIN_BATCH_SIZE=""
ENABLE_FSDP=""

case $MODEL_NAME in
    "encodon_80m")
        NUM_NODES=1
        LR="1.0e-5"
        # Set batch size based on finetuning strategy
        case $FINETUNE_STRATEGY in
            "full")
                TRAIN_BATCH_SIZE=8
                CHECKPOINT_EVERY_N_TRAIN_STEPS=3200
                VAL_CHECK_INTERVAL=3200
                VAL_BATCHES=500
                ;;
            "head_only_random"|"head_only_pretrained")
                TRAIN_BATCH_SIZE=32
                CHECKPOINT_EVERY_N_TRAIN_STEPS=800
                VAL_CHECK_INTERVAL=800
                VAL_BATCHES=100
                ;;
            "lora")
                TRAIN_BATCH_SIZE=8
                CHECKPOINT_EVERY_N_TRAIN_STEPS=800
                VAL_CHECK_INTERVAL=800
                VAL_BATCHES=100
                ;;
        esac
        ;;
    "encodon_600m")
        NUM_NODES=1
        LR="1.0e-5"
        TRAIN_BATCH_SIZE=8
        ;;
    "encodon_1b")
        NUM_NODES=4
        LR="1.0e-5"
        # Set batch size based on finetuning strategy
        case $FINETUNE_STRATEGY in
            "full")
                TRAIN_BATCH_SIZE=4
                LR="2.0e-5"
                CHECKPOINT_EVERY_N_TRAIN_STEPS=1000
                VAL_CHECK_INTERVAL=1000
                VAL_BATCHES=500
                ;;
            "head_only_random"|"head_only_pretrained")
                TRAIN_BATCH_SIZE=64
                LR="1e-4"
                CHECKPOINT_EVERY_N_TRAIN_STEPS=200
                VAL_CHECK_INTERVAL=200
                VAL_BATCHES=100
                ;;
            "lora")
                TRAIN_BATCH_SIZE=4
                LR="2.0e-5"
                CHECKPOINT_EVERY_N_TRAIN_STEPS=1000
                VAL_CHECK_INTERVAL=1000
                VAL_BATCHES=500
                ;;
        esac
        ;;
    "encodon_5b")
        NUM_NODES=8
        LR="1.0e-5"
        ENABLE_FSDP="--enable_fsdp"
        # Set batch size based on finetuning strategy
        case $FINETUNE_STRATEGY in
            "full")
                TRAIN_BATCH_SIZE=2
                LR="2.0e-5"
                CHECKPOINT_EVERY_N_TRAIN_STEPS=1000
                VAL_CHECK_INTERVAL=1000
                VAL_BATCHES=500
                ;;
            "head_only_random"|"head_only_pretrained")
                TRAIN_BATCH_SIZE=16
                LR="1e-4"
                CHECKPOINT_EVERY_N_TRAIN_STEPS=200
                VAL_CHECK_INTERVAL=200
                VAL_BATCHES=100
                ;;
            "lora")
                TRAIN_BATCH_SIZE=2
                LR="2.0e-5"
                CHECKPOINT_EVERY_N_TRAIN_STEPS=1000
                VAL_CHECK_INTERVAL=1000
                VAL_BATCHES=500
                ;;
        esac
        ;;

    *)
        echo "Invalid model_name: $MODEL_NAME"
        exit 1
        ;;
esac


NUM_JOBS=2
if [ "$CLUSTER" == "local" ]; then
    NUM_NODES=1
    NUM_GPUS=2
    NUM_JOBS=1
    TRAIN_BATCH_SIZE=8
fi
VAL_BATCH_SIZE=$TRAIN_BATCH_SIZE

# Derive EXP_NAME from CHECKPOINT_PATH, FINETUNE_STRATEGY and DATA_PATH
DIR=$(dirname "$CHECKPOINT_PATH")
while [[ "$(basename "$DIR")" == *"checkpoint"* ]]; do
  DIR=$(dirname "$DIR")
done
CKPT_NAME=$(basename "$CHECKPOINT_PATH" .ckpt)
DATASET_NAME=$(basename "$DATA_PATH" .csv)
PROCESS_ITEM="missense_seq"
N_VARIANTS=1
CENTER_WEIGHT_THRESHOLD=0.7
USE_PAIV1="--missense_use_paiv1"
USE_AM="--missense_use_am"
N_PER_BENIGN=1
VERSION=102
EXP_NAME="$(basename "$DIR")_${CKPT_NAME}_${DATASET_NAME}_${FINETUNE_STRATEGY}_${LR}_${LOSS_TYPE}_N${N_VARIANTS}_center${CENTER_WEIGHT_THRESHOLD}"
if [ ! -z "$USE_PAIV1" ]; then
    EXP_NAME="${EXP_NAME}_paiv1"
fi
if [ ! -z "$USE_AM" ]; then
    EXP_NAME="${EXP_NAME}_am"
fi
if [ $N_PER_BENIGN -gt 1 ]; then
    EXP_NAME="${EXP_NAME}_npb${N_PER_BENIGN}"
fi
EXP_NAME="${EXP_NAME}_v${VERSION}"

CMD=(
    "python" "-m" "src.runner" "finetune"
    "--user" "$USER"
    "--cluster" "$CLUSTER"
    "--exp_name" "${EXP_NAME}"
    "--time" "03:55:00"
    "--num_nodes" "$NUM_NODES"
    "--num_gpus" "$NUM_GPUS"
    "--num_jobs" "$NUM_JOBS"
    "--enable_wandb"
    "--seed" "42"
    "--project_name" "codon-fm-missense-finetune"
    "--data_path" "$DATA_PATH"
    "--process_item" "$PROCESS_ITEM"
    "--dataset_name" "MissenseDataset"
    "--train_batch_size" "$TRAIN_BATCH_SIZE"
    "--val_batch_size" "$VAL_BATCH_SIZE"
    "--max_steps" "$MAX_STEPS"
    "--checkpoint_path" "$CHECKPOINT_PATH"
    "--model_name" "$MODEL_NAME"
    "--loss_type" "$LOSS_TYPE"
    "--train_val_test_ratio" "0.9" "0.1" "0"
    "--checkpoint_every_n_train_steps" "$CHECKPOINT_EVERY_N_TRAIN_STEPS"
    "--val_check_interval" "$VAL_CHECK_INTERVAL"
    "--log_every_n_steps" "10"
    "--bf16"
    "--limit_val_batches" "$VAL_BATCHES"
    "--missense_use_weights"
    "--missense_center_weight_threshold" "$CENTER_WEIGHT_THRESHOLD"
    "--missense_n_per_benign" "$N_PER_BENIGN"
    "--missense_variants_per_seq" "$N_VARIANTS"
)

if [ $LOSS_TYPE == "missense_synom_agg" ]; then
    CMD+=("--mask_mutation")
fi

if [ ! -z "$USE_PAIV1" ]; then
    CMD+=("$USE_PAIV1")
fi
if [ ! -z "$USE_AM" ]; then
    CMD+=("$USE_AM")
fi

case $FINETUNE_STRATEGY in
    "lora")
        CMD+=("--finetune_strategy" "lora" "--lora_alpha" "64.0" "--lora_r" "64" "--lora_dropout" "0.1")
        ;;
    "head_only_random")
        CMD+=("--finetune_strategy" "head_only_random")
        ;;
    "head_only_pretrained")
        CMD+=("--finetune_strategy" "head_only_pretrained")
        ;;
    "full")
        CMD+=("--finetune_strategy" "full" --warmup_iterations "10000")
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