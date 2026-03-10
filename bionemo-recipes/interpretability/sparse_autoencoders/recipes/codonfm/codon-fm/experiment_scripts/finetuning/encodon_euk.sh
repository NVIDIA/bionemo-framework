#!/bin/bash

# This script is for finetuning a pre-trained Encodon model on the eukaryotic subset of the data.

# --- Helper functions --- #
function print_help {
    echo "Usage: $0 <user> <cluster> <checkpoint_path> <model_name> <num_jobs> [local_out_dir]"
    echo "NOTE: This script must be run from the root directory of the project."
    echo "Example (local): $0 myuser local /path/to/checkpoint encodon_80m 1 /path/to/output"
    echo "Example (slurm): $0 myuser slurm /path/to/checkpoint encodon_80m 1"
    echo "local_out_dir is required if cluster is 'local'."
}

# Function to generate abbreviated taxon group names
function generate_taxon_abbrev {
    local groups=("$@")
    local abbrev=""
    for group in "${groups[@]}"; do
        case $group in
            "Primates") abbrev+="P" ;;
            "vertebrate_mammalian") abbrev+="VM" ;;
            "vertebrate_other") abbrev+="VO" ;;
            "invertebrates") abbrev+="I" ;;
            "fungi") abbrev+="F" ;;
            "plant") abbrev+="PL" ;;
            "protozoa") abbrev+="PR" ;;
            *) abbrev+="X" ;;  # fallback for unknown groups
        esac
    done
    echo "$abbrev"
}

# Function to run a single experiment
function run_experiment {
    local user=$1
    local cluster=$2
    local checkpoint_path=$3
    local model_name=$4
    local num_jobs=$5
    local local_out_dir=$6
    shift 6
    local taxon_groups=("$@")
    
    echo "Running experiment with taxon groups: ${taxon_groups[*]}"
    
    # Generate abbreviated name for taxon groups
    local taxon_abbrev=$(generate_taxon_abbrev "${taxon_groups[@]}")
    echo "Generated taxon abbreviation: '$taxon_abbrev'"
    
    # Configuration based on model
    case $model_name in
        "encodon_80m")
            num_nodes=4
            num_gpus=8
            train_batch_size=32
            lr=1e-4
            ;;
        "encodon_600m")
            num_nodes=16
            num_gpus=8
            train_batch_size=8
            lr=7.5e-5
            ;;
        "encodon_1b")
            num_nodes=48
            num_gpus=8
            train_batch_size=4
            lr=7.5e-5
            ;;
        *)
            echo "Invalid model_name: $model_name"
            return 1
            ;;
    esac

    if [ "$cluster" == "local" ]; then
        num_gpus=2
        num_nodes=1
        train_batch_size=16
    fi

    # Derive EXP_NAME from CHECKPOINT_PATH
    local dir=$(dirname "$checkpoint_path")
    while [[ "$(basename "$dir")" == *"checkpoint"* ]]; do
      dir=$(dirname "$dir")
    done
    local exp_name="$(basename "$dir")_euk_${taxon_abbrev}"
    exp_name="${exp_name}_lr_${lr}_bs_${train_batch_size}"
    echo "Final experiment name: '$exp_name'"

    local data_path="/data/ncbi/processed_unfiltered/"
    local max_steps=10000000
    local time="03:55:00"
    local project_name="codon-fm-finetune-euk-new"

    # Build command
    local cmd=(python -m src.runner finetune
        --user "${user}"
        --cluster "${cluster}"
        --exp_name "${exp_name}"
        --num_nodes "${num_nodes}"
        --num_gpus "${num_gpus}"
        --train_batch_size "${train_batch_size}"
        --time "${time}"
        --project_name "${project_name}"
        --data_path "${data_path}"
        --dataset_name "CodonMemmapDataset"
        --process_item "mlm_memmap"
        --model_name "${model_name}"
        --lr "${lr}"
        --groups_to_use "${taxon_groups[@]}"
        --checkpoint_path "${checkpoint_path}"
        --resume_trainer_state
        --bf16
        --num_jobs "${num_jobs}"
        --max_steps "${max_steps}"
        --split_name_prefix "nopathogen"
        --taxid_exclusion_file "/data/ncbi/taxids_to_remove.json"
        --enable_wandb
    )

    # Add codon weights file only if 'cdswts' is in the checkpoint name
    if [[ "$checkpoint_path" == *"cdswt"* ]]; then
        cmd+=(--codon_weights_file "/data/ncbi/codon_counts_nopathogen.json")
    fi

    if [ "$enable_fsdp" = true ]; then
        cmd+=("--enable_fsdp")
    fi

    if [ "$cluster" == "local" ]; then
        cmd+=(--local_out_dir "${local_out_dir}")
    fi

    echo "Running command: ${cmd[*]}" 
    "${cmd[@]}"
}

# --- Argument Validation --- #
if [ "$#" -eq 0 ]; then
    print_help
    exit 1
fi

if [ "$2" == "local" ] && [ "$#" -ne 6 ]; then
    echo "Error: For local cluster, exactly 6 arguments are required: <user> local <checkpoint_path> <model_name> <num_jobs> <local_out_dir>"
    exit 1
fi

if [ "$2" != "local" ] && [ "$#" -ne 5 ]; then
    echo "Error: For non-local cluster, exactly 5 arguments are required: <user> <cluster> <checkpoint_path> <model_name> <num_jobs>"
    exit 1
fi

#--- Configuration ---#
USER=$1
CLUSTER=$2
CHECKPOINT_PATH=$3
MODEL_NAME=$4
NUM_JOBS=$5

if [ "$CLUSTER" == "local" ]; then
    LOCAL_OUT_DIR=$6
fi

# Define the 4 taxon group experiments
# uncomment the experiments you want to run
TAXON_EXPERIMENTS=(
    # "Primates vertebrate_mammalian"
    # "Primates vertebrate_mammalian vertebrate_other"
    # "Primates vertebrate_mammalian vertebrate_other invertebrate"
    "Primates vertebrate_mammalian vertebrate_other invertebrate fungi plant protozoa"
)

echo "Running 4 experiments with different taxon group combinations..."

# Run each experiment
for i in "${!TAXON_EXPERIMENTS[@]}"; do
    echo "=== Experiment $((i+1))/4 ==="
    IFS=' ' read -ra TAXON_GROUPS <<< "${TAXON_EXPERIMENTS[$i]}"
    
    if [ "$CLUSTER" == "local" ]; then
        run_experiment "$USER" "$CLUSTER" "$CHECKPOINT_PATH" "$MODEL_NAME" "$NUM_JOBS" "$LOCAL_OUT_DIR" "${TAXON_GROUPS[@]}"
    else
        run_experiment "$USER" "$CLUSTER" "$CHECKPOINT_PATH" "$MODEL_NAME" "$NUM_JOBS" "" "${TAXON_GROUPS[@]}"
    fi
    
    echo "=== Experiment $((i+1))/4 completed ==="
    echo
done

echo "All 4 experiments submitted!"
