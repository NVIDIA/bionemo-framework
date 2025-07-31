#!/bin/bash

sleep 3600

DEST_DIR="/workspaces/bionemo-framework/ft-attack/checkpoints/"


# Define relative checkpoint paths
CHECKPOINTS=(
    "ft_checkpoints/evo2_7b_1m_100_ncbi_virus_human_host_full_species/evo2/checkpoints/epoch=0-step=99-consumed_samples=800.0-last"
    "ft_checkpoints/evo2_7b_1m_200_ncbi_virus_human_host_full_species/evo2/checkpoints/epoch=0-step=199-consumed_samples=1600.0-last"
    "ft_checkpoints/evo2_7b_1m_500_ncbi_virus_human_host_full_species/evo2/checkpoints/epoch=0-step=499-consumed_samples=4000.0-last"
    "ft_checkpoints/evo2_7b_1m_1000_ncbi_virus_human_host_full_species/evo2/checkpoints/epoch=0-step=999-consumed_samples=8000.0-last"
    # "ft_checkpoints/evo2_7b_1m_2000_ncbi_virus_human_host_full_species/evo2/checkpoints/epoch=0-step=1999-consumed_samples=16000.0-last"
)

# Function to run evaluation
run_evaluation() {
    local checkpoint_path="$1"
    local gpu_devices="$2"
    local model_dir=$(echo "$checkpoint_path" | cut -d'/' -f1)
    
    echo "Starting evaluation of $model_dir on GPUs $gpu_devices..."
    
    CUDA_VISIBLE_DEVICES="$gpu_devices" python eval/eval_fitness.py \
        --ckpt-dir "$DEST_DIR$checkpoint_path" \
        --model-size 7b_arc_longcontext \
        --batch-size 16 \
        --tensor-parallel-size 4 \
        --DMS_filenames /workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv \
        --DMS_scores_folder /workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/nucleotides/
    
    echo "Completed evaluation of $model_dir"
}

# OPTION 1: Simple sequential (no parallel, one at a time)
# echo "Starting sequential evaluations..."
# for checkpoint_path in "${CHECKPOINTS[@]}"; do
#     run_evaluation "$checkpoint_path" "0,1,2,3"
# done

# OPTION 2: Simple batched parallel (comment out the advanced version below to use this)
# echo "Starting simple batched parallel evaluations..."
# for i in "${!CHECKPOINTS[@]}"; do
#     checkpoint_path="${CHECKPOINTS[$i]}"
#     
#     # Alternate between GPU sets: 0,1,2,3 and 4,5,6,7
#     if (( i % 2 == 0 )); then
#         gpu_devices="0,1,2,3"
#     else
#         gpu_devices="4,5,6,7"
#     fi
#     
#     # Run in background
#     run_evaluation "$checkpoint_path" "$gpu_devices" &
#     
#     # If we've started 2 processes, wait for them to complete before starting next batch
#     if (( (i + 1) % 2 == 0 )); then
#         wait
#     fi
# done
# wait

#OPTION 3: Advanced parallel with dynamic GPU allocation (currently active)
# Run evaluations with maximum parallelization


echo "Starting parallel evaluations..."

# Track background job PIDs and their GPU sets
declare -A job_gpus
gpu_set1_free=true
gpu_set2_free=true

for checkpoint_path in "${CHECKPOINTS[@]}"; do
    # Wait for at least one GPU set to be available
    while [[ "$gpu_set1_free" == false && "$gpu_set2_free" == false ]]; do
        # Check if any background jobs have finished
        for pid in "${!job_gpus[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                # Job finished, free up its GPUs
                if [[ "${job_gpus[$pid]}" == "0,1,2,3" ]]; then
                    gpu_set1_free=true
                else
                    gpu_set2_free=true
                fi
                unset job_gpus[$pid]
            fi
        done
        
        # Brief sleep to avoid busy waiting
        [[ "$gpu_set1_free" == false && "$gpu_set2_free" == false ]] && sleep 1
    done
    
    # Assign to available GPU set
    if [[ "$gpu_set1_free" == true ]]; then
        gpu_devices="0,1,2,3"
        gpu_set1_free=false
    else
        gpu_devices="4,5,6,7"
        gpu_set2_free=false
    fi
    
    # Start evaluation in background
    run_evaluation "$checkpoint_path" "$gpu_devices" &
    job_gpus[$!]="$gpu_devices"
done

# Wait for all remaining jobs to complete
wait

echo "All evaluations completed!"