#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8      # n gpus per machine <required>
#SBATCH --mail-type=FAIL
#SBATCH --time=04:00:00
#SBATCH --partition=batch_block1
#SBATCH --account=convai_bionemo_training
#SBATCH --job-name=bionemo_equidock
#SBATCH --mem=0                 # all mem avail
#SBATCH --overcommit
#SBATCH --exclusive             # exclusive node access

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -x

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--prop)
            echo 'Overwriting values from $2.'
            PROPERTY_FILES=$2
            shift
            shift
            ;;
        *)
            echo 'Invalid input'
            exit 1
            ;;
    esac
done

# All variables with default values must be defined in this section
#=========================
BIONEMO_IMAGE="nvcr.io/nvidian/clara-lifesciences/bionemo_equidock:0.1.0"
STORAGE_ROOT="" # Add dataset path (download possible through ngc dataset: processed DATASETID: 1610980, raw DATASETID: 1611195)
WANDB_API_KEY= # Add WANDB API KEY
MICRO_BATCH_SIZE=32 # Please check GPU mem size. 256 is recommended for A100 with 80 GB mem.
JOB_TYPE='nemo-bionemo'
EXP_NAME_PREFIX='bionemo'
EXP_DIR="" # Add mount path for results
#=========================

set -e
if [ -z "${STORAGE_ROOT}" ];
then
    echo "STORAGE_ROOT is invaild. STORAGE_ROOT=${STORAGE_ROOT}. Please check the properties file."
    exit 1
fi

EXP_NAME=${EXP_NAME_PREFIX}_node_${SLURM_JOB_NUM_NODES}
DATA_PATH="${STORAGE_ROOT}"
MOUNTS="$DATA_PATH:/data,$EXP_DIR:/result"

# NeMo and BioNeMo code is picked from the container. To use code from a shared
# folder instead, please NEMO_CODE and BIONEMO_CODE in the properties file.
if [ ! -z "${BIONEMO_CODE}" ];
then
    MOUNTS="${MOUNTS},${BIONEMO_CODE}:$BIONEMO_HOME"
fi

echo "INFO: bionemo code: ${BIONEMO_CODE}"

set -x
srun \
    --output slurm-%j-%n.out \
    --error error-%j-%n.out \
    --container-image ${BIONEMO_IMAGE} \
    --container-mounts ${MOUNTS} \
    --container-workdir /workspace/bionemo/examples/protein/equidock \
    --export WANDB_API_KEY="${WANDB_API_KEY}" \
    bash -c "cd   \$BIONEMO_HOME/examples/protein/equidock ;
    python pretrain.py  --config-path=conf    --config-name=pretrain    exp_manager.wandb_logger_kwargs.job_type="${JOB_TYPE}"    exp_manager.wandb_logger_kwargs.name=${EXP_NAME}    trainer.num_nodes=${SLURM_JOB_NUM_NODES}    trainer.devices=${SLURM_NTASKS_PER_NODE}    model.micro_batch_size=${MICRO_BATCH_SIZE}"
set +x
