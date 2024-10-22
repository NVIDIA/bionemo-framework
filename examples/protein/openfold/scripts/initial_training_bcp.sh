#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Below is a sample set of parameters for launching OpenFold model training with BioNeMo on BCP clusters
# Replace all ?? with appropriate values prior to launching a job
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/protein/openfold/conf/openfold_initial_training.yaml

BIONEMO_IMAGE=?? # BioNeMo container image
WANDB_API_KEY=?? # Add your WANDB API KEY

CONFIG_NAME='openfold_initial_training' # name of the yaml config file with parameters

# NGC specific parameters
TIME_LIMIT="2h"
REPLICAS=16  #number of nodes for the job
NGC_GPUS_PER_NODE=8 #number of gpus per node
ACE=nv-us-east-2
INSTANCE="dgxa100.80g.8.norm"
ORG=nvidian
TEAM=clara-lifesciences
LABEL=ml___openfold
WL_LABEL=wl___other___bionemo
JOB_NAME=ml-model.openfold-bionemo
WORKSPACE=?? # NGC workspace ID goes here, it will be mounted under "/result". Please use RESULTS_PATH to specify this run direcory

# Training parameters
# =========================
MICRO_BATCH_SIZE=1 # micro batch size per GPU, for best efficiency should be set to occupy ~85% of GPU memory. OpenFold saturates A100 with just a single example
ACCUMULATE_GRAD_BATCHES=1 # gradient accumulation
VAL_CHECK_INTERVAL=200 # how often validation step is performed, including downstream task validation
MAX_STEPS=80_000 # duration of training as the number of training steps
# =========================

# Logging
# =========================
PROJECT_NAME="openfold-initial-training"  # project name, will be used for logging
EXP_TAG="-base" # any additional experiment info, can be empty
EXP_NAME="openfold_gradacc${ACCUMULATE_GRAD_BATCHES}_nodes${SLURM_JOB_NUM_NODES}${EXP_TAG}"
CREATE_WANDB_LOGGER=True
WANDB_LOGGER_OFFLINE=False
# =========================

# Mounts
# =========================
OPEN_PROTEIN_SET_DATASET_ID=1612279 # ID of the dataset with processed OpenProteinSet
PDB_MMCIF_DATASET_ID=1612359 # ID of the dataset with processed pdb_mmcif dataset

DATA_MOUNT=/data  # path to training data in the container
OPEN_PROTEIN_SET_MOUNT=${DATA_MOUNT}/open_protein_set/processed # do not change it. It assumes correct paths structure.
PDB_MMCIF_DATASET_MOUNT=${DATA_MOUNT}/pdb_mmcif/processed # do not change it. It assumes correct paths structure.


RESULTS_PATH="/result/nemo_experiments/${PROJECT_NAME}/${EXP_NAME}" # directory to store logs, checkpoints and results

# =========================

# if $LOCAL_ENV file exists, source it to specify my environment
if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
fi

# Command
# =========================
read -r -d '' COMMAND <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB_API_KEY} \
&& echo "Starting training" \
&& cd \$BIONEMO_HOME \
&& cd examples/protein/openfold \
&& python \$BIONEMO_HOME/examples/protein/openfold/train.py \
    --config-path=\$BIONEMO_HOME/examples/protein/openfold/conf \
    --config-name=${CONFIG_NAME} \
    exp_manager.exp_dir=${RESULTS_PATH} \
    exp_manager.create_wandb_logger=${CREATE_WANDB_LOGGER} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.offline=${WANDB_LOGGER_OFFLINE} \
    trainer.num_nodes=${REPLICAS} \
    trainer.devices=${NGC_GPUS_PER_NODE} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
    trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    model.data.dataset_path=${DATA_MOUNT} \

EOF
# =========================

BCP_COMMAND="bcprun --debug --nnodes=${REPLICAS} \
             --npernode=${NGC_GPUS_PER_NODE} \
             -e WANDB_API_KEY=${WANDB_API_KEY} --cmd '"${COMMAND}"'"

echo "ngc batch run --name "${JOB_NAME}" \
      --priority NORMAL --preempt RUNONCE \
      --total-runtime ${TIME_LIMIT} --ace "${ACE}" \
      --instance "${INSTANCE}" --array-type "PYTORCH" --commandline "\"${BCP_COMMAND}"\" \
      --result /result/ngc_log --replicas "${REPLICAS}" \
      --image "${BIONEMO_IMAGE}" --org ${ORG} --team ${TEAM} \
      --workspace ${WORKSPACE}:/result --datasetid ${DATASET_ID}:${DATA_MOUNT} \
      --datasetid ${DWNSTR_TASK_DATASET_ID}:${DWNSTR_TASK_DATA_MOUNT} \
      --label ${LABEL} --label ${WL_LABEL}" | bash
