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

####
# Example shell script to launch training of
# retrosynthesis prediction with
# MegaMolBart model on NGC.
####

# NOTE: parameters must be adjusted for customer training
# NOTE: this script assumes that model checkpoint is already in the container

LOCAL_ENV=.env

#set container image
CONTAINER_IMAGE=

# Model architecture -- should be megamolbart
MODEL=megamolbart

#Experiment description
DWNSTRM_TASK=retrosynthesis
EXP_TAG="" # customize EXP_TAG to add additional information

# NGC specific parameters
NGC_ARRAY_SIZE=1  #number of nodes for the job
NGC_GPUS_PER_NODE=2 #number of gpus per node
REPLICAS=1 #equal to the number of nodes
ACE=nv-us-west-2
INSTANCE="dgx1v.32g.2.norm"
ORG=nvidian
TEAM=clara-lifesciences
LABEL=ml__bionemo
WL_LABEL=wl___other___bionemo
JOB_NAME=ml-model.bionemo-fw-${MODEL}-downstream-${DWNSTRM_TASK}-${EXP_TAG}
WORKSPACE= # Your NGC workspace ID goes here

# Model specific parameters
ACCUMULATE_GRAD_BATCHES=1
TENSOR_MODEL_PARALLEL_SIZE=1 # tensor model parallel size,  model checkpoint must be compatible with tensor model parallel size
RESTORE_FROM_PATH=/model/molecule/${MODEL}/${MODEL}.nemo
MICRO_BATCH_SIZE=256
MAX_STEPS=20000 # duration of training as the number of training steps
VAL_CHECK_INTERVAL=100 # how often validation step is performed


# Dataset and logging parameters
WANDB_API_KEY= # Your personal WandB API key goes here
DATASET_ID=1612025 # ID of the dataset with USPTO50k data
DATASET_PATH=/data/uspto_50k_dataset
INDEX_MAPPING_DIR=/result/index_files #set index mapping directory
DATA_INPUT_NAME=products
DATA_TARGET_NAME=reactants
EXP_DIR=/workspace/nemo_experiments/${MODEL}/MegaMolBARTRetro_uspto50k/MegaMolBARTRetro_uspto50k_batch${MICRO_BATCH_SIZE}_grad_acc${ACCUMULATE_GRAD_BATCHES}
WANDB_LOGGER_NAME=${MODEL}_MegaMolBARTRetro_uspto50k_batch${MICRO_BATCH_SIZE}_grad_acc${ACCUMULATE_GRAD_BATCHES}_${EXP_TAG}
CONFIG_PATH=./conf

# if $LOCAL_ENV file exists, source it to specify my environment
if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
fi

read -r -d '' COMMAND <<EOF
 python downstream_retro.py --config-path=${CONFIG_PATH} \
 --config-name=downstream_retro_uspto50k exp_manager.exp_dir=${EXP_DIR} \
 exp_manager.wandb_logger_kwargs.offline=False \
 trainer.devices=${NGC_GPUS_PER_NODE} \
 trainer.num_nodes=${NGC_ARRAY_SIZE} \
 model.micro_batch_size=${MICRO_BATCH_SIZE} \
 model.data.dataset_path=${DATASET_PATH} \
 model.data.index_mapping_dir=${INDEX_MAPPING_DIR} \
 model.data.input_name=${DATA_INPUT_NAME} \
 model.data.target_name=${DATA_TARGET_NAME} \
 exp_manager.wandb_logger_kwargs.name=${WANDB_LOGGER_NAME} \
 trainer.max_steps=${MAX_STEPS} model.global_batch_size=null \
 trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
 model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
 restore_from_path=${RESTORE_FROM_PATH} \
 trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}
EOF

BCP_COMMAND="bcprun --debug --nnodes=${NGC_ARRAY_SIZE} --npernode=${NGC_GPUS_PER_NODE} -w /workspace/bionemo/examples/molecule/megamolbart -e WANDB_API_KEY=${WANDB_API_KEY} --cmd '"${COMMAND}"'"

#Add --array-type "PYTORCH" to command below for multinode jobs
echo "ngc batch run --name "${JOB_NAME}" --priority NORMAL --preempt RUNONCE --total-runtime 2h --ace "${ACE}" --instance "${INSTANCE}" --commandline "\"${BCP_COMMAND}"\" --result /result/ngc_log --replicas "${REPLICAS}" --image "${CONTAINER_IMAGE}" --org ${ORG} --team ${TEAM} --workspace ${WORKSPACE}:/result --datasetid ${DATASET_ID}:/data/uspto_50k_dataset/ --label ${LABEL} --label ${WL_LABEL}" | bash
