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

# Below is a sample set of parameters for launching ESM1nv model training with BioNeMo on BCP clusters
# Replace all ?? with appropriate values prior to launching a job
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/protein/esm1nv/conf/pretrain_small.yaml

BIONEMO_IMAGE=?? # BioNeMo container image
WANDB_API_KEY=?? # Add your WANDB API KEY

CONFIG_NAME='pretrain_small' # name of the yaml config file with parameters

# NGC specific parameters
TIME_LIMIT="2h"
NGC_ARRAY_SIZE=1  #number of nodes for the job
NGC_GPUS_PER_NODE=2 #number of gpus per node
REPLICAS=1 #equal to the number of nodes
ACE=nv-us-west-2
INSTANCE="dgx1v.32g.2.norm"
ORG=nvidian
TEAM=clara-lifesciences
LABEL=ml__bionemo
WL_LABEL=wl___other___bionemo
JOB_NAME=ml-model.bionemo-fw-esm1nv-pretrain
WORKSPACE=?? # NGC workspace ID goes here

# Training parameters
# =========================
MICRO_BATCH_SIZE=16 # micro batch size per GPU, for best efficiency should be set to occupy ~85% of GPU memory. Suggested value for A100 80GB is 256
ACCUMULATE_GRAD_BATCHES=1 # gradient accumulation
TENSOR_MODEL_PARALLEL_SIZE=1 # tensor model parallel size
VAL_CHECK_INTERVAL=500 # how often validation step is performed, including downstream task validation
MAX_STEPS=1000000 # duration of training as the number of training steps
# =========================

# Logging
# =========================
PROJECT_NAME="esm1nv_pretraining" # project name, will be used for logging
EXP_TAG="-small" # any additional experiment info, can be empty
EXP_NAME="esm1nv_batch${MICRO_BATCH_SIZE}_gradacc${ACCUMULATE_GRAD_BATCHES}_nodes${NGC_ARRAY_SIZE}${EXP_TAG}"
CREATE_WANDB_LOGGER=True # set to False if you don't want to log results with WandB
WANDB_LOGGER_OFFLINE=False # set to True if there are issues uploading to WandB during training
# =========================

# Mounts
# =========================
DATASET_ID=110553 # ID of the dataset with Uniref50 data
DWNSTR_TASK_DATASET_ID=1612756 # ID of the dataset with FLIP secondary structure data for downstream task validation
DATA_MOUNT=/data # path to training data in the container
DWNSTR_TASK_DATA_MOUNT=/FLIP # path to downstream task validation data in the container
DATASET=uniref202205_0256 # folder containing data for model training
DWNSTR_TASK_DATASET=secondary_structure # folder containing data for downstream task validation
TRAIN_FILES='x_OP_000..049_CL_' # Range for the train dataset
TEST_FILES='x_OP_000..049_CL_'  # Range for the test dataset
VAL_FILES='x_OP_000..049_CL_'   # Range for the val dataset
INDEX_MAPPING_DIR=/result/index_files/${PROJECT_NAME}/${EXP_NAME}

RESULTS_PATH="/result/nemo_experiments/${PROJECT_NAME}/${EXP_NAME}" # directory to store logs, checkpoints and results

# =========================

# if $LOCAL_ENV file exists, source it to specify my environment
if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
fi

# Note: BIONEMO_HOME is set inside the container to the correct repo path (typically /workspace/bionemo)
read -r -d '' COMMAND <<EOF
echo "*******STARTING********" \
&& cd \$BIONEMO_HOME/examples/protein/esm1nv \
&& python pretrain.py \
    --config-path=\$BIONEMO_HOME/examples/protein/esm1nv/conf \
    --config-name=${CONFIG_NAME} \
    exp_manager.exp_dir=${RESULTS_PATH} \
    exp_manager.create_wandb_logger=${CREATE_WANDB_LOGGER} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.offline=${WANDB_LOGGER_OFFLINE} \
    trainer.num_nodes=${NGC_ARRAY_SIZE} \
    trainer.devices=${NGC_GPUS_PER_NODE} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
    trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    model.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    ++model.data.index_mapping_dir=${INDEX_MAPPING_DIR} \
    model.dwnstr_task_validation.dataset.dataset_path=${DWNSTR_TASK_DATA_MOUNT}/${DWNSTR_TASK_DATASET} \
    model.data.dataset_path=${DATA_MOUNT}/${DATASET} \
    model.data.dataset.train=${TRAIN_FILES} \
    model.data.dataset.val=${VAL_FILES} \
    model.data.dataset.test=${TEST_FILES}

EOF

BCP_COMMAND="bcprun --debug --nnodes=${NGC_ARRAY_SIZE} \
             --npernode=${NGC_GPUS_PER_NODE} \
             -e WANDB_API_KEY=${WANDB_API_KEY} --cmd '"${COMMAND}"'"

#Add --array-type "PYTORCH" to command below for multinode jobs
echo "ngc batch run --name "${JOB_NAME}" \
      --priority NORMAL --preempt RUNONCE \
      --total-runtime ${TIME_LIMIT} --ace "${ACE}" \
      --instance "${INSTANCE}" --commandline "\"${BCP_COMMAND}"\" \
      --result /result/ngc_log --replicas "${REPLICAS}" \
      --image "${BIONEMO_IMAGE}" --org ${ORG} --team ${TEAM} \
      --workspace ${WORKSPACE}:/result --datasetid ${DATASET_ID}:${DATA_MOUNT} \
      --datasetid ${DWNSTR_TASK_DATASET_ID}:${DWNSTR_TASK_DATA_MOUNT} \
      --label ${LABEL} --label ${WL_LABEL}" | bash
