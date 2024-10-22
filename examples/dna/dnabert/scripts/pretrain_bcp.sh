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

# Below is a sample set of parameters for launching DNABERT model training with BioNeMo on BCP clusters
# Replace all ?? with appropriate values prior to launching a job
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/dna/dnabert/conf/dnabert_config.yaml

BIONEMO_IMAGE=???
WANDB_API_KEY=???

CONFIG_NAME='dnabert_config' # name of the yaml config file with parameters

# NGC specific parameters
TIME_LIMIT="2h"
NGC_ARRAY_SIZE=1  #number of nodes for the job
NGC_GPUS_PER_NODE=2 #number of gpus per node
REPLICAS=1 #equal to the number of nodes
ACE=nv-us-east-3
INSTANCE="dgxa100.80g.4.norm"
ORG=nvidian
TEAM=cvai_bnmo_trng
LABEL=ml__bionemo
WL_LABEL=wl___other___bionemo
JOB_NAME=ml-model.bionemo-fw-dnabert-pretrain
RESULT_WORKSPACE=???

# Training parameters
# =========================
MICRO_BATCH_SIZE=16 # micro batch size per GPU, for best efficiency should be set to occupy ~85% of GPU memory. Suggested value for A100 80GB is 256
ACCUMULATE_GRAD_BATCHES=1 # gradient accumulation
TENSOR_MODEL_PARALLEL_SIZE=1 # tensor model parallel size
VAL_CHECK_INTERVAL=50 # how often validation step is performed, including downstream task validation
MAX_STEPS=100 # duration of training as the number of training steps
# =========================

# Logging
# =========================
PROJECT_NAME="dnabert_pretraining" # project name, will be used for logging
EXP_TAG="-full" # any additional experiment info, can be empty
EXP_NAME="dnabert_batch${MICRO_BATCH_SIZE}_gradacc${ACCUMULATE_GRAD_BATCHES}_nodes${NGC_ARRAY_SIZE}${EXP_TAG}"
CREATE_WANDB_LOGGER=True # set to False if you don't want to log results with WandB
WANDB_LOGGER_OFFLINE=False # set to True if there are issues uploading to WandB during training
# =========================


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
mkdir -p /bionemo_fw_qa/steven/result &&
&& cd \$BIONEMO_HOME/examples/dna/dnabert\
&& python pretrain.py \
    --config-path=\$BIONEMO_HOME/examples/dna/dnabert/conf \
    --config-name=${CONFIG_NAME} \
    ++do_training=False \
    ++model.data.dataset_path=/bionemo_fw_qa/steven/result/GRCh38.p13 \
&&
python pretrain.py \
    --config-path=\$BIONEMO_HOME/examples/dna/dnabert/conf \
    --config-name=${CONFIG_NAME} \
    ++do_training=True \
    ++model.data.dataset_path=/bionemo_fw_qa/steven/result/GRCh38.p13 \
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
    model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}

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
      --workspace ${RESULT_WORKSPACE}:/result \
      --label ${LABEL} --label ${WL_LABEL}" | bash
