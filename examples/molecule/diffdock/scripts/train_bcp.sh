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


# Below is a sample set of parameters for launching DiffDock model training with BioNeMo on BCP clusters
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/molecule/diffdock/conf/train_score.yaml and
# examples/molecule/diffdock/conf/train_confidence.yaml

# set your Weights and Biases API KEY
if [[ "${WANDB_API_KEY}" == "" ]]; then
  echo "ERROR: Must set WANDB_API_KEY"
  exit 1
fi

if [[ "${BIONEMO_IMAGE}" == "" ]]; then
  echo "ERROR: Must set a specific BIONEMO_IMAGE. This image name must be complete and the image must reside on NGC."
  exit 1
fi

# NGC workspace ID goes here, it will be mounted under "/result".
# Please use RESULTS_PATH to specify this run directory
if [[ "${WORKSPACE}" == "" ]]; then
  echo "ERROR: Must set an NGC workspace as WORKSPACE"
  exit 1
fi

set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-"score"} # "score", "confidence", to train score or confidence model
if [[ "${MODEL_TYPE}" != "score" && "${MODEL_TYPE}" != "confidence" ]]; then
  echo "ERROR: Invalid MODEL_TYPE, only 'score' or 'confidence' accepted. Found: ${MODEL_TYPE}"
  exit 1
fi

CONFIG_NAME="train_${MODEL_TYPE}" # name of the yaml config file with parameters

# NGC specific parameters
TIME_LIMIT=${TIME_LIMIT:-"2h"}
REPLICAS=${REPLICAS:-1} # number of nodes for the job
NGC_GPUS_PER_NODE=${NGC_GPUS_PER_NODE:-8} # number of gpus per node
ACE=${ACE:-"nv-us-east-2"}
INSTANCE=${INSTANCE:-"dgxa100.80g.8.norm"}
ORG=${ORG:-"nvidian"}
TEAM=${TEAM:-"clara-lifesciences"}
LABEL=${LABEL:-"ml___diffdock"}
WL_LABEL=${WL_LABEL:-"wl___other___bionemo"}
JOB_NAME=${JOB_NAME:-"ml-model.diffdock-bionemo"}

# Training parameters
# =========================
# micro batch size per GPU, for best efficiency should be set to occupy ~85% of GPU memory.
# Suggested value for small span model for A100 80GB is 32
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-12}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-1} # gradient accumulation
MAX_STEPS=${MAX_STEPS:-1000000} # duration of training as the number of training steps
# =========================

# Logging
# =========================
PROJECT_NAME="diffdock_${MODEL_TYPE}_training" # project name, will be used for logging
EXP_TAG=${EXP_TAG:-"-large"} # any additional experiment info
EXP_NAME="diffdock_${MODEL_TYPE}_batch${MICRO_BATCH_SIZE}_gradacc${ACCUMULATE_GRAD_BATCHES}_nodes1${EXP_TAG}"
CREATE_WANDB_LOGGER=True
WANDB_LOGGER_OFFLINE=False
# =========================

# Note: BIONEMO_HOME is set inside the container to the correct repo path (typically /workspace/bionemo)
BIONEMO_HOME==${BIONEMO_HOME:/workspace/bionemo}

# Mounts
# =========================
DATASET_ID=1617183 # processed sample dataset for diffdock score and confidence model training
RESULTS_PATH="${BIONEMO_HOME}/result/nemo_experiments/${PROJECT_NAME}/${EXP_NAME}" # directory to store logs, checkpoints and results

# =========================

# if $LOCAL_ENV file exists, source it to specify my environment
if [ -e "./${LOCAL_ENV}" ]
then
    echo sourcing environment from "./${LOCAL_ENV}"
    source ./${LOCAL_ENV}
fi


# Command
# =========================
read -r -d '' BCP_COMMAND <<EOF
bcprun --debug --nnodes=${REPLICAS} --npernode=${NGC_GPUS_PER_NODE} \
    -w ${BIONEMO_HOME}/examples/molecule/diffdock -e WANDB_API_KEY=${WANDB_API_KEY} \
    --cmd 'export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync && \
    python train.py --config-name=${CONFIG_NAME} \
    exp_manager.exp_dir=${RESULTS_PATH} \
    exp_manager.create_wandb_logger=${CREATE_WANDB_LOGGER} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.offline=${WANDB_LOGGER_OFFLINE} \
    trainer.num_nodes=${REPLICAS} \
    trainer.devices=${NGC_GPUS_PER_NODE} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
    model.micro_batch_size=${MICRO_BATCH_SIZE}'
EOF
# =========================

#Add --array-type "PYTORCH" to command below for multinode jobs
echo "ngc batch run --name "${JOB_NAME}" \
      --priority NORMAL --preempt RUNONCE \
      --total-runtime ${TIME_LIMIT} --ace "${ACE}" \
      --instance "${INSTANCE}" --commandline "\"${BCP_COMMAND}"\" \
      --result ${BIONEMO_HOME}/result/ngc_log --replicas "${REPLICAS}" \
      --image "${BIONEMO_IMAGE}" --org ${ORG} --team ${TEAM} \
      --workspace ${WORKSPACE}:${BIONEMO_HOME}/result --datasetid ${DATASET_ID}:${BIONEMO_HOME}/data \
      --label ${LABEL} --label ${WL_LABEL}" | bash
