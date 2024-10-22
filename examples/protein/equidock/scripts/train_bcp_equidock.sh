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

BIONEMO_IMAGE=?? # BioNeMo container image
WANDB_API_KEY=?? # Add your WANDB API KEY

# Model & Dataset
# =========================
DATASET_NAME=dips
MODEL_NAME=EquiDock_${DATASET_NAME}
MICRO_BATCH_SIZE=32
# processed EquiDock datasets (1610980)
# raw data (1611195) should be processed using:
# python examples/protein/equidock/pretrain.py do_training=False
DATASET_ID=1610980

# NGC specific parameters
# =========================
NGC_ARRAY_SIZE=1 # number of nodes for the job
NGC_GPUS_PER_NODE=2 # number of gpus per node
REPLICAS=${NGC_ARRAY_SIZE} # equal to the number of nodes
ACE=nv-us-west-2
INSTANCE="dgx1v.32g.2.norm" # choose instance based on NGC_ARRAY_SIZE and NGC_GPUS_PER_NODE and available GPUs
NGC_CLI_ORG=nvidian
NGC_CLI_TEAM=cvai_bnmo_trng

LABEL=ml___equidock
JOB_NAME=ml-model.bionemo-fw-${MODEL_NAME}-train
WORKSPACE=?? # Your NGC workspace ID goes here

# Logging parameters
# =========================
WANDB_LOGGER_NAME=${MODEL_NAME}_nnodes_${NGC_ARRAY_SIZE}_ndevices_${NGC_GPUS_PER_NODE}_bs_${MICRO_BATCH_SIZE}
EXP_DIR=/workspace/nemo_experiments/${WANDB_LOGGER_NAME}

# Note: BIONEMO_HOME is set inside the container to the correct repo path (typically /workspace/bionemo)
BIONEMO_HOME=/workspace/bionemo

read -r -d '' BCP_COMMAND <<EOF
bcprun --debug --nnodes=${NGC_ARRAY_SIZE} --npernode=${NGC_GPUS_PER_NODE} \
    -w ${BIONEMO_HOME}/examples/protein/equidock -e WANDB_API_KEY=${WANDB_API_KEY} \
    --cmd 'python pretrain.py --config-name=pretrain data.data_name=${DATASET_NAME} \
    trainer.devices=${NGC_GPUS_PER_NODE} trainer.num_nodes=${NGC_ARRAY_SIZE} \
    exp_manager.exp_dir=${EXP_DIR} exp_manager.wandb_logger_kwargs.offline=False \
    trainer.devices=${NGC_GPUS_PER_NODE} trainer.num_nodes=${NGC_ARRAY_SIZE} \
    model.micro_batch_size=${MICRO_BATCH_SIZE}'
EOF

#Add --array-type "PYTORCH" to command below for multinode jobs
echo "ngc batch run --name "${JOB_NAME}" --priority NORMAL --preempt RUNONCE \
    --ace "${ACE}" --instance "${INSTANCE}" --commandline "\"${BCP_COMMAND}"\" \
    --result ${BIONEMO_HOME}/result/ngc_log --replicas "${REPLICAS}" \
    --image "${BIONEMO_IMAGE}" --org ${NGC_CLI_ORG} --team ${NGC_CLI_TEAM} \
    --workspace ${WORKSPACE}:${BIONEMO_HOME}/result \
    --datasetid ${DATASET_ID}:${BIONEMO_HOME}/data --label ${LABEL} --label wl___other___bionemo" | bash
