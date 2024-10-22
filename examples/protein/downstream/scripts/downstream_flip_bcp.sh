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

# Usage
# =========================
# This is an example script for running a predictive task on BCP.
# Below is a sample set of parameters for launching ESM-{1,2}nv or ProtT5nv
# finetuning for a FLIP downstream task with BioNeMo on BCP clusters.
# Replace all ?? with appropriate values prior to launching a job.
# Any parameters not specified in this script, such as the ESM-2 checkpoint size,
# can be changed in the yaml config file located in examples/protein/MODEL/conf/
# directory where MODEL is {esm1nv, esm2nv, prott5nv}.

BIONEMO_IMAGE=?? # BioNeMo container image
WANDB_API_KEY=?? # Add your WANDB API KEY

CONFIG_NAME='downstream_flip_sec_str' # yaml config file {downstream_flip_sec_str, downstream_flip_scl, downstream_flip_meltome}, should be aligned with TASK_NAME parameter.
PROTEIN_MODEL=esm2nv # protein LM name, can be {esm1nv, esm2nv, prott5nv}
# =========================

# NGC specific parameters
# =========================
TIME_LIMIT="2h"
NGC_ARRAY_SIZE=1  # number of nodes for the job
NGC_GPUS_PER_NODE=2  # number of gpus per node
REPLICAS=1  # equal to the number of nodes
ACE=nv-us-west-2
INSTANCE="dgx1v.32g.2.norm"
NGC_CLI_API_KEY=??
NGC_CLI_ORG=nvidian
NGC_CLI_TEAM=cvai_bnmo_trng

LABEL=ml__bionemo
WL_LABEL=wl___other___bionemo
JOB_NAME=ml-model.bionemo-fw-${PROTEIN_MODEL}-finetune
WORKSPACE=??  # Your NGC workspace ID goes here
# =========================

# Training parameters
# =========================
ACCUMULATE_GRAD_BATCHES=1 # gradient accumulation
ENCODER_FROZEN=True # encoder can be frozen or trainable
RESTORE_FROM_PATH=/model/protein/${PROTEIN_MODEL}/esm2nv_650M_converted.nemo # Path to the pretrained model checkpoint in the container
TENSOR_MODEL_PARALLEL_SIZE=1 # tensor model parallel size,  model checkpoint must be compatible with tensor model parallel size
MICRO_BATCH_SIZE=32 # micro batch size per GPU, for best efficiency should be set to occupy ~85% of GPU memory. Suggested value for A100 80GB is 256
MAX_STEPS=2000 # duration of training as the number of training steps
VAL_CHECK_INTERVAL=20 # how often validation step is performed, including downstream task validation
# =========================

# Logging parameters
# =========================
DATA_PATH=/data/FLIP
TASK_NAME=secondary_structure # FLIP task name: secondary_structure, scl, meltome, etc.
EXP_DIR=/workspace/nemo_experiments/${PROTEIN_MODEL}/${CONFIG_NAME}_encoder_frozen-${ENCODER_FROZEN}_tp${TENSOR_MODEL_PARALLEL_SIZE}_grad_acc${ACCUMULATE_GRAD_BATCHES}
WANDB_LOGGER_NAME=${PROTEIN_MODEL}_${CONFIG_NAME}_finetune_encoder_frozen-${ENCODER_FROZEN}_tp${TENSOR_MODEL_PARALLEL_SIZE}_grad_acc${ACCUMULATE_GRAD_BATCHES}
WANDB_LOGGER_OFFLINE=False # set to True if uploading results to WandB online is undesired

CONFIG_PATH=../${PROTEIN_MODEL}/conf
# =========================

# Instructions for downloading checkpoint, preprocessing the data and running finetuning
# =========================
read -r -d '' BCP_COMMAND <<EOF
bcprun --debug --nnodes=${NGC_ARRAY_SIZE} --npernode=${NGC_GPUS_PER_NODE} -w /workspace/bionemo -e WANDB_API_KEY=${WANDB_API_KEY} --cmd 'export NGC_CLI_ORG=$NGC_CLI_ORG NGC_CLI_API_KEY=$NGC_CLI_API_KEY MODEL_PATH=/model; ./launch.sh download';
bcprun --debug --nnodes=${NGC_ARRAY_SIZE} --npernode=1 -w /workspace/bionemo/examples/protein/downstream -e WANDB_API_KEY=${WANDB_API_KEY} --cmd 'python downstream_flip.py do_training=False';
bcprun --debug --nnodes=${NGC_ARRAY_SIZE} --npernode=${NGC_GPUS_PER_NODE} -w /workspace/bionemo/examples/protein/downstream -e WANDB_API_KEY=${WANDB_API_KEY} --cmd 'python downstream_flip.py --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} exp_manager.exp_dir=${EXP_DIR} exp_manager.wandb_logger_kwargs.offline=${WANDB_LOGGER_OFFLINE} \
    trainer.devices=${NGC_GPUS_PER_NODE} trainer.num_nodes=${NGC_ARRAY_SIZE} ++model.dwnstr_task_validation.enabled=False \
    model.micro_batch_size=${MICRO_BATCH_SIZE} model.data.task_name=${TASK_NAME} model.data.dataset_path=${DATA_PATH}/${TASK_NAME} \
    exp_manager.wandb_logger_kwargs.name=${WANDB_LOGGER_NAME} trainer.val_check_interval=${VAL_CHECK_INTERVAL} model.global_batch_size=null \
    trainer.max_steps=${MAX_STEPS} model.encoder_frozen=${ENCODER_FROZEN} model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    restore_from_path=${RESTORE_FROM_PATH} trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}'
EOF
# =========================

# Add --array-type "PYTORCH" to command below for multinode jobs
echo "ngc batch run --name "${JOB_NAME}" --priority NORMAL \
      --preempt RUNONCE --total-runtime ${TIME_LIMIT} --ace "${ACE}" \
      --instance "${INSTANCE}" --commandline "\"${BCP_COMMAND}"\" \
      --result /result/ngc_log --replicas "${REPLICAS}" \
      --image "${BIONEMO_IMAGE}" --org ${NGC_CLI_ORG} --team ${NGC_CLI_TEAM} \
      --workspace ${WORKSPACE}:/result --label ${LABEL} --label ${WL_LABEL}" | bash
