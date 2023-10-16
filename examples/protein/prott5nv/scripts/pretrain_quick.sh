#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
# Example shell script to run ProtT5 training.
####

### CONFIG ###
CONFIG_FILE=pretrain_small
PROJECT=ProtT5
DATA_MOUNT=/data/uniref2022_05
PROJECT_MOUNT=/workspace/bionemo # /workspace/bionemo if library mounted or /opt/nvidia/bionemo
OUTPUT_MOUNT=/result
WANDB_OFFLINE=True # set to False to upload to WandB while training, otherwise True
RESULTS_MOUNT=${OUTPUT_MOUNT}/nemo_experiments/${PROJECT}/${CONFIG_FILE}
DATA_FILES_SELECTED=x_OP_000..049_CL_  # x000 for a single file for x_OP_000..049_CL_ for a range
### END CONFIG ###

# Don't change these
BIONEMO_HOME=/opt/nvidia/bionemo # Where BioNeMo is installed in container, set the same as Docker container
BIONEMO_WORKSPACE=/workspace/bionemo # Location of examples / config files and where BioNeMo code can be mounted for development
RUN_SCRIPT="pretrain.py"
RUN_SCRIPT_DIRECTORY="${PROJECT_MOUNT}/examples/protein/prott5nv"

usage() {
cat <<EOF
USAGE: pretrain_quick.sh
ProtT5 pretrain script
----------------------------------------
pretrain_quick.sh [command]
    valid commands:
        preprocess
        train

    default command:
        train

    options:
        -f|--data-files
            List of data files to use without csv file extension
            e.g. x_OP_000..001_CL_ for x000.csv and x001.csv, or can be single file
        -c|--config
            Name of YAML configuration file without file extension
            e.g. prott5_small_config
        -a|--args
            Additional training arguments to be added, repeat flag for additional arguments
            e.g. --args "++trainer.devices=2" --args "++model.tensor_model_parallel_size=2"

EOF
}


execute() {
    TRAINING_ARGS="${TRAINING_ARGS} model.data.dataset_path=${DATA_MOUNT}" # Works even if $TRAINING_ARGS is empty
    TRAINING_ARGS="${TRAINING_ARGS} ++model.data.dataset.train=${DATA_FILES_SELECTED} ++model.data.dataset.val=${DATA_FILES_SELECTED} ++model.data.dataset.test=${DATA_FILES_SELECTED}"
    TRAINING_ARGS="${TRAINING_ARGS} exp_manager.exp_dir=${RESULTS_MOUNT}"
    TRAINING_ARGS="${TRAINING_ARGS} ++exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE}"

    set -x
    python ${RUN_SCRIPT} \
        --config-path=conf \
        --config-name=${CONFIG_FILE} \
	    do_training=${DO_TRAINING} \
        ${TRAINING_ARGS}
    set +x
}


preprocess() {
    DO_TRAINING="False"
    execute
}


train() {
    DO_TRAINING="True"
    execute
}


parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--data-files)
                DATA_FILES_SELECTED=$2
                shift
                shift
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift
                shift
                ;;
            -a|--args)
                TRAINING_ARGS="${TRAINING_ARGS} $2"
                shift
                shift
                ;;
            *)
                usage
                exit 1
                ;;
        esac
    done
}


mkdir -p ${RESULTS_MOUNT}
export PROJECT_MOUNT=${PROJECT_MOUNT}

if [[ ${PROJECT_MOUNT} != ${BIONEMO_HOME} ]]; then
    echo "Prepending ${PROJECT_MOUNT} to PYTHONPATH for development"
    DEV_PYTHONPATH=${PROJECT_MOUNT}
else
    DEV_PYTHONPATH=""
fi
export PYTHONPATH="${DEV_PYTHONPATH}:${BIONEMO_WORKSPACE}:${BIONEMO_WORKSPACE}/generated:$PYTHONPATH"

export HYDRA_FULL_ERROR=1
cd ${RUN_SCRIPT_DIRECTORY}

if [ $# -eq 0 ]; then
    ARGS=train
    parse_args "$@"
else
    ARGS=$1
    shift
    parse_args "$@"
fi

case $ARGS in
    preprocess)
        preprocess
        ;;
    train)
        train
        ;;
    *)
        usage
        exit 1
        ;;
esac
