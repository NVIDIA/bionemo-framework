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

#
# name: preprocess_esm2_bpc.sh
# usage:
#   cd workspace
#   ${BIONEMO_HOME}/examples/protein/esm2nv/scripts/preprocess_esm2_bcp.sh
#
# notes:
#   Step (2) does the preprocessing of raw data
#     Input files and key parameters:
#       model.data.uf50_datapath=${dataset_dir_local_full_path}/${UF50_TRAIN_SEQUENCES_BASE_FILENAME}
#       model.data.uf90_datapath=${dataset_dir_local_full_path}/${UF90_AND_UF50_SEQUENCES_BASE_FILENAME}
#       model.data.cluster_mapping_tsv=${dataset_dir_local_full_path}/${MAPPING_BASE_FILENAME}
#
#     Output files: program will create and populate these directories
#       model.data.dataset_path=${dataset_dir_local_full_path}/${UF50}
#       model.data.uf90.uniref90_path=${dataset_dir_local_full_path}/${UF90}
#
#   See also https://google.github.io/styleguide/shellguide.html
#

# define message template
MESSAGE_TEMPLATE='********preprocess_esm2_bcp.sh: %s\n'
printf "${MESSAGE_TEMPLATE}" "begin"

# file level constants
CONFIG_NAME=pretrain_esm2_650M
UF50_TRAIN_SEQUENCES_BASE_FILENAME=uniref50_train_filt.fasta
UF90_AND_UF50_SEQUENCES_BASE_FILENAME=uniref90membersandreps_ur50trainfiltreps.fasta
MAPPING_BASE_FILENAME=mapping.tsv
UF50=uf50
UF90=uf90
FORMAT_DATETIME_STD='%Y-%m-%d %H:%M:%S'

# default values may not be correct for NVIDIA-external users
DRY_RUN="${DRY_RUN:-'false'}"    # 'true' or 'false'
BIONEMO_HOME_THIS="${BIONEMO_HOME:-/workspace/bionemo}"
DATA_DIR_IN_REMOTE_HD="${DATA_DIR_IN_REMOTE_HD:-/ngc_workspace_mount/data_from_ngc}"  # ngc workspace mount
DATA_DIR_IN_LOCAL_HD="${DATA_DIR_IN_LOCAL_HD:-/workspace/data_from_ngc}"
DATASET_DIR="${DATASET_DIR:-uniref50_90_202104_esm2nv_v1.0}"

# timer functions
# ToDo: replace with parsing 'time' output
timer_start () {
    datetime_before_task="$(date +"${FORMAT_DATETIME_STD}")"
    seconds_before_task="$(date --date="${datetime_before_task}" "+%s")"
}
timer_end() {
    datetime_after_task="$(date +"${FORMAT_DATETIME_STD}")"
    seconds_after_task="$(date --date="${datetime_after_task}" "+%s")"
    delta_seconds="$((seconds_after_task - seconds_before_task))"
}

# print output for checks
printf "${MESSAGE_TEMPLATE}" "BIONEMO_HOME_THIS=${BIONEMO_HOME_THIS}"
printf "${MESSAGE_TEMPLATE}" "DATASET_DIR=${DATASET_DIR}"
printf "${MESSAGE_TEMPLATE}" "DATA_DIR_IN_REMOTE_HD=${DATA_DIR_IN_REMOTE_HD}"
printf "${MESSAGE_TEMPLATE}" "DATA_DIR_IN_LOCAL_HD=${DATA_DIR_IN_LOCAL_HD}"

# check for presence of input dataset on workspace
if [ ! -d "${DATA_DIR_IN_REMOTE_HD}/${DATASET_DIR}" ]; then
    printf "${MESSAGE_TEMPLATE}" "did not find dataset at ${DATA_DIR_IN_REMOTE_HD}/${DATASET_DIR}"
    exit 1
fi

# make a directory in local hd
if [ ! -d "${DATA_DIR_IN_LOCAL_HD}" ]; then
    mkdir -p "${DATA_DIR_IN_LOCAL_HD}"
else
    rm -rf "${DATA_DIR_IN_LOCAL_HD}"/*
fi
# -----------------------------------------------------------------------------
# (1) copy raw data to local hd -----------------------------------------------
raw_dataset_copy_seconds=''
printf "${MESSAGE_TEMPLATE}" "copy data from workspace to local hd, begin"
timer_start
cp -r "${DATA_DIR_IN_REMOTE_HD}/${DATASET_DIR}" "${DATA_DIR_IN_LOCAL_HD}"
timer_end
raw_dataset_copy_seconds="${delta_seconds}"
ls -alF "${DATA_DIR_IN_LOCAL_HD}"
printf "${MESSAGE_TEMPLATE}" "raw_dataset_copy_seconds: ${raw_dataset_copy_seconds}"
printf "${MESSAGE_TEMPLATE}" "copy data from workspace to local hd, end"


# -----------------------------------------------------------------------------
# (2) preprocess raw data, read from local hd ---------------------------------
dataset_dir_local_full_path="${DATA_DIR_IN_LOCAL_HD}/${DATASET_DIR}"

read -r -d '' PYTHON_COMMAND <<EOF
python ${BIONEMO_HOME_THIS}/examples/protein/esm2nv/pretrain.py \\
    --config-name=${CONFIG_NAME} \\
    ++do_preprocess=True \\
    ++do_training=False \\
    ++model.data.uf50_datapath=${dataset_dir_local_full_path}/${UF50_TRAIN_SEQUENCES_BASE_FILENAME} \\
    ++model.data.uf90_datapath=${dataset_dir_local_full_path}/${UF90_AND_UF50_SEQUENCES_BASE_FILENAME} \\
    ++model.data.cluster_mapping_tsv=${dataset_dir_local_full_path}/${MAPPING_BASE_FILENAME} \\
    ++model.data.dataset_path=${dataset_dir_local_full_path}/${UF50} \\
    ++model.data.uf90.uniref90_path=${dataset_dir_local_full_path}/${UF90}
EOF

printf "${MESSAGE_TEMPLATE}" "PYTHON_COMMAND="
printf "${PYTHON_COMMAND}"
printf "\n"

preproc_delta_seconds=''
if [ ! "${DRY_RUN}" == 'true' ]; then
    printf "${MESSAGE_TEMPLATE}" "preprocess raw data read from local hd, begin"

    export PYTHONPATH=".:${BIONEMO_HOME_THIS}:${PYTHONPATH}"
    printf "${MESSAGE_TEMPLATE}" "entering dir=${BIONEMO_HOME_THIS}"
    pushd "${BIONEMO_HOME_THIS}"

    HYDRA_FULL_ERROR=1
    timer_start
    eval "${PYTHON_COMMAND}"
    timer_end
    preproc_delta_seconds="${delta_seconds}"
    popd
    ls -alF "${dataset_dir_local_full_path}"

    printf "${MESSAGE_TEMPLATE}" "preprocess raw data read from local hd, end"
fi

# -----------------------------------------------------------------------------
# (3) transfer preprocessed dataset to workspace, and rename ------------------
printf "${MESSAGE_TEMPLATE}" "copy data from local hd to workspace, begin"
preproc_dataset_copy_seconds=''

dataset_dir_local_full_path_with_preproc_tag="${DATA_DIR_IN_LOCAL_HD}/${DATASET_DIR}_preproc"
dataset_dir_remote_full_path_with_preproc_tag="${DATA_DIR_IN_REMOTE_HD}/${DATASET_DIR}_preproc"
mv "${dataset_dir_local_full_path}" "${dataset_dir_local_full_path_with_preproc_tag}"

# dataset transfer
timer_start
cp -r "${dataset_dir_local_full_path_with_preproc_tag}" "${DATA_DIR_IN_REMOTE_HD}"
timer_end
preproc_dataset_copy_seconds="${delta_seconds}"
printf "${MESSAGE_TEMPLATE}" "preproc_dataset_copy_seconds: ${preproc_dataset_copy_seconds}"
printf "${MESSAGE_TEMPLATE}" "copy data from local hd to workspace, end"

printf "${MESSAGE_TEMPLATE}" "summary:"
printf "${MESSAGE_TEMPLATE}" "preproc dataset is located on ngc workspace"
printf "${MESSAGE_TEMPLATE}" "at: ${dataset_dir_remote_full_path_with_preproc_tag}"
printf "${MESSAGE_TEMPLATE}" "raw_dataset_copy_seconds: ${raw_dataset_copy_seconds}"
printf "${MESSAGE_TEMPLATE}" "preproc_delta_seconds: ${preproc_delta_seconds}"
printf "${MESSAGE_TEMPLATE}" "preproc_dataset_copy_seconds: ${preproc_dataset_copy_seconds}"
printf "${MESSAGE_TEMPLATE}" "end"
