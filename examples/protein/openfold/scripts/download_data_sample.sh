#!/bin/bash
#
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

# This unset is necessary because this script may be called by other scripts.
unset DATA_PATH
# Function to display help
display_help() {
    echo "Usage: $0 [-data_path <path>] [-pbss <value>] [-help]"
    echo "  -data_path <path>   Specify the data path, \$BIONEMO_HOME by default"
    echo "  -pbss <value>       If set, data will be download from PBSS. If unset, public sources by default."
    echo "  -help               Display this help message"
    exit 1
}

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        -pbss)
            PBSS=true
            shift
            ;;
        -help)
            display_help
            ;;
        *)
            echo "Unknown parameter: $1"
            display_help
            ;;
    esac
done

if [ -n "$DATA_PATH" ]; then
    echo "Data path specified: $DATA_PATH"
elif [ -n "$BIONEMO_HOME" ]; then
    echo "Data path defaulting to \$BIONEMO_HOME repo base: $BIONEMO_HOME"
    DATA_PATH=$BIONEMO_HOME
else
    echo "\$BIONEMO_HOME is unset and -data_path was not provided. Exiting."
    exit 1
fi

cmd="python download_artifacts.py --data openfold_sample --data_dir ${DATA_PATH} --verbose"

if [ -n "$PBSS" ]; then
  cmd="${cmd} --source pbss"
fi
$cmd

echo "Downloading sample inference data from public sources to $DATA_PATH (estimate download time < 5 mins)"

# Download cif from RCSB
PDB_DIR=${DATA_PATH}/protein/openfold/inference/pdb
mkdir -p $PDB_DIR
for pdb_code in "7b4q" "7dnu"; do
    wget -O ${PDB_DIR}/${pdb_code}.cif "https://files.rcsb.org/download/${pdb_code}.cif"
done

# Download msa from OpenProteinSet
MSA_DIR=${DATA_PATH}/protein/openfold/inference/msas
mkdir -p $MSA_DIR
for pdb_chain_code in "7b4q_A" "7dnu_A"; do
    aws s3 sync --no-sign-request s3://openfold/pdb/${pdb_chain_code}/a3m ${MSA_DIR}/${pdb_chain_code}
done

fi
