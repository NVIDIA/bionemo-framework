#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# This unset is necessary because this script may be called by other scripts.
unset DATA_PATH
# Function to display help
display_help() {
    echo "Usage: $0 [-data_path <path>] [-pbss <value>] [-help]"
    echo "  -data_path <path>   Specify the data path, \$BIONEMO_HOME by default"
    echo "  -pbss <value>       If set, data will be download from PBSS. If unset, NGC by default."
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

cmd="python download_artifacts.py --data singlecell_sample --data_dir ${DATA_PATH} --verbose"

if [ -n "$PBSS" ]; then
  cmd="${cmd} --source pbss"
fi
$cmd
