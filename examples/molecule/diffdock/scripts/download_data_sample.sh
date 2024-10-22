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

# Function to display help
display_help() {
    echo "Usage: $0 [-data_path <path>] [-pbss <value>] [-help]"
    echo "  -pbss <value>       If set, data will be download from PBSS. If unset, NGC by default."
    echo "  -help               Display this help message"
    exit 1
}

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
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


CMD="python download_artifacts.py --data diffdock_preprocessing_sample diffdock_training_sample diffdock_cugraph_sample --data_dir $BIONEMO_HOME --verbose"

if [ -n "$PBSS" ]; then
    CMD="${CMD} --source pbss"
fi

${CMD}
