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

set -xueo pipefail
source "$(dirname "$0")/utils.sh"

display_help() {
    echo "Usage: $0 [-pbss <value>] [-help]"
    echo "  -pbss <value>       If set, data will be download from PBSS. If unset, NGC by default."
    echo "  -help               Display this help message"
    exit 1
}

PBSS=false
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

if ! set_bionemo_home; then
    exit 1
fi

examples/protein/openfold/scripts/install_third_party.sh

MODEL_PATH=${BIONEMO_HOME}/models
MODELS="openfold_finetuning_inhouse esm2nv_3b esm2nv_8m_lora esm2nv_8m_untrained esm2nv_650m diffdock_confidence diffdock_score equidock_db5 equidock_dips megamolbart molmim_70m_24_3 prott5nv esm1nv dnabert geneformer_10M_240530 dsmbind esm2_650m_huggingface esm2_3b_huggingface"
CMD="python download_artifacts.py --models ${MODELS} --model_dir ${MODEL_PATH} --data all --data_dir ${BIONEMO_HOME} --verbose"

if [ -n "$PBSS" ]; then
  CMD="${CMD} --source pbss"
fi
$CMD

python examples/singlecell/geneformer/scripts/get_pt_neighbors_data.py
