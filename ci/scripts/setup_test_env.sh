#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
