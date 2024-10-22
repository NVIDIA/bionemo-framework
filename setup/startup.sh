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

MODEL=$1

usage() {
cat <<EOF
Invalid command. Docker container only takes the following commands currently.
1. esm-1nv
2. esm-2nv
3. megamolbart
4. prott5nv
5. bash
6. Any command to be executed.
----------------------------------------
esm-1nv:     launches the inference notebook service for ESM-1nv model
esm-2nv:     launches the inference notebook service for ESM-2nv model
megamolbart: launches the inference notebook service for MegaMolBART model
prott5nv:    launches the inference notebook service for ProtT5nv model
bash:        starts an interactive bash shell

Otherwise, one can provide a command to be executed inside the docker container.
Note that this command is executed and then the container will immediately exit.

EOF
}

if [ -z "${BIONEMO_HOME}" ]; then
    echo "\$BIONEMO_HOME is unset. Please set the variable and run the script again. This variable should be set to the base of the repo path."
    exit 1
fi

EXAMPLE_BASE="${BIONEMO_HOME}/examples"

# Kill all pytriton services that may be lingering
ps -ef | grep -e "python[3]* -m bionemo.triton.*" -e "tritonserver" | grep -v "grep" | awk '{print $2}' | xargs kill -9 2&> /dev/null | true

# This function handles:
#   - starting the model's inference server
#   - setting up an exit trap to kill this server
#   - starts a jupyter lab notebook for the model
# Specify the examples location for the model (e.g. ${EXAMPLE_BASE}/...) as the first argument.
function run_server_kill_on_exit_start_notebook() {
    BASE_MODEL_PATH="${1}"
    CONFIG_PATH="${BASE_MODEL_PATH}/conf"
    NB_PATH="${BASE_MODEL_PATH}/nbs"

    if [[ ! -d "${CONFIG_PATH}" ]]; then
        echo "ERROR: Invalid base example config & notebook input: ${BASE_MODEL_PATH}"
        echo "ERROR: No configuration dir exists at: ${CONFIG_PATH}"
        exit 1 # EXIT program, don't just return!
    fi

    if [[ ! -f "${CONFIG_PATH}/infer.yaml" ]]; then
        echo "ERROR: Invalid base example config & notebook input: ${BASE_MODEL_PATH}"
        echo "ERROR: No configuration file exists at: ${CONFIG_PATH}/infer.yaml"
        exit 1 # EXIT program, don't just return!
    fi

    if [[ ! -d "${NB_PATH}" ]]; then
        echo "ERROR: Invalid base example config & notebook input: ${BASE_MODEL_PATH}"
        echo "ERROR: No notebook dir exists at: ${NB_PATH}"
        exit 1 # EXIT program, don't just return!
    fi

    echo "Starting PyTrition inference wrapper server for '${CONFIG_PATH}'"
    python -m bionemo.triton.inference_wrapper \
        --config-path "${CONFIG_PATH}" \
        --config-name infer.yaml &
    SERVER_PID="$!"
    echo "Server started as ${SERVER_PID}"
    # stop the inference server when this program exits
    trap "kill -9 ${SERVER_PID}" EXIT

    echo "Starting Jupyter notebook for '${NB_PATH}' on port 8888"
    jupyter lab \
        --no-browser \
        --port=8888 \
        --ip=0.0.0.0 \
        --allow-root \
        --notebook-dir="${NB_PATH}" \
        --NotebookApp.password='' \
        --NotebookApp.token='' \
        --NotebookApp.password_required=False
}

# Check the input and either start the inference server & notebook for the model,
# start an interactive bash shell, or execute an arbitrary command & then exit.
# If no input is specified, then the help text is displayed & exited with code 1.
if [[ "$MODEL" == "esm-1nv" ]]; then

    run_server_kill_on_exit_start_notebook "${EXAMPLE_BASE}/protein/esm1nv/"

elif [[ "$MODEL" == "esm-2nv" ]]; then

    run_server_kill_on_exit_start_notebook "${EXAMPLE_BASE}/protein/esm2nv/"

elif [[ "$MODEL" == "prott5nv" ]]; then

    run_server_kill_on_exit_start_notebook "${EXAMPLE_BASE}/protein/prott5nv/"

elif [[ "$MODEL" == "megamolbart" ]]; then

    run_server_kill_on_exit_start_notebook "${EXAMPLE_BASE}/molecule/megamolbart"

elif [[ "$MODEL" == "molmim" ]]; then

    run_server_kill_on_exit_start_notebook "${EXAMPLE_BASE}/molecule/molmim"

elif [[ "$MODEL" == "bash" ]]; then
    # In the case user runs bash for an interactive terminal, add a tag
    # to terminal prompt for ease of use.
    bash --rcfile /docker_bashrc

elif [[ $# -gt 0 ]]; then
    ${MODEL}

else
    usage
    exit 1
fi
