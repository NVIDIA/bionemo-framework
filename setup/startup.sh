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

MODEL=$1

usage() {
cat <<EOF
Invalid command. Docker container only takes the following commands currently.
1. esm-1nv
2. megamolbart
3. prott5nv
4. Any command to be executed.
----------------------------------------
esm-1nv : launches the inference notebook service for ESM-1nv model
megamolbart : launches the inference notebook service for MegaMolBART model
prott5nv : launches the inference notebook service for ProtT5nv model

Else, one can provide a command to be executed inside the docker container.
That command is executed and the container exits.

EOF
}

EXAMPLE_BASE="/workspace/bionemo/examples"

# Kill all grpc service that may be lingering
# TODO: This is a WAR
ps -ef | grep 'python3 -m bionemo.model.*.grpc.service' | awk '{print $2}' | xargs kill -9 2>/dev/null

if [ "$MODEL" = "esm-1nv" ]; then
    python3 -m bionemo.model.protein.esm1nv.grpc.service & \
        jupyter lab \
            --no-browser \
            --port=8888 \
            --ip=0.0.0.0 \
            --allow-root \
            --notebook-dir=${EXAMPLE_BASE}/protein/esm1nv/nbs/ \
            --NotebookApp.password='' \
            --NotebookApp.token='' \
            --NotebookApp.password_required=False

elif [ "$MODEL" = "prott5nv" ]; then
    python3 -m bionemo.model.protein.prott5nv.grpc.service & \
        jupyter lab \
            --no-browser \
            --port=8888 \
            --ip=0.0.0.0 \
            --allow-root \
            --notebook-dir=${EXAMPLE_BASE}/protein/prott5nv/nbs/ \
            --NotebookApp.password='' \
            --NotebookApp.token='' \
            --NotebookApp.password_required=False

elif [ "$MODEL" = "megamolbart" ]; then
    python3 -m bionemo.model.molecule.megamolbart.grpc.service & \
        jupyter lab \
            --no-browser \
            --port=8888 \
            --ip=0.0.0.0 \
            --allow-root \
            --notebook-dir=${EXAMPLE_BASE}/molecule/megamolbart/nbs/ \
            --NotebookApp.password='' \
            --NotebookApp.token='' \
            --NotebookApp.password_required=False

# In the case user runs bash for an interactive terminal, add a tag
# to terminal prompt for ease of use.
elif [ "$MODEL" = "bash" ]; then
    bash --rcfile /docker_bashrc

elif [ "$#" -gt 0 ]; then
    ${MODEL}

else
    usage
fi
