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

# Function to check if Git repository is clean
check_git_repository() {
    if ! git diff-index --quiet HEAD --; then
        if [ $? -eq 128 ]; then
            echo "ERROR: Not in a git repository!" >&2
            return 1
        else
            echo "Warning: Repository is dirty! Commit all changes before building the image!" >&2
            return 0
        fi
    fi
}

set_bionemo_home() {
    set +u
    if [ -z "$BIONEMO_HOME" ]; then
        echo "\$BIONEMO_HOME is unset. Setting \$BIONEMO_HOME to repository root."

        # Ensure repository is clean
        if ! check_git_repository; then
            echo "Failed to set \$BIONEMO_HOME due to repository state." >&2
            return 1
        fi

        REPOSITORY_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
        if [ $? -ne 0 ]; then
            echo "ERROR: Could not determine the repository root. Ensure you're in a Git repository." >&2
            return 1
        fi

        BIONEMO_HOME="${REPOSITORY_ROOT}"
        echo "Setting \$BIONEMO_HOME to: $BIONEMO_HOME"
    fi
    set -u

    # Change directory to BIONEMO_HOME or exit if failed
    cd "${BIONEMO_HOME}" || { echo "ERROR: Could not change directory to \$BIONEMO_HOME: $BIONEMO_HOME" >&2; return 1; }
}


version_ge() {
        # Returns 0 (true) if $1 >= $2, 1 (false) otherwise
        [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
    }


verify_required_docker_version(){

    #required docker version
    required_docker_version="23.0.1"
    #required docker buidx version
    required_buildx_version="0.10.2"

    # Check Docker version
    docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')

    if ! version_ge "$docker_version" "$required_docker_version"; then
        echo "Error: Docker version $required_docker_version or higher is required. Current version: $docker_version"
        return 1
    fi

    # Check Buildx version
    buildx_version=$(docker buildx version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')

    if ! version_ge "$buildx_version" "$required_buildx_version"; then
        echo "Error: Docker Buildx version $required_buildx_version or higher is required. Current version: $buildx_version"
        return 1
    fi
}
