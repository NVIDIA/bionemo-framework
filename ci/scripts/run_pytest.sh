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


# Enable strict mode with better error handling
set -euox pipefail

# Function to display usage information
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --skip-docs    Skip running tests in the docs directory
    --no-nbval     Skip jupyter notebook validation tests
    --skip-slow    Skip tests marked as slow (@pytest.mark.slow)

Note: Documentation tests (docs/) are only run when notebook validation
      is enabled (--no-nbval not set) and docs are not skipped
      (--skip-docs not set)
    -h, --help     Display this help message
EOF
    exit "${1:-0}"
}

# Set default environment variables
: "${BIONEMO_DATA_SOURCE:=pbss}"
: "${PYTHONDONTWRITEBYTECODE:=1}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"

# Export necessary environment variables
export BIONEMO_DATA_SOURCE PYTHONDONTWRITEBYTECODE PYTORCH_CUDA_ALLOC_CONF

# Initialize variables
declare -a coverage_files
SKIP_DOCS=false
NO_NBVAL=false
SKIP_SLOW=false
error=false

# Parse command line arguments
while (( $# > 0 )); do
    case "$1" in
        --skip-docs) SKIP_DOCS=true ;;
        --no-nbval) NO_NBVAL=true ;;
        --skip-slow) SKIP_SLOW=true ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
    shift
done

# Source utility functions
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/utils.sh" || { echo "Failed to source utils.sh" >&2; exit 1; }

# Set up BioNeMo home directory
set_bionemo_home || exit 1

# Echo some useful information
lscpu
nvidia-smi
uname -a

# Set up pytest options
PYTEST_OPTIONS=(
    -v
    --cov=bionemo
    --cov-append
    --cov-report=xml:coverage.xml
)
[[ "$NO_NBVAL" != true ]] && PYTEST_OPTIONS+=(--nbval-lax)
[[ "$SKIP_SLOW" == true ]] && PYTEST_OPTIONS+=(-m "not slow")

# Define test directories
TEST_DIRS=(./sub-packages/bionemo-*/)
if [[ "$NO_NBVAL" != true && "$SKIP_DOCS" != true ]]; then
    TEST_DIRS+=(docs/)
fi

echo "Test directories: ${TEST_DIRS[*]}"

# Run tests with coverage
for dir in "${TEST_DIRS[@]}"; do
    echo "Running pytest in $dir"

    if ! pytest "${PYTEST_OPTIONS[@]}" --junitxml=$(basename $dir).junit.xml -o junit_family=legacy "$dir"; then
        error=true
    fi
done

# Exit with appropriate status
$error && exit 1
exit 0
