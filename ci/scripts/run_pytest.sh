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
set -ueo pipefail

# Function to display usage information
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --skip-docs    Skip running tests in the docs directory
    --no-nbval     Skip jupyter notebook validation tests

Note: Documentation tests (docs/) are only run when notebook validation
      is enabled (--no-nbval not set) and docs are not skipped
      (--skip-docs not set)
    -h, --help     Display this help message
EOF
    exit "${1:-0}"
}

# Function to handle cleanup on script exit
cleanup() {
    local exit_code=$?
    [ -n "${coverage_files[*]:-}" ] && rm -f "${coverage_files[@]:-}"
    exit "$exit_code"
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
error=false

# Parse command line arguments
while (( $# > 0 )); do
    case "$1" in
        --skip-docs) SKIP_DOCS=true ;;
        --no-nbval) NO_NBVAL=true ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
    shift
done

# Set up trap for cleanup
trap cleanup EXIT

# Source utility functions
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/utils.sh" || { echo "Failed to source utils.sh" >&2; exit 1; }

# Set up BioNeMo home directory
set_bionemo_home || exit 1

# Clear previous coverage data
python -m coverage erase

# Set up pytest options
PYTEST_OPTIONS=(
    -v
    --durations=0
    --durations-min=60.0
)
[[ "$NO_NBVAL" != true ]] && PYTEST_OPTIONS+=(--nbval-lax)

# Define test directories
TEST_DIRS=(./sub-packages/bionemo-*/)
if [[ "$NO_NBVAL" != true && "$SKIP_DOCS" != true ]]; then
    TEST_DIRS+=(docs/)
fi

echo "Test directories: ${TEST_DIRS[*]}"

# Run tests with coverage
for dir in "${TEST_DIRS[@]}"; do
    echo "Running pytest in $dir"
    coverage_file=".coverage.${dir//\//_}"
    coverage_files+=("$coverage_file")

    if ! python -m coverage run \
        --parallel-mode \
        --source=bionemo \
        --data-file="$coverage_file" \
        -m pytest "${PYTEST_OPTIONS[@]}" "$dir"; then
        error=true
    fi
done

# Combine and report coverage
python -m coverage combine
python -m coverage report --show-missing

# Exit with appropriate status
$error && exit 1
exit 0
