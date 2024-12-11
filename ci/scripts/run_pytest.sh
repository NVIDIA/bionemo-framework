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


# Parse arguments
CODECOV_DISABLED=0
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-codecov) CODECOV_DISABLED=1 ;;
        --help)
            echo "Usage: $0 [--no-codecov] [--help]"
            echo "  --no-codecov    Disables installing and uploading coverage results to codecov"
            echo "  --help          Display this help message"
            exit 0
            ;;
    esac
    shift
done

set -xueo pipefail


export PYTHONDONTWRITEBYTECODE=1
# NOTE: if a non-nvidia user wants to run the test suite, just run `export BIONEMO_DATA_SOURCE=ngc` prior to this call.
export BIONEMO_DATA_SOURCE="${BIONEMO_DATA_SOURCE:-pbss}"
source "$(dirname "$0")/utils.sh"

if ! set_bionemo_home; then
    exit 1
fi

# download Codecov CLI
if [[ $CODECOV_DISABLED -eq 0 ]]; then
    curl -Os https://cli.codecov.io/latest/linux/codecov

    # integrity check
    curl https://keybase.io/codecovsecurity/pgp_keys.asc | gpg --no-default-keyring --keyring trustedkeys.gpg --import # One-time step
    curl -Os https://cli.codecov.io/latest/linux/codecov
    curl -Os https://cli.codecov.io/latest/linux/codecov.SHA256SUM
    curl -Os https://cli.codecov.io/latest/linux/codecov.SHA256SUM.sig
    gpgv codecov.SHA256SUM.sig codecov.SHA256SUM

    shasum -a 256 -c codecov.SHA256SUM
    sudo chmod +x codecov
fi

# If pytests fail, we still want to ensure that the report is uploaded to codecov, so we set +e from here through the
# upload.
set +e
error=false
for dir in docs/ ./sub-packages/bionemo-*/; do
    echo "Running pytest in $dir"
    pytest -v --nbval-lax --cov=bionemo --cov-append --junitxml=$(basename $dir).junit.xml -o junit_family=legacy $dir || error=true
done

# Merge all sub-directory test results into a single xml file.
junitparser merge *.junit.xml combined.junit.xml || error=true

if [[ $CODECOV_DISABLED -eq 0 ]]; then

    # Upload test analytics to codecov.
    ./codecov do-upload --commit-sha $(git rev-parse HEAD) --report-type test_results --file combined.junit.xml || error=true

    # Upload coverage results to codecov.
    ./codecov upload-process --commit-sha $(git rev-parse HEAD) || error=true
fi
set -e

if [ "$error" = true ]; then
    exit 1
fi
