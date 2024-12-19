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

set -xueo pipefail
export PYTHONDONTWRITEBYTECODE=1
# NOTE: if a non-nvidia user wants to run the test suite, just run `export BIONEMO_DATA_SOURCE=ngc` prior to this call.
export BIONEMO_DATA_SOURCE="${BIONEMO_DATA_SOURCE:-pbss}"
# flexible GPU memory management, reducing the risk of fragmentation-related CUDA OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname "$0")/utils.sh"

if ! set_bionemo_home; then
    exit 1
fi

python -m coverage erase

error=false
for dir in docs/ ./sub-packages/bionemo-*/; do
    echo "Running pytest in $dir"
    python -m coverage run --parallel-mode --source=bionemo \
    -m pytest -v --nbval-lax --durations=0 --durations-min=60.0 "$dir" || error=true
done

python -m coverage combine
python -m coverage report --show-missing

if [ "$error" = true ]; then
    exit 1
fi
