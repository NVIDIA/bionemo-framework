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

# Build wheel files for PyG packages:
# torch geometric, torch cluster, torch sparse, torch scatter and torch spline conv
# and upload to gitlab package registry, to reduce the time cost when building container
# Usage: bash build_wheels.sh <base container> <path to save wheel files>

BASE_IMAGE=${1:-nvcr.io/nvidia/nemo:23.10}
WHEEL_FILE_PATH=${2:-$(pwd)/build-dir}

set -euo pipefail

packages=" \
 git+https://github.com/rusty1s/pytorch_cluster.git@1.6.3 \
 torch-sparse==0.6.18 \
 git+https://github.com/pyg-team/pytorch_geometric.git@2.5.0 \
 git+https://github.com/rusty1s/pytorch_scatter.git@2.1.2 \
 torch-spline-conv==1.2.2 \
"

CMD="\
  cd /build-dir;
  for package in ${packages}; do \
  echo Building \${package}...; \
  pip wheel --no-deps \${package}; \
  done
"

mkdir -p $WHEEL_FILE_PATH

docker run -v ${WHEEL_FILE_PATH}:/build-dir ${BASE_IMAGE} bash -c "${CMD}"

echo "All wheels ready in ${WHEEL_FILE_PATH} !"
echo "You can now publish them with:"
echo "TWINE_PASSWORD=<gitlab token> TWINE_USERNAME=<gitlab username> \
python -m twine upload --repository-url https://gitlab-master.nvidia.com/api/v4/projects/65301/packages/pypi \
<path to .whl file>"
