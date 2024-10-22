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

# If there are C-like issues when running on cluster environments, it may be
# necessary to recompile the Megatron helpers, which can be done by running
# this script. This recompilation should also be done immediately before
# training starts on clusters as a best practice.

# Find NeMo installation location and re-combile Megatron helpers
NEMO_PATH=$(python -c 'import nemo; print(nemo.__path__[0])')
cd ${NEMO_PATH}/collections/nlp/data/language_modeling/megatron
make
