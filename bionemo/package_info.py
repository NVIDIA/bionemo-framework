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

MAJOR = 0
MINOR = 2
PATCH = 0
PRE_RELEASE = 'dev'

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'bionemo'
__contact_names__ = 'NVIDIA'
__homepage__ = 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/megamolbart'  # FIXME GA
__repository_url__ = 'https://github.com/NVIDIA/MegaMolBART'  # FIXME GA
__download_url__ = 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/megamolbart'  # FIXME GA
__description__ = 'BioNeMo, a library for large language models in drug discovery. Trainable with NeMo.'
__license__ = 'Apache2'
__keywords__ = (
    'drug discovery, cheminformatics, proeteomics, deep learning, machine learning, gpu, NeMo, nvidia, pytorch, torch'
)
