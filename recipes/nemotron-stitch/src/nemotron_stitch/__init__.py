# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The LLaVA recipe, generalized past vision.

Base package import must stay cheap: framework integrations (AutoModel, NeMo
RL, vLLM) live behind lazy imports in their subpackages so the base wheel
imports cleanly without those frameworks.
Only stdlib-only modules are re-exported here.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from nemotron_stitch import contracts, provenance

try:
    __version__ = version("nemotron-stitch")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__", "contracts", "provenance"]
