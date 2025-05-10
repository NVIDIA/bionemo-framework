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


from enum import Enum


class AugmentationType(Enum):
    """An enumeration representing the type ofOptimal Transport that can be used in Continuous Flow Matching.

    - **EXACT_OT**: Standard mini batch optimal transport defined in  https://arxiv.org/pdf/2302.00482.
    - **EQUIVARIANT_OT**: Adding roto/translation optimization to mini batch OT see https://arxiv.org/pdf/2306.15030  https://arxiv.org/pdf/2312.07168 4.2.
    - **KABSCH**: Simple Kabsch alignment between each data and noise point, No permuation # https://arxiv.org/pdf/2410.22388 Sec 3.2

    These prediction types can be used to train neural networks for specific tasks, such as denoising, image synthesis, or time-series forecasting.
    """

    EXACT_OT = "exact_ot"
    EQUIVARIANT_OT = "equivariant_ot"
    KABSCH = "kabsch"
