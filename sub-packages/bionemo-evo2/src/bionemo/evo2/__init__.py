# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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
from dataclasses import dataclass
from typing import Callable

import torch.nn.functional as F
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS, HyenaNV1bConfig


@dataclass
class HyenaNV1bConfig2(HyenaNV1bConfig):
    """A parallel friendly version of the HyenaNV1bConfig."""

    hidden_size: int = 2048
    num_groups_hyena: int = 2048
    num_attention_heads: int = 16
    ffn_hidden_size: int = 6144
    # Spike-no-more-embedding init by default.
    share_embeddings_and_output_weights: bool = False
    embedding_init_method_std: float = 1.0
    # activation_func_clamp_value: Optional[float] = 7.0
    # glu_linear_offset: float = 1.0


# TODO move this to a more permanent location.
HYENA_MODEL_OPTIONS["striped_hyena_1b_nv_parallel"] = HyenaNV1bConfig2
