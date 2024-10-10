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


# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

import bionemo.model.protein.openfold.inductor as inductor
from bionemo.model.protein.openfold.linear import Linear


class BackboneUpdate(nn.Module):
    """Backbone Update module.

    Supplementary '1.8.3 Backbone update': Algorithm 23.

    Args:
        c_s: Single representation dimension (channels).

    """

    def __init__(self, c_s: int) -> None:
        super(BackboneUpdate, self).__init__()
        self.linear = Linear(c_s, 6, bias=True, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: [optim-hub] re-check condition: can we safely ignore dap / do we take into account hopper
        if inductor.is_enabled_on_hopper():  #  and dap.size() in {2, 8}:
            forward_fn = _forward_jit
        else:
            forward_fn = _forward_eager
        return forward_fn(s, self.linear.weight, self.linear.bias)


def _forward_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, w, b)


_forward_jit = torch.compile(_forward_eager)
