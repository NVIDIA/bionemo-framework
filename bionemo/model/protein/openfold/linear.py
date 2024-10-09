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

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm  # this function uses np.random internally


# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0.0, scale=1.0)
TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978


class Linear(nn.Linear):
    """Linear transformation with extra non-standard initializations.

    Supplementary '1.11.4 Parameters initialization': Linear layers.

    Args:
        in_features: Last dimension of the input tensor.
        out_features: Last dimension of the output tensor.
        bias: Whether to learn an additive bias. Default: `True`.
        init: Parameter initialization method.
            One of:
            - "default": LeCun (fan-in) with a truncated normal distribution
            - "relu": He initialization with a truncated normal distribution
            - "glorot": fan-average Glorot uniform initialization
            - "gating": Weights=0, Bias=1
            - "normal": Normal initialization with std=1/sqrt(fan_in)
            - "final": Weights=0, Bias=0

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        # By default, the biases of the Linear layers are filled with zeros.
        if bias:
            self.bias.data.fill_(0.0)

        if init == "default":
            lecun_normal_init_(self.weight.data)
        elif init == "relu":
            he_normal_init_(self.weight.data)
        elif init == "glorot":
            glorot_uniform_init_(self.weight.data)
        elif init == "gating":
            gating_init_(self.weight.data)
            if bias:
                self.bias.data.fill_(1.0)
        elif init == "normal":
            normal_init_(self.weight.data)
        elif init == "final":
            final_init_(self.weight.data)
        else:
            raise ValueError(f"unknown init {repr(init)}")


def lecun_normal_init_(weight_data: torch.Tensor) -> None:
    trunc_normal_init_(weight_data, scale=1.0)


def he_normal_init_(weight_data: torch.Tensor) -> None:
    trunc_normal_init_(weight_data, scale=2.0)


def glorot_uniform_init_(weight_data: torch.Tensor) -> None:
    nn.init.xavier_uniform_(weight_data, gain=1.0)


def final_init_(weight_data: torch.Tensor) -> None:
    weight_data.fill_(0.0)


def gating_init_(weight_data: torch.Tensor) -> None:
    weight_data.fill_(0.0)


def normal_init_(weight_data: torch.Tensor) -> None:
    nn.init.kaiming_normal_(weight_data, nonlinearity="linear")


def trunc_normal_init_(
    weight_data: torch.Tensor,
    scale: float = 1.0,
    fan: str = "fan_in",
) -> None:
    assert isinstance(weight_data, torch.Tensor)
    assert not isinstance(weight_data, nn.Parameter)
    weight_shape = weight_data.shape
    weight_numel = weight_data.numel()
    fan_value = _calculate_fan(weight_shape, fan)
    scale = scale / max(1, fan_value)
    stddev = math.sqrt(scale) / TRUNCATED_NORMAL_STDDEV_FACTOR
    values = truncnorm.rvs(a=-2, b=2, loc=0, scale=stddev, size=weight_numel)
    values = np.reshape(values, newshape=weight_shape)
    weight_data.copy_(torch.tensor(values, device=weight_data.device))


def _calculate_fan(linear_weight_shape: torch.Size, fan: str = "fan_in") -> int:
    fan_out, fan_in = linear_weight_shape
    if fan == "fan_in":
        fan_value = fan_in
    elif fan == "fan_out":
        fan_value = fan_out
    elif fan == "fan_avg":
        fan_value = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")
    return fan_value
