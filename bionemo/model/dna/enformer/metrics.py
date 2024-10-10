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


from typing import Optional

import torch
from torchmetrics import Metric


class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable: Optional[bool] = False
    full_state_update: bool = False
    higher_is_better: Optional[bool] = True

    def __init__(self, n_channels: int, dist_sync_on_step=False, process_group=None):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.reduce_dims = (0, 1)
        self.n_channels = n_channels
        self.add_state(
            "product",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "true",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "true_squared",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_squared",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state("count", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = self.product - true_mean * self.pred - pred_mean * self.true + self.count * true_mean * pred_mean

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation
