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

import numpy as np
import ot
import torch


def compute_sq_dist_mat(X_1, X_2):
    """Computes the l2 squared cost matrix between two point cloud inputs.
    Args:
        X_1: [n, #features] point cloud, tensor
        X_2: [m, #features] point cloud, tensor
    Output:
        [n, m] matrix of the l2 distance between point pairs
    """
    n_1, _ = X_1.size()
    n_2, _ = X_2.size()
    X_1 = X_1.view(n_1, 1, -1)
    X_2 = X_2.view(1, n_2, -1)
    squared_dist = (X_1 - X_2) ** 2
    cost_mat = torch.sum(squared_dist, dim=2)
    return cost_mat


def compute_ot_emd(cost_mat, device):
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    a = np.ones([cost_mat.shape[0]])
    b = np.ones([cost_mat.shape[1]])
    a = a / a.sum()
    b = b / b.sum()
    ot_mat = ot.emd(a=a, b=b, M=cost_mat_detach, numItermax=10000)
    ot_mat_attached = torch.tensor(ot_mat, device=device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)
    return ot_dist, ot_mat_attached
