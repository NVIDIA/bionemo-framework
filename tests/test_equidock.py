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
import pytest
import torch

from bionemo.model.protein.equidock.loss_metrics.eval import rigid_transform_Kabsch_3D, rigid_transform_Kabsch_3D_torch
from bionemo.model.protein.equidock.loss_metrics.intersection_loss import compute_body_intersection_loss
from bionemo.model.protein.equidock.model.utils import compute_cross_attention
from bionemo.model.protein.equidock.utils.remove_clashes import remove_clashes_optimizer
from bionemo.utils.tests import Deterministic


@pytest.mark.needs_gpu
@pytest.mark.parametrize("data_type", [torch.float32, torch.float64])
def test_Kabsch_algorithm(data_type):
    with Deterministic():
        for _ in range(10):
            # Random points
            A = np.random.randn(3, 10)
            B = np.random.randn(3, 10)

            # Get transformations from both functions
            R_numpy, t_numpy = rigid_transform_Kabsch_3D(A, B)
            R_torch, t_torch = rigid_transform_Kabsch_3D_torch(
                torch.from_numpy(A).to("cuda").to(data_type), torch.from_numpy(B).to("cuda").to(data_type)
            )

            # Convert the torch tensors to numpy arrays for easy comparison
            R_torch = R_torch.detach().cpu().numpy()
            t_torch = t_torch.detach().cpu().numpy()

            # Assert if matrices and vectors are close
            assert np.allclose(
                R_numpy, R_torch, atol=1e-4 if data_type == torch.float32 else 1e-6
            ), f"Rotation matrices differ: \n{R_numpy}\n{R_torch}"
            assert np.allclose(
                t_numpy, t_torch, atol=1e-4 if data_type == torch.float32 else 1e-6
            ), f"Translation vectors differ: \n{t_numpy}\n{t_torch}"


def test_body_intersection_loss():
    with Deterministic():
        model_ligand_coors_deform = torch.zeros(1, 3)
        bound_receptor_repres_nodes_loc_array = 5.0 * torch.ones(1, 3)
        sigmas = torch.linspace(1.0, 10.0, 10)
        old_res = compute_body_intersection_loss(
            model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigmas[0], 1.0
        )

        for sigma in sigmas[1:]:
            new_res = compute_body_intersection_loss(
                model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma, 1.0
            )
            if new_res < old_res:
                assert False, "Body intersection loss should be increasing with sigma increase!"
            old_res = new_res


@pytest.mark.parametrize("cross_msgs", [True, False])
def test_cross_attention(cross_msgs: bool):
    N = 21
    M = 11
    D = 16
    d = 4
    with Deterministic():
        q = np.random.randn(N, D)
        k = np.random.randn(M, D)
        v = np.random.randn(M, d)
        mask = np.random.randint(0, 2, size=(N, M))

        a = mask * (q @ k.transpose([1, 0])) - 1000.0 * (1.0 - mask)

        a_x = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)
        if cross_msgs:
            expected = a_x @ v
        else:
            expected = (
                q * 0
            )  # TODO: should it be v * 0? The paper has q*0, but maybe that's just because q.shape[1:] == v.shape[1:]?

        q = torch.from_numpy(q)
        k = torch.from_numpy(k)
        v = torch.from_numpy(v)
        mask = torch.from_numpy(mask).bool()

        result = compute_cross_attention(q, k, v, mask, cross_msgs).detach().cpu().numpy()

        assert np.allclose(expected, result, atol=1e-4), f"Cross message attention fails: \n{expected}\n{result}"


@pytest.mark.parametrize("fast_optimizer", [False, True])
@pytest.mark.parametrize("half_precision", [False, True])
def test_remove_clashes(fast_optimizer, half_precision):
    with Deterministic():
        dtype = torch.half if half_precision else torch.float32
        x = torch.rand(100, 3).cuda().to(dtype)
        y = torch.rand(50, 3).cuda().to(dtype)
        x_new = remove_clashes_optimizer(
            x, y, iterations=2000, min_loss=0.5, fast_optimizer=fast_optimizer, half_precision=half_precision
        )
        assert (
            compute_body_intersection_loss(x_new, y, 8.0, 8.0).item()
            < compute_body_intersection_loss(x, y, 8.0, 8.0).item()
        )
        x = torch.rand(100, 3).cuda().to(dtype)
        y = x.detach() + torch.randn(1, 3, device="cuda", dtype=dtype)

        x_new = remove_clashes_optimizer(
            x, y, iterations=2000, min_loss=0.5, fast_optimizer=fast_optimizer, half_precision=half_precision
        )
        # compute_body_intersection_loss is minimized to value below min_loss in remove_clashes_optimizer
        # so, we expect its value reaches below min_loss
        assert compute_body_intersection_loss(x_new, y, 8.0, 8.0).item() < 0.5
