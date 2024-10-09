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


import torch
from nemo.utils import logging
from torch import Tensor

from bionemo.data.equidock.protein_utils import get_coords, get_rot_mat
from bionemo.model.protein.equidock.loss_metrics.intersection_loss import compute_body_intersection_loss


def remove_clashes_optimizer(
    x: Tensor,
    y: Tensor,
    iterations: int = 2000,
    min_loss: float = 0.5,
    lr: float = 0.001,
    fast_optimizer: bool = False,
    half_precision: bool = True,
):
    """
    Remove steric clashes by minimizing body intersection loss.

    Args:
        x (Tensor): ligand
        y (Tensor): protein
        iterations (int, optional): max number of iteration. Defaults to 2000.
        min_loss (float, optional): minimum loss value. Defaults to 0.5.
        lr (float): Learning rate
        fast_optimizer (bool): Use fast optimizer
        half_precision (bool): Use torch.float16 instead of torch.float32

    Returns:
        ligand position after steric clash removal
    """

    dtype = torch.half if half_precision else torch.float

    euler_angles_finetune = torch.nn.Parameter(torch.zeros(3, device="cuda", dtype=dtype))
    translation_finetune = torch.nn.Parameter(torch.zeros(3, device="cuda", dtype=dtype))

    optimizer = torch.optim.SGD([euler_angles_finetune, translation_finetune], lr=lr)

    if not fast_optimizer:
        for iter in range(iterations):
            optimizer.zero_grad()
            ligand_th = (get_rot_mat(euler_angles_finetune) @ x.T).T + translation_finetune
            non_int_loss = compute_body_intersection_loss(ligand_th, y, sigma=8, surface_ct=8)

            non_int_loss.backward()

            if iter % 100 == 0:
                logging.info(f"{iter:4} {non_int_loss.item():10.4f}")
            if non_int_loss.item() < min_loss:
                break

            optimizer.step()

            if non_int_loss.item() < min_loss + 1.5:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr / 10

            if iter > 1500:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * 10
            else:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

    else:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for _ in range(3):
                ligand_th = (get_rot_mat(euler_angles_finetune) @ x.T).T + translation_finetune
                non_int_loss = compute_body_intersection_loss(ligand_th, y, sigma=8, surface_ct=8)
                non_int_loss.backward()
                optimizer.step()

        if non_int_loss.item() < min_loss:
            return ligand_th

        torch.cuda.current_stream().wait_stream(s)

        # capture
        g = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            ligand_th = (get_rot_mat(euler_angles_finetune) @ x.T).T + translation_finetune
            non_int_loss = compute_body_intersection_loss(ligand_th, y, sigma=8, surface_ct=8)
            non_int_loss.backward()
            optimizer.step()

        for iter in range(4, iterations):
            g.replay()

            if iter % 100 == 0:
                logging.info(f"{iter:4} {non_int_loss.item():10.4f}")
            if non_int_loss.item() < min_loss:
                break

            if non_int_loss.item() < min_loss + 1.5:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr / 10

            if iter > 1500:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * 10
            else:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
    return ligand_th


def remove_clashes(
    unbound_ligand_new_pos,
    receptor_file_name: str,
    iterations: int = 2000,
    min_loss: float = 0.5,
    lr: float = 0.001,
    fast_optimizer: bool = False,
    half_precision: bool = True,
):
    """
    Remove steric clashes by minimizing body intersection loss.

    Args:
        unbound_ligand_new_pos (np.array): coordinates of unbound ligand
        receptor_file_name (str): file name for receptro
        iterations (int, optional): max number of iteration. Defaults to 2000.
        min_loss (float, optional): minimum loss value. Defaults to 0.5.
        lr (float): Learning rate
        fast_optimizer (bool): Use fast optimizer
        half_precision (bool): Use torch.float16 instead of torch.float32

    Returns:
        ligand position after steric clash removal
    """

    dtype = torch.half if half_precision else torch.float
    gt_receptor_nodes_coors = get_coords(receptor_file_name, True).to(device="cuda", dtype=dtype)
    initial_pos = torch.from_numpy(unbound_ligand_new_pos).to(device="cuda", dtype=dtype)
    ligand_th = remove_clashes_optimizer(
        initial_pos, gt_receptor_nodes_coors, iterations, min_loss, lr, fast_optimizer, half_precision
    )

    return ligand_th.cpu().detach().numpy()
