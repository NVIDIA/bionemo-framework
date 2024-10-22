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


def G_fn(protein_coords, x, sigma):
    r"""
    Compute the surface of a protein point
    Ligand residue locations: a_i in R^3. Receptor: b_j in R^3
    Ligand: G_l(x) = -sigma * ln( \sum_i  exp(- ||x - a_i||^2 / sigma)  ), same for G_r(x)
    Ligand surface: x such that G_l(x) = surface_ct
    Other properties: G_l(a_i) < 0, G_l(x) = infinity if x is far from all a_i

    Args:
        protein_coords (torch.Tensor): Coordinates of protein
        x (torch.Tensor): Coordiante of the second protein
        sigma (torch.Tensor or float): Gaussian variance

    Returns:
        (torch.Tensor):
    """
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(-torch.sum((protein_coords.view(1, -1, 3) - x.view(-1, 1, 3)) ** 2, dim=2) / float(sigma))  # (m, n)
    return -sigma * torch.log(1e-3 + e.sum(dim=1))
    # replace with torch.logsumexp


def compute_body_intersection_loss(
    model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma, surface_ct
):
    """
    Intersection of ligand and receptor: points x such that G_l(x) < surface_ct && G_r(x) < surface_ct
    Intersection loss: IL = \avg_i max(0, surface_ct - G_r(a_i)) + \avg_j max(0, surface_ct - G_l(b_j))


    Args:
        model_ligand_coors_deform  (torch.Tensor): Coordinates of deformed ligand
        bound_receptor_repres_nodes_loc_array (torch.Tensor): Coordinates of bounded receptor
        sigma (float): Gaussian variance
        surface_ct (float): _description_

    Returns:
        torch.Tensor: Intersection loss
    """
    loss = torch.mean(
        torch.clamp(surface_ct - G_fn(bound_receptor_repres_nodes_loc_array, model_ligand_coors_deform, sigma), min=0)
    ) + torch.mean(
        torch.clamp(surface_ct - G_fn(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma), min=0)
    )
    return loss
