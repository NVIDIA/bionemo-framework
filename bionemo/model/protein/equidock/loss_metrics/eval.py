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


#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List

import numpy as np
import torch
from torchmetrics import Metric

from bionemo.data.equidock.protein_utils import rigid_transform_Kabsch_3D, rigid_transform_Kabsch_3D_torch


__all__ = ["Meter_Unbound_Bound", "Torchmetrics_Unbound_Bound"]


def metrics_statistics(metrics_lst: List[Metric], dataset_type: str = "val"):
    if len(metrics_lst) != 3:
        raise ValueError(f"len of metrics should be 3, but it is {len(metrics_lst)}!")

    complex_rmsd_tensors, ligand_rmsd_tensors, receptor_rmsd_tensors = (metrics_lst[i].compute() for i in range(3))
    complex_rmsd_tensors, ligand_rmsd_tensors, receptor_rmsd_tensors = (
        complex_rmsd_tensors.view(-1),
        ligand_rmsd_tensors.view(-1),
        receptor_rmsd_tensors.view(-1),
    )

    complex_rmsd_mean, ligand_rmsd_mean, receptor_rmsd_mean = (
        complex_rmsd_tensors.mean(),
        ligand_rmsd_tensors.mean(),
        receptor_rmsd_tensors.mean(),
    )
    complex_rmsd_median, ligand_rmsd_median, receptor_rmsd_median = (
        complex_rmsd_tensors.median(),
        ligand_rmsd_tensors.median(),
        receptor_rmsd_tensors.median(),
    )
    complex_rmsd_std, ligand_rmsd_std, receptor_rmsd_std = (
        complex_rmsd_tensors.std(),
        ligand_rmsd_tensors.std(),
        receptor_rmsd_tensors.std(),
    )

    rmsd_log = {
        dataset_type + "_ligand_rmsd_mean": ligand_rmsd_mean.cpu().detach(),
        dataset_type + "_receptor_rmsd_mean": receptor_rmsd_mean.cpu().detach(),
        dataset_type + "_complex_rmsd_mean": complex_rmsd_mean.cpu().detach(),
        dataset_type + "_complex_rmsd_median": complex_rmsd_median.cpu().detach(),
        dataset_type + "_ligand_rmsd_median": ligand_rmsd_median.cpu().detach(),
        dataset_type + "_receptor_rmsd_median": receptor_rmsd_median.cpu().detach(),
        dataset_type + "_complex_rmsd_std": complex_rmsd_std.cpu().detach(),
        dataset_type + "_ligand_rmsd_std": ligand_rmsd_std.cpu().detach(),
        dataset_type + "_receptor_rmsd_std": receptor_rmsd_std.cpu().detach(),
        dataset_type + "_shape": complex_rmsd_tensors.shape[0],
    }

    for i in range(3):
        metrics_lst[i].reset()

    return rmsd_log


def rmsd_compute(ligand_coors_pred, receptor_coors_pred, ligand_coors_true, receptor_coors_true):
    """
    Computes rmsd of complex, ligand, and receptor

    Args:
        ligand_coors_pred (Tensor): model prediction of coordiantes of ligand
        receptor_coors_pred (Tensor): model prediction of coordinates of receptor
        ligand_coors_true (Tensor): ground truth coordinates of liganad
        receptor_coors_true (Tensor): ground truth coordinates of receptor

    Returns:
        Tuple(Tensor): Computed rmsd for complex, ligand, and receptor
    """

    ligand_coors_pred = ligand_coors_pred.detach()
    receptor_coors_pred = receptor_coors_pred.detach()

    ligand_coors_true = ligand_coors_true.detach()
    receptor_coors_true = receptor_coors_true.detach()

    ligand_rmsd = torch.sqrt(((ligand_coors_pred - ligand_coors_true) ** 2).sum(dim=1).mean())
    receptor_rmsd = torch.sqrt(((receptor_coors_pred - receptor_coors_true) ** 2).sum(dim=1).mean())

    complex_coors_pred = torch.cat([ligand_coors_pred, receptor_coors_pred], 0)
    complex_coors_true = torch.cat([ligand_coors_true, receptor_coors_true], 0)

    R, b = rigid_transform_Kabsch_3D_torch(complex_coors_pred.T, complex_coors_true.T)
    complex_coors_pred_aligned = ((R @ complex_coors_pred.T) + b).T

    complex_rmsd = torch.sqrt(((complex_coors_pred_aligned - complex_coors_true) ** 2).sum(dim=1).mean())

    return complex_rmsd, ligand_rmsd, receptor_rmsd


class Torchmetrics_Unbound_Bound(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("complex_rmsd_list", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ligand_rmsd_list", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("receptor_rmsd_list", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, complex_rmsd, ligand_rmsd, receptor_rmsd):
        self.complex_rmsd_list += torch.tensor(complex_rmsd)
        self.ligand_rmsd_list += torch.tensor(ligand_rmsd)
        self.receptor_rmsd_list += torch.tensor(receptor_rmsd)

        self.total += 1

    def compute(self):
        """
        Returns mean of complex, ligand, & receptor rmsd

        Returns:
            _type_: _description_
        """
        sum_complex_rmsd_list = self.complex_rmsd_list.float()
        sum_ligand_rmsd_list = self.ligand_rmsd_list.float()
        sum_receptor_rmsd_list = self.receptor_rmsd_list.float()

        return (
            sum_complex_rmsd_list / self.total,
            sum_ligand_rmsd_list / self.total,
            sum_receptor_rmsd_list / self.total,
        )


class Meter_Unbound_Bound:
    def __init__(self):
        self.complex_rmsd_list = []
        self.ligand_rmsd_list = []
        self.receptor_rmsd_list = []

    def reset(self):
        self.complex_rmsd_list = []
        self.ligand_rmsd_list = []
        self.receptor_rmsd_list = []

    def update_rmsd(self, ligand_coors_pred, receptor_coors_pred, ligand_coors_true, receptor_coors_true):
        ligand_coors_pred = ligand_coors_pred.detach().cpu().numpy()
        receptor_coors_pred = receptor_coors_pred.detach().cpu().numpy()

        ligand_coors_true = ligand_coors_true.detach().cpu().numpy()
        receptor_coors_true = receptor_coors_true.detach().cpu().numpy()

        ligand_rmsd = np.sqrt(np.mean(np.sum((ligand_coors_pred - ligand_coors_true) ** 2, axis=1)))
        receptor_rmsd = np.sqrt(np.mean(np.sum((receptor_coors_pred - receptor_coors_true) ** 2, axis=1)))

        complex_coors_pred = np.concatenate((ligand_coors_pred, receptor_coors_pred), axis=0)
        complex_coors_true = np.concatenate((ligand_coors_true, receptor_coors_true), axis=0)

        R, b = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
        complex_coors_pred_aligned = ((R @ complex_coors_pred.T) + b).T

        complex_rmsd = np.sqrt(np.mean(np.sum((complex_coors_pred_aligned - complex_coors_true) ** 2, axis=1)))

        self.complex_rmsd_list.append(complex_rmsd)
        self.ligand_rmsd_list.append(ligand_rmsd)
        self.receptor_rmsd_list.append(receptor_rmsd)

        return complex_rmsd

    def summarize(self, reduction_rmsd="median"):
        if reduction_rmsd == "mean":
            complex_rmsd_array = np.array(self.complex_rmsd_list)
            complex_rmsd_summarized = np.mean(complex_rmsd_array)

            ligand_rmsd_array = np.array(self.ligand_rmsd_list)
            ligand_rmsd_summarized = np.mean(ligand_rmsd_array)

            receptor_rmsd_array = np.array(self.receptor_rmsd_list)
            receptor_rmsd_summarized = np.mean(receptor_rmsd_array)
        elif reduction_rmsd == "median":
            complex_rmsd_array = np.array(self.complex_rmsd_list)
            complex_rmsd_summarized = np.median(complex_rmsd_array)

            ligand_rmsd_array = np.array(self.ligand_rmsd_list)
            ligand_rmsd_summarized = np.median(ligand_rmsd_array)

            receptor_rmsd_array = np.array(self.receptor_rmsd_list)
            receptor_rmsd_summarized = np.median(receptor_rmsd_array)
        else:
            raise ValueError("Meter_Unbound_Bound: reduction_rmsd mis specified!")
        return ligand_rmsd_summarized, receptor_rmsd_summarized, complex_rmsd_summarized

    def summarize_with_std(self, reduction_rmsd="median"):
        complex_rmsd_array = np.array(self.complex_rmsd_list)
        if reduction_rmsd == "mean":
            complex_rmsd_summarized = np.mean(complex_rmsd_array)
        elif reduction_rmsd == "median":
            complex_rmsd_summarized = np.median(complex_rmsd_array)
        else:
            raise ValueError("Meter_Unbound_Bound: reduction_rmsd mis specified!")
        return complex_rmsd_summarized, np.std(complex_rmsd_array)
