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

from typing import Dict, List, Optional, Set

import torch

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.model.protein.openfold.loss import lddt_ca
from bionemo.model.protein.openfold.superimposition import superimpose


VALIDATION_METRICS_NAMES = {"lddt_ca", "drmsd_ca", "alignment_rmsd", "gdt_ts", "gdt_ha"}


def compute_validation_metrics(
    predicted_atom_positions: torch.Tensor,
    target_atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
    metrics_names: Set[str],
) -> Dict[str, torch.Tensor]:
    """Computes a set of metrics that compare predicted and ground truth atom positions

    Args:
        predicted_atom_positions (torch.Tensor): tensor of predicted atom positions
        target_atom_positions (torch.Tensor): ground truth for atom positions
        atom_mask (torch.Tensor): mask indicating on which atoms metrics should be computed over.
        Only atoms listed in bionemo.data.protein.openfold.residue_constants.ATOM_ORDER and
        selenium in selenomethionine are considered.
        metrics_names (Set[str]): Set of metrics to calculate. Available are:
        "lddt_ca", "drmsd_ca", "alignment_rmsd", "gdt_ts", "gdt_ha".

    Raises:
        ValueError: if requested metric is not available.

    Returns:
        Dict[str, torch.Tensor]: dictonary of metrics names their values.
    """

    val_metrics = {}

    assert isinstance(metrics_names, set)
    if len(metrics_names) == 0:
        raise ValueError(
            "Validation `metrics_names` set is empty." f" VALIDATION_METRICS_NAMES={VALIDATION_METRICS_NAMES}"
        )
    assert metrics_names.issubset(VALIDATION_METRICS_NAMES)

    pred_coords = predicted_atom_positions
    gt_coords = target_atom_positions
    all_atom_mask = atom_mask

    if "lddt_ca" in metrics_names:
        val_metrics["lddt_ca"] = (
            lddt_ca(
                all_atom_pred_pos=pred_coords,
                all_atom_positions=gt_coords,
                all_atom_mask=all_atom_mask,
                eps=1e-8,
                per_residue=False,
            )
            * 100
        )
        if metrics_names == {"lddt_ca"}:
            return val_metrics

    gt_coords_masked = gt_coords * all_atom_mask[..., None]
    pred_coords_masked = pred_coords * all_atom_mask[..., None]
    ca_pos = rc.ATOM_ORDER["CA"]
    gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
    pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
    all_atom_mask_ca = all_atom_mask[..., ca_pos]

    if "drmsd_ca" in metrics_names:
        val_metrics["drmsd_ca"] = drmsd(
            structure_1=pred_coords_masked_ca,
            structure_2=gt_coords_masked_ca,
            mask=all_atom_mask_ca,  # still required here to compute n
        )

    superimposition_metric_names = {
        "alignment_rmsd",
        "gdt_ts",
        "gdt_ha",
    } & metrics_names

    if superimposition_metric_names:
        superimposed_pred, alignment_rmsd = superimpose(
            reference=gt_coords_masked_ca,
            coords=pred_coords_masked_ca,
            mask=all_atom_mask_ca,
        )

        if "alignment_rmsd" in metrics_names:
            val_metrics["alignment_rmsd"] = alignment_rmsd

        if "gdt_ts" in metrics_names:
            val_metrics["gdt_ts"] = gdt_ts(
                p1=superimposed_pred,
                p2=gt_coords_masked_ca,
                mask=all_atom_mask_ca,
            )

        if "gdt_ha" in metrics_names:
            val_metrics["gdt_ha"] = gdt_ha(
                p1=superimposed_pred,
                p2=gt_coords_masked_ca,
                mask=all_atom_mask_ca,
            )

    return val_metrics


def drmsd(
    structure_1: torch.Tensor,
    structure_2: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """distance‐based root mean square deviation (dRMSD) metric"""

    def prep_d(structure):
        d = structure[..., :, None, :] - structure[..., None, :, :]
        d = d**2
        d = torch.sqrt(torch.sum(d, dim=-1))
        return d

    d1 = prep_d(structure_1)
    d2 = prep_d(structure_2)

    drmsd = d1 - d2
    drmsd = drmsd**2
    if mask is not None:
        drmsd = drmsd * (mask[..., None] * mask[..., None, :])
    drmsd = torch.sum(drmsd, dim=(-1, -2))
    n = d1.shape[-1] if mask is None else torch.sum(mask, dim=-1)
    drmsd = drmsd * (1 / (n * (n - 1))) if n > 1 else (drmsd * 0.0)
    drmsd = torch.sqrt(drmsd)

    return drmsd


def gdt(
    p1: torch.Tensor,
    p2: torch.Tensor,
    mask: torch.Tensor,
    cutoffs: List[float],
) -> torch.Tensor:
    p1 = p1.float()
    p2 = p2.float()
    n = torch.sum(mask, dim=-1)
    distances = torch.sqrt(torch.sum((p1 - p2) ** 2, dim=-1))
    scores = []
    for c in cutoffs:
        score = torch.sum((distances <= c) * mask, dim=-1) / n
        score = torch.mean(score)
        scores.append(score)
    return sum(scores) / len(scores)


def gdt_ts(
    p1: torch.Tensor,
    p2: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """A Global Distance Test with fixed `cutoffs` values of 1, 2, 4, and 8 angstroms"""
    return gdt(p1, p2, mask, cutoffs=[1.0, 2.0, 4.0, 8.0])


def gdt_ha(
    p1: torch.Tensor,
    p2: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """A Global Distance Test with fixed `cutoffs` values of 0.5, 1, 2, and 4 angstroms."""
    return gdt(p1, p2, mask, cutoffs=[0.5, 1.0, 2.0, 4.0])
