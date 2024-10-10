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

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.utils import logging

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.model.protein.openfold.utils.numpy_utils import map_array_tree
from bionemo.model.protein.openfold.utils.rigid_utils import Rigid, Rotation
from bionemo.model.protein.openfold.utils.torch_utils import map_tensor_tree


# fmt: off
class AlphaFoldLoss(nn.Module):
    """AlphaFold loss module.

    Supplementary '1.9 Loss functions and auxiliary heads'.

    """

    def __init__(self, config) -> None:
        super(AlphaFoldLoss, self).__init__()
        self.fape_loss_config = config.fape_loss_config
        self.supervised_chi_loss_config = config.supervised_chi_loss_config
        self.distogram_loss_config = config.distogram_loss_config
        self.masked_msa_loss_config = config.masked_msa_loss_config
        self.plddt_loss_config = config.plddt_loss_config
        self.experimentally_resolved_loss_config = config.experimentally_resolved_loss_config
        self.violation_loss_config = config.violation_loss_config
        self.tm_loss_config = config.tm_loss_config

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """AlphaFold loss forward pass.

        Args:
            outputs: forward pass output dict
            batch: train batch dict

        Returns:
            scaled_weighted_total_loss: total loss connected to the graph
            losses: dict with loss breakdown detached from the graph

        """
        batch_size = batch["aatype"].size(0)

        if "violations" not in outputs.keys():
            outputs["violations"] = find_structural_violations(
                batch=batch,
                atom14_pred_positions=outputs["sm_positions"][:, -1],
                violation_tolerance_factor=self.violation_loss_config.violation_tolerance_factor,
                clash_overlap_tolerance=self.violation_loss_config.clash_overlap_tolerance,
            )

        if "renamed_atom14_gt_positions" not in outputs.keys():
            batch.update(
                compute_renamed_ground_truth(
                    batch=batch,
                    atom14_pred_positions=outputs["sm_positions"][:, -1],
                )
            )

        losses = {}

        losses["fape"] = fape_loss(
            outputs=outputs,
            batch=batch,
            backbone_clamp_distance=self.fape_loss_config.backbone_clamp_distance,
            backbone_loss_unit_distance=self.fape_loss_config.backbone_loss_unit_distance,
            backbone_weight=self.fape_loss_config.backbone_weight,
            sidechain_clamp_distance=self.fape_loss_config.sidechain_clamp_distance,
            sidechain_length_scale=self.fape_loss_config.sidechain_length_scale,
            sidechain_weight=self.fape_loss_config.sidechain_weight,
            eps=self.fape_loss_config.eps,
        )

        losses["supervised_chi"] = supervised_chi_loss(
            angles_sin_cos=outputs["sm_angles"],
            unnormalized_angles_sin_cos=outputs["sm_unnormalized_angles"],
            aatype=batch["aatype"],
            seq_mask=batch["seq_mask"],
            chi_mask=batch["chi_mask"],
            chi_angles_sin_cos=batch["chi_angles_sin_cos"],
            chi_weight=self.supervised_chi_loss_config.chi_weight,
            angle_norm_weight=self.supervised_chi_loss_config.angle_norm_weight,
            eps=self.supervised_chi_loss_config.eps,
        )

        losses["distogram"] = distogram_loss(
            logits=outputs["distogram_logits"],
            pseudo_beta=batch["pseudo_beta"],
            pseudo_beta_mask=batch["pseudo_beta_mask"],
            min_bin=self.distogram_loss_config.min_bin,
            max_bin=self.distogram_loss_config.max_bin,
            num_bins=self.distogram_loss_config.num_bins,
            eps=self.distogram_loss_config.eps,
        )

        losses["masked_msa"] = masked_msa_loss(
            logits=outputs["masked_msa_logits"],
            true_msa=batch["true_msa"],
            bert_mask=batch["bert_mask"],
            eps=self.masked_msa_loss_config.eps,
        )

        losses["plddt_loss"] = lddt_loss(
            logits=outputs["lddt_logits"],
            all_atom_pred_pos=outputs["final_atom_positions"],
            all_atom_positions=batch["all_atom_positions"],
            all_atom_mask=batch["all_atom_mask"],
            resolution=batch["resolution"],
            cutoff=self.plddt_loss_config.cutoff,
            num_bins=self.plddt_loss_config.num_bins,
            min_resolution=self.plddt_loss_config.min_resolution,
            max_resolution=self.plddt_loss_config.max_resolution,
            eps=self.plddt_loss_config.eps,
        )

        losses["experimentally_resolved"] = experimentally_resolved_loss(
            logits=outputs["experimentally_resolved_logits"],
            atom37_atom_exists=batch["atom37_atom_exists"],
            all_atom_mask=batch["all_atom_mask"],
            resolution=batch["resolution"],
            min_resolution=self.experimentally_resolved_loss_config.min_resolution,
            max_resolution=self.experimentally_resolved_loss_config.max_resolution,
            eps=self.experimentally_resolved_loss_config.eps,
        )

        losses["violation"] = violation_loss(
            violations=outputs["violations"],
            atom14_atom_exists=batch["atom14_atom_exists"],
            eps=self.violation_loss_config.eps,
        )

        if self.tm_loss_config.enabled:
            losses["tm"] = tm_loss(
                logits=outputs["tm_logits"],
                final_affine_tensor=outputs["final_affine_tensor"],
                backbone_rigid_tensor=batch["backbone_rigid_tensor"],
                backbone_rigid_mask=batch["backbone_rigid_mask"],
                resolution=batch["resolution"],
                max_bin=self.tm_loss_config.max_bin,
                num_bins=self.tm_loss_config.num_bins,
                min_resolution=self.tm_loss_config.min_resolution,
                max_resolution=self.tm_loss_config.max_resolution,
                eps=self.tm_loss_config.eps,
            )

        for loss in losses.values():
            assert loss.size() == (batch_size,)

        weighted_losses = {}
        weighted_losses["fape"] = losses["fape"] * self.fape_loss_config.weight
        weighted_losses["supervised_chi"] = (
            losses["supervised_chi"] * self.supervised_chi_loss_config.weight
        )
        weighted_losses["distogram"] = losses["distogram"] * self.distogram_loss_config.weight
        weighted_losses["masked_msa"] = losses["masked_msa"] * self.masked_msa_loss_config.weight
        weighted_losses["plddt_loss"] = losses["plddt_loss"] * self.plddt_loss_config.weight
        weighted_losses["experimentally_resolved"] = (
            losses["experimentally_resolved"] * self.experimentally_resolved_loss_config.weight
        )
        weighted_losses["violation"] = losses["violation"] * self.violation_loss_config.weight
        if self.tm_loss_config.enabled:
            weighted_losses["tm"] = losses["tm"] * self.tm_loss_config.weight

        for name in list(weighted_losses.keys()):
            loss = weighted_losses[name]
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.warning(f"LOSS WARNING! weighted_losses['{name}']: {loss}")
                loss = torch.zeros_like(loss, requires_grad=True)
                weighted_losses[name] = loss

        weighted_total_loss = sum(weighted_losses.values())

        # "To decrease the relative importance of short sequences,
        # we multiply the final loss of each training example
        # by the square root of the number of residues after cropping.
        # This implies equal weighting for all proteins that
        # are longer than the crop size,
        # and a square-root penalty for the shorter ones."
        assert batch["seq_length"].size() == (batch_size,)
        seq_length = batch["seq_length"].float()
        crop_size = torch.ones_like(seq_length) * batch["aatype"].size(1)
        scale = torch.sqrt(torch.minimum(seq_length, crop_size))
        scaled_weighted_total_loss = scale * weighted_total_loss

        losses = {key: tensor.detach().clone().mean() for key, tensor in losses.items()}
        losses["weighted_total_loss"] = weighted_total_loss.detach().clone().mean()
        losses["scaled_weighted_total_loss"] = scaled_weighted_total_loss.detach().clone().mean()

        scaled_weighted_total_loss = scaled_weighted_total_loss.mean()
        return scaled_weighted_total_loss, losses


def softmax_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert logits.size() == targets.size()
    return -torch.sum(targets * torch.log_softmax(logits, dim=-1), dim=-1)


def sigmoid_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert logits.size() == targets.size()
    return F.binary_cross_entropy_with_logits(input=logits, target=targets, reduction="none")


def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes FAPE loss.

    Args:
        pred_frames: Rigid object of predicted frames.      [*, N_frames]
        target_frames: Rigid object of ground truth frames. [*, N_frames]
        frames_mask: Binary mask for the frames.            [*, N_frames]
        pred_positions: Predicted atom positions.           [*, N_pts, 3]
        target_positions: Ground truth positions.           [*, N_pts, 3]
        positions_mask: Positions mask.                     [*, N_pts]
        length_scale: Length scale by which the loss is divided.
        l1_clamp_distance: Cutoff above which distance errors are disregarded.
        eps: Small value used to regularize denominators.

    Returns:
        FAPE loss tensor.

    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(pred_positions[..., None, :, :])
    local_target_pos = target_frames.invert()[..., None].apply(target_positions[..., None, :, :])

    error_dist = torch.sqrt(torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps)

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # "roughly" because eps is necessarily duplicated in the latter
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    pred_aff = Rigid.from_tensor_7(traj)
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    fape_value = compute_fape(
        pred_frames=pred_aff,
        target_frames=gt_aff[:, None],
        frames_mask=backbone_rigid_mask[:, None],
        pred_positions=pred_aff.get_trans(),
        target_positions=gt_aff[:, None].get_trans(),
        positions_mask=backbone_rigid_mask[:, None],
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )

    if use_clamped_fape is not None:
        unclamped_fape_value = compute_fape(
            pred_frames=pred_aff,
            target_frames=gt_aff[:, None],
            frames_mask=backbone_rigid_mask[:, None],
            pred_positions=pred_aff.get_trans(),
            target_positions=gt_aff[:, None].get_trans(),
            positions_mask=backbone_rigid_mask[:, None],
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        use_clamped_fape = use_clamped_fape.unsqueeze(-1)

        fape_value = (
            fape_value * use_clamped_fape
            + unclamped_fape_value * (1 - use_clamped_fape)
        )

    fape_value = torch.mean(fape_value, dim=1)

    return fape_value


def sidechain_loss(
    sidechain_frames: torch.Tensor,
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    renamed_gt_frames = (
        (1.0 - alt_naming_is_better[..., None, None, None]) * rigidgroups_gt_frames
        + alt_naming_is_better[..., None, None, None] * rigidgroups_alt_gt_frames
    )

    sidechain_frames = sidechain_frames[:, -1]
    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    sidechain_atom_pos = sidechain_atom_pos[:, -1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(*batch_dims, -1, 3)
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)

    fape_value = compute_fape(
        pred_frames=sidechain_frames,
        target_frames=renamed_gt_frames,
        frames_mask=rigidgroups_gt_exists,
        pred_positions=sidechain_atom_pos,
        target_positions=renamed_atom14_gt_positions,
        positions_mask=renamed_atom14_gt_exists,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )

    return fape_value


def fape_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    backbone_clamp_distance: float,
    backbone_loss_unit_distance: float,
    backbone_weight: float,
    sidechain_clamp_distance: float,
    sidechain_length_scale: float,
    sidechain_weight: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    backbone_loss_value = backbone_loss(
        backbone_rigid_tensor=batch["backbone_rigid_tensor"],
        backbone_rigid_mask=batch["backbone_rigid_mask"],
        traj=outputs["sm_frames"],
        use_clamped_fape=batch.get("use_clamped_fape", None),
        clamp_distance=backbone_clamp_distance,
        loss_unit_distance=backbone_loss_unit_distance,
        eps=eps,
    )

    sidechain_loss_value = sidechain_loss(
        sidechain_frames=outputs["sm_sidechain_frames"],
        sidechain_atom_pos=outputs["sm_positions"],
        rigidgroups_gt_frames=batch["rigidgroups_gt_frames"],
        rigidgroups_alt_gt_frames=batch["rigidgroups_alt_gt_frames"],
        rigidgroups_gt_exists=batch["rigidgroups_gt_exists"],
        renamed_atom14_gt_positions=batch["renamed_atom14_gt_positions"],
        renamed_atom14_gt_exists=batch["renamed_atom14_gt_exists"],
        alt_naming_is_better=batch["alt_naming_is_better"],
        clamp_distance=sidechain_clamp_distance,
        length_scale=sidechain_length_scale,
        eps=eps,
    )

    fape_loss_value = (
        backbone_loss_value * backbone_weight
        + sidechain_loss_value * sidechain_weight
    )

    return fape_loss_value


def supervised_chi_loss(
    angles_sin_cos: torch.Tensor,
    unnormalized_angles_sin_cos: torch.Tensor,
    aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_angles_sin_cos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Torsion Angle Loss.

    Supplementary '1.9.1 Side chain and backbone torsion angle loss':
    Algorithm 27 Side chain and backbone torsion angle loss.

    Args:
        angles_sin_cos: Predicted angles.        [*, N, 7, 2]
        unnormalized_angles_sin_cos:             [*, N, 7, 2]
            The same angles, but unnormalized.
        aatype: Residue indices.                 [*, N]
        seq_mask: Sequence mask.                 [*, N]
        chi_mask: Angle mask.                    [*, N, 7]
        chi_angles_sin_cos: Ground truth angles. [*, N, 7, 2]
        chi_weight: Weight for the angle component of the loss.
        angle_norm_weight: Weight for the normalization component of the loss.

    Returns:
        Torsion angle loss tensor.

    """
    pred_angles = angles_sin_cos[..., 3:, :]
    residue_type_one_hot = F.one_hot(aatype, rc.RESTYPE_NUM + 1)
    chi_pi_periodic = torch.einsum(
        "ijk,kl->ijl",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(rc.CHI_PI_PERIODIC),
    )

    true_chi = chi_angles_sin_cos[:, None]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1).unsqueeze(-4)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum((true_chi_shifted - pred_angles) ** 2, dim=-1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

    sq_chi_loss = _masked_mean(chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3))

    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(
        torch.sum(unnormalized_angles_sin_cos**2, dim=-1) + eps
    )
    norm_error = torch.abs(angle_norm - 1.0)
    angle_norm_loss = _masked_mean(
        seq_mask[..., None, :, None], norm_error, dim=(-1, -2, -3)
    )

    loss = loss + angle_norm_weight * angle_norm_loss

    return loss


def distogram_loss(
    logits: torch.Tensor,
    pseudo_beta: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
    num_bins: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        num_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries**2

    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(logits=logits, targets=F.one_hot(true_bins, num_bins))

    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    return mean


def masked_msa_loss(
    logits: torch.Tensor,
    true_msa: torch.Tensor,
    bert_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes BERT-style masked MSA loss.

    Supplementary '1.9.9 Masked MSA prediction'.

    Args:
        logits:    [*, N_seq, N_res, 23] predicted residue distribution
        true_msa:  [*, N_seq, N_res] true MSA
        bert_mask: [*, N_seq, N_res] MSA mask

    Returns:
        Masked MSA loss

    """
    errors = softmax_cross_entropy(
        logits=logits,
        targets=F.one_hot(true_msa, num_classes=23),
    )

    # FP16-friendly averaging. Equivalent to:
    # loss = (
    #     torch.sum(errors * bert_mask, dim=(-1, -2)) /
    #     (eps + torch.sum(bert_mask, dim=(-1, -2)))
    # )
    loss = errors * bert_mask
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = eps + torch.sum(scale * bert_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    return loss


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=(0.5 * bin_width),
        end=1.0,
        step=bin_width,
        device=logits.device,
    )
    probs = torch.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * (probs.ndim - 1)), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * torch.swapdims(all_atom_mask, -2, -1)
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    ca_pos = rc.ATOM_ORDER["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )


def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    cutoff: float = 15.0,
    num_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
) -> torch.Tensor:

    ca_pos = rc.ATOM_ORDER["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    score = lddt(
        all_atom_pred_pos=all_atom_pred_pos,
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask,
        cutoff=cutoff,
        eps=eps,
    )

    score = score.detach()

    bin_index = torch.floor(score * num_bins).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    lddt_ca_one_hot = F.one_hot(bin_index, num_classes=num_bins)

    errors = softmax_cross_entropy(logits=logits, targets=lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
        eps + torch.sum(all_atom_mask, dim=-1)
    )

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    return loss


def experimentally_resolved_loss(
    logits: torch.Tensor,
    atom37_atom_exists: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    min_resolution: float,
    max_resolution: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    errors = sigmoid_cross_entropy(logits=logits, targets=all_atom_mask)
    loss = torch.sum(errors * atom37_atom_exists, dim=-1)
    loss = loss / (eps + torch.sum(atom37_atom_exists, dim=(-1, -2)).view(-1, 1))
    loss = torch.sum(loss, dim=-1)
    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )
    return loss


def _calculate_bin_centers(boundaries: torch.Tensor) -> torch.Tensor:
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [
            bin_centers,
            (bin_centers[-1] + step).unsqueeze(-1),
        ],
        dim=0,
    )
    return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    num_bins: int = 64,
) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      num_bins: Number of bins

    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.

    """
    boundaries = torch.linspace(
        0, max_bin, steps=(num_bins - 1), device=logits.device
    )

    aligned_confidence_probs = torch.softmax(logits, dim=-1)

    expected_aligned_error = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    predicted_aligned_error = expected_aligned_error[0]
    max_predicted_aligned_error = expected_aligned_error[1]

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    num_bins: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(
        start=0,
        end=max_bin,
        steps=(num_bins - 1),
        device=logits.device,
    )

    bin_centers = _calculate_bin_centers(boundaries)

    num_res = logits.shape[-2]
    clipped_n = max(num_res, 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == torch.max(weighted)).nonzero()[0]

    return per_alignment[tuple(argmax)]


def tm_loss(
    logits: torch.Tensor,
    final_affine_tensor: torch.Tensor,
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    resolution: torch.Tensor,
    max_bin: int = 31,
    num_bins: int = 64,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_affine = Rigid.from_tensor_7(final_affine_tensor)
    backbone_rigid = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum(
        (_points(pred_affine) - _points(backbone_rigid)) ** 2,
        dim=-1,
    )

    sq_diff = sq_diff.detach()

    boundaries = torch.linspace(
        start=0,
        end=max_bin,
        steps=(num_bins - 1),
        device=logits.device,
    )
    boundaries = boundaries**2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits=logits,
        targets=F.one_hot(true_bins, num_bins),
    )

    square_mask = (
        backbone_rigid_mask[..., None] * backbone_rigid_mask[..., None, :]
    )

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    return loss


def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,  # (*, N, 37/14, 3)
    pred_atom_mask: torch.Tensor,  # (*, N, 37/14)
    residue_index: torch.Tensor,  # (*, N)
    aatype: torch.Tensor,  # (*, N)
    tolerance_factor_soft: float = 12.0,
    tolerance_factor_hard: float = 12.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    equation 44 & 45 (Supplementary '1.9.11 Structural violations').

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation present.

    """
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = aatype[..., 1:] == rc.RESNAME_TO_IDX["PRO"]
    gt_length = (
        (~next_is_proline) * rc.BETWEEN_RES_BOND_LENGTH_C_N[0]
        + next_is_proline * rc.BETWEEN_RES_BOND_LENGTH_C_N[1]
    )
    gt_stddev = (
        (~next_is_proline) * rc.BETWEEN_RES_BOND_LENGTH_STDDEV_C_N[0]
        + next_is_proline * rc.BETWEEN_RES_BOND_LENGTH_STDDEV_C_N[1]
    )
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.relu(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    c_n_violation_mask = mask * (c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1))
    n_ca_bond_length = torch.sqrt(eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1))

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = rc.BETWEEN_RES_COS_ANGLES_CA_C_N[0]
    gt_stddev = rc.BETWEEN_RES_BOND_LENGTH_STDDEV_C_N[0]
    ca_c_n_cos_angle_error = torch.sqrt(eps + (ca_c_n_cos_angle - gt_angle) ** 2)
    ca_c_n_loss_per_residue = torch.relu(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = rc.BETWEEN_RES_COS_ANGLES_C_N_CA[0]
    gt_stddev = rc.BETWEEN_RES_COS_ANGLES_C_N_CA[1]
    c_n_ca_cos_angle_error = torch.sqrt(eps + torch.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = torch.relu(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both neighbouring residues).
    per_residue_loss_sum = (c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue)
    per_residue_loss_sum = 0.5 * (F.pad(per_residue_loss_sum, (0, 1)) + F.pad(per_residue_loss_sum, (1, 0)))

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [
                c_n_violation_mask,
                ca_c_n_violation_mask,
                c_n_ca_violation_mask,
            ],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        F.pad(violation_mask, (0, 1)),
        F.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }


def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    overlap_tolerance_soft: float = 1.5,
    overlap_tolerance_hard: float = 1.5,
    eps: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of equation 46 (Supplementary '1.9.11 Structural violations').

    Args:
      atom14_pred_positions: Predicted positions of atoms in global prediction frame.
      atom14_atom_exists: Mask denoting whether atom at positions exists for given amino acid type.
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom shape (N, 14)

    """
    dtype = atom14_pred_positions.dtype

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(dtype)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None]
        < residue_index[..., None, :, None, None]
    )

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = F.one_hot(residue_index.new_tensor(2), num_classes=14)
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])),
        *c_one_hot.shape,
    )
    c_one_hot = c_one_hot.type(dtype)
    n_one_hot = F.one_hot(residue_index.new_tensor(0), num_classes=14)
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])),
        *n_one_hot.shape,
    )
    n_one_hot = n_one_hot.type(dtype)

    neighbour_mask = (
        (residue_index[..., :, None, None, None] + 1)
        == residue_index[..., None, :, None, None]
    )
    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = rc.RESTYPE_NAME_TO_ATOM14_NAMES["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(*((1,) * len(residue_index.shape[:-1])), 1).squeeze(-1)
    cys_sg_one_hot = F.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = (
        torch.sum(dists_to_low_error, dim=(-4, -2))
        + torch.sum(dists_to_low_error, axis=(-3, -1))
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, axis=(-4, -2)),
        torch.amax(clash_mask, axis=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }


def within_residue_violations(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss: float = 0.0,
    eps: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with the same residues of
    equation 46 (Supplementary '1.9.11 Structural violations').

    Args:
        atom14_pred_positions ([*, N, 14, 3]):
            Predicted positions of atoms in global prediction frame.
        atom14_atom_exists ([*, N, 14]):
            Mask denoting whether atom at positions exists for given
            amino acid type
        atom14_dists_lower_bound ([*, N, 14]):
            Lower bound on allowed distances.
        atom14_dists_upper_bound ([*, N, 14]):
            Upper bound on allowed distances
        tighten_bounds_for_loss ([*, N]):
            Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum' ([*, N, 14]):
              sum of all clash losses per atom, shape
        * 'per_atom_clash_mask' ([*, N, 14]):
              mask whether atom clashes with any other atom shape

    """
    # Compute the mask for each residue.
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])),
        *dists_masks.shape,
    )
    dists_masks = (
        atom14_atom_exists[..., :, :, None]
        * atom14_atom_exists[..., :, None, :]
        * dists_masks
    )

    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, :, None, :]
                - atom14_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.relu(atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = torch.relu(dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * (
        (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
    )

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0],
        torch.max(violations, axis=-1)[0],
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
    }


def find_structural_violations(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
        aatype=batch["aatype"],
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [
        rc.VAN_DER_WAALS_RADIUS[name[0]]
        for name in rc.ATOM_TYPES
    ]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = (
        batch["atom14_atom_exists"]
        * atomtype_radius[batch["residx_atom14_to_atom37"]]
    )
    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch["residue_index"],
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = rc.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_atom_exists = batch["atom14_atom_exists"]
    atom14_dists_lower_bound = (
        atom14_pred_positions.new_tensor(restype_atom14_bounds["lower_bound"])[batch["aatype"]]
    )
    atom14_dists_upper_bound = (
        atom14_pred_positions.new_tensor(restype_atom14_bounds["upper_bound"])[batch["aatype"]]
    )
    residue_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(between_residue_clashes["per_atom_clash_mask"], dim=-1)[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],  # ()
            "angles_ca_c_n_loss_mean": connection_violations["ca_c_n_loss_mean"],  # ()
            "angles_c_n_ca_loss_mean": connection_violations["c_n_ca_loss_mean"],  # ()
            "connections_per_residue_loss_sum": connection_violations["per_residue_loss_sum"],  # (N)
            "connections_per_residue_violation_mask": connection_violations["per_residue_violation_mask"],  # (N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # ()
            "clashes_per_atom_loss_sum": between_residue_clashes["per_atom_loss_sum"],  # (N, 14)
            "clashes_per_atom_clash_mask": between_residue_clashes["per_atom_clash_mask"],  # (N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations["per_atom_loss_sum"],  # (N, 14)
            "per_atom_violations": residue_violations["per_atom_violations"],  # (N, 14),
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
    }


def find_structural_violations_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
) -> Dict[str, np.ndarray]:
    violations = find_structural_violations(
        batch=map_array_tree(fn=torch.tensor, tree=batch),
        atom14_pred_positions=torch.tensor(atom14_pred_positions),
        violation_tolerance_factor=violation_tolerance_factor,
        clash_overlap_tolerance=clash_overlap_tolerance,
    )
    violations = map_tensor_tree(fn=np.array, tree=violations)
    return violations


def extreme_ca_ca_distance_violations(
    pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
    pred_atom_mask: torch.Tensor,  # (N, 37(14))
    residue_index: torch.Tensor,  # (N)
    max_angstrom_tolerance: float = 1.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are
    more than 'max_angstrom_tolerance' apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.

    Returns:
      Fraction of consecutive CA-CA pairs with violation.

    """
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0
    ca_ca_distance = torch.sqrt(eps + torch.sum((this_ca_pos - next_ca_pos) ** 2, dim=-1))
    violations = (ca_ca_distance - rc.CA_CA) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    return _masked_mean(mask, violations, -1)


def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
    )
    violation_metrics = {}
    violation_metrics["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
    violation_metrics["violations_between_residue_bond"] = _masked_mean(
        mask=batch["seq_mask"],
        value=violations["between_residues"]["connections_per_residue_violation_mask"],
        dim=-1,
    )
    violation_metrics["violations_between_residue_clash"] = _masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    violation_metrics["violations_within_residue"] = _masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["within_residues"]["per_atom_violations"],
            dim=-1,
        )[0],
        dim=-1,
    )
    violation_metrics["violations_per_residue"] = _masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return violation_metrics


def compute_violation_metrics_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    violations: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    violation_metrics = compute_violation_metrics(
        batch=map_array_tree(fn=torch.tensor, tree=batch),
        atom14_pred_positions=torch.tensor(atom14_pred_positions),
        violations=map_array_tree(fn=torch.tensor, tree=violations),
    )
    violation_metrics = map_tensor_tree(fn=np.array, tree=violation_metrics)
    return violation_metrics


def violation_loss(
    violations: Dict[str, torch.Tensor],
    atom14_atom_exists: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_atoms = torch.sum(atom14_atom_exists, dim=(-1, -2))

    l_clash = torch.sum(
        violations["between_residues"]["clashes_per_atom_loss_sum"]
        + violations["within_residues"]["per_atom_loss_sum"],
        dim=(-1, -2),
    )
    l_clash = l_clash / (eps + num_atoms)

    loss = (
        violations["between_residues"]["bonds_c_n_loss_mean"]
        + violations["between_residues"]["angles_ca_c_n_loss_mean"]
        + violations["between_residues"]["angles_c_n_ca_loss_mean"]
        + l_clash
    )

    return loss


def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    eps: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """Find optimal renaming of ground truth based on the predicted positions.

    Supplementary '1.8.5 Rename symmetric ground truth atoms': Algorithm 26.

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape

    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.

    """
    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = batch["atom14_gt_positions"]
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_gt_positions[..., None, :, None, :]
                - atom14_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"]
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_alt_gt_positions[..., None, :, None, :]
                - atom14_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = batch["atom14_gt_exists"]
    atom14_atom_is_ambiguous = batch["atom14_atom_is_ambiguous"]
    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
        (1.0 - alt_naming_is_better[..., None, None]) * atom14_gt_positions
        + alt_naming_is_better[..., None, None] * atom14_alt_gt_positions
    )

    renamed_atom14_gt_mask = (
        (1.0 - alt_naming_is_better[..., None]) * atom14_gt_exists
        + alt_naming_is_better[..., None] * batch["atom14_alt_gt_exists"]
    )

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_atom14_gt_positions": renamed_atom14_gt_positions,
        "renamed_atom14_gt_exists": renamed_atom14_gt_mask,
    }


def _masked_mean(
    mask: torch.Tensor,
    value: torch.Tensor,
    dim: Union[int, Tuple[int, ...]],
    eps: float = 1e-4,
) -> torch.Tensor:
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
