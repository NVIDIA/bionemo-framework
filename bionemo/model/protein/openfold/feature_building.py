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

from typing import Dict

import torch
import torch.nn.functional as F

import bionemo.data.protein.openfold.residue_constants as rc
import bionemo.model.protein.openfold.inductor as inductor


def _pseudo_beta_eager(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    is_gly = torch.eq(aatype, rc.RESTYPE_ORDER["G"])
    ca_idx = rc.ATOM_ORDER["CA"]
    cb_idx = rc.ATOM_ORDER["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly.unsqueeze(-1), [1] * is_gly.ndim + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )
    return pseudo_beta.to(dtype=dtype)


_pseudo_beta_jit = torch.compile(_pseudo_beta_eager)


def _pseudo_beta(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    dtype,
) -> torch.Tensor:
    if inductor.is_enabled():
        pseudo_beta_fn = _pseudo_beta_jit
    else:
        pseudo_beta_fn = _pseudo_beta_eager
    return pseudo_beta_fn(
        aatype,
        all_atom_positions,
        dtype,
    )


def _build_extra_msa_feat_eager(
    extra_msa: torch.Tensor,
    extra_has_deletion: torch.Tensor,
    extra_deletion_value: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    msa_1hot = F.one_hot(input=extra_msa, num_classes=num_classes)
    msa_feat = [
        msa_1hot,
        extra_has_deletion.unsqueeze(-1),
        extra_deletion_value.unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)


_build_extra_msa_feat_jit = torch.compile(_build_extra_msa_feat_eager)


def _build_extra_msa_feat(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    if inductor.is_enabled():
        build_extra_msa_feat_fn = _build_extra_msa_feat_jit
    else:
        build_extra_msa_feat_fn = _build_extra_msa_feat_eager
    return build_extra_msa_feat_fn(
        extra_msa=feats["extra_msa"],
        extra_has_deletion=feats["extra_has_deletion"],
        extra_deletion_value=feats["extra_deletion_value"],
        num_classes=23,
    )


def _build_template_angle_feat(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    template_aatype = feats["template_aatype"]
    torsion_angles_sin_cos = feats["template_torsion_angles_sin_cos"]
    alt_torsion_angles_sin_cos = feats["template_alt_torsion_angles_sin_cos"]
    torsion_angles_mask = feats["template_torsion_angles_mask"]
    template_angle_feat = torch.cat(
        [
            F.one_hot(input=template_aatype, num_classes=22),
            torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14),
            alt_torsion_angles_sin_cos.reshape(*alt_torsion_angles_sin_cos.shape[:-2], 14),
            torsion_angles_mask,
        ],
        dim=-1,
    )
    return template_angle_feat


def _atom14_to_atom37(
    atom14_positions: torch.Tensor,
    residx_atom37_to_atom14: torch.Tensor,
    atom37_atom_exists: torch.Tensor,
) -> torch.Tensor:
    # atom14_positions: [batch, N_res, 14, 3]
    # residx_atom37_to_atom14: [batch, N_res, 37]
    # atom37_atom_exists: [batch, N_res, 37]

    indices = residx_atom37_to_atom14.unsqueeze(-1)
    # indices: [batch, N_res, 37, 1]
    indices = indices.expand(-1, -1, -1, 3)
    # indices: [batch, N_res, 37, 3]

    atom37_positions = torch.gather(atom14_positions, 2, indices)
    # atom37_positions: [batch, N_res, 37, 3]

    atom37_mask = atom37_atom_exists.unsqueeze(-1)
    # atom37_mask: [batch, N_res, 37, 1]

    atom37_positions = atom37_positions * atom37_mask
    # atom37_positions: [batch, N_res, 37, 3]

    return atom37_positions


def _apply_template_mask_eager(t: torch.Tensor, template_mask: torch.Tensor) -> torch.Tensor:
    t_mask = (torch.sum(template_mask, dim=1) > 0).to(dtype=t.dtype)
    t_mask = t_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    t = t * t_mask
    return t


_apply_template_mask_jit = torch.compile(_apply_template_mask_eager)


def _apply_template_mask(t: torch.Tensor, template_mask: torch.Tensor) -> torch.Tensor:
    if inductor.is_enabled():
        apply_template_mask_fn = _apply_template_mask_jit
    else:
        apply_template_mask_fn = _apply_template_mask_eager
    return apply_template_mask_fn(t, template_mask)
