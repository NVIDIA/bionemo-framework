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

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.model.protein.openfold.angle_resnet import AngleResnet
from bionemo.model.protein.openfold.backbone_update import BackboneUpdate
from bionemo.model.protein.openfold.invariant_point_attention import InvariantPointAttention
from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.linear import Linear
from bionemo.model.protein.openfold.single_transition import SingleTransition
from bionemo.model.protein.openfold.utils.rigid_utils import Rigid, Rotation


class StructureModule(nn.Module):
    """Structure Module.

    Supplementary '1.8 Structure module': Algorithm 20.

    Args:
        c_s: Single representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden_ipa: Hidden dimension in invariant point attention.
        c_hidden_ang_res: Hidden dimension in angle resnet.
        num_heads_ipa: Number of heads used in invariant point attention.
        num_qk_points: Number of query/key points in invariant point attention.
        num_v_points: Number of value points in invariant point attention.
        dropout_rate: Dropout rate in structure module.
        num_blocks: Number of shared blocks in the forward pass.
        num_ang_res_blocks: Number of blocks in angle resnet.
        num_angles: Number of angles in angle resnet.
        scale_factor: Scale translation factor.
        inf: Safe infinity value.
        eps: Epsilon to prevent division by zero.

    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_ipa: int,
        c_hidden_ang_res: int,
        num_heads_ipa: int,
        num_qk_points: int,
        num_v_points: int,
        dropout_rate: float,
        num_blocks: int,
        num_ang_res_blocks: int,
        num_angles: int,
        scale_factor: float,
        inf: float,
        eps: float,
    ) -> None:
        super(StructureModule, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden_ipa = c_hidden_ipa
        self.c_hidden_ang_res = c_hidden_ang_res
        self.num_heads_ipa = num_heads_ipa
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_ang_res_blocks = num_ang_res_blocks
        self.num_angles = num_angles
        self.scale_factor = scale_factor
        self.inf = inf
        self.eps = eps
        self.layer_norm_s = LayerNorm(c_s)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_in = Linear(c_s, c_s, bias=True, init="default")
        self.ipa = InvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden_ipa,
            num_heads=num_heads_ipa,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            inf=inf,
            eps=eps,
        )
        self.ipa_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_ipa = LayerNorm(c_s)
        self.transition = SingleTransition(
            c_s=c_s,
            dropout_rate=dropout_rate,
        )
        self.bb_update = BackboneUpdate(
            c_s=c_s,
        )
        self.angle_resnet = AngleResnet(
            c_s=c_s,
            c_hidden=c_hidden_ang_res,
            num_blocks=num_ang_res_blocks,
            num_angles=num_angles,
            eps=eps,
        )
        # buffers:
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        aatype: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Structure Module forward pass.

        Args:
            s: [batch, N_res, c_s] single representation
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res] sequence mask
            aatype: [batch, N_res] amino acid indices

        Returns:
            dictionary containing:
                "sm_frames": [batch, num_blocks, N_res, 7]
                "sm_sidechain_frames": [batch, num_blocks, N_res, 8, 4, 4]
                "sm_unnormalized_angles": [batch, num_blocks, N_res, 7, 2]
                "sm_angles": [batch, num_blocks, N_res, 7, 2]
                "sm_positions": [batch, num_blocks, N_res, 14, 3]
                "sm_states": [batch, num_blocks, N_res, c_s]
                "sm_single": [batch, N_res, c_s] updated single representation

        """
        self._initialize_buffers(dtype=s.dtype, device=s.device)

        s = self.layer_norm_s(s)
        # s: [batch, N_res, c_s]

        z = self.layer_norm_z(z)
        # z: [batch, N_res, N_res, c_z]

        s_initial = s
        # s_initial: [batch, N_res, c_s]

        s = self.linear_in(s)
        # s: [batch, N_res, c_s]

        rigids = Rigid.identity(
            shape=s.shape[:-1],
            dtype=s.dtype,
            device=s.device,
            requires_grad=self.training,
            fmt="quat",
        )

        outputs = []

        for i in range(self.num_blocks):
            s = s + self.ipa(s=s, z=z, r=rigids, mask=mask)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            # s: [batch, N_res, c_s]

            s = self.transition(s)
            # s: [batch, N_res, c_s]

            rigids = rigids.compose_q_update_vec_jit(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones here
            backb_to_global = Rigid(
                rots=Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                trans=rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(self.scale_factor)

            unnormalized_angles, angles = self.angle_resnet(s=s, s_initial=s_initial)
            # unnormalized_angles: [batch, N_res, num_angles, 2]
            # angles: [batch, N_res, num_angles, 2]

            all_frames_to_global = _torsion_angles_to_frames(
                r=backb_to_global,
                alpha=angles,
                aatype=aatype,
                rrgdf=self.default_frames,
            )

            pred_xyz = _frames_and_literature_positions_to_atom14_pos(
                r=all_frames_to_global,
                aatype=aatype,
                default_frames=self.default_frames,
                group_idx=self.group_idx,
                atom_mask=self.atom_mask,
                lit_positions=self.lit_positions,
            )
            # pred_xyz: [batch, N_res, 14, 3]

            scaled_rigids = rigids.scale_translation(self.scale_factor)

            output = {
                "sm_frames": scaled_rigids.to_tensor_7(),
                "sm_sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "sm_unnormalized_angles": unnormalized_angles,
                "sm_angles": angles,
                "sm_positions": pred_xyz,
                "sm_states": s,
            }

            outputs.append(output)

            rigids = rigids.stop_rot_gradient()

        outputs = self._stack_tensors(outputs, dim=1)

        outputs["sm_single"] = s

        return outputs

    def _initialize_buffers(self, dtype: torch.dtype, device: torch.device) -> None:
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    rc.RESTYPE_RIGID_GROUP_DEFAULT_FRAME,
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    rc.RESTYPE_ATOM14_TO_RIGID_GROUP,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    rc.RESTYPE_ATOM14_MASK,
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    rc.RESTYPE_ATOM14_RIGID_GROUP_POSITIONS,
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def _stack_tensors(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        dim: int,
    ) -> Dict[str, torch.Tensor]:
        stacked = {}
        for key in list(outputs[0].keys()):
            stacked[key] = torch.stack(
                tensors=[output[key] for output in outputs],
                dim=dim,
            )
        return stacked


def _torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
) -> Rigid:
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement,
    # which uses different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose_jit(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose_jit(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose_jit(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose_jit(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose_jit(all_frames_to_bb)

    return all_frames_to_global


def _frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
    group_idx: torch.Tensor,
    atom_mask: torch.Tensor,
    lit_positions: torch.Tensor,
) -> torch.Tensor:
    # [*, N, 14, 4, 4]
    # default_4x4 = default_frames[aatype, ...]

    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = F.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
