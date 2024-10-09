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


FEATURE_SHAPES = {
    "aatype": ("N_res",),
    "all_atom_mask": ("N_res", 37),
    "all_atom_positions": ("N_res", 37, 3),
    "atom14_alt_gt_exists": ("N_res", 14),
    "atom14_alt_gt_positions": ("N_res", 14, 3),
    "atom14_atom_exists": ("N_res", 14),
    "atom14_atom_is_ambiguous": ("N_res", 14),
    "atom14_gt_exists": ("N_res", 14),
    "atom14_gt_positions": ("N_res", 14, 3),
    "atom37_atom_exists": ("N_res", 37),
    "backbone_rigid_mask": ("N_res",),
    "backbone_rigid_tensor": ("N_res", 4, 4),
    "bert_mask": ("N_clust", "N_res"),
    "chi_angles_sin_cos": ("N_res", 4, 2),
    "chi_mask": ("N_res", 4),
    "extra_deletion_value": ("N_extra_seq", "N_res"),
    "extra_has_deletion": ("N_extra_seq", "N_res"),
    "extra_msa": ("N_extra_seq", "N_res"),
    "extra_msa_mask": ("N_extra_seq", "N_res"),
    "extra_msa_row_mask": ("N_extra_seq",),
    "is_distillation": (),
    "msa_feat": ("N_clust", "N_res", 49),
    "msa_mask": ("N_clust", "N_res"),
    "msa_row_mask": ("N_clust",),
    "pseudo_beta": ("N_res", 3),
    "pseudo_beta_mask": ("N_res",),
    "residue_index": ("N_res",),
    "residx_atom14_to_atom37": ("N_res", 14),
    "residx_atom37_to_atom14": ("N_res", 37),
    "resolution": (),
    "rigidgroups_alt_gt_frames": ("N_res", 8, 4, 4),
    "rigidgroups_group_exists": ("N_res", 8),
    "rigidgroups_group_is_ambiguous": ("N_res", 8),
    "rigidgroups_gt_exists": ("N_res", 8),
    "rigidgroups_gt_frames": ("N_res", 8, 4, 4),
    "seq_length": (),
    "seq_mask": ("N_res",),
    "target_feat": ("N_res", 22),
    "template_aatype": ("N_templ", "N_res"),
    "template_all_atom_mask": ("N_templ", "N_res", 37),
    "template_all_atom_positions": ("N_templ", "N_res", 37, 3),
    "template_alt_torsion_angles_sin_cos": ("N_templ", "N_res", 7, 2),
    "template_mask": ("N_templ",),
    "template_pseudo_beta": ("N_templ", "N_res", 3),
    "template_pseudo_beta_mask": ("N_templ", "N_res"),
    "template_sum_probs": ("N_templ", 1),
    "template_torsion_angles_mask": ("N_templ", "N_res", 7),
    "template_torsion_angles_sin_cos": ("N_templ", "N_res", 7, 2),
    "true_msa": ("N_clust", "N_res"),
}
