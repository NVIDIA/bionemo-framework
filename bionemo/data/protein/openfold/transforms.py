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

import itertools
from functools import reduce, wraps
from operator import add
from typing import Dict, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.model.protein.openfold.utils.rigid_utils import Rigid, Rotation
from bionemo.model.protein.openfold.utils.torch_utils import TORCH_SEED_MODULUS


MSA_FEATURE_NAMES = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "msa_row_mask",
    "bert_mask",
    "true_msa",
]


def cast_to_64bit_ints(protein):
    # We keep all ints as int64
    for k, v in protein.items():
        if v.dtype == torch.int32:
            protein[k] = v.type(torch.int64)

    return protein


def make_one_hot(x, num_classes):
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


def make_seq_mask(protein):
    protein["seq_mask"] = torch.ones(protein["aatype"].shape, dtype=torch.float32)
    return protein


def make_template_mask(protein):
    protein["template_mask"] = torch.ones(protein["template_aatype"].shape[0], dtype=torch.float32)
    return protein


def curry1(f):
    """Supply all arguments but the first."""

    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


def make_all_atom_aatype(protein):
    protein["all_atom_aatype"] = protein["aatype"]
    return protein


def fix_templates_aatype(protein):
    # Map one-hot to indices
    num_templates = protein["template_aatype"].shape[0]
    if num_templates > 0:
        protein["template_aatype"] = torch.argmax(protein["template_aatype"], dim=-1)
        # Map hhsearch-aatype to our aatype.
        new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
        new_order = torch.tensor(
            new_order_list,
            dtype=torch.int64,
            device=protein["aatype"].device,
        ).expand(num_templates, -1)
        protein["template_aatype"] = torch.gather(new_order, 1, index=protein["template_aatype"])

    return protein


def correct_msa_restypes(protein):
    """Correct MSA restype to have the same order as rc."""
    new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = torch.tensor(
        [new_order_list] * protein["msa"].shape[1],
        device=protein["msa"].device,
    ).transpose(0, 1)
    protein["msa"] = torch.gather(new_order, 0, protein["msa"])

    perm_matrix = np.zeros((22, 22), dtype=np.float32)
    perm_matrix[range(len(new_order_list)), new_order_list] = 1.0

    for k in protein:
        if "profile" in k:
            num_dim = protein[k].shape.as_list()[-1]
            assert num_dim in [
                20,
                21,
                22,
            ], "num_dim for %s out of expected range: %s" % (k, num_dim)
            protein[k] = torch.dot(protein[k], perm_matrix[:num_dim, :num_dim])

    return protein


def squeeze_features(protein):
    """Remove singleton and repeated dimensions in protein features."""
    protein["aatype"] = torch.argmax(protein["aatype"], dim=-1)
    for k in [
        "domain_name",
        "msa",
        "num_alignments",
        "seq_length",
        "sequence",
        "superfamily",
        "deletion_matrix",
        "resolution",
        "between_segment_residues",
        "residue_index",
        "template_all_atom_mask",
    ]:
        if k in protein:
            final_dim = protein[k].shape[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                if torch.is_tensor(protein[k]):
                    protein[k] = torch.squeeze(protein[k], dim=-1)
                else:
                    protein[k] = np.squeeze(protein[k], axis=-1)

    for k in ["seq_length", "num_alignments"]:
        if k in protein:
            protein[k] = protein[k][0]

    return protein


@curry1
def randomly_replace_msa_with_unknown(protein, replace_proportion, seed=None):
    """Replace a portion of the MSA with 'X'."""
    g = torch.Generator(device=protein["msa"].device)
    assert seed is not None
    g.manual_seed(seed % TORCH_SEED_MODULUS)
    msa_mask = torch.rand(protein["msa"].shape, generator=g) < replace_proportion
    x_idx = 20
    gap_idx = 21
    msa_mask = torch.logical_and(msa_mask, protein["msa"] != gap_idx)
    protein["msa"] = torch.where(
        msa_mask,
        torch.ones_like(protein["msa"]) * x_idx,
        protein["msa"],
    )
    aatype_mask = torch.rand(protein["aatype"].shape, generator=g) < replace_proportion

    protein["aatype"] = torch.where(
        aatype_mask,
        torch.ones_like(protein["aatype"]) * x_idx,
        protein["aatype"],
    )
    return protein


@curry1
def sample_msa(protein, max_seq, keep_extra, seed=None):
    """Sample MSA randomly, remaining sequences are stored as `extra_*`."""
    num_seq = protein["msa"].shape[0]
    g = torch.Generator(device=protein["msa"].device)
    assert seed is not None
    g.manual_seed(seed % TORCH_SEED_MODULUS)
    shuffled = torch.randperm(num_seq - 1, generator=g) + 1
    index_order = torch.cat(
        (torch.tensor([0], device=shuffled.device), shuffled),
        dim=0,
    )
    num_sel = min(max_seq, num_seq)
    sel_seq, not_sel_seq = torch.split(index_order, [num_sel, num_seq - num_sel])

    for k in MSA_FEATURE_NAMES:
        if k in protein:
            if keep_extra:
                protein["extra_" + k] = torch.index_select(protein[k], 0, not_sel_seq)
            protein[k] = torch.index_select(protein[k], 0, sel_seq)

    return protein


@curry1
def add_distillation_flag(protein, distillation):
    protein["is_distillation"] = distillation
    return protein


@curry1
def sample_msa_distillation(protein, max_seq, seed):
    if protein["is_distillation"] == 1:
        protein = sample_msa(max_seq, keep_extra=False, seed=seed)(protein)
    return protein


@curry1
def crop_extra_msa(protein, max_extra_msa, seed=None):
    num_seq = protein["extra_msa"].shape[0]
    num_sel = min(max_extra_msa, num_seq)
    g = torch.Generator(device=protein["msa"].device)
    assert seed is not None
    g.manual_seed(seed % TORCH_SEED_MODULUS)
    select_indices = torch.randperm(num_seq, generator=g)[:num_sel]
    for k in MSA_FEATURE_NAMES:
        if "extra_" + k in protein:
            protein["extra_" + k] = torch.index_select(protein["extra_" + k], 0, select_indices)

    return protein


def delete_extra_msa(protein):
    for k in MSA_FEATURE_NAMES:
        if "extra_" + k in protein:
            del protein["extra_" + k]
    return protein


# Not used in inference
@curry1
def block_delete_msa(protein, config, seed=None):
    num_seq = protein["msa"].shape[0]
    block_num_seq = torch.floor(
        torch.tensor(num_seq, dtype=torch.float32, device=protein["msa"].device) * config.msa_fraction_per_block
    ).to(torch.int32)
    g = torch.Generator(device=protein["msa"].device)
    assert seed is not None
    g.manual_seed(seed % TORCH_SEED_MODULUS)
    if config.randomize_num_blocks:
        nb = torch.randint(0, config.num_blocks + 1, size=[1], generator=g)[0].item()
    else:
        nb = config.num_blocks

    del_block_starts = torch.randint(0, num_seq, size=[nb], generator=g)
    del_blocks = del_block_starts[:, None] + torch.range(block_num_seq)
    del_blocks = torch.clip(del_blocks, 0, num_seq - 1)
    del_indices = torch.unique(torch.sort(torch.reshape(del_blocks, [-1])))[0]

    # Make sure we keep the original sequence
    combined = torch.cat((torch.range(1, num_seq)[None], del_indices[None]))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    # intersection = uniques[counts > 1]
    keep_indices = torch.squeeze(difference, 0)

    for k in MSA_FEATURE_NAMES:
        if k in protein:
            protein[k] = torch.gather(protein[k], keep_indices)

    return protein


@curry1
def nearest_neighbor_clusters(protein, gap_agreement_weight=0.0):
    weights = torch.cat(
        [
            torch.ones(21, device=protein["msa"].device),
            gap_agreement_weight * torch.ones(1, device=protein["msa"].device),
            torch.zeros(1, device=protein["msa"].device),
        ],
        0,
    )

    # Make agreement score as weighted Hamming distance
    msa_one_hot = make_one_hot(protein["msa"], 23)
    sample_one_hot = protein["msa_mask"][:, :, None] * msa_one_hot
    extra_msa_one_hot = make_one_hot(protein["extra_msa"], 23)
    extra_one_hot = protein["extra_msa_mask"][:, :, None] * extra_msa_one_hot

    num_seq, num_res, _ = sample_one_hot.shape
    extra_num_seq, _, _ = extra_one_hot.shape

    # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
    # in an optimized fashion to avoid possible memory or computation blowup.
    agreement = torch.matmul(
        torch.reshape(
            input=extra_one_hot,
            shape=[extra_num_seq, num_res * 23],
        ),
        torch.reshape(
            input=sample_one_hot * weights,
            shape=[num_seq, num_res * 23],
        ).transpose(0, 1),
    )

    # Assign each sequence in the extra sequences to the closest MSA sample
    protein["extra_cluster_assignment"] = torch.argmax(agreement, dim=1).to(torch.int64)

    return protein


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Similar to
    tf.unsorted_segment_sum, but only supports 1-D indices.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The 1-D segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert len(segment_ids.shape) == 1 and segment_ids.shape[0] == data.shape[0]
    segment_ids = segment_ids.view(segment_ids.shape[0], *((1,) * len(data.shape[1:])))
    segment_ids = segment_ids.expand(data.shape)
    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape, device=segment_ids.device).scatter_add_(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


@curry1
def summarize_clusters(protein):
    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = protein["msa"].shape[0]

    def csum(x):
        return unsorted_segment_sum(x, protein["extra_cluster_assignment"], num_seq)

    mask = protein["extra_msa_mask"]
    mask_counts = 1e-6 + protein["msa_mask"] + csum(mask)  # Include center

    msa_sum = csum(mask[:, :, None] * make_one_hot(protein["extra_msa"], 23))
    msa_sum += make_one_hot(protein["msa"], 23)  # Original sequence
    protein["cluster_profile"] = msa_sum / mask_counts[:, :, None]
    del msa_sum

    del_sum = csum(mask * protein["extra_deletion_matrix"])
    del_sum += protein["deletion_matrix"]  # Original sequence
    protein["cluster_deletion_mean"] = del_sum / mask_counts
    del del_sum

    return protein


def make_msa_mask(protein):
    """Mask features are all ones, but will later be zero-padded."""
    protein["msa_mask"] = torch.ones(protein["msa"].shape, dtype=torch.float32)
    protein["msa_row_mask"] = torch.ones((protein["msa"].shape[0]), dtype=torch.float32)
    return protein


def pseudo_beta_fn(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: Optional[torch.Tensor],
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create pseudo beta features."""
    is_gly = torch.eq(aatype, rc.RESTYPE_ORDER["G"])
    ca_idx = rc.ATOM_ORDER["CA"]
    cb_idx = rc.ATOM_ORDER["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly.unsqueeze(-1), [1] * is_gly.ndim + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_mask[..., ca_idx],
            all_atom_mask[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


@curry1
def make_pseudo_beta(protein, prefix=""):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ["", "template_"]
    (
        protein[prefix + "pseudo_beta"],
        protein[prefix + "pseudo_beta_mask"],
    ) = pseudo_beta_fn(
        protein["template_aatype" if prefix else "aatype"],
        protein[prefix + "all_atom_positions"],
        protein["template_all_atom_mask" if prefix else "all_atom_mask"],
    )
    return protein


@curry1
def add_constant_field(protein, key, value):
    protein[key] = torch.tensor(value, device=protein["msa"].device)
    return protein


def shaped_categorical(probs, generator, epsilon=1e-10):
    ds = probs.shape
    num_classes = ds[-1]
    probs = torch.reshape(probs + epsilon, [-1, num_classes])
    check_result = torch.distributions.constraints.simplex.check(probs)
    assert check_result.all()
    counts = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator)
    counts = counts.squeeze(-1)
    return torch.reshape(counts, ds[:-1])


def make_hhblits_profile(protein):
    """Compute the HHblits MSA profile if not already present."""
    if "hhblits_profile" in protein:
        return protein

    # Compute the profile for every residue (over all MSA sequences).
    msa_one_hot = make_one_hot(protein["msa"], 22)

    protein["hhblits_profile"] = torch.mean(msa_one_hot, dim=0)
    return protein


@curry1
def make_masked_msa(protein, profile_prob, same_prob, uniform_prob, replace_fraction, seed=None):
    """Create data for BERT on raw MSA."""
    # Add a random amino acid uniformly.
    g = torch.Generator(device=protein["seq_length"].device)
    assert seed is not None
    g.manual_seed(seed % TORCH_SEED_MODULUS)

    random_aa = torch.tensor(
        [0.05] * 20 + [0.0, 0.0],
        dtype=torch.float32,
        device=protein["aatype"].device,
    )

    categorical_probs = (
        uniform_prob * random_aa
        + profile_prob * protein["hhblits_profile"]
        + same_prob * make_one_hot(protein["msa"], 22)
    )

    # Put all remaining probability on [MASK] which is a new column
    pad_shapes = list(reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))]))
    pad_shapes[1] = 1
    mask_prob = 1.0 - profile_prob - same_prob - uniform_prob
    assert mask_prob >= 0.0

    categorical_probs = F.pad(categorical_probs, pad_shapes, value=mask_prob)

    sh = protein["msa"].shape
    mask_position = torch.rand(sh, generator=g) < replace_fraction

    bert_msa = shaped_categorical(categorical_probs, generator=g)
    bert_msa = torch.where(mask_position, bert_msa, protein["msa"])

    # Mix real and masked MSA
    protein["bert_mask"] = mask_position.to(torch.float32)
    protein["true_msa"] = protein["msa"]
    protein["msa"] = bert_msa

    return protein


@curry1
def pad_to_schema_shape(
    protein: Dict[str, torch.Tensor],
    feature_schema_shapes: Dict[str, tuple],
    num_residues: int,
    num_clustered_msa_seq: int,
    num_extra_msa_seq: int,
    num_templates: int,
) -> Dict[str, torch.Tensor]:
    """Guess at the MSA and sequence dimension to make fixed size."""

    pad_size_map = {
        "N_res": num_residues,
        "N_clust": num_clustered_msa_seq,
        "N_extra_seq": num_extra_msa_seq,
        "N_templ": num_templates,
    }

    for key, tensor in protein.items():
        if key == "extra_cluster_assignment":
            continue

        tensor_shape = list(tensor.shape)
        schema_shape = feature_schema_shapes[key]
        assert len(tensor_shape) == len(schema_shape)

        pad_shape = [
            pad_size_map.get(dim_schema, dim_size) for (dim_schema, dim_size) in zip(schema_shape, tensor_shape)
        ]

        padding = [(0, pad_size - dim_size) for pad_size, dim_size in zip(pad_shape, tensor_shape)]

        padding.reverse()
        padding = list(itertools.chain(*padding))

        if padding:
            protein[key] = F.pad(tensor, padding)

    return protein


@curry1
def make_msa_feat(protein):
    """Create and concatenate MSA features."""
    # Whether there is a domain break. Always zero for chains, but keeping for
    # compatibility with domain datasets.
    has_break = torch.clip(protein["between_segment_residues"].to(torch.float32), 0, 1)
    aatype_1hot = make_one_hot(protein["aatype"], 21)

    target_feat = [
        torch.unsqueeze(has_break, dim=-1),
        aatype_1hot,  # Everyone gets the original sequence.
    ]

    msa_1hot = make_one_hot(protein["msa"], 23)
    has_deletion = torch.clip(protein["deletion_matrix"], 0.0, 1.0)
    deletion_value = torch.atan(protein["deletion_matrix"] / 3.0) * (2.0 / np.pi)

    msa_feat = [
        msa_1hot,
        torch.unsqueeze(has_deletion, dim=-1),
        torch.unsqueeze(deletion_value, dim=-1),
    ]

    if "cluster_profile" in protein:
        deletion_mean_value = torch.atan(protein["cluster_deletion_mean"] / 3.0) * (2.0 / np.pi)
        msa_feat.extend(
            [
                protein["cluster_profile"],
                torch.unsqueeze(deletion_mean_value, dim=-1),
            ]
        )

    if "extra_deletion_matrix" in protein:
        protein["extra_has_deletion"] = torch.clip(protein["extra_deletion_matrix"], 0.0, 1.0)
        protein["extra_deletion_value"] = torch.atan(protein["extra_deletion_matrix"] / 3.0) * (2.0 / np.pi)

    protein["msa_feat"] = torch.cat(msa_feat, dim=-1)
    protein["target_feat"] = torch.cat(target_feat, dim=-1)
    return protein


@curry1
def filter_features(
    protein: Dict[str, torch.Tensor],
    allowed_feature_names: Set[str],
) -> Dict[str, torch.Tensor]:
    return {key: tensor for key, tensor in protein.items() if key in allowed_feature_names}


@curry1
def crop_templates(protein, max_templates):
    for k, v in protein.items():
        if k.startswith("template_"):
            protein[k] = v[:max_templates]
    return protein


def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.RESTYPES:
        atom_names = rc.RESTYPE_NAME_TO_ATOM14_NAMES[rc.RESTYPE_1TO3[rt]]
        restype_atom14_to_atom37.append([(rc.ATOM_ORDER[name] if name else 0) for name in atom_names])
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0) for name in rc.ATOM_TYPES]
        )

        restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )
    protein_aatype = protein["aatype"].to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=protein["aatype"].device)
    for restype, restype_letter in enumerate(rc.RESTYPES):
        restype_name = rc.RESTYPE_1TO3[restype_letter]
        atom_names = rc.RESIDUE_ATOMS[restype_name]
        for atom_name in atom_names:
            atom_type = rc.ATOM_ORDER[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask

    return protein


def make_atom14_positions(protein):
    """Constructs denser atom positions (14 dimensions instead of 37)."""
    residx_atom14_mask = protein["atom14_atom_exists"]
    residx_atom14_to_atom37 = protein["residx_atom14_to_atom37"]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * _batched_gather(
        protein["all_atom_mask"],
        residx_atom14_to_atom37,
        dim=-1,
        no_batch_dims=len(protein["all_atom_mask"].shape[:-1]),
    )

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[..., None] * (
        _batched_gather(
            protein["all_atom_positions"],
            residx_atom14_to_atom37,
            dim=-2,
            no_batch_dims=len(protein["all_atom_positions"].shape[:-2]),
        )
    )

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["atom14_gt_exists"] = residx_atom14_gt_mask
    protein["atom14_gt_positions"] = residx_atom14_gt_positions

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [rc.RESTYPE_1TO3[res] for res in rc.RESTYPES]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {
        res: torch.eye(
            14,
            dtype=protein["all_atom_mask"].dtype,
            device=protein["all_atom_mask"].device,
        )
        for res in restype_3
    }
    for resname, swap in rc.RESIDUE_ATOM_RENAMING_SWAPS.items():
        correspondences = torch.arange(14, device=protein["all_atom_mask"].device)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = rc.RESTYPE_NAME_TO_ATOM14_NAMES[resname].index(source_atom_swap)
            target_index = rc.RESTYPE_NAME_TO_ATOM14_NAMES[resname].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = protein["all_atom_mask"].new_zeros((14, 14))
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix

    renaming_matrices = torch.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[protein["aatype"]]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = torch.einsum("...rac,...rab->...rbc", residx_atom14_gt_positions, renaming_transform)
    protein["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = torch.einsum("...ra,...rab->...rb", residx_atom14_gt_mask, renaming_transform)
    protein["atom14_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = protein["all_atom_mask"].new_zeros((21, 14))
    for resname, swap in rc.RESIDUE_ATOM_RENAMING_SWAPS.items():
        for atom_name1, atom_name2 in swap.items():
            restype = rc.RESTYPE_ORDER[rc.RESTYPE_3TO1[resname]]
            atom_idx1 = rc.RESTYPE_NAME_TO_ATOM14_NAMES[resname].index(atom_name1)
            atom_idx2 = rc.RESTYPE_NAME_TO_ATOM14_NAMES[resname].index(atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    protein["atom14_atom_is_ambiguous"] = restype_atom14_is_ambiguous[protein["aatype"]]

    return protein


def atom37_to_frames(protein, eps=1e-8):
    aatype = protein["aatype"]
    all_atom_positions = protein["all_atom_positions"]
    all_atom_mask = protein["all_atom_mask"]

    batch_dims = len(aatype.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(rc.RESTYPES):
        resname = rc.RESTYPE_1TO3[restype_letter]
        for chi_idx in range(4):
            if rc.CHI_ANGLES_MASK[restype][chi_idx]:
                names = rc.CHI_ANGLES_ATOMS[resname][chi_idx]
                restype_rigidgroup_base_atom_names[restype, chi_idx + 4, :] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*aatype.shape[:-1], 21, 8),
    )
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(rc.CHI_ANGLES_MASK)

    lookuptable = rc.ATOM_ORDER.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom37_idx,
    )
    restype_rigidgroup_base_atom37_idx = restype_rigidgroup_base_atom37_idx.view(
        *((1,) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape
    )

    residx_rigidgroup_base_atom37_idx = _batched_gather(
        restype_rigidgroup_base_atom37_idx,
        aatype,
        dim=-3,
        no_batch_dims=batch_dims,
    )

    base_atom_pos = _batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom37_idx,
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )

    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    group_exists = _batched_gather(
        restype_rigidgroup_mask,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = _batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom37_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_frames = gt_frames.compose(Rigid(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(*((1,) * batch_dims), 21, 8)
    restype_rigidgroup_rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 21, 8, 1, 1),
    )

    for resname, _ in rc.RESIDUE_ATOM_RENAMING_SWAPS.items():
        restype = rc.RESTYPE_ORDER[rc.RESTYPE_3TO1[resname]]
        chi_idx = int(sum(rc.CHI_ANGLES_MASK[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = _batched_gather(
        restype_rigidgroup_is_ambiguous,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = _batched_gather(
        restype_rigidgroup_rots,
        aatype,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = Rotation(rot_mats=residx_rigidgroup_ambiguity_rot)
    alt_gt_frames = gt_frames.compose(Rigid(residx_rigidgroup_ambiguity_rot, None))

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    protein["rigidgroups_gt_frames"] = gt_frames_tensor
    protein["rigidgroups_gt_exists"] = gt_exists
    protein["rigidgroups_group_exists"] = group_exists
    protein["rigidgroups_group_is_ambiguous"] = residx_rigidgroup_is_ambiguous
    protein["rigidgroups_alt_gt_frames"] = alt_gt_frames_tensor

    return protein


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.RESTYPES + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.RESTYPES:
        residue_name = rc.RESTYPE_1TO3[residue_name]
        residue_chi_angles = rc.CHI_ANGLES_ATOMS[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.ATOM_ORDER[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices


@curry1
def atom37_to_torsion_angles(
    protein,
    prefix="",
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    aatype = protein[prefix + "aatype"]
    all_atom_positions = protein[prefix + "all_atom_positions"]
    all_atom_mask = protein[prefix + "all_atom_mask"]

    aatype = torch.clamp(aatype, max=20)

    pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
    prev_all_atom_positions = torch.cat([pad, all_atom_positions[..., :-1, :, :]], dim=-3)

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(prev_all_atom_mask[..., 1:3], dim=-1) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
    psi_mask = torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype) * all_atom_mask[..., 4]

    chi_atom_indices = torch.as_tensor(get_chi_atom_indices(), device=aatype.device)

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = _batched_gather(all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2]))

    chi_angles_mask = list(rc.CHI_ANGLES_MASK)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = _batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype)
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack([fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1)

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        rc.CHI_PI_PERIODIC,
    )[aatype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*aatype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = torsion_angles_sin_cos * mirror_torsion_angles[..., None]

    protein[prefix + "torsion_angles_sin_cos"] = torsion_angles_sin_cos
    protein[prefix + "alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
    protein[prefix + "torsion_angles_mask"] = torsion_angles_mask

    return protein


def get_backbone_frames(protein):
    # DISCREPANCY: AlphaFold uses tensor_7s here. I don't know why.
    protein["backbone_rigid_tensor"] = protein["rigidgroups_gt_frames"][..., 0, :, :]
    protein["backbone_rigid_mask"] = protein["rigidgroups_gt_exists"][..., 0]

    return protein


def get_chi_angles(protein):
    dtype = protein["all_atom_mask"].dtype
    protein["chi_angles_sin_cos"] = protein["torsion_angles_sin_cos"][..., 3:, :].to(dtype)
    protein["chi_mask"] = protein["torsion_angles_mask"][..., 3:].to(dtype)
    return protein


@curry1
def random_crop_and_template_subsampling(
    protein: Dict[str, torch.Tensor],
    feature_schema_shapes: Dict[str, tuple],
    sequence_crop_size: int,
    max_templates: int,
    subsample_templates: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Randomly crop to `sequence_crop_size` and optionally subsample templates."""
    # We want each ensemble to be cropped the same way
    g = torch.Generator(device=protein["seq_length"].device)
    assert seed is not None
    g.manual_seed(seed % TORCH_SEED_MODULUS)

    seq_length = int(protein["seq_length"])
    num_res_crop_size = min(seq_length, sequence_crop_size)

    if "template_mask" in protein:
        num_templates = protein["template_mask"].shape[-1]
    else:
        num_templates = 0

    # No need to subsample templates if there aren't any
    subsample_templates = subsample_templates and num_templates > 0

    def _randint(lower, upper):
        return int(
            torch.randint(
                low=lower,
                high=upper + 1,
                size=(1,),
                generator=g,
                device=protein["seq_length"].device,
            )[0]
        )

    if subsample_templates:
        templates_crop_start = _randint(0, num_templates)
        templates_select_indices = torch.randperm(
            n=num_templates,
            generator=g,
            device=protein["seq_length"].device,
        )
    else:
        templates_crop_start = 0

    num_templates_crop_size = min(num_templates - templates_crop_start, max_templates)

    assert seq_length >= num_res_crop_size
    num_res_crop_start = _randint(0, seq_length - num_res_crop_size)

    for key, tensor in protein.items():
        assert key in feature_schema_shapes
        schema_shape = feature_schema_shapes[key]
        assert isinstance(schema_shape, tuple)

        if "template" not in key and "N_res" not in schema_shape:
            continue

        # randomly permute the templates before cropping them.
        if key.startswith("template") and subsample_templates:
            tensor = tensor[templates_select_indices]

        assert len(schema_shape) == len(tensor.shape)

        slices = []
        for dim_schema, dim_size in zip(schema_shape, tensor.shape):
            is_N_res = bool(dim_schema == "N_res")
            is_N_templ = bool(dim_schema == "N_templ")
            if is_N_templ and key.startswith("template"):
                crop_size = num_templates_crop_size
                crop_start = templates_crop_start
            else:
                crop_start = num_res_crop_start if is_N_res else 0
                crop_size = num_res_crop_size if is_N_res else dim_size
            slices.append(slice(crop_start, crop_start + crop_size))
        protein[key] = tensor[slices]

    protein["seq_length"].fill_(num_res_crop_size)

    return protein


def _batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]
