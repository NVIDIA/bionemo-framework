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

import datetime
import random
import time
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig

import bionemo.data.protein.openfold.residue_constants as rc
import bionemo.data.protein.openfold.transforms as data_transforms
from bionemo.data.protein.openfold.helpers import datetime_from_string, get_seed_randomly
from bionemo.data.protein.openfold.mmcif import zero_center_atom_positions
from bionemo.data.protein.openfold.parsers import parse_a3m, parse_hhr
from bionemo.data.protein.openfold.templates import TemplateHitFeaturizer
from bionemo.model.protein.openfold.schema import FEATURE_SHAPES


def create_sequence_features(sequence: str, domain_name: str) -> dict:
    seqlen = len(sequence)  # num residues
    sequence_features = {}
    sequence_features["aatype"] = rc.sequence_to_onehot(
        sequence=sequence,
        mapping=rc.RESTYPE_ORDER_WITH_X,
        map_unknown_to_x=True,
    )
    sequence_features["between_segment_residues"] = np.zeros(shape=(seqlen), dtype=np.int32)
    sequence_features["domain_name"] = np.array([domain_name.encode("utf-8")], dtype=np.object_)
    sequence_features["residue_index"] = np.arange(seqlen, dtype=np.int32)
    sequence_features["seq_length"] = np.full(shape=(seqlen), fill_value=seqlen, dtype=np.int32)
    sequence_features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)
    return sequence_features


def create_mmcif_features(
    mmcif_dict: dict,
    author_chain_id: str,
    zero_center: bool = False,
) -> dict:
    mmcif_features = {}

    pdb_chain_id = mmcif_dict["pdb_id"] + author_chain_id
    sequence = mmcif_dict["sequences"][author_chain_id]

    sequence_features = create_sequence_features(
        sequence=sequence,
        domain_name=pdb_chain_id,
    )
    mmcif_features.update(sequence_features)

    all_atom_positions = mmcif_dict["atoms"][author_chain_id]["all_atom_positions"]
    all_atom_mask = mmcif_dict["atoms"][author_chain_id]["all_atom_mask"]
    if zero_center:
        all_atom_positions = zero_center_atom_positions(
            all_atom_positions=all_atom_positions,
            all_atom_mask=all_atom_mask,
        )
    mmcif_features["all_atom_positions"] = all_atom_positions.astype(np.float32)
    mmcif_features["all_atom_mask"] = all_atom_mask.astype(np.float32)

    mmcif_features["resolution"] = np.array([mmcif_dict["resolution"]], dtype=np.float32)
    mmcif_features["release_date"] = np.array([mmcif_dict["release_date"].encode("utf-8")], dtype=np.object_)
    mmcif_features["is_distillation"] = np.array(0.0, dtype=np.float32)

    return mmcif_features


def create_target_features(
    target_dict: dict,
    plddt_threshold: float,
    zero_center: bool = False,
) -> dict:
    # This function returns mmcif-like features,
    # but for targets used in self-distillation dataset.
    target_features = {}

    target_id = target_dict["id"]
    sequence = target_dict["sequence"]

    sequence_features = create_sequence_features(sequence=sequence, domain_name=target_id)
    target_features.update(sequence_features)

    all_atom_positions = target_dict["atom_positions"]
    all_atom_mask = target_dict["atom_mask"]
    if zero_center:
        all_atom_positions = zero_center_atom_positions(all_atom_positions, all_atom_mask)
    plddt = target_dict["plddt"]
    high_confidence_mask = plddt > plddt_threshold
    high_confidence_mask = np.any(high_confidence_mask, axis=-1)
    all_atom_mask *= high_confidence_mask[..., None]
    target_features["all_atom_positions"] = all_atom_positions.astype(np.float32)
    target_features["all_atom_mask"] = all_atom_mask.astype(np.float32)

    target_features["resolution"] = np.array([0.0], dtype=np.float32)
    target_features["release_date"] = np.array(["unreleased".encode("utf-8")], dtype=np.object_)
    target_features["is_distillation"] = np.array(1.0, dtype=np.float32)

    return target_features


def create_template_features(
    sequence: str,
    hhr_string: str,
    template_hit_featurizer: TemplateHitFeaturizer,
    release_date: str,
    pdb_id: Optional[str],
    days_before_release: int,
    shuffling_seed: Optional[int],
) -> dict:
    query_release_date = datetime_from_string(release_date, "%Y-%m-%d")
    timedelta = datetime.timedelta(days=days_before_release)
    max_template_date = query_release_date - timedelta
    template_hits = parse_hhr(hhr_string)
    template_features = template_hit_featurizer.get_template_features(
        query_sequence=sequence,
        template_hits=template_hits,
        max_template_date=max_template_date,
        query_pdb_id=pdb_id,
        shuffling_seed=shuffling_seed,
    )
    return template_features


def create_msa_features(
    sequence: str,
    a3m_strings: List[str],
) -> dict:
    msas = []
    deletion_matrices = []
    for a3m_string in a3m_strings:
        if not a3m_string:
            continue
        msa, deletion_matrix = parse_a3m(a3m_string)
        msas.append(msa)
        deletion_matrices.append(deletion_matrix)

    if len(msas) == 0:
        msas.append([sequence])
        deletion_matrices.append([[0 for _ in sequence]])

    int_msa = []
    deletion_matrix = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f"MSA {msa_index} must contain at least one sequence.")
        for sequence_index, sequence in enumerate(msa):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append([rc.HHBLITS_AA_TO_ID[res] for res in sequence])
            deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

    num_res = len(msas[0][0])
    num_alignments = len(int_msa)

    msa_features = {}
    msa_features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    msa_features["msa"] = np.array(int_msa, dtype=np.int32)
    msa_features["num_alignments"] = np.array([num_alignments] * num_res, dtype=np.int32)
    return msa_features


def process_features(
    raw_features: dict,
    alphafold_config: Union[DictConfig, Dict],
    mode: str,  # "train", "eval" or "predict"
    seed: Optional[int],
) -> dict:
    assert mode in {"train", "eval", "predict"}

    if "deletion_matrix_int" in raw_features:
        deletion_matrix_int = raw_features.pop("deletion_matrix_int")
        raw_features["deletion_matrix"] = deletion_matrix_int.astype(np.float32)

    raw_feature_names = _get_raw_feature_names(
        alphafold_config=alphafold_config,
        mode=mode,
    )

    raw_feature_tensors = {
        raw_feature_name: torch.tensor(array)
        for raw_feature_name, array in raw_features.items()
        if raw_feature_name in raw_feature_names
    }

    if seed is None:
        seed = get_seed_randomly()

    features = _process_raw_feature_tensors(
        tensors=raw_feature_tensors,
        alphafold_config=alphafold_config,
        mode=mode,
        seed=seed,
    )

    # Set `use_clamped_fape` randomly when training:
    if mode == "train":
        use_clamped_fape_probability = alphafold_config.use_clamped_fape_probability
        rng = random.Random(seed)
        p = rng.uniform(0, 1)
        use_clamped_fape_value = float(p < use_clamped_fape_probability)
        features["use_clamped_fape"] = torch.full(
            size=[alphafold_config.num_recycling_iters + 1],
            fill_value=use_clamped_fape_value,
            dtype=torch.float32,
        )

    return features


def _get_raw_feature_names(
    alphafold_config: Union[DictConfig, Dict],
    mode: str,
) -> List[str]:
    raw_feature_names = deepcopy(alphafold_config.primary_raw_feature_names)
    if alphafold_config.templates_enabled:
        raw_feature_names += alphafold_config.template_raw_feature_names
    if mode in {"train", "eval"}:
        raw_feature_names += alphafold_config.supervised_raw_features_names
    return raw_feature_names


def _process_raw_feature_tensors(
    tensors: Dict[str, torch.Tensor],
    alphafold_config: Union[DictConfig, Dict],
    mode: str,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Based on the config, apply filters and transformations to the data."""

    if mode == "train":
        sequence_crop_size = alphafold_config.train_sequence_crop_size
    elif mode in {"eval", "predict"}:
        sequence_crop_size = tensors["seq_length"][0].item()

    # nonensembled transformations:
    _compose_nonensembled_perf = -time.perf_counter()
    nonensembled = _nonensembled_transform_fns(
        alphafold_config=alphafold_config,
        mode=mode,
        seed=seed,
    )
    tensors = _compose(nonensembled)(tensors)
    _compose_nonensembled_perf += time.perf_counter()

    # ensembled transformations:
    _compose_ensembled_perf = -time.perf_counter()
    ensembles = []
    for i in range(alphafold_config.num_recycling_iters + 1):
        ensembled = _ensembled_transform_fns(
            alphafold_config=alphafold_config,
            sequence_crop_size=sequence_crop_size,
            mode=mode,
            seed=seed,
            ensemble_iter=i,
        )
        ensembles.append(_compose(ensembled)(deepcopy(tensors)))
    tensors = {}
    for key in ensembles[0].keys():
        tensors[key] = torch.stack([d[key] for d in ensembles], dim=-1)
    _compose_ensembled_perf += time.perf_counter()

    return tensors


def _nonensembled_transform_fns(
    alphafold_config: Union[DictConfig, Dict],
    mode: str,
    seed: int,
) -> List[Callable]:
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.correct_msa_restypes,
        data_transforms.squeeze_features,
        data_transforms.randomly_replace_msa_with_unknown(0.0, seed),
        data_transforms.make_seq_mask,
        data_transforms.make_msa_mask,
        data_transforms.make_hhblits_profile,
    ]
    if alphafold_config.templates_enabled:
        transforms.extend(
            [
                data_transforms.fix_templates_aatype,
                data_transforms.make_template_mask,
                data_transforms.make_pseudo_beta("template_"),
            ]
        )
        if alphafold_config.embed_template_torsion_angles:
            transforms.extend(
                [
                    data_transforms.atom37_to_torsion_angles("template_"),
                ]
            )

    transforms.extend(
        [
            data_transforms.make_atom14_masks,
        ]
    )

    if mode in {"train", "eval"}:
        transforms.extend(
            [
                data_transforms.make_atom14_positions,
                data_transforms.atom37_to_frames,
                data_transforms.atom37_to_torsion_angles(""),
                data_transforms.make_pseudo_beta(""),
                data_transforms.get_backbone_frames,
                data_transforms.get_chi_angles,
            ]
        )

    return transforms


def _ensembled_transform_fns(
    alphafold_config: Union[DictConfig, Dict],
    sequence_crop_size: int,
    mode: str,
    seed: int,
    ensemble_iter: int,
) -> List[Callable]:
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []

    if mode == "train":
        transforms.append(
            data_transforms.sample_msa_distillation(
                alphafold_config.max_distillation_msa_clusters,
                (seed + ensemble_iter),
            )
        )

    transforms.append(
        data_transforms.sample_msa(
            alphafold_config.max_msa_clusters,
            keep_extra=True,
            seed=(seed + ensemble_iter),
        )
    )

    if alphafold_config.masked_msa_enabled:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_transforms.make_masked_msa(
                alphafold_config.masked_msa_profile_prob,
                alphafold_config.masked_msa_same_prob,
                alphafold_config.masked_msa_uniform_prob,
                alphafold_config.masked_msa_replace_fraction,
                (seed + ensemble_iter),
            )
        )

    if alphafold_config.msa_cluster_features:
        transforms.append(data_transforms.nearest_neighbor_clusters())
        transforms.append(data_transforms.summarize_clusters())

    # Crop after creating the cluster profiles.
    if alphafold_config.max_extra_msa:
        transforms.append(
            data_transforms.crop_extra_msa(
                alphafold_config.max_extra_msa,
                (seed + ensemble_iter),
            )
        )
    else:
        transforms.append(data_transforms.delete_extra_msa)

    transforms.append(data_transforms.make_msa_feat())

    transforms.append(
        data_transforms.filter_features(
            allowed_feature_names=set(FEATURE_SHAPES.keys()),
        )
    )

    if mode == "train":
        subsample_templates = True
    elif mode in {"eval", "predict"}:
        subsample_templates = False

    transforms.append(
        data_transforms.random_crop_and_template_subsampling(
            feature_schema_shapes=FEATURE_SHAPES,
            sequence_crop_size=sequence_crop_size,
            max_templates=alphafold_config.max_templates,
            subsample_templates=subsample_templates,
            seed=seed,
        )
    )
    transforms.append(
        data_transforms.pad_to_schema_shape(
            feature_schema_shapes=FEATURE_SHAPES,
            num_residues=sequence_crop_size,
            num_clustered_msa_seq=alphafold_config.max_msa_clusters,
            num_extra_msa_seq=alphafold_config.max_extra_msa,
            num_templates=alphafold_config.max_templates,
        )
    )

    return transforms


@data_transforms.curry1
def _compose(x, fs):
    for f in fs:
        x = f(x)
    return x
