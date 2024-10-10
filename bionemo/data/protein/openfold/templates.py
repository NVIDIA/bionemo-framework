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

import dataclasses
import datetime
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from nemo.utils import logging

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.data.protein.openfold.helpers import datetime_from_string
from bionemo.data.protein.openfold.mmcif import load_mmcif_dict, zero_center_atom_positions
from bionemo.data.protein.openfold.parsers import TemplateHit, parse_a3m
from bionemo.data.protein.openfold.tools.kalign import Kalign


class NoChainsError(Exception):
    """An error indicating that template mmCIF didn't have any chains."""


class SequenceNotInTemplateError(Exception):
    """An error indicating that template mmCIF didn't contain the sequence."""


class NoAtomDataInTemplateError(Exception):
    """An error indicating that template mmCIF didn't contain atom positions."""


class TemplateAtomMaskAllZerosError(Exception):
    """An error indicating that template mmCIF had all atom positions masked."""


class QueryToTemplateAlignError(Exception):
    """An error indicating that the query can't be aligned to the template."""


class CaDistanceError(Exception):
    """An error indicating that a CA atom distance exceeds a threshold."""


@dataclasses.dataclass(frozen=True)
class TemplateFeaturesResult:
    features: Optional[dict]
    error: Optional[str]
    warning: Optional[str]


class TemplateHitFeaturizer:
    """A class for computing template features from `.hhr` template hits."""

    def __init__(
        self,
        max_template_hits: int,
        pdb_mmcif_dicts_dirpath: Path,
        template_pdb_chain_ids: Set[str],
        pdb_release_dates: Dict[str, datetime.datetime],
        pdb_obsolete_filepath: Optional[Path] = None,
        shuffle_top_k_prefiltered: Optional[int] = None,
        kalign_executable_path: str = "kalign",
        realign_when_required: bool = True,
        verbose: bool = False,
    ) -> None:
        if pdb_obsolete_filepath is not None:
            pdb_obsolete_mapping = _load_pdb_obsolete_mapping(pdb_obsolete_filepath)
        else:
            pdb_obsolete_mapping = {}
        self.max_template_hits = max_template_hits
        self.pdb_mmcif_dicts_dirpath = pdb_mmcif_dicts_dirpath
        self.template_pdb_chain_ids = template_pdb_chain_ids
        self.pdb_release_dates = pdb_release_dates
        self.pdb_obsolete_mapping = pdb_obsolete_mapping
        self.shuffle_top_k_prefiltered = shuffle_top_k_prefiltered
        self.kalign_executable_path = kalign_executable_path
        self.realign_when_required = realign_when_required
        self.verbose = verbose

    def get_template_features(
        self,
        query_sequence: str,
        template_hits: List[TemplateHit],
        max_template_date: datetime.datetime,
        query_pdb_id: Optional[str] = None,
        shuffling_seed: Optional[int] = None,
    ) -> dict:
        if len(template_hits) == 0:
            return create_empty_template_feats(len(query_sequence))

        prefiltered_template_hits = _prefilter_template_hits(
            template_hits=template_hits,
            query_sequence=query_sequence,
            query_pdb_id=query_pdb_id,
            max_template_date=max_template_date,
            pdb_release_dates=self.pdb_release_dates,
            pdb_obsolete_mapping=self.pdb_obsolete_mapping,
            template_pdb_chain_ids=self.template_pdb_chain_ids,
            verbose=self.verbose,
        )

        prefiltered_template_hits = sorted(
            prefiltered_template_hits,
            key=lambda x: x.sum_probs,
            reverse=True,
        )

        if self.shuffle_top_k_prefiltered is not None:
            top = prefiltered_template_hits[: self.shuffle_top_k_prefiltered]
            bottom = prefiltered_template_hits[self.shuffle_top_k_prefiltered :]
            shuffling_rng = random.Random(shuffling_seed)
            shuffling_rng.shuffle(top)
            prefiltered_template_hits = top + bottom

        template_features = {
            "template_domain_names": [],
            "template_sequence": [],
            "template_aatype": [],
            "template_all_atom_positions": [],
            "template_all_atom_mask": [],
            "template_sum_probs": [],
        }
        num_featurized_templates = 0
        errors = []
        warnings = []
        for template_hit in prefiltered_template_hits:
            result = _featurize_template_hit(
                template_hit=template_hit,
                query_sequence=query_sequence,
                pdb_mmcif_dicts_dirpath=self.pdb_mmcif_dicts_dirpath,
                max_template_date=max_template_date,
                pdb_release_dates=self.pdb_release_dates,
                pdb_obsolete_mapping=self.pdb_obsolete_mapping,
                kalign_executable_path=self.kalign_executable_path,
                realign_when_required=self.realign_when_required,
                verbose=self.verbose,
            )

            if result.error:
                errors.append(result.error)

            if result.warning:
                warnings.append(result.warning)

            if result.features is not None:
                for key in list(template_features.keys()):
                    template_features[key].append(result.features[key])
                num_featurized_templates += 1

            if num_featurized_templates >= self.max_template_hits:
                # We got all the templates we wanted, stop processing template hits.
                break

        if num_featurized_templates > 0:
            for key in (
                "template_aatype",
                "template_all_atom_positions",
                "template_all_atom_mask",
                "template_sum_probs",
            ):
                template_features[key] = np.stack(template_features[key], axis=0)
        else:
            template_features = create_empty_template_feats(seqlen=len(query_sequence))

        if self.verbose:
            errors_str = f" errors: {errors}" if errors else ""
            warnings_str = f" warnings: {warnings}" if warnings else ""
            logging.debug(
                "get_template_features:"
                f" num_featurized_templates={num_featurized_templates}"
                f" template_domain_names={template_features['template_domain_names']}"
                f"{errors_str}"
                f"{warnings_str}"
            )

        return template_features


def _load_pdb_obsolete_mapping(pdb_obsolete_filepath: Path) -> Dict[str, str]:
    """Parses the data file from PDB that lists which PDB ids are obsolete."""
    mapping = {}
    with open(pdb_obsolete_filepath) as f:
        lines = f.read().strip().split("\n")
    for line in lines:
        line = line.strip()
        # Skipping obsolete entries that don't contain a mapping to a new entry.
        if line.startswith("OBSLTE") and len(line) > 30:
            # Format:    Date      From     To
            # 'OBSLTE    31-JUL-94 116L     216L'
            from_id = line[20:24].lower()
            to_id = line[29:33].lower()
            mapping[from_id] = to_id
    return mapping


def create_empty_template_feats(seqlen: int) -> dict:
    return {
        "template_domain_names": [],
        "template_sequence": [],
        "template_aatype": np.zeros(shape=(0, seqlen), dtype=np.int32),
        "template_all_atom_positions": np.zeros(
            shape=(0, seqlen, rc.ATOM_TYPE_NUM, 3),
            dtype=np.float32,
        ),
        "template_all_atom_mask": np.zeros(
            shape=(0, seqlen, rc.ATOM_TYPE_NUM),
            dtype=np.float32,
        ),
        "template_sum_probs": np.zeros(shape=(0, 1), dtype=np.float32),
    }


def _prefilter_template_hits(
    template_hits: List[TemplateHit],
    query_sequence: str,
    query_pdb_id: Optional[str],
    max_template_date: datetime.datetime,
    pdb_release_dates: Dict[str, datetime.datetime],
    pdb_obsolete_mapping: Dict[str, str],
    template_pdb_chain_ids: Set[str],
    verbose: bool,
) -> List[TemplateHit]:
    prefiltered_template_hits = []
    rejection_messages_grouped = defaultdict(list)
    for template_hit in template_hits:
        result = _check_if_template_hit_is_acceptable(
            template_hit=template_hit,
            query_sequence=query_sequence,
            query_pdb_id=query_pdb_id,
            max_template_date=max_template_date,
            pdb_release_dates=pdb_release_dates,
            pdb_obsolete_mapping=pdb_obsolete_mapping,
            template_pdb_chain_ids=template_pdb_chain_ids,
        )
        is_template_hit_acceptable = result[0]
        rejection_message = result[1]
        if is_template_hit_acceptable:
            prefiltered_template_hits.append(template_hit)
        else:
            key = rejection_message.split(";")[0]
            rejection_messages_grouped[key].append(rejection_message)
    if verbose:
        num_template_hits = len(template_hits)
        num_accepted_hits = len(prefiltered_template_hits)
        num_rejected_hits = num_template_hits - num_accepted_hits
        rejection_messages_cnt = Counter()
        for key, rejection_messages in rejection_messages_grouped.items():
            rejection_messages_cnt[key] += len(rejection_messages)
        logging.debug(
            "_prefilter_template_hits:"
            f" query_pdb_id={repr(query_pdb_id)}"
            f" num_template_hits={num_template_hits}"
            f" num_accepted_hits={num_accepted_hits}"
            f" num_rejected_hits={num_rejected_hits}"
            f" rejection_messages_stats={rejection_messages_cnt}"
        )
    return prefiltered_template_hits


def _check_if_template_hit_is_acceptable(
    template_hit: TemplateHit,
    query_sequence: str,
    query_pdb_id: Optional[str],
    max_template_date: datetime.datetime,
    pdb_release_dates: Dict[str, datetime.datetime],
    pdb_obsolete_mapping: Dict[str, str],
    template_pdb_chain_ids: Set[str],
    max_subsequence_ratio: float = 0.95,
    min_align_ratio: float = 0.1,
) -> Tuple[bool, str]:
    """Checks if template is acceptable (without parsing the template mmcif file).

    Args:
        template_hit: HHSearch template hit.
        query_sequence: Amino acid sequence of the query.
        query_pdb_id: Optional 4 letter pdb code of the query.
        max_template_date: Max template release date after which template hits are discarded.
        pdb_release_dates: Dictionary mapping from pdb codes to their release dates.
        template_pdb_chain_ids: Set of `pdb_chain_id`s allowed to be used as templates.
        pdb_obsolete_mapping: Mapping from obsolete pdb ids to new ids.
        max_subsequence_ratio: Exclude any exact matches with this much overlap.
        min_align_ratio: Minimum overlap between the template and query.

    Returns:
        is_template_hit_acceptable: Flag denoting is template hit is acceptable.
        rejection_message: Rejection message if template is not acceptable.

    """
    template_hit_pdb_chain = _get_pdb_id_and_chain_id(template_hit)
    template_hit_pdb_id = template_hit_pdb_chain[0]
    template_hit_chain_id = template_hit_pdb_chain[1]

    if template_hit_pdb_id not in pdb_release_dates:
        if template_hit_pdb_id in pdb_obsolete_mapping:
            template_hit_pdb_id = pdb_obsolete_mapping[template_hit_pdb_id]

    template_hit_pdb_chain_id = template_hit_pdb_id + "_" + template_hit_chain_id
    if template_hit_pdb_chain_id not in template_pdb_chain_ids:
        # Template hit not present in downloaded PDB
        rejection_message = f"not in template_pdb_chain_ids;{template_hit_pdb_chain_id}"
        return False, rejection_message

    if query_pdb_id is not None:
        if template_hit_pdb_id.lower() == query_pdb_id.lower():
            # Template PDB ID identical to query PDB ID
            rejection_message = f"template == query;{template_hit_pdb_id};{query_pdb_id}"
            return False, rejection_message

    if template_hit_pdb_id in pdb_release_dates:
        template_release_date = pdb_release_dates[template_hit_pdb_id]
        if template_release_date > max_template_date:
            # Template release date is after max allowed date
            rejection_message = "{msg};{trd};{mtd}".format(
                msg="template_release_date > max_template_date",
                trd=template_release_date.date(),
                mtd=max_template_date.date(),
            )
            return False, rejection_message

    aligned_cols = template_hit.aligned_cols
    align_ratio = aligned_cols / len(query_sequence)
    if align_ratio <= min_align_ratio:
        # Proportion of residues aligned to query too small
        rejection_message = "{msg};{ar:.6f};{minar:.6f}".format(
            msg="align_ratio <= min_align_ratio",
            ar=align_ratio,
            minar=min_align_ratio,
        )
        return False, rejection_message

    template_sequence = template_hit.hit_sequence.replace("-", "")
    length_ratio = float(len(template_sequence)) / len(query_sequence)
    # Check whether the template is a large subsequence or duplicate of original query.
    # This can happen due to duplicate entries in the PDB database.
    is_duplicate = template_sequence in query_sequence and length_ratio > max_subsequence_ratio
    if is_duplicate:
        # Template is an exact subsequence of query with large coverage
        rejection_message = "{msg};{lr:.6f};{maxsr:.6f}".format(
            msg="is_duplicate",
            lr=length_ratio,
            maxsr=max_subsequence_ratio,
        )
        return False, rejection_message

    if len(template_sequence) < 10:
        rejection_message = f"template too short;{len(template_sequence)}"
        return False, rejection_message

    return True, ""


def _get_pdb_id_and_chain_id(template_hit: TemplateHit) -> Tuple[str, str]:
    """Get PDB id and chain id from HHSearch Hit."""
    # PDB ID: 4 letters. Chain ID: 1+ alphanumeric letters or "." if unknown.
    id_match = re.match(r"[a-zA-Z\d]{4}_[a-zA-Z0-9.]+", template_hit.name)
    if not id_match:
        # Fail hard if we can't get the PDB ID and chain name from the hit.
        raise ValueError(f"hit.name did not start with PDBID_chain: {template_hit.name}")
    pdb_id, chain_id = id_match.group(0).split("_")
    return pdb_id.lower(), chain_id


def _featurize_template_hit(
    template_hit: TemplateHit,
    query_sequence: str,
    pdb_mmcif_dicts_dirpath: Path,
    max_template_date: datetime.datetime,
    pdb_release_dates: Dict[str, datetime.datetime],
    pdb_obsolete_mapping: Dict[str, str],
    kalign_executable_path: str,
    realign_when_required: bool,
    verbose: bool,
) -> TemplateFeaturesResult:
    """Featurizes single HHSearch template hit.

    Args:
        template_hit: HHSearch template hit.
        query_sequence: Amino acid sequence of the query.
        pdb_mmcif_dicts_dirpath: Path to mmcif dicts directory (see `scripts/preprocess_pdb_mmcif.py`).
        max_template_date: Max template release date after which template hits are discarded.
        pdb_release_dates: Dictionary mapping from pdb codes to their release dates.
        pdb_obsolete_mapping: Mapping from obsolete pdb ids to new ids.
        kalign_executable_path: The path to a kalign executable used for template realignment.
        verbose: Whether to print relevant details.

    Returns:
        template_features_result: TemplateFeaturesResult

    """
    template_hit_pdb_chain = _get_pdb_id_and_chain_id(template_hit)
    template_hit_pdb_id = template_hit_pdb_chain[0]
    template_hit_chain_id = template_hit_pdb_chain[1]

    if template_hit_pdb_id not in pdb_release_dates:
        if template_hit_pdb_id in pdb_obsolete_mapping:
            template_hit_pdb_id = pdb_obsolete_mapping[template_hit_pdb_id]

    index_mapping = _build_query_to_hit_index_mapping(
        original_query_sequence=query_sequence,
        hit_query_sequence=template_hit.query,
        hit_sequence=template_hit.hit_sequence,
        indices_hit=template_hit.indices_hit,
        indices_query=template_hit.indices_query,
    )

    # The mapping is from the query to the actual hit sequence, so we need to
    # remove gaps (which regardless have a missing confidence score).
    template_hit_sequence = template_hit.hit_sequence.replace("-", "")

    mmcif_dict = load_mmcif_dict(
        mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
        pdb_id=template_hit_pdb_id,
    )

    template_hit_releaste_date = datetime_from_string(mmcif_dict["release_date"], "%Y-%m-%d")
    assert template_hit_releaste_date <= max_template_date

    try:
        template_features, realign_warning = _extract_template_features(
            mmcif_dict=mmcif_dict,
            index_mapping=index_mapping,
            query_sequence=query_sequence,
            template_sequence=template_hit_sequence,
            template_pdb_id=template_hit_pdb_id,
            template_chain_id=template_hit_chain_id,
            kalign_executable_path=kalign_executable_path,
            realign_when_required=realign_when_required,
            verbose=verbose,
        )

        # if we do not realign as per user request, template_features will be None
        if template_features is not None:
            template_features["template_sum_probs"] = np.array([template_hit.sum_probs], dtype=np.float32)
        # It is possible there were some errors when parsing the other chains in the
        # mmCIF file, but the template features for the chain we want were still
        # computed. In such case the mmCIF parsing errors are not relevant.
        return TemplateFeaturesResult(features=template_features, error=None, warning=realign_warning)
    except (
        NoChainsError,
        NoAtomDataInTemplateError,
        TemplateAtomMaskAllZerosError,
    ) as e:
        # These 3 errors indicate missing mmCIF experimental data rather than
        # a problem with the template search.
        error = "{type_e}: {str_e} ;" " sum_probs={sum_probs:.2f}," " rank_index={rank_index}".format(
            type_e=type(e),
            str_e=str(e),
            sum_probs=template_hit.sum_probs,
            rank_index=template_hit.index,
        )
        return TemplateFeaturesResult(features=None, error=error, warning=None)
    except QueryToTemplateAlignError as e:
        error = "{type_e}: {str_e} ;" " sum_probs={sum_probs:.2f}," " rank_index={rank_index}".format(
            type_e=type(e),
            str_e=str(e),
            sum_probs=template_hit.sum_probs,
            rank_index=template_hit.index,
        )
        return TemplateFeaturesResult(features=None, error=error, warning=None)


def _build_query_to_hit_index_mapping(
    original_query_sequence: str,
    hit_query_sequence: str,
    hit_sequence: str,
    indices_hit: Sequence[int],
    indices_query: Sequence[int],
) -> Dict[int, int]:
    """Gets mapping from indices in original query sequence to indices in the hit.

    `hit_query_sequence` and `hit_sequence` are two aligned sequences containing gap characters.
    `hit_query_sequence` contains only the part of the `original_query_sequence` that matched the hit.
    When interpreting the indices from the `.hhr`, we need to correct for this to recover a mapping
    from `original_query_sequence` to the `hit_sequence`.

    Args:
        original_query_sequence: String describing the original query sequence.
        hit_query_sequence: The portion of the original query sequence that is in the `.hhr` file.
        hit_sequence: The portion of the matched hit sequence that is in the `.hhr` file.
        indices_hit: The indices for each amino acid relative to the `hit_sequence`.
        indices_query: The indices for each amino acid relative to the original query sequence.

    Returns:
        index_mapping: Dictionary with indices in the `original_query_sequence` as keys
            and indices in the `hit_sequence` as values.

    """
    # If the hit is empty (no aligned residues), return empty mapping
    if not hit_query_sequence:
        return {}

    # Remove gaps and find the offset of hit.query relative to original query.
    hhsearch_query_sequence = hit_query_sequence.replace("-", "")
    hit_sequence = hit_sequence.replace("-", "")
    hhsearch_query_offset = original_query_sequence.find(hhsearch_query_sequence)

    # Index of -1 used for gap characters. Subtract the min index ignoring gaps.
    min_idx = min(x for x in indices_hit if x > -1)
    fixed_indices_hit = [x - min_idx if x > -1 else -1 for x in indices_hit]

    min_idx = min(x for x in indices_query if x > -1)
    fixed_indices_query = [x - min_idx if x > -1 else -1 for x in indices_query]

    # Zip the corrected indices, ignore case where both seqs have gap characters.
    index_mapping = {}
    for q_i, q_t in zip(fixed_indices_query, fixed_indices_hit):
        if q_t != -1 and q_i != -1:
            if q_t >= len(hit_sequence):
                continue
            elif q_i + hhsearch_query_offset >= len(original_query_sequence):
                continue
            index_mapping[q_i + hhsearch_query_offset] = q_t

    return index_mapping


def _extract_template_features(
    mmcif_dict: dict,
    index_mapping: Dict[int, int],
    query_sequence: str,
    template_sequence: str,
    template_pdb_id: str,
    template_chain_id: str,
    kalign_executable_path: str,
    realign_when_required: bool,
    verbose: bool,
) -> Tuple[dict, Optional[str]]:
    """Extracts template features from a single HHSearch hit.

    Args:
        mmcif_dict: mmcif dict representing the template (see `load_mmcif_dict`).
        index_mapping: Dictionary mapping indices in the query sequence
            to indices in the template sequence.
        query_sequence: String describing the amino acid sequence for the query protein.
        template_sequence: String describing the amino acid sequence for the template protein.
        template_pdb_id: PDB code for the template.
        template_chain_id: String ID describing which chain of the structure should be used.
        kalign_executable_path: The path to a kalign executable used for template realignment.
        verbose: Whether to print relevant details.

    Returns:
        A tuple with:
        * A dictionary containing the features derived from the template protein structure.
        * A warning message if the hit was realigned to the actual mmCIF sequence.
            Otherwise None.

    Raises:
        NoChainsError: If the `mmcif_dict` doesn't contain any chains.
        SequenceNotInTemplateError: If the given chain id / sequence can't
            be found in the `mmcif_dict`.
        QueryToTemplateAlignError: If the actual template in the mmCIF file
            can't be aligned to the query.
        NoAtomDataInTemplateError: If the `mmcif_dict` doesn't contain
            atom positions.
        TemplateAtomMaskAllZerosError: If the `mmcif_dict` doesn't have any
            unmasked residues.

    """
    template_id = f"{template_pdb_id}_{template_chain_id}"
    if not mmcif_dict["sequences"]:
        raise NoChainsError(f"empty mmcif_dict['sequences'] for {repr(template_id)}")

    try:
        seqres, chain_id, mapping_offset = _find_template_in_pdb(
            template_chain_id=template_chain_id,
            template_sequence=template_sequence,
            mmcif_dict=mmcif_dict,
        )
        # update `template_id`
        template_id = f"{template_pdb_id}_{chain_id}"
        warning = None
    except SequenceNotInTemplateError:
        # If PDB70 contains a different version of the template, we use the sequence
        # from the `mmcif_dict`.
        if not realign_when_required:
            # return empty features
            return None, None
        if verbose:
            logging.warning(
                "_extract_template_features:"
                f" The exact template_sequence={repr(template_sequence)}"
                f" was not found in {repr(template_id)}."
                f" Realigning the template to the actual sequence"
                f" via {repr(kalign_executable_path)}..."
            )
        chain_id = template_chain_id
        # This throws an exception if it fails to realign the hit.
        seqres, index_mapping = _realign_pdb_template_to_query(
            old_template_sequence=template_sequence,
            template_chain_id=template_chain_id,
            mmcif_dict=mmcif_dict,
            old_mapping=index_mapping,
            kalign_executable_path=kalign_executable_path,
            verbose=verbose,
        )
        if verbose:
            logging.info(
                "_extract_template_features: ...realigned" f" {repr(template_id)} to {repr(seqres)} successfully!"
            )
        # The template sequence changed.
        template_sequence = seqres
        # No mapping offset, the query is aligned to the actual sequence.
        mapping_offset = 0
        # set warning message:
        warning = (
            "Realignment Warning ;"
            f" realigned template ({repr(template_id)})"
            f" sequence={repr(template_sequence)} to {repr(seqres)}"
        )

    try:
        # Essentially set to infinity - we don't want to reject templates unless
        # they're really really bad.
        all_atom_positions, all_atom_mask = _get_atom_positions(
            mmcif_dict=mmcif_dict,
            auth_chain_id=chain_id,
            max_ca_ca_distance=150.0,
            zero_center=True,
        )
    except (CaDistanceError, KeyError) as e:
        raise NoAtomDataInTemplateError(f"Could not get atom data {repr(template_id)}: {str(e)}") from e

    all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])
    all_atom_mask = np.split(all_atom_mask, all_atom_mask.shape[0])

    output_template_sequence = []
    template_all_atom_positions = []
    template_all_atom_mask = []

    for _ in query_sequence:
        # Residues in the query_sequence that are not in the template_sequence:
        template_all_atom_positions.append(np.zeros((rc.ATOM_TYPE_NUM, 3)))
        template_all_atom_mask.append(np.zeros(rc.ATOM_TYPE_NUM))
        output_template_sequence.append("-")

    for k, v in index_mapping.items():
        template_index = v + mapping_offset
        template_all_atom_positions[k] = all_atom_positions[template_index][0]
        template_all_atom_mask[k] = all_atom_mask[template_index][0]
        output_template_sequence[k] = template_sequence[v]

    # Alanine (AA with the lowest number of atoms) has 5 atoms (C, CA, CB, N, O).
    if np.sum(template_all_atom_mask) < 5:
        rmin = min(index_mapping.values()) + mapping_offset
        rmax = max(index_mapping.values()) + mapping_offset
        raise TemplateAtomMaskAllZerosError(
            f"Template all atom mask was all zeros: {repr(template_id)}." f" Residue range: {rmin}-{rmax}"
        )

    output_template_sequence = "".join(output_template_sequence)

    template_aatype = rc.sequence_to_onehot(output_template_sequence, rc.HHBLITS_AA_TO_ID)

    template_features = {
        "template_domain_names": f"{template_id}",
        "template_sequence": output_template_sequence,
        "template_aatype": np.array(template_aatype, dtype=np.int32),
        "template_all_atom_positions": np.array(template_all_atom_positions, dtype=np.float32),
        "template_all_atom_mask": np.array(template_all_atom_mask, dtype=np.float32),
    }

    return template_features, warning


def _find_template_in_pdb(
    template_chain_id: str,
    template_sequence: str,
    mmcif_dict: dict,
) -> Tuple[str, str, int]:
    """Tries to find the template chain in the given pdb file (`mmcif_dict`).

    This method tries the three following things in order:
        1. Tries if there is an exact match in both the chain ID and the sequence.
             If yes, the chain sequence is returned. Otherwise:
        2. Tries if there is an exact match only in the sequence.
             If yes, the chain sequence is returned. Otherwise:
        3. Tries if there is a fuzzy match (X = wildcard) in the sequence.
             If yes, the chain sequence is returned.
    If none of these succeed, a SequenceNotInTemplateError is thrown.

    Args:
        template_chain_id: The template chain ID.
        template_sequence: The template chain sequence.
        mmcif_dict: The PDB object to search for the template in.

    Returns:
        A tuple with:
        * The chain sequence that was found to match the template in the PDB object.
        * The ID of the chain that is being returned.
        * The offset where the template sequence starts in the chain sequence.

    Raises:
        SequenceNotInTemplateError: If no match is found after the steps described above.

    """
    # Try if there is an exact match in both the chain ID and the (sub)sequence.
    chain_sequence = mmcif_dict["sequences"].get(template_chain_id)
    if chain_sequence and (template_sequence in chain_sequence):
        # Found an exact template match
        mapping_offset = chain_sequence.find(template_sequence)
        return chain_sequence, template_chain_id, mapping_offset

    # Try if there is an exact match in the (sub)sequence only.
    for chain_id, chain_sequence in mmcif_dict["sequences"].items():
        if chain_sequence and (template_sequence in chain_sequence):
            # Found a sequence-only match
            mapping_offset = chain_sequence.find(template_sequence)
            return chain_sequence, chain_id, mapping_offset

    # Return a chain sequence that fuzzy matches (X = wildcard) the template.
    # Make parentheses unnamed groups (?:_) to avoid the 100 named groups limit.
    regex = ["." if aa == "X" else f"(?:{aa}|X)" for aa in template_sequence]
    regex = re.compile("".join(regex))
    for chain_id, chain_sequence in mmcif_dict["sequences"].items():
        match = re.search(regex, chain_sequence)
        if match:
            # Found a fuzzy sequence-only match
            mapping_offset = match.start()
            return chain_sequence, chain_id, mapping_offset

    # No hits, raise an error.
    pdb_id = mmcif_dict["pdb_id"]
    template_pdb_chain_id = f"{pdb_id}_{template_chain_id}"
    raise SequenceNotInTemplateError(
        f"Could not find the template sequence in {repr(template_pdb_chain_id)}."
        f" template_sequence={repr(template_sequence)}"
        f" mmcif_dict['sequences']={repr(mmcif_dict['sequences'])}."
    )


def _get_atom_positions(
    mmcif_dict: dict,
    auth_chain_id: str,
    max_ca_ca_distance: float,
    zero_center: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gets atom positions and mask from a list of Biopython Residues."""
    all_atom_positions = mmcif_dict["atoms"][auth_chain_id]["all_atom_positions"]
    all_atom_mask = mmcif_dict["atoms"][auth_chain_id]["all_atom_mask"]
    if zero_center:
        all_atom_positions = zero_center_atom_positions(
            all_atom_positions=all_atom_positions,
            all_atom_mask=all_atom_mask,
        )
    all_atom_mask = all_atom_mask.astype(np.float32)
    _check_residue_distances(
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask,
        max_ca_ca_distance=max_ca_ca_distance,
    )
    return all_atom_positions, all_atom_mask


def _check_residue_distances(
    all_atom_positions: np.ndarray,
    all_atom_mask: np.ndarray,
    max_ca_ca_distance: float,
) -> None:
    """Checks if the distance between unmasked neighbor residues is ok."""
    ca_position = rc.ATOM_ORDER["CA"]
    prev_is_unmasked = False
    prev_calpha = None
    for i, (coords, mask) in enumerate(zip(all_atom_positions, all_atom_mask)):
        this_is_unmasked = bool(mask[ca_position])
        if this_is_unmasked:
            this_calpha = coords[ca_position]
            if prev_is_unmasked:
                distance = np.linalg.norm(this_calpha - prev_calpha)
                if distance > max_ca_ca_distance:
                    raise CaDistanceError(
                        f"The distance between residues {i} and {i + 1}"
                        f" is {distance:.6f} > limit {max_ca_ca_distance:.6f}."
                    )
            prev_calpha = this_calpha
        prev_is_unmasked = this_is_unmasked


def _realign_pdb_template_to_query(
    old_template_sequence: str,
    template_chain_id: str,
    mmcif_dict: dict,
    old_mapping: Dict[int, int],
    kalign_executable_path: str,
    verbose: bool,
) -> Tuple[str, Dict[int, int]]:
    """Aligns template from the `mmcif_dict` to the query.

    In case PDB70 contains a different version of the template sequence, we need
    to perform a realignment to the actual sequence that is in the mmCIF file.
    This method performs such realignment, but returns the new sequence and
    mapping only if the sequence in the mmCIF file is 90% identical to the old
    sequence.

    Note that the old_template_sequence comes from the hit, and contains only that
    part of the chain that matches with the query while the new_template_sequence
    is the full chain.

    Args:
        old_template_sequence: The template sequence that was returned by the PDB
            template search (typically done using HHSearch).
        template_chain_id: The template chain id was returned by the PDB template
            search (typically done using HHSearch). This is used to find the right
            chain in the `mmcif_dict` chain_to_seqres mapping.
        mmcif_dict: A mmcif dict which holds the actual template data.
        old_mapping: A mapping from the query sequence to the template sequence.
            This mapping will be used to compute the new mapping from the query
            sequence to the actual `mmcif_dict` template sequence by aligning the
            old_template_sequence and the actual template sequence.
        kalign_executable_path: The path to a kalign executable.

    Returns:
        A tuple (new_template_sequence, new_query_to_template_mapping) where:
        * new_template_sequence is the actual template sequence that was found in
            the `mmcif_dict`.
        * new_query_to_template_mapping is the new mapping from the query to the
            actual template found in the `mmcif_dict`.

    Raises:
        QueryToTemplateAlignError:
        * If there was an error thrown by the alignment tool.
        * Or if the actual template sequence differs by more than 10% from the
            old_template_sequence.

    """
    aligner = Kalign(binary_path=kalign_executable_path, verbose=verbose)
    new_template_sequence = mmcif_dict["sequences"].get(template_chain_id, "")

    # Sometimes the template chain id is unknown. But if there is only a single
    # sequence within the `mmcif_dict`, it is safe to assume it is that one.
    if not new_template_sequence:
        if len(mmcif_dict["sequences"]) == 1:
            # Could not find `template_chain_id` in `mmcif_dict["sequences"]`,
            # but there is only 1 sequence, so using that one.
            new_template_sequence = list(mmcif_dict["sequences"].values())[0]
        else:
            raise QueryToTemplateAlignError(
                f"Could not find chain {repr(template_chain_id)} in {repr(mmcif_dict['pdb_id'])}."
                " If there are no mmCIF parsing errors, it is possible it was not"
                " a protein chain."
            )

    try:
        (old_aligned_template, new_aligned_template), _ = parse_a3m(
            aligner.align([old_template_sequence, new_template_sequence])
        )
    except (RuntimeError, ValueError) as e:
        raise QueryToTemplateAlignError(
            "Could not align old template {ots} to template {nts} ({tid})." " {type_e}: {str_e}".format(
                ots=repr(old_template_sequence),
                nts=repr(new_template_sequence),
                tid=repr(mmcif_dict["pdb_id"] + "_" + template_chain_id),
                type_e=type(e),
                str_e=str(e),
            )
        )

    if verbose:
        logging.debug(
            f"_realign_pdb_template_to_query: old_aligned_template={repr(old_aligned_template)}\n"
            f"_realign_pdb_template_to_query: new_aligned_template={repr(new_aligned_template)}"
        )

    old_to_new_template_mapping = {}
    old_template_index = -1
    new_template_index = -1
    num_same = 0
    for old_template_aa, new_template_aa in zip(old_aligned_template, new_aligned_template):
        if old_template_aa != "-":
            old_template_index += 1
        if new_template_aa != "-":
            new_template_index += 1
        if old_template_aa != "-" and new_template_aa != "-":
            old_to_new_template_mapping[old_template_index] = new_template_index
            if old_template_aa == new_template_aa:
                num_same += 1

    # Require at least 90% sequence identity wrt to the shorter of the sequences.
    shorter_seqlen = min(len(old_template_sequence), len(new_template_sequence))
    similarity = num_same / shorter_seqlen
    if similarity < 0.9:
        raise QueryToTemplateAlignError(
            "Insufficient similarity {sim:.6f} of the sequence in the database:"
            " {ots} to the actual sequence in the mmCIF file {tid}: {nts}."
            " We require at least 90% similarity wrt to the shorter of the sequences."
            " This is not a problem unless you think this is a template that should be included.".format(
                sim=similarity,
                ots=repr(old_template_sequence),
                tid=repr(mmcif_dict["pdb_id"] + "_" + template_chain_id),
                nts=repr(new_template_sequence),
            )
        )

    new_query_to_template_mapping = {}
    for query_index, old_template_index in old_mapping.items():
        new_template_index = old_to_new_template_mapping.get(old_template_index, -1)
        new_query_to_template_mapping[query_index] = new_template_index

    new_template_sequence = new_template_sequence.replace("-", "")

    return new_template_sequence, new_query_to_template_mapping
