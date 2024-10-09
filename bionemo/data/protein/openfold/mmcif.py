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
import gzip
import io
import math
import pickle
import zlib
from collections import defaultdict, namedtuple
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import PDB
from Bio.Data import SCOPData
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.Structure import Structure as PDBStructure
from nemo.utils import logging

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.data.protein.openfold.helpers import list_zip


# internal types:
_EntityId = str
_MMCIFChainId = str
_AuthorChainId = str
_ChainIdsMapping = Dict[_MMCIFChainId, _AuthorChainId]
_Monomer = namedtuple("Monomer", ["num", "id"])
_Polymer = List[_Monomer]
_LegalPolymers = Dict[_MMCIFChainId, _Polymer]
_AtomSiteList = List[dict]
_Sequences = Dict[_AuthorChainId, str]
_AuthorChainIds = List[_AuthorChainId]
_EntityIdToChainIds = Dict[_EntityId, List[_MMCIFChainId]]
_ResidueIndex = int
_ResidueKey = Tuple[str, int, str]
_ResidueKeys = Dict[_AuthorChainId, Dict[_ResidueIndex, _ResidueKey]]
_AtomsNumpy = Dict[_AuthorChainId, Dict[str, np.ndarray]]
_AtomsCompressed = Dict[_AuthorChainId, Dict[str, Tuple[bytearray, tuple]]]


def load_mmcif_file(mmcif_filepath: Path) -> str:
    """Load `.cif` file into mmcif string."""
    if not isinstance(mmcif_filepath, Path):
        raise TypeError(f"mmcif_filepath should be of type {Path}," f" but is of type {type(mmcif_filepath)}")
    assert mmcif_filepath.suffix == ".cif"
    with open(mmcif_filepath, "r") as f:
        mmcif_string = f.read()
    return mmcif_string


def load_mmcif_gz_file(mmcif_gz_filepath: Path) -> str:
    """Load `.cif.gz` file into mmcif string."""
    if not isinstance(mmcif_gz_filepath, Path):
        raise TypeError(f"mmcif_gz_filepath should be of type {Path}," f" but is of type {type(mmcif_gz_filepath)}")
    assert "".join(mmcif_gz_filepath.suffixes) == ".cif.gz"
    with gzip.open(mmcif_gz_filepath, "rb") as f:
        mmcif_bytes = f.read()
    mmcif_string = mmcif_bytes.decode("utf-8")
    return mmcif_string


def load_mmcif_dict(mmcif_dicts_dirpath: Path, pdb_id: str) -> dict:
    """Load mmcif dict for `pdb_id` created by `scripts/preprocess_pdb_mmcif.py`."""
    mmcif_dicts_filename = pdb_id[1:3]
    mmcif_dicts_filepath = mmcif_dicts_dirpath / mmcif_dicts_filename
    with open(mmcif_dicts_filepath, "rb") as f:
        mmcif_dicts = pickle.load(f)
    mmcif_dict = mmcif_dicts[pdb_id]
    mmcif_dict["atoms"] = decompress_mmcif_dict_atoms(mmcif_dict["atoms"])
    return mmcif_dict


def load_mmcif_chains_df(
    mmcif_chains_filepath: Path,
    verbose: bool = False,
    pprefix: str = "",
) -> pd.DataFrame:
    """Load mmcif chains CSV file created by `scripts/preprocess_pdb_mmcif.py`."""
    if verbose:
        logging.debug(f"{pprefix}Loading {repr(mmcif_chains_filepath)}...")
    mmcif_chains_df = pd.read_csv(
        mmcif_chains_filepath,
        na_values="",
        keep_default_na=False,
    )
    if verbose:
        logging.debug(
            f"{pprefix}mmcif_chains_df {mmcif_chains_df.shape} loaded from {mmcif_chains_filepath} successfully!"
        )
    return mmcif_chains_df


def parse_mmcif_string(mmcif_string: str) -> dict:
    """Parse mmcif string into mmcif dict."""
    mmcif_parser = PDB.MMCIFParser(QUIET=True)
    handle = io.StringIO(mmcif_string)
    full_structure = mmcif_parser.get_structure(None, handle)
    mmcif_parser_dict = mmcif_parser._mmcif_dict
    pdb_id = _get_pdb_id(mmcif_parser_dict)
    exptl_method = _get_exptl_method(mmcif_parser_dict)
    release_date = _get_release_date(mmcif_parser_dict)
    resolution = _get_resolution(mmcif_parser_dict)
    entity_id_to_mmcif_chain_ids: _EntityIdToChainIds = _get_entity_id_to_chain_ids(mmcif_parser_dict)
    legal_polymers: _LegalPolymers = _get_legal_polymers(mmcif_parser_dict, entity_id_to_mmcif_chain_ids)
    if not legal_polymers:
        raise RuntimeError("legal_polymers are empty")
    atom_site_list: _AtomSiteList = _get_atom_site_list(mmcif_parser_dict)
    atom_site_list = _filter_atom_site_list(atom_site_list)
    chain_ids_mapping: _ChainIdsMapping = _get_chain_ids_mapping(atom_site_list)
    sequences: _Sequences = _get_sequences(legal_polymers, chain_ids_mapping)
    residue_keys: _ResidueKeys = _get_residue_keys(legal_polymers, atom_site_list, chain_ids_mapping)
    author_chain_ids: _AuthorChainIds = list(sequences.keys())
    first_model: PDBModel = _get_first_model(full_structure)
    atoms: _AtomsNumpy = _get_atoms(first_model, author_chain_ids, residue_keys)
    _assert_sequences_and_atoms(sequences, atoms)
    mmcif_dict = {
        "pdb_id": pdb_id,
        "exptl_method": exptl_method,
        "release_date": release_date,
        "resolution": resolution,
        "author_chain_ids": author_chain_ids,
        "entity_id_to_mmcif_chain_ids": entity_id_to_mmcif_chain_ids,
        "mmcif_chain_id_to_author_chain_id": chain_ids_mapping,
        "sequences": sequences,
        "atoms": atoms,
    }
    return mmcif_dict


def compress_mmcif_dict_atoms(atoms: _AtomsNumpy) -> _AtomsCompressed:
    atoms_compressed = {}
    for chain_id in atoms.keys():
        all_atom_positions_array = atoms[chain_id]["all_atom_positions"]
        all_atom_mask_array = atoms[chain_id]["all_atom_mask"]
        all_atom_positions_bytearray = all_atom_positions_array.tobytes()
        all_atom_mask_bytearray = all_atom_mask_array.tobytes()
        assert all_atom_positions_array.dtype == np.float32
        assert all_atom_mask_array.dtype == bool
        all_atom_positions_compressed = zlib.compress(all_atom_positions_bytearray)
        all_atom_mask_compressed = zlib.compress(all_atom_mask_bytearray)
        all_atom_positions_shape = all_atom_positions_array.shape
        all_atom_mask_shape = all_atom_mask_array.shape
        atoms_compressed[chain_id] = {
            "all_atom_positions": (
                all_atom_positions_compressed,
                all_atom_positions_shape,
            ),
            "all_atom_mask": (all_atom_mask_compressed, all_atom_mask_shape),
        }
    return atoms_compressed


def decompress_mmcif_dict_atoms(atoms: _AtomsCompressed) -> _AtomsNumpy:
    atoms_decompressed = {}
    for chain_id in atoms.keys():
        all_atom_positions = atoms[chain_id]["all_atom_positions"]
        all_atom_positions_compressed, all_atom_positions_shape = all_atom_positions
        all_atom_mask_compressed, all_atom_mask_shape = atoms[chain_id]["all_atom_mask"]
        all_atom_positions_bytearray = zlib.decompress(all_atom_positions_compressed)
        all_atom_mask_bytearray = zlib.decompress(all_atom_mask_compressed)
        all_atom_positions_array = np.frombuffer(
            buffer=all_atom_positions_bytearray,
            dtype=np.float32,
        )
        all_atom_mask_array = np.frombuffer(
            buffer=all_atom_mask_bytearray,
            dtype=bool,
        )
        all_atom_positions_array = all_atom_positions_array.reshape(all_atom_positions_shape).copy()
        all_atom_mask_array = all_atom_mask_array.reshape(all_atom_mask_shape).copy()
        atoms_decompressed[chain_id] = {
            "all_atom_positions": all_atom_positions_array,
            "all_atom_mask": all_atom_mask_array,
        }
    return atoms_decompressed


def zero_center_atom_positions(
    all_atom_positions: np.ndarray,
    all_atom_mask: np.ndarray,
) -> np.ndarray:
    all_atom_positions = all_atom_positions.copy()
    translation = all_atom_positions[all_atom_mask].mean(axis=0)
    all_atom_positions[all_atom_mask] -= translation
    return all_atom_positions


def _get_pdb_id(mmcif_parser_dict: dict) -> str:
    entry_id = mmcif_parser_dict["_entry.id"]
    assert isinstance(entry_id, list)
    assert len(entry_id) == 1
    pdb_id = entry_id[0]
    assert len(pdb_id) == 4
    pdb_id = pdb_id.lower()
    return pdb_id


def _get_exptl_method(mmcif_parser_dict: dict) -> str:
    exptl_method_list = mmcif_parser_dict["_exptl.method"]
    assert isinstance(exptl_method_list, list)
    assert len(exptl_method_list) > 0
    if len(exptl_method_list) == 1:
        exptl_method = exptl_method_list[0]
    elif len(exptl_method_list) > 1:
        exptl_method = ";".join(exptl_method_list)
    return exptl_method


def _get_release_date(mmcif_parser_dict: dict) -> str:
    revision_dates = mmcif_parser_dict["_pdbx_audit_revision_history.revision_date"]
    assert isinstance(revision_dates, list)
    assert len(revision_dates) > 0
    for revision_date in revision_dates:
        datetime.datetime.strptime(revision_date, "%Y-%m-%d")
    release_date = min(revision_dates)
    return release_date


def _get_resolution(mmcif_parser_dict: dict) -> float:
    resolutions = []

    if "_refine.ls_d_res_high" in mmcif_parser_dict:
        resolutions = mmcif_parser_dict["_refine.ls_d_res_high"]
    elif "_em_3d_reconstruction.resolution" in mmcif_parser_dict:
        resolutions = mmcif_parser_dict["_em_3d_reconstruction.resolution"]
    elif "_reflns.d_resolution_high" in mmcif_parser_dict:
        resolutions = mmcif_parser_dict["_reflns.d_resolution_high"]

    assert isinstance(resolutions, list)

    def _parse_resolution(resolution: str) -> float:
        try:
            return float(resolution)
        except ValueError:
            return float("nan")

    resolutions = [_parse_resolution(resolution) for resolution in resolutions]
    resolutions = [resolution for resolution in resolutions if not math.isnan(resolution)]

    if len(resolutions) == 1:
        return resolutions[0]
    elif len(resolutions) > 1:
        max_resolution = max(resolutions)
        return max_resolution

    return float("nan")


def _get_entity_id_to_chain_ids(mmcif_parser_dict: dict) -> _EntityIdToChainIds:
    # Create mapping: entity id -> mmcif chain ids
    entity_id_to_mmcif_chain_ids: _EntityIdToChainIds = defaultdict(list)
    struct_asym = list_zip(
        mmcif_parser_dict["_struct_asym.id"],
        mmcif_parser_dict["_struct_asym.entity_id"],
    )
    for mmcif_chain_id, entity_id in struct_asym:
        entity_id_to_mmcif_chain_ids[entity_id].append(mmcif_chain_id)
    return dict(entity_id_to_mmcif_chain_ids)


def _get_legal_polymers(
    mmcif_parser_dict: dict,
    entity_id_to_mmcif_chain_ids: _EntityIdToChainIds,
) -> _LegalPolymers:
    # Group polymer information for each entity in the structure
    polymers: Dict[str, _Polymer] = defaultdict(list)
    entity_poly_seq = list_zip(
        mmcif_parser_dict["_entity_poly_seq.entity_id"],
        mmcif_parser_dict["_entity_poly_seq.num"],
        mmcif_parser_dict["_entity_poly_seq.mon_id"],
    )
    for entity_id, num, mon_id in entity_poly_seq:
        monomer = _Monomer(id=mon_id, num=int(num))
        polymers[entity_id].append(monomer)

    # Create mapping: monomer id -> monomer type
    chem_comp = list_zip(
        mmcif_parser_dict["_chem_comp.id"],
        mmcif_parser_dict["_chem_comp.type"],
    )
    chem_types = dict(chem_comp)

    # Identify and return the correct protein chains (polymers)
    legal_polymers: _LegalPolymers = {}
    for entity_id, polymer in polymers.items():
        # Legal polymer should have at least one peptide-like component/monomer
        if any("peptide" in chem_types[monomer.id] for monomer in polymer):
            mmcif_chain_ids = entity_id_to_mmcif_chain_ids[entity_id]
            for mmcif_chain_id in mmcif_chain_ids:
                assert mmcif_chain_id not in legal_polymers
                legal_polymers[mmcif_chain_id] = deepcopy(polymer)

    # Return legal polymers grouped by chain id
    return legal_polymers


def _get_atom_site_list(mmcif_parser_dict: dict) -> _AtomSiteList:
    atom_site_tuple_list = list_zip(
        mmcif_parser_dict["_atom_site.label_comp_id"],  # residue_name
        mmcif_parser_dict["_atom_site.auth_asym_id"],  # author_chain_id
        mmcif_parser_dict["_atom_site.label_asym_id"],  # mmcif_chain_id
        mmcif_parser_dict["_atom_site.auth_seq_id"],  # author_seq_num
        mmcif_parser_dict["_atom_site.label_seq_id"],  # mmcif_seq_num
        mmcif_parser_dict["_atom_site.pdbx_PDB_ins_code"],  # insertion_code
        mmcif_parser_dict["_atom_site.group_PDB"],  # hetatm_atom
        mmcif_parser_dict["_atom_site.pdbx_PDB_model_num"],  # model_num
    )
    atom_site_list: _AtomSiteList = [
        {
            "residue_name": atom_site_tuple[0],
            "author_chain_id": atom_site_tuple[1],
            "mmcif_chain_id": atom_site_tuple[2],
            "author_seq_num": atom_site_tuple[3],
            "mmcif_seq_num": atom_site_tuple[4],
            "insertion_code": atom_site_tuple[5],
            "hetatm_atom": atom_site_tuple[6],
            "model_num": atom_site_tuple[7],
        }
        for atom_site_tuple in atom_site_tuple_list
    ]
    return atom_site_list


def _filter_atom_site_list(atom_site_list: _AtomSiteList) -> _AtomSiteList:
    # because only "first_model" is used
    filtered_atom_site_list = [atom_site for atom_site in atom_site_list if atom_site["model_num"] == "1"]
    return filtered_atom_site_list


def _get_chain_ids_mapping(atom_site_list: _AtomSiteList) -> _ChainIdsMapping:
    chain_ids_mapping: _ChainIdsMapping = {}
    for atom_site in atom_site_list:
        author_chain_id: _AuthorChainId = atom_site["author_chain_id"]
        mmcif_chain_id: _MMCIFChainId = atom_site["mmcif_chain_id"]
        if mmcif_chain_id in chain_ids_mapping:
            assert chain_ids_mapping[mmcif_chain_id] == author_chain_id
        else:
            chain_ids_mapping[mmcif_chain_id] = author_chain_id
    # Return mapping from internal mmCIF chain ids
    # to chain ids used by the authors / Biopython.
    return chain_ids_mapping


def _get_sequences(
    legal_polymers: _LegalPolymers,
    chain_ids_mapping: _ChainIdsMapping,
) -> _Sequences:
    sequences: _Sequences = {}
    for mmcif_chain_id, polymer in legal_polymers.items():
        author_chain_id = chain_ids_mapping[mmcif_chain_id]
        sequence = []
        for monomer in polymer:
            code = SCOPData.protein_letters_3to1.get(monomer.id, "X")
            sequence.append(code if len(code) == 1 else "X")
        sequence = "".join(sequence)
        assert author_chain_id not in sequences
        sequences[author_chain_id] = sequence
    return sequences


def _get_residue_keys(
    legal_polymers: _LegalPolymers,
    atom_site_list: _AtomSiteList,
    chain_ids_mapping: _ChainIdsMapping,
) -> _ResidueKeys:
    residue_keys: _ResidueKeys = {}
    seq_start_num = {
        mmcif_chain_id: min(monomer.num for monomer in polymer) for mmcif_chain_id, polymer in legal_polymers.items()
    }
    for atom_site in atom_site_list:
        residue_name = atom_site["residue_name"]
        author_chain_id = atom_site["author_chain_id"]
        mmcif_chain_id = atom_site["mmcif_chain_id"]
        author_seq_num = atom_site["author_seq_num"]
        mmcif_seq_num = atom_site["mmcif_seq_num"]
        insertion_code = atom_site["insertion_code"]
        hetatm_atom = atom_site["hetatm_atom"]

        if mmcif_chain_id in legal_polymers:
            hetflag = " "
            if hetatm_atom == "HETATM":
                # Water atoms are assigned a special hetflag of 'W' in Biopython.
                # We need to do the same, so that this hetflag can be used to fetch
                # a residue from the Biopython structure by id.
                if residue_name in ("HOH", "WAT"):
                    hetflag = "W"
                else:
                    hetflag = "H_" + residue_name

            if insertion_code in (".", "?"):  # insertion code unset
                insertion_code = " "

            if author_chain_id not in residue_keys:
                residue_keys[author_chain_id] = {}

            residue_index: _ResidueIndex = int(mmcif_seq_num) - seq_start_num[mmcif_chain_id]
            residue_key: _ResidueKey = (hetflag, int(author_seq_num), insertion_code)

            # The original code overrides existing `residue_key` breezily,
            # even if the incoming value is different that the previous one.
            # https://github.com/deepmind/alphafold/blob/v2.2.2/alphafold/data/mmcif_parsing.py#L239
            # This behaviour is also present here.
            residue_keys[author_chain_id][residue_index] = residue_key

    # Fill missing information in residue keys
    for mmcif_chain_id, polymer in legal_polymers.items():
        author_chain_id = chain_ids_mapping[mmcif_chain_id]
        for residue_index, monomer in enumerate(polymer):
            if residue_index not in residue_keys[author_chain_id]:
                residue_keys[author_chain_id][residue_index] = None

    return residue_keys


def _get_first_model(structure: PDBStructure) -> PDBModel:
    # Return the first model in a Biopython structure
    return next(structure.get_models())


def _get_atoms(
    first_model: PDBModel,
    author_chain_ids: List[_AuthorChainId],
    residue_keys: _ResidueKeys,
) -> _AtomsNumpy:
    atoms: _AtomsNumpy = {}
    for author_chain_id in author_chain_ids:
        chains = list(first_model.get_chains())
        chains = [chain for chain in chains if chain.id == author_chain_id]
        assert len(chains) == 1
        chain = chains[0]
        num_residues = len(residue_keys[author_chain_id])
        all_atom_positions = np.zeros([num_residues, rc.ATOM_TYPE_NUM, 3], dtype=np.float32)
        all_atom_mask = np.zeros([num_residues, rc.ATOM_TYPE_NUM], dtype=bool)
        for residue_index in range(num_residues):
            residue_key = residue_keys[author_chain_id][residue_index]
            if residue_key is not None:
                residue = chain[residue_key]
                for atom in residue.get_atoms():
                    name = atom.get_name()
                    x, y, z = atom.get_coord()
                    if name in rc.ATOM_ORDER:
                        atom_index = rc.ATOM_ORDER[name]
                        all_atom_positions[residue_index, atom_index] = (x, y, z)
                        all_atom_mask[residue_index, atom_index] = True
                    elif name.upper() == "SE" and residue.get_resname() == "MSE":
                        # Put the coords of the selenium atom in the sulphur column
                        atom_index = rc.ATOM_ORDER["SD"]
                        all_atom_positions[residue_index, atom_index] = (x, y, z)
                        all_atom_mask[residue_index, atom_index] = True
        atoms[author_chain_id] = {
            "all_atom_positions": all_atom_positions,
            "all_atom_mask": all_atom_mask,
        }
    return atoms


def _assert_sequences_and_atoms(sequences: _Sequences, atoms: _AtomsNumpy) -> None:
    assert sequences.keys() == atoms.keys()
    for author_chain_id in sequences.keys():
        seqlen = len(sequences[author_chain_id])
        for array_name in atoms[author_chain_id].keys():
            assert seqlen == len(atoms[author_chain_id][array_name])
