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


"""
Process raw data to data format that can be used by DSMBind.

How your raw data should be structured?
    In your dataset folder, each subfolder should contain a .pdb file and .sdf file, corresponding to the protein and ligand respectively. The subfolder name should start with the PDB id of the protein (otherwise, the code needs a minor change to get the correct PDB id from the subfolder name).

    A dataset structure example:
    # your_dataset/             # directory of raw data
    # ├── 4r0i_3KM_A/           # complex 1
    # │   ├── 4r0i_3KM_A_protein.pdb
    # │   └── 4r0i_3KM_A_ligand.sdf
    # ├── 5cpr_539_B/           # complex 2
    # │   ├── 5cpr_539_B_protein.pdb
    # │   └── 5cpr_539_B_ligand.sdf
    # ...
"""

import os
import pickle

import numpy as np
from Bio import PDB
from rdkit.Chem import PandasTools
from tqdm import tqdm

from bionemo.data.dsmbind.constants import ALPHABET, RES_ATOM14


def get_residue_atoms(residue, expected_atoms):
    """
    Get the coordinates of atoms for a given residue. Appended values are zeros.
    """
    coords = np.full((14, 3), 0, dtype=np.float32)  # Prepare a zero-filled array for coordinates
    atom_dict = {atom.get_name(): atom.get_coord() for atom in residue.get_atoms()}
    for i, atom_name in enumerate(expected_atoms):
        if atom_name in atom_dict:
            coords[i] = atom_dict[atom_name]
    return coords


def process_folder(folder: str):
    """
    A function to process one complex data.

    Args:
        folder (str): the folder path of the data.

    Returns:
        pdb (str): PDB id of the target.
        binder_mol (Mol): RDKit molecule representing the ligand.
        target_seq (str): the residue sequence of the target.
        target_coords (numpy.array): the coordinate array of the target atoms.
    """
    files = os.listdir(folder)
    pdb_file = [file for file in files if file.endswith(".pdb")][0]
    sdf_file = [file for file in files if file.endswith(".sdf")][0]
    pdb = os.path.basename(folder)[:4]

    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb, os.path.join(folder, pdb_file))
    model = structure[0]  # Assuming only one model in the PDB file

    # Collecting residue sequence and all atom coordinates
    sequence = []
    coordinates = []
    for residue in model.get_residues():
        resname = (
            PDB.Polypeptide.three_to_one(residue.resname)
            if residue.resname in PDB.Polypeptide.standard_aa_names
            else "#"
        )
        res_index = ALPHABET.index(resname) if resname in ALPHABET else 0
        expected_atoms = RES_ATOM14[res_index]
        sequence.append(resname)
        res_coords = get_residue_atoms(residue, expected_atoms)
        coordinates.append(res_coords)

    target_seq = "".join(sequence)
    target_coords = np.array(coordinates)  # Shape will be (len(target_seq), 14, 3)

    # Load and process the ligand
    df = PandasTools.LoadSDF(os.path.join(folder, sdf_file), molColName="Molecule", includeFingerprints=False)
    try:
        mol = df["Molecule"][0]
        if mol.GetNumHeavyAtoms() < 3:
            return None
    except Exception as e:
        print(e)
        return None

    return {
        "pdb": pdb,
        "binder_mol": mol,
        "target_seq": target_seq,
        "target_coords": target_coords,
    }


def preprocess(raw_data_dir: str):
    """
    A function for data preprocessing. The processed data will be saved under the same directory.

    Args:
        raw_data_dir (str): the directory to the raw data.
    """
    data = []
    folder_names = [f.path for f in os.scandir(raw_data_dir) if f.is_dir()]
    for folder in tqdm(folder_names):
        # Skip folders with missing files
        if len(os.listdir(folder)) < 2:
            continue
        entry = process_folder(folder)
        if entry is not None:
            data.append(entry)

    with open(os.path.join(raw_data_dir, "processed.pkl"), "wb") as f:
        pickle.dump(data, f)
