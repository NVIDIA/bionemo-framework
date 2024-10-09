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


"""The dataset class and customized collate function for preparing dataset that can be handled by DSMBind."""

import pickle
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from bionemo.data.dsmbind.constants import ALPHABET, ATOM_TYPES, RES_ATOM14


def featurize_tgt(batch, vocab=ALPHABET):
    """
    A function for preparing target pocket info.

    Args:
        batch (List[Dict]): a list of data (Dict) as constructed by the __init__ function of the DSMBindDataset class.

    Returns:
       X (torch.Tensor): a tensor with shape (batch_size, max_residue_length_in_batch, 14, 3), which represents the target atom coordinates.
       S (torch.Tensor): a tensor with shape (batch_size, max_residue_length_in_batch), which represents the residue types.
       A (torch.Tensor): a tensor with shape (batch_size, max_residue_length_in_batch, 14), which represents the target atom types.
    """
    B = len(batch)
    L_max = max([len(b["pocket_seq"]) for b in batch])
    X = torch.zeros([B, L_max, 14, 3])
    S = torch.zeros([B, L_max]).long()
    A = torch.zeros([B, L_max, 14]).long()

    # Build the batch
    for i, b in enumerate(batch):
        l = len(b["pocket_seq"])
        indices = torch.tensor([vocab.index(a) for a in b["pocket_seq"]])
        S[i, :l] = indices
        X[i, :l] = b["pocket_coords"]
        A[i, :l] = b["pocket_atypes"]

    return X, S, A


class DSMBindDataset(Dataset):
    def __init__(
        self,
        processed_data_path: str,
        aa_size: int,
        max_residue_atoms: int = 14,
        patch_size: int = 50,
        affinity_label: bool = False,
    ):
        """
        A dataset class for DSMBind training.

        Args:
            processed_data_path (str): the path to the processed data file.
            aa_size (int): number of residue types.
            max_residue_atoms (int): maximum number of atoms of residues.
            patch_size (int): number of residues to be considered as in pocket.
            affinity_label (bool): if affinity labels are provided.
        """
        with open(processed_data_path, "rb") as f:
            data = pickle.load(f)

        self.data = []
        self.aa_size = aa_size
        self.max_residue_atoms = max_residue_atoms
        self.affinity_label = affinity_label
        for entry in tqdm(data):
            entry["target_coords"] = torch.tensor(entry["target_coords"]).float()
            entry["target_atypes"] = torch.tensor(
                [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry["target_seq"]]
            )
            mol = entry["binder_mol"]
            conf = mol.GetConformer()
            coords = [conf.GetAtomPosition(i) for i, atom in enumerate(mol.GetAtoms())]
            entry["binder_coords"] = torch.tensor([[p.x, p.y, p.z] for p in coords]).float()
            # make pocket
            dist = entry["target_coords"][:, 1] - entry["binder_coords"].mean(dim=0, keepdims=True)
            entry["pocket_idx"] = idx = dist.norm(dim=-1).sort().indices[:patch_size].sort().values
            entry["pocket_seq"] = "".join([entry["target_seq"][i] for i in idx.tolist()])
            entry["pocket_coords"] = entry["target_coords"][idx]
            entry["pocket_atypes"] = entry["target_atypes"][idx]
            if affinity_label:
                entry["affinity"] = torch.tensor(entry["affinity"])
            self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pl_collate_fn(self, batch: List[Dict]):
        """
        Custom collate function for handling batches of protein-ligand data for DSMBind model.

        Args:
            batch (List[Dict]): a list of data (Dict) as constructed by the __init__ function.

        Returns:
            batched_binder (Tuple[torch.Tensor, List[Mol], torch.Tensor]): the batched ligand info. The first tensor is the ligand atom coordinates. The second list is a list of RDKit molecules. The third tensor is a mask for indicating ligand atoms.
            batched_target (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the batched target pocket info. The first tensor is the target atom coordinates. The second tensor is a one-hot residue embedding. The third tensor represents all target atoms in the pocket.
            affinity (Optional[torch.Tensor]): the batched affinity values.
        """
        mols = [entry["binder_mol"] for entry in batch]
        N = max([mol.GetNumAtoms() for mol in mols])
        bind_X = torch.zeros([len(batch), N, self.max_residue_atoms, 3])
        bind_A = torch.zeros([len(batch), N, self.max_residue_atoms]).long()
        tgt_X, tgt_S, tgt_A = featurize_tgt(batch)
        tgt_S = torch.zeros([tgt_S.size(0), tgt_S.size(1), self.aa_size])
        for i, b in enumerate(batch):
            L = b["binder_mol"].GetNumAtoms()
            bind_X[i, :L, 1, :] = b["binder_coords"]
            bind_A[i, :L, 1] = 1
            L = len(b["pocket_seq"])
            residue_embedding = torch.zeros(len(b["pocket_seq"]), self.aa_size)
            for j, aa in enumerate(b["pocket_seq"]):
                residue_embedding[j, ALPHABET.index(aa)] = 1  # One-hot embedding for residue
            tgt_S[i, :L] = residue_embedding
        batched_binder = (bind_X, mols, bind_A)
        batched_target = (tgt_X, tgt_S, tgt_A)
        if self.affinity_label:
            affinity = torch.stack([entry["affinity"] for entry in batch])
            return batched_binder, batched_target, affinity
        else:
            return batched_binder, batched_target
